import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c11_mission as C11
import continuous_prm_c12_refiner as C12


def _toy_graph():
    # Forward chain 0 -> 1 -> 2 -> 3.  Value information must move in the
    # reverse direction, from a destination to its predecessor/source.
    node_feats = torch.zeros(4, 14)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_feats = torch.zeros(3, 3)
    h_legsum = torch.zeros(4)
    return node_feats, edge_index, edge_feats, h_legsum


def test_c12_config_does_not_mutate_c11_defaults():
    before = C11.C11MissionConfig()
    cfg = C12.C12RefinerConfig()
    after = C11.C11MissionConfig()
    assert cfg.k_values == (2, 8, 16)
    assert cfg.k_max == 16
    assert before.k_values == after.k_values == (0, 2, 4, 8)
    assert before.k_max == after.k_max == 8


def test_c12_cell_grid_is_only_a_and_c():
    cells = C12.build_cell_grid()
    assert [(c["config_label"], c["K"]) for c in cells] == [
        ("A", 2), ("A", 8), ("A", 16),
        ("C", 2), ("C", 8), ("C", 16),
    ]


def test_seed_streams_are_deterministic_disjoint_and_reuse_c11_test_train():
    seen = set()
    for split in C12.SPLITS:
        for config_idx in (0, 2):
            for K in (2, 8, 16):
                values = [C12.world_seed(split, w, config_idx, K) for w in range(20)]
                assert values == [C12.world_seed(split, w, config_idx, K) for w in range(20)]
                assert not (seen & set(values))
                seen.update(values)
    for config_idx in (0, 2):
        for K in (2, 8):
            for w in range(20):
                assert C12.world_seed("train", w, config_idx, K) == C11.train_seed(w, config_idx, K)
                assert C12.world_seed("test", w, config_idx, K) == C11.test_seed(w, config_idx, K)


def test_reverse_aggregation_moves_destination_values_to_sources():
    _, edge_index, _, _ = _toy_graph()
    values = torch.tensor([[0.0], [0.0], [0.0], [7.0]])
    one = C12.reverse_mean_aggregate(values, edge_index)
    assert one[:, 0].tolist() == [0.0, 0.0, 7.0, 0.0]
    two = C12.reverse_mean_aggregate(one, edge_index)
    assert two[:, 0].tolist() == [0.0, 7.0, 0.0, 0.0]


def test_final_transition_distance_uses_forward_product_hops():
    # N=2, K=2. Edges: start flat 0 -> 1 -> 2 -> 4.  The last edge enters
    # stage K, so its source (flat 2) is two hops from the start.
    edge_index = np.asarray([[0, 1, 2, 4], [1, 2, 4, 5]], dtype=np.int64)
    assert C12.final_transition_distance(edge_index, n_roadmap_nodes=2, K=2) == 2
    disconnected = np.asarray([[1, 2], [2, 4]], dtype=np.int64)
    assert np.isinf(C12.final_transition_distance(disconnected, n_roadmap_nodes=2, K=2))


def test_g0b_gate_boundary_conditions():
    cfg = C12.C12RefinerConfig()
    passing = C12.evaluate_g0b_cell(
        valid_worlds=cfg.g0_min_worlds,
        expansion_ratio=cfg.g0_max_expansion_ratio,
        median_final_transition_hops=cfg.g0_min_median_hops + 1e-6,
        max_label_wall_s=cfg.g0_max_label_wall_s,
        max_peak_rss_bytes=cfg.g0_max_peak_rss_bytes,
        max_graph_bytes=cfg.g0_max_graph_bytes,
        degenerate_budget=False,
        cfg=cfg,
    )
    assert passing["passed"] is True
    assert all(passing["checks"].values())

    too_shallow = C12.evaluate_g0b_cell(
        valid_worlds=cfg.g0_min_worlds,
        expansion_ratio=0.1,
        median_final_transition_hops=cfg.g0_min_median_hops,
        max_label_wall_s=1.0,
        max_peak_rss_bytes=1,
        max_graph_bytes=1,
        degenerate_budget=False,
        cfg=cfg,
    )
    assert too_shallow["passed"] is False
    assert too_shallow["checks"]["deep_enough"] is False


def test_tied_refiner_outputs_registered_cycles_and_reuses_one_block():
    cfg = C12.C12RefinerConfig(gnn_hidden=16)
    model = C12.TiedGraphRefiner(cfg)
    x, edges, edge_feats, hl = _toy_graph()
    outputs = model(x, edges, edge_feats, hl)
    assert tuple(outputs) == cfg.report_cycles
    assert all(v.shape == (4,) for v in outputs.values())
    assert len([m for m in model.modules() if isinstance(m, C12.SharedGraphBlock)]) == 1


def test_shallow_is_parameter_matched_and_untied_is_compute_matched():
    cfg = C12.C12RefinerConfig(gnn_hidden=16)
    tied = C12.TiedGraphRefiner(cfg)
    shallow = C12.ShallowParamMatch(cfg)
    untied = C12.UntiedComputeMatch(cfg)
    assert C12.parameter_count(tied) == C12.parameter_count(shallow)
    assert tied.edge_applications == shallow.edge_applications * cfg.refinement_cycles
    assert tied.edge_applications == untied.edge_applications
    assert len(untied.blocks) == cfg.refinement_cycles
    assert len({id(block) for block in untied.blocks}) == cfg.refinement_cycles


def test_cycle_one_matches_explicit_one_cycle_and_cycle_eight_backprops():
    cfg = C12.C12RefinerConfig(gnn_hidden=16)
    torch.manual_seed(4)
    model = C12.TiedGraphRefiner(cfg)
    x, edges, edge_feats, hl = _toy_graph()
    all_outputs = model(x, edges, edge_feats, hl)
    one = model(x, edges, edge_feats, hl, max_cycles=1)
    assert torch.equal(all_outputs[1], one[1])
    loss = all_outputs[8].sum()
    loss.backward()
    assert all(p.grad is not None for p in model.block.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.block.parameters()) > 0.0


def test_deep_supervision_weights_are_frozen_and_normalized():
    cfg = C12.C12RefinerConfig()
    assert cfg.deep_supervision_weights == {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4}
    assert sum(cfg.deep_supervision_weights.values()) == pytest.approx(1.0)


def test_c12b_verdict_truth_table():
    assert C12.c12b_verdict(g0=True, g1=True, g2=True)["code"] == "shared_refinement_positive"
    assert C12.c12b_verdict(g0=True, g1=True, g2=False)["code"] == "propagation_only"
    assert C12.c12b_verdict(g0=True, g1=False, g2=False)["code"] == "no_progressive_refinement"
    assert C12.c12b_verdict(g0=False, g1=False, g2=False)["code"] == "not_authorized"
