import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c12_refiner as C12
import continuous_prm_c12_refiner_pipeline as P


def _sample(n_nodes, offset=0):
    return P.GraphSample(
        bundle=None,
        node_feats=np.full((n_nodes, 14), offset, dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        edge_feats=np.zeros((2, 3), dtype=np.float32),
        h_legsum_norm=np.zeros(n_nodes, dtype=np.float32),
        target_flat_ids=np.asarray([0, n_nodes - 1], dtype=np.int64),
        target_y=np.asarray([0.0, 1.0], dtype=np.float32),
        final_transition_hops=12.0,
    )


def test_graph_batch_offsets_edges_and_targets():
    batch = P.batch_samples([_sample(4), _sample(4, 1)], torch.device("cpu"))
    assert batch.node_feats.shape == (8, 14)
    assert batch.edge_index[:, 2:].tolist() == [[4, 5], [5, 6]]
    assert batch.target_flat_ids.tolist() == [0, 3, 4, 7]


def test_common_node_budget_packing_is_deterministic():
    samples = [_sample(4), _sample(4), _sample(7), _sample(2)]
    assert P.pack_sample_indices(samples, [0, 1, 2, 3], 8) == [[0, 1], [2], [3]]


def test_deep_supervision_loss_uses_registered_weights():
    cfg = C12.C12RefinerConfig()
    batch = P.batch_samples([_sample(4)], torch.device("cpu"))
    outputs = {
        1: torch.zeros(4),
        2: torch.ones(4),
        4: torch.full((4,), 2.0),
        8: torch.full((4,), 3.0),
    }
    actual = P.supervised_loss(outputs, batch, cfg)
    expected = sum(
        cfg.deep_supervision_weights[cycle]
        * torch.nn.functional.smooth_l1_loss(
            values[batch.target_flat_ids], batch.target_y, beta=cfg.smooth_l1_beta
        )
        for cycle, values in outputs.items()
    )
    assert actual.item() == pytest.approx(expected.item())


def test_pilot_eval_stream_is_disjoint_from_all_registered_streams():
    pilot = {
        P._pilot_eval_seed(w, config_idx, K)
        for w in range(20)
        for config_idx in (0, 2)
        for K in (2, 8)
    }
    registered = {
        C12.world_seed(split, w, config_idx, K)
        for split in C12.SPLITS
        for w in range(20)
        for config_idx in (0, 2)
        for K in (2, 8, 16)
    }
    assert pilot.isdisjoint(registered)


def _synthetic_positive_rows():
    eval_rows = []
    state_rows = []
    cycle_values = {
        2: {1: 0.70, 2: 0.69, 4: 0.68, 8: 0.67},
        8: {1: 0.90, 2: 0.75, 4: 0.58, 8: 0.40},
    }
    for config in ("A", "C"):
        for K in (2, 8):
            for world in range(25):
                noise = world * 1e-4
                for seed in (0, 1, 2):
                    arms = [
                        ("c11_gnn8", 8, 0.86),
                        ("shallow_param_match", 1, 0.88),
                        ("untied_compute_match", 1, 0.84),
                        ("untied_compute_match", 2, 0.80),
                        ("untied_compute_match", 4, 0.76),
                        ("untied_compute_match", 8, 0.72),
                    ] + [
                        ("tied_refiner", cycle, burden)
                        for cycle, burden in cycle_values[K].items()
                    ]
                    for arm, cycle, burden in arms:
                        common = {
                            "config": config,
                            "K": K,
                            "world_idx": world,
                            "world_seed": 100000 * K + world,
                            "model_seed": seed,
                            "arm": arm,
                            "cycle": cycle,
                            "binding_budget": 100,
                            "final_transition_hops": K * 12 + world,
                        }
                        eval_rows.append(
                            dict(
                                common,
                                found=True,
                                cost=1.0,
                                optimal_cost=1.0,
                                cost_ratio=1.0,
                                expansions=int((burden + noise) * 100),
                                closed=int((burden + noise) * 100),
                                expansion_burden=burden + noise,
                                completion=1.0,
                            )
                        )
                        state_rows.append(
                            dict(
                                common,
                                state_mae=burden,
                                rank_corr=1.0 - burden / 2,
                                bellman_residual=burden / 3,
                            )
                        )
                for reference, burden in (("h_legsum", 1.0), ("h_oracle", 0.05)):
                    common = {
                        "config": config,
                        "K": K,
                        "world_idx": world,
                        "world_seed": 100000 * K + world,
                        "model_seed": -1,
                        "arm": reference,
                        "cycle": 0,
                        "binding_budget": 100,
                        "final_transition_hops": K * 12 + world,
                    }
                    eval_rows.append(
                        dict(
                            common,
                            found=True,
                            cost=1.0,
                            optimal_cost=1.0,
                            cost_ratio=1.0,
                            expansions=int(burden * 100),
                            closed=int(burden * 100),
                            expansion_burden=burden,
                            completion=1.0,
                        )
                    )
                    state_rows.append(
                        dict(common, state_mae=burden, rank_corr=1.0, bellman_residual=burden)
                    )
    return state_rows, eval_rows


def test_analysis_positive_truth_table_and_world_clustering():
    state_rows, eval_rows = _synthetic_positive_rows()
    summary, significance = P.analyze_results(state_rows, eval_rows, scale="full")
    assert summary["gates"]["G1_B"]["passed"] is True
    assert summary["gates"]["G2_B"]["passed"] is True
    assert summary["gates"]["G3_B"]["code"] == "shared_refinement_positive"
    assert all(row["n_worlds"] == 25 for row in significance)
    assert all(row["p_bh"] < 0.05 for row in significance)


def test_bh_adjustment_is_monotone_in_rank():
    adjusted = P._bh_adjust([0.01, 0.04, 0.03, 0.002])
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.008])

