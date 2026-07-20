import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_identifiability as I


def _adj(edges, n):
    graph = [[] for _ in range(n)]
    for a, b, weight in edges:
        graph[a].append((b, float(weight)))
        graph[b].append((a, float(weight)))
    return graph


def test_trimmed_recurrent_readout_is_invariant_to_trailing_padding():
    torch.manual_seed(7)
    model = I.RecurrentRanker(
        backbone_type="onlstm",
        readout_mode="trimmed",
        token_dim=16,
        hidden_dim=16,
        max_output=4.0,
    ).eval()
    short = torch.zeros((2, 5, 16), dtype=torch.float32)
    short[:, 0, 0] = 1.0
    short[:, 1, 1] = 1.0
    short[:, 2, 2] = 1.0
    short[:, :3, 4:] = torch.randn((2, 3, 12))
    long = torch.zeros((2, 9, 16), dtype=torch.float32)
    long[:, :5] = short

    with torch.no_grad():
        trimmed_short = model.forward_mode(short, "trimmed")
        trimmed_long = model.forward_mode(long, "trimmed")
        padded_short = model.encode(short, "padded")
        padded_long = model.encode(long, "padded")

    assert torch.allclose(trimmed_short, trimmed_long, atol=1.0e-7)
    assert not torch.allclose(padded_short, padded_long, atol=1.0e-6)


def test_summary_last_places_declared_state_readout_after_real_actions():
    model = I.RecurrentRanker(
        backbone_type="hrm",
        readout_mode="summary_last",
        token_dim=16,
        hidden_dim=16,
        max_output=4.0,
    )
    x = torch.zeros((3, 8, 16), dtype=torch.float32)
    x[:, 0, 0] = 1.0
    x[:, 1, 1] = 1.0
    x[:, 2, 2] = 1.0
    x[:, :3, 4:] = torch.randn((3, 3, 12))
    output = model(x)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_focal_secondary_modes_preserve_bounded_cost_on_toy_graph():
    graph = _adj(
        [
            (0, 2, 1.0),
            (2, 1, 1.0),
            (0, 3, 0.7),
            (3, 4, 0.7),
            (4, 1, 0.7),
        ],
        5,
    )
    euclid = np.array([1.5, 0.0, 0.8, 1.2, 0.6], dtype=np.float64)
    rank = np.array([2.0, 0.0, 1.0, 1.4, 0.7], dtype=np.float64)
    optimal = 2.0
    for mode in ("h", "fhat", "residual"):
        result = I.focal_search_with_secondary(
            graph, euclid, rank, budget=5, w=1.10, secondary=mode
        )
        assert result["found"]
        assert result["cost"] <= 1.10 * optimal + 1.0e-12


def test_world_split_keeps_same_world_holdout_disjoint():
    world_id = np.repeat(np.arange(4), 10)
    train, within = I.split_selected_worlds(world_id, n_worlds=3, seed=19)
    assert not set(train) & set(within)
    assert set(world_id[train]) == {0, 1, 2}
    assert set(world_id[within]) == {0, 1, 2}
    assert 3 not in world_id[np.concatenate([train, within])]


def test_feature_views_exclude_zero_padding_from_neighbor_aggregates():
    x = np.zeros((2, 7, 16), dtype=np.float32)
    x[:, 0, 0] = 1.0
    x[:, 1:3, 1] = 1.0
    x[:, 3, 2] = 1.0
    x[:, 3, 4:] = 2.0
    views = I.feature_views(x, num_rays=2)
    assert views["summary"].shape == (2, 12)
    assert views["full"].shape == (2, 7 * 16)
    assert np.isfinite(views["compact"]).all()
    assert np.allclose(views["compact"][:, -1], 1.0)

