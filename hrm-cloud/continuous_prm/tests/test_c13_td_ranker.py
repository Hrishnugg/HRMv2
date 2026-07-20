import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c13_td_ranker as TD


def _undirected_adj(points, edges):
    adj = [[] for _ in range(len(points))]
    for a, b in edges:
        weight = float(np.linalg.norm(points[a] - points[b]))
        adj[a].append((b, weight))
        adj[b].append((a, weight))
    return adj


def _corridor_graph():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.4, 0.2],
            [0.7, 0.1],
        ],
        dtype=np.float64,
    )
    return points, _undirected_adj(points, [(0, 2), (2, 3), (3, 1)])


def test_rollout_is_an_actually_traversed_feasible_trajectory():
    points, adj = _corridor_graph()
    policy = TD.RolloutPolicyConfig(
        rollouts_per_start=1,
        max_steps_factor=4,
        epsilon=0.0,
        temperature=1.0e-6,
    )
    episode = TD.local_behavior_rollout(
        points,
        adj,
        start_idx=0,
        side_len=1.0,
        rng=random.Random(7),
        cfg=policy,
    )

    assert episode.found
    assert episode.nodes[0] == 0
    assert episode.nodes[-1] == 1
    for u, v, cost in zip(episode.nodes, episode.nodes[1:], episode.edge_costs):
        matching = [w for neighbor, w in adj[u] if neighbor == v]
        assert matching
        assert np.isclose(cost, matching[0])

    returns = TD.realized_returns(episode)
    assert returns[-1] == 0.0
    assert np.allclose(returns[:-1] - returns[1:], np.asarray(episode.edge_costs))
    assert np.isclose(returns[0], episode.total_cost)


def test_rollout_target_collector_cannot_read_shortest_path_oracle():
    points, adj = _corridor_graph()

    class NoOracleRoadmap:
        def __init__(self):
            self.points = points
            self.adj = adj

        @property
        def dist_to_goal(self):
            raise AssertionError("rollout target attempted to read shortest-path distance")

    policy = TD.RolloutPolicyConfig(
        rollouts_per_start=4,
        max_steps_factor=4,
        epsilon=0.0,
        temperature=1.0e-6,
    )
    indices, returns, counts, stats = TD.collect_rollout_targets_from_roadmap(
        NoOracleRoadmap(),
        side_len=1.0,
        rng=random.Random(11),
        cfg=policy,
    )

    assert stats["successful_episodes"] >= 1
    assert 1 in indices
    assert len(indices) == len(returns) == len(counts)
    assert np.all(returns >= 0.0)
    assert TD.SHORTEST_PATH_TARGET is False
    assert TD.ORACLE_POLICY_ACCESS is False


def test_return_aggregation_uses_median_not_minimum():
    episodes = [
        TD.RolloutEpisode((0, 1), (10.0,), True),
        TD.RolloutEpisode((0, 1), (4.0,), True),
        TD.RolloutEpisode((0, 1), (7.0,), True),
        TD.RolloutEpisode((0,), (), False),
    ]
    values, counts, stats = TD.aggregate_successful_start_returns(
        {0: episodes, 1: [TD.RolloutEpisode((1,), (), True)]}
    )

    assert values[0] == 7.0
    assert values[0] != 4.0
    assert values[1] == 0.0
    assert counts[0] == 3
    assert stats["aggregation"] == "median_successful_fresh_start_realized_return"


def test_rollout_residual_is_log_transformed_euclidean_excess_and_clipped():
    points, _ = _corridor_graph()
    indices = np.array([0, 1, 2], dtype=np.int64)
    euclid = np.linalg.norm(points[indices] - points[1], axis=1)
    realized = euclid + np.array([100.0, 0.0, 0.25])
    clipped, raw = TD.rollout_residual_targets(
        points,
        indices,
        realized,
        side_len=1.0,
        max_norm_residual=4.0,
    )

    assert np.allclose(raw, [100.0, 0.0, 0.25])
    assert np.allclose(clipped, [4.0, 0.0, np.log1p(0.25)])


def test_hrm_and_onlstm_rankers_emit_finite_nonnegative_values():
    x = torch.zeros((3, 9, 16), dtype=torch.float32)
    configs = [
        C.BackboneConfig(
            name="hrm",
            backbone_type="hrm",
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            head_hidden=16,
        ),
        C.BackboneConfig(
            name="onlstm",
            backbone_type="onlstm",
            hidden_dim=16,
            num_layers=1,
            chunk_size=8,
            head_hidden=16,
        ),
    ]
    for config in configs:
        model = C.ContinuousHeuristicModel(config, token_dim=16, max_norm_residual=4.0)
        output = model(x)
        assert output.shape == (3,)
        assert torch.isfinite(output).all()
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 4.0)


def test_smoke_configuration_keeps_focal_search_euclidean_anchored():
    cfg = TD.apply_smoke_overrides(TD.C13TDConfig(smoke_test=True))
    assert cfg.focal_ws == "1.10"
    assert cfg.density_nodes == "192,211"
    assert TD.TARGET_SOURCE == "successful_fresh_start_local_behavior_rollout_return_mc"
    assert TD.TARGET_TRANSFORM == "log1p_normalized_return_excess_over_euclidean"


def test_summary_compares_learned_focal_against_same_search_control():
    def make_row(provider, mode, expansions, focal_w=""):
        return {
            "suite": "toy",
            "world_index": 0,
            "requested_nodes": 8,
            "budget": 8,
            "provider": provider,
            "mode": mode,
            "focal_w": focal_w,
            "found": True,
            "expansions": expansions,
            "cost_ratio": 1.0,
            "inference_seconds": 0.0,
            "search_seconds": 0.0,
        }

    rows = [
        make_row("euclid", "astar", 100),
        make_row("euclid_focal_rank", "focal", 80, 1.1),
        make_row("hrm_td_rank", "focal", 70, 1.1),
    ]
    summary = TD.summarize_evaluation(rows)
    learned = next(row for row in summary if row["provider"] == "hrm_td_rank")
    assert learned["expansion_delta_vs_euclid_focal_mean"] == -10.0
    assert learned["expansion_delta_vs_euclid_mean"] == -30.0
