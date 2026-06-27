import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C  # noqa: E402
import continuous_prm_focal as focal  # noqa: E402


def _line_graph(n=6):
    # 0=start ... goal at index 1 placed at the far end via relabeling.
    # Build a simple chain 0-2-3-4-5-1 with unit edges; goal_idx=1.
    adj = [[] for _ in range(n)]
    order = [0, 2, 3, 4, 5, 1]
    for a, b in zip(order, order[1:]):
        adj[a].append((b, 1.0))
        adj[b].append((a, 1.0))
    return adj, order


def _euclid_like_admissible(adj, goal_idx=1):
    # True cost-to-go is admissible and consistent; use it as the OPEN ordering h.
    return C.dijkstra_to_goal(adj, goal_idx=goal_idx)


def test_focal_w1_matches_optimal_cost():
    adj, order = _line_graph()
    h = _euclid_like_admissible(adj)
    rank = np.zeros(len(adj))  # uninformative ranker
    res = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.0)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"] and opt["found"]
    assert math.isclose(res["cost"], opt["cost"], rel_tol=1e-9)


def test_focal_bound_never_violated():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rng = np.random.default_rng(0)
    rank = rng.random(len(adj))  # adversarial-ish ranker
    w = 2.0
    res = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=w)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"]
    assert res["cost"] <= w * opt["cost"] + 1e-9


def test_focal_completeness_and_budget():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rank = np.zeros(len(adj))
    # Budget too small to reach goal -> not found, expansions capped.
    res = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1, w=1.5)
    assert res["expansions"] <= 1
    assert res["found"] is False
    # Ample budget -> found.
    res2 = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.5)
    assert res2["found"]


def test_focal_determinism():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rank = np.linspace(0, 1, len(adj))
    a = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.3)
    b = focal.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.3)
    assert a == b


def test_focal_constant_rank_degrades_to_astar_expansions():
    # Collapsed ranker (constant) -> selection falls through to f -> behaves like A*.
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    const_rank = np.full(len(adj), 3.14)
    res = focal.focal_astar_search(adj, euclid_h=h, rank_h=const_rank, budget=1000, w=1.0)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"]
    assert math.isclose(res["cost"], opt["cost"], rel_tol=1e-9)
    assert res["expansions"] <= opt["expansions"] + 1
