import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c11_headroom as C11


# ---------------------------------------------------------------------------
# Shared fixtures: build a small deterministic world+roadmap+mission.
# ---------------------------------------------------------------------------

H7.install_c7_hard_maps()
ROADMAP_CFG = C.RoadmapConfig(n_nodes=192, k_neighbors=7)


def _build_world_and_roadmap(spec_name: str, seed: int):
    """Retry seeds until a valid, goal-connected world+roadmap is found."""
    specs = C.build_anchor_specs()
    spec = specs[spec_name]
    for attempt in range(50):
        s = seed + attempt * 97
        world = C.build_world(spec, s, 0.45)
        if world is None:
            continue
        rm = C.build_prm(world, ROADMAP_CFG, s + 17)
        if rm is None or not rm.connected_to_goal[0]:
            continue
        return world, rm
    raise RuntimeError(f"could not build a valid world/roadmap for {spec_name} near seed {seed}")


def _mission(spec_name: str, seed: int, K: int):
    world, rm = _build_world_and_roadmap(spec_name, seed)
    wp = C11.sample_mission(rm, world, K, seed)
    return world, rm, wp


WORLD_SEEDS = [1234, 5678, 9999]
SPEC_NAMES = ["C_hard_maze", "C_hard_rooms_large"]


# ---------------------------------------------------------------------------
# Test 1: oracle at the goal state is 0.
# ---------------------------------------------------------------------------

def test_oracle_goal_state_is_zero():
    world, rm, wp = _mission("C_hard_maze", WORLD_SEEDS[0], K=4)
    oracle = C11.product_oracle(rm, wp)
    assert oracle[len(wp), 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 2: forward-consistency -- greedy descent on h* from (0,0) reaches
# (1,K) with total cost == h*(0,0) within 1e-6.
# ---------------------------------------------------------------------------

def test_forward_consistency_greedy_descent_matches_h_star():
    world, rm, wp = _mission("C_hard_maze", WORLD_SEEDS[1], K=4)
    K = len(wp)
    oracle = C11.product_oracle(rm, wp)
    h_star_start = oracle[0, 0]
    assert h_star_start < C.INF

    node, stage = 0, 0
    total_cost = 0.0
    visited_states = set()
    max_steps = rm.points.shape[0] * (K + 1) + 10
    for _ in range(max_steps):
        if (node, stage) == (1, K):
            break
        assert (node, stage) not in visited_states, "cycle detected in greedy descent"
        visited_states.add((node, stage))
        best = None  # (edge_cost + h_next, edge_cost, neighbor, next_stage)
        for j, w in rm.adj[node]:
            next_stage = stage + 1 if (stage < K and j == wp[stage]) else stage
            cand_h = oracle[next_stage, j]
            score = w + cand_h
            key = (score, w, j, next_stage)
            if best is None or key < best[0]:
                best = (key, j, w, next_stage)
        assert best is not None, "no outgoing edges from a non-goal state"
        _, nxt_node, edge_w, nxt_stage = best
        total_cost += edge_w
        node, stage = nxt_node, nxt_stage
    else:
        raise AssertionError("greedy descent did not reach the goal state in time")

    assert (node, stage) == (1, K)
    assert total_cost == pytest.approx(h_star_start, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: admissibility chain h_next <= h_legsum <= h_oracle + 1e-9 across
# >=200 sampled reachable states over >=3 worlds/K values. Load-bearing.
# ---------------------------------------------------------------------------

def test_admissibility_chain_holds_across_worlds_and_k():
    rng = np.random.RandomState(20260707)
    configs = []
    for spec_name in SPEC_NAMES:
        for seed in WORLD_SEEDS:
            for K in (2, 4, 8):
                configs.append((spec_name, seed, K))

    n_sampled = 0
    for spec_name, seed, K in configs:
        world, rm, wp = _mission(spec_name, seed, K)
        oracle = C11.product_oracle(rm, wp)
        hn = C11.h_next(rm, wp)
        hl = C11.h_legsum(rm, wp)
        N = rm.points.shape[0]
        Kk = len(wp)

        # Sample reachable (i, s) states: reachable means oracle finite AND
        # (for realism) the node is connected to the goal on the roadmap.
        reachable = [
            (i, s)
            for s in range(Kk + 1)
            for i in range(N)
            if oracle[s, i] < C.INF and rm.connected_to_goal[i]
        ]
        if not reachable:
            continue
        take = min(len(reachable), 40)
        idxs = rng.choice(len(reachable), size=take, replace=False)
        for idx in idxs:
            i, s = reachable[idx]
            v_next = hn(i, s)
            v_legsum = hl(i, s)
            v_oracle = oracle[s, i]
            assert v_next <= v_legsum + 1e-9, (spec_name, seed, K, i, s, v_next, v_legsum)
            assert v_legsum <= v_oracle + 1e-9, (spec_name, seed, K, i, s, v_legsum, v_oracle)
            n_sampled += 1

    assert n_sampled >= 200, f"only sampled {n_sampled} states, need >= 200"


# ---------------------------------------------------------------------------
# Test 4: monotone in K -- h*(0,0) for K=4 >= h*(0,0) for K=2 on the same
# world/waypoint-prefix (more waypoints to visit costs at least as much).
# ---------------------------------------------------------------------------

def test_monotone_in_k_same_waypoint_prefix():
    world, rm = _build_world_and_roadmap("C_hard_maze", WORLD_SEEDS[2])
    wp_full = C11.sample_mission(rm, world, 4, WORLD_SEEDS[2])
    wp_prefix = wp_full[:2]

    oracle_full = C11.product_oracle(rm, wp_full)
    oracle_prefix = C11.product_oracle(rm, wp_prefix)

    h_star_k4 = oracle_full[0, 0]
    h_star_k2 = oracle_prefix[0, 0]
    assert h_star_k4 < C.INF and h_star_k2 < C.INF
    assert h_star_k4 >= h_star_k2 - 1e-9


# ---------------------------------------------------------------------------
# Test 5: waypoint transition-rule consistency at arrival.
#
# The transition rule is edge-triggered on ARRIVAL: moving along edge i->j
# advances the stage iff j == wp[s] (checked at the destination j, not the
# source). Consequently merely standing at node wp[0] while at stage 0 does
# NOT itself satisfy waypoint 0 -- stage only advances to 1 by actually
# traversing an edge that lands on wp[0]. So (wp[0], 0) and (wp[0], 1) are
# genuinely different product states: from (wp[0], 0) the agent must still
# leave and come back around to re-arrive at wp[0] (or reach it again via
# some other predecessor edge) to trigger the stage-1 transition, which
# costs a strictly positive amount on a roadmap with positive edge weights.
# The precise, always-true invariant this implies:
#   oracle(wp[0], 0) > oracle(wp[0], 1)   (strictly, since completing the
#   waypoint from an already-arrived position requires at least one more
#   positive-cost edge traversal that re-triggers arrival at wp[0]).
# This pins down "arrival, not occupancy, triggers the transition" and
# distinguishes the edge-triggered rule from a (wrong) state-triggered one,
# under which the two values would trivially coincide.
# ---------------------------------------------------------------------------

def test_waypoint_arrival_transition_consistency():
    world, rm, wp = _mission("C_hard_maze", WORLD_SEEDS[0], K=4)
    oracle = C11.product_oracle(rm, wp)
    w0 = wp[0]

    # Sanity: w0 is not also a later mission target (waypoints are distinct
    # and goal node 1 is excluded from waypoint sampling), so standing at w0
    # triggers exactly the stage 0->1 transition and nothing further.
    assert w0 not in wp[1:]
    assert w0 != 1

    assert oracle[1, w0] < C.INF
    # Arrival-triggered: re-completing the waypoint from a position already
    # AT wp[0] costs strictly more than having already completed it.
    assert oracle[0, w0] > oracle[1, w0] + 1e-9

    # And the amount by which it costs more is exactly bounded below by the
    # cheapest possible round trip through any single neighbor of wp[0]
    # (leave along edge w0->j, then satisfy the transition by re-arriving at
    # w0 via edge j->w0): a direct, precise consequence of edge-triggered
    # arrival semantics, not a tautology (it fails under a state-triggered
    # rule where oracle[0, w0] would equal oracle[1, w0]).
    cheapest_roundtrip = min(2.0 * w for _, w in rm.adj[w0])
    assert oracle[0, w0] <= oracle[1, w0] + cheapest_roundtrip + 1e-9


# ---------------------------------------------------------------------------
# Task 2: product A*, calibration, matched three-arm eval.
# ---------------------------------------------------------------------------

def _valid_worlds(spec_name: str, K: int, config_idx: int, n_worlds: int):
    """Mirror eval_cell's own world-validity loop (pre-eval_cell existing):
    advance world_idx, seed via the pre-registered formula, skip on build_world
    / build_prm / sample_mission failure or unreachable oracle."""
    specs = C.build_anchor_specs()
    spec = specs[spec_name]
    out = []
    world_idx = 0
    attempts = 0
    while len(out) < n_worlds and attempts < 200:
        seed = 1234 + 7919 * world_idx + 104729 * config_idx + 15485863 * K
        attempts += 1
        world_idx += 1
        world = C.build_world(spec, seed, 0.45)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=192, k_neighbors=7), seed + 17)
        if rm is None:
            continue
        try:
            wp = C11.sample_mission(rm, world, K, seed)
        except RuntimeError:
            continue
        oracle = C11.product_oracle(rm, wp)
        if not C11.mission_reachable(oracle):
            continue
        out.append((seed, world, rm, wp, oracle))
    assert len(out) >= n_worlds, f"only found {len(out)} valid worlds, need {n_worlds}"
    return out


def test_astar_product_oracle_optimal_and_fewest_expansions():
    worlds = _valid_worlds("C_hard_maze", K=4, config_idx=0, n_worlds=3)
    for seed, world, rm, wp, oracle in worlds:
        opt_cost = float(oracle[0, 0])
        assert opt_cost < C.INF / 10.0

        h_oracle = oracle
        h_next_fn = C11.h_next(rm, wp)
        h_legsum_fn = C11.h_legsum(rm, wp)
        K_ = len(wp)
        N = rm.points.shape[0]
        h_next_arr = np.array([[h_next_fn(i, s) for i in range(N)] for s in range(K_ + 1)])
        h_legsum_arr = np.array([[h_legsum_fn(i, s) for i in range(N)] for s in range(K_ + 1)])

        res_oracle = C11.astar_product(rm.adj, wp, h_oracle, budget=3200)
        res_next = C11.astar_product(rm.adj, wp, h_next_arr, budget=3200)
        res_legsum = C11.astar_product(rm.adj, wp, h_legsum_arr, budget=3200)

        assert res_oracle["found"]
        assert res_oracle["cost"] == pytest.approx(opt_cost, abs=1e-6)
        assert res_oracle["expansions"] <= res_next["expansions"], (seed, res_oracle, res_next)
        assert res_oracle["expansions"] <= res_legsum["expansions"], (seed, res_oracle, res_legsum)

        if res_next["found"]:
            assert res_next["cost"] == pytest.approx(opt_cost, abs=1e-6)
        if res_legsum["found"]:
            assert res_legsum["cost"] == pytest.approx(opt_cost, abs=1e-6)


def test_astar_product_budget_exhaustion_returns_not_found():
    worlds = _valid_worlds("C_hard_maze", K=4, config_idx=0, n_worlds=1)
    seed, world, rm, wp, oracle = worlds[0]
    res = C11.astar_product(rm.adj, wp, oracle, budget=1)
    assert res["found"] is False
    assert res["expansions"] <= 1
    assert np.isnan(res["cost"])


def test_calibrate_binding_budget_thresholds_at_five_percent():
    budgets = (100, 200, 400, 800, 1600, 3200)

    def _records(success_by_budget):
        recs = []
        for b in budgets:
            n_success = success_by_budget[b]
            for w in range(25):
                recs.append({
                    "arm": "h_legsum", "budget": b,
                    "found": w < n_success,
                })
        return recs

    # 0/25 at 100 (0.00), 1/25 at 200 (0.04, below 0.05), 2/25 at 400 (0.08, >= 0.05).
    recs_below_then_pass = _records({100: 0, 200: 1, 400: 2, 800: 2, 1600: 2, 3200: 2})
    budget, degenerate = C11.calibrate_binding_budget(recs_below_then_pass, budgets)
    assert budget == 400
    assert degenerate is False

    # All-zero success at every budget -> largest budget, DEGENERATE.
    recs_all_zero = _records({b: 0 for b in budgets})
    budget, degenerate = C11.calibrate_binding_budget(recs_all_zero, budgets)
    assert budget == 3200
    assert degenerate is True


def _records_equal_nan_aware(recs_a, recs_b) -> bool:
    """Dict equality where NaN == NaN (plain `==` would call two runs with
    identical not-found records "different" merely because `nan != nan`)."""
    if len(recs_a) != len(recs_b):
        return False
    for a, b in zip(recs_a, recs_b):
        if a.keys() != b.keys():
            return False
        for k in a:
            va, vb = a[k], b[k]
            if isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb):
                continue
            if va != vb:
                return False
    return True


def test_eval_cell_small_integration():
    budgets = (400, 3200)
    records1 = C11.eval_cell("C_hard_maze", config_idx=0, K=2, n_worlds=3, budgets=budgets)
    records2 = C11.eval_cell("C_hard_maze", config_idx=0, K=2, n_worlds=3, budgets=budgets)

    assert len(records1) == 3 * 3 * 2  # worlds x arms x budgets
    assert _records_equal_nan_aware(records1, records2)  # deterministic

    seen_worlds = set()
    for r in records1:
        assert r["config"] == "C_hard_maze"
        assert r["config_idx"] == 0
        assert r["K"] == 2
        assert r["arm"] in ("h_next", "h_legsum", "h_oracle")
        assert r["budget"] in budgets
        assert r["opt_cost"] < C.INF / 10.0
        seen_worlds.add(r["world_idx"])
        if r["found"]:
            assert r["cost"] == pytest.approx(r["opt_cost"], abs=1e-6)
    assert len(seen_worlds) == 3


# ---------------------------------------------------------------------------
# Task 3: keys->doors stage-dependent adjacency.
#
# A fixture mission (K=2) on C_hard_maze, built the same way eval_cell would
# (T2 seed formula: seed = 1234 + 7919*world_idx + 104729*config_idx +
# 15485863*K, config_idx=2 for config C). We search a handful of world_idx
# candidates and keep the first that (a) yields a valid world/roadmap/mission
# and (b) accepts a door placement -- mirroring eval_cell's own skip-on-
# failure loop, so a fixture failure here would also mean config C is
# unusable for T4, which is exactly what these tests should catch.
# ---------------------------------------------------------------------------

DOORS_CONFIG_IDX = 2  # matches config C's slot in the eventual eval_cell("C_hard_maze", 2, K, ...) call.


def _doors_fixture(K: int = 2, spec_name: str = "C_hard_maze", max_world_idx: int = 40):
    """First valid (world, rm, wp, seed, placement) for config C, T2 seed formula."""
    specs = C.build_anchor_specs()
    spec = specs[spec_name]
    for world_idx in range(max_world_idx):
        seed = 1234 + 7919 * world_idx + 104729 * DOORS_CONFIG_IDX + 15485863 * K
        world = C.build_world(spec, seed, 0.45)
        if world is None:
            continue
        rm = C.build_prm(world, ROADMAP_CFG, seed + 17)
        if rm is None:
            continue
        try:
            wp = C11.sample_mission(rm, world, K, seed)
        except RuntimeError:
            continue
        try:
            placement = C11.place_doors(rm, world, wp, seed)
        except RuntimeError:
            continue
        return world, rm, wp, seed, placement
    raise RuntimeError(f"no valid config-C fixture found in {max_world_idx} world_idx attempts")


def test_doors_block_at_least_three_edges_and_open_at_later_stage():
    world, rm, wp, seed, placement = _doors_fixture(K=2)

    assert len(placement.rects) == 2
    assert len(placement.blocked) == 2
    for d in range(2):
        assert len(placement.blocked[d]) >= 3

    # Pick a specific edge blocked by door 0 and check it is invalid at s=0
    # (s <= d=0 -> blocked) and valid again at s=2 (s=K=2 > 1 >= both d's ->
    # both doors open).
    i, j = next(iter(placement.blocked[0]))
    assert placement.adj_valid(i, j, 0) is False
    K = len(wp)
    assert placement.adj_valid(i, j, K) is True


def test_oracle_with_doors_is_finite_and_elementwise_geq_plain_oracle():
    world, rm, wp, seed, placement = _doors_fixture(K=2)

    oracle_plain = C11.product_oracle(rm, wp)
    oracle_doors = C11.product_oracle(rm, wp, placement.adj_valid)

    assert C11.mission_reachable(oracle_doors)

    finite_mask = (oracle_plain < C.INF / 10.0) & (oracle_doors < C.INF / 10.0)
    assert finite_mask.any()
    assert np.all(oracle_doors[finite_mask] >= oracle_plain[finite_mask] - 1e-9)

    # Lenient: if any blocked edge lies on a plain-optimal shortest path
    # somewhere in the product graph, doors should strictly raise the cost
    # to go at least at one reachable state. Not forced if it doesn't hold
    # naturally on this fixture.
    strictly_worse = (oracle_doors[finite_mask] > oracle_plain[finite_mask] + 1e-12).any()
    if not strictly_worse:
        print(
            "note: doors did not strictly raise h* anywhere on this fixture "
            "(no blocked edge fell on a plain-optimal path) -- allowed, not forced."
        )


def test_admissibility_preserved_with_doors_on_sampled_states():
    world, rm, wp, seed, placement = _doors_fixture(K=2)
    oracle_doors = C11.product_oracle(rm, wp, placement.adj_valid)
    hl = C11.h_legsum(rm, wp)

    N = rm.points.shape[0]
    K = len(wp)
    reachable = [
        (i, s)
        for s in range(K + 1)
        for i in range(N)
        if oracle_doors[s, i] < C.INF / 10.0
    ]
    assert len(reachable) >= 100, f"only {len(reachable)} reachable states, need >= 100"

    rng = np.random.RandomState(1357)
    take = min(len(reachable), 150)
    idxs = rng.choice(len(reachable), size=take, replace=False)
    for idx in idxs:
        i, s = reachable[idx]
        assert hl(i, s) <= oracle_doors[s, i] + 1e-9, (i, s, hl(i, s), oracle_doors[s, i])


def test_astar_product_optimal_under_doors_with_oracle_and_legsum():
    world, rm, wp, seed, placement = _doors_fixture(K=2)
    oracle_doors = C11.product_oracle(rm, wp, placement.adj_valid)
    assert C11.mission_reachable(oracle_doors)
    opt_cost = float(oracle_doors[0, 0])

    hl = C11.h_legsum(rm, wp)
    K = len(wp)
    N = rm.points.shape[0]
    h_legsum_arr = np.array([[hl(i, s) for i in range(N)] for s in range(K + 1)])

    res_oracle = C11.astar_product(rm.adj, wp, oracle_doors, budget=3200, adj_valid=placement.adj_valid)
    assert res_oracle["found"]
    assert res_oracle["cost"] == pytest.approx(opt_cost, abs=1e-6)

    res_legsum = C11.astar_product(rm.adj, wp, h_legsum_arr, budget=3200, adj_valid=placement.adj_valid)
    if res_legsum["found"]:
        assert res_legsum["cost"] == pytest.approx(opt_cost, abs=1e-6)


def test_door_validator_rejects_bad_placements():
    world, rm, wp, seed, placement = _doors_fixture(K=2)
    N = rm.points.shape[0]

    # A rect centered exactly on wp[0] (tiny but non-degenerate) must be
    # rejected: it CONTAINS a waypoint position.
    wp0_pt = rm.points[wp[0]]
    tiny = 1e-6
    bad_rect_on_waypoint = (
        float(wp0_pt[0]) - tiny, float(wp0_pt[1]) - tiny,
        float(wp0_pt[0]) + tiny, float(wp0_pt[1]) + tiny,
    )
    assert C11._door_rect_contains_any_special_point(bad_rect_on_waypoint, rm, wp) is True

    # A tiny rect far outside the world (side_len is 1.0 or 2.0 for all specs
    # used here; world coordinates live in [0, side_len]) blocks 0 edges and
    # must be rejected on the min-blocked-edges predicate.
    side = float(world.side_len)
    empty_rect = (side * 10.0, side * 10.0, side * 10.0 + tiny, side * 10.0 + tiny)
    blocked_empty = C11._edges_blocked_by_rect(rm, empty_rect)
    assert len(blocked_empty) == 0

    cfg = C11.C11ProbeConfig()
    assert len(blocked_empty) < cfg.door_min_blocked_edges


def test_place_doors_is_deterministic():
    world, rm, wp, seed, placement1 = _doors_fixture(K=2)
    placement2 = C11.place_doors(rm, world, wp, seed)

    assert placement1.rects == placement2.rects
    assert placement1.blocked == placement2.blocked
    # Cross-check the callables agree pointwise on a sample of (i, j, s).
    N = rm.points.shape[0]
    K = len(wp)
    for i in range(0, N, 17):
        for j, _ in rm.adj[i]:
            for s in range(K + 1):
                assert placement1.adj_valid(i, j, s) == placement2.adj_valid(i, j, s)


def test_door_adj_valid_factory_matches_place_doors_and_plugs_into_eval_cell():
    world, rm, wp, seed, placement = _doors_fixture(K=2)

    factory_adj_valid = C11.door_adj_valid_factory(rm, world, wp, seed)
    assert factory_adj_valid is not None
    N = rm.points.shape[0]
    K = len(wp)
    for i in range(0, N, 17):
        for j, _ in rm.adj[i]:
            for s in range(K + 1):
                assert factory_adj_valid(i, j, s) == placement.adj_valid(i, j, s)

    # Full integration: config C runs end-to-end through eval_cell, and a
    # RuntimeError from the factory (placement exhausted its resample
    # attempts) surfaces as a world-skip rather than propagating.
    records = C11.eval_cell(
        "C_hard_maze", config_idx=DOORS_CONFIG_IDX, K=2, n_worlds=2,
        budgets=(400, 3200), adj_valid_factory=C11.door_adj_valid_factory,
    )
    assert len(records) == 2 * 3 * 2  # worlds x arms x budgets
    seen_worlds = set()
    for r in records:
        assert r["arm"] in ("h_next", "h_legsum", "h_oracle")
        seen_worlds.add(r["world_idx"])
        if r["found"]:
            assert r["cost"] == pytest.approx(r["opt_cost"], abs=1e-6)
    assert len(seen_worlds) == 2
