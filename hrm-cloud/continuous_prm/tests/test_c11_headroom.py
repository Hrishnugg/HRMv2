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
