import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c11_headroom as C11P
import continuous_prm_c11_mission as M


# ---------------------------------------------------------------------------
# Test 1: cell grid shape.
# ---------------------------------------------------------------------------

def test_cell_grid_shape():
    cells = M.build_cell_grid()
    assert len(cells) == 11

    # No ("C", 0) cell -- config C is dropped at K=0 (degenerates to A).
    c_k0 = [c for c in cells if c["config_label"] == "C" and c["K"] == 0]
    assert c_k0 == []

    expected_keys = {"config_label", "spec_name", "config_idx", "K", "doors"}
    for cell in cells:
        assert set(cell.keys()) == expected_keys

    config_idx_by_label = {c["config_label"]: c["config_idx"] for c in cells}
    assert config_idx_by_label["A"] == 0
    assert config_idx_by_label["B"] == 1
    assert config_idx_by_label["C"] == 2

    # A and B present at all four K values; C only at K in (2, 4, 8).
    a_ks = sorted(c["K"] for c in cells if c["config_label"] == "A")
    b_ks = sorted(c["K"] for c in cells if c["config_label"] == "B")
    c_ks = sorted(c["K"] for c in cells if c["config_label"] == "C")
    assert a_ks == [0, 2, 4, 8]
    assert b_ks == [0, 2, 4, 8]
    assert c_ks == [2, 4, 8]

    # doors flag matches config label.
    for cell in cells:
        expected_doors = cell["config_label"] == "C"
        assert cell["doors"] == expected_doors

    # spec_name mapping.
    for cell in cells:
        if cell["config_label"] in ("A", "C"):
            assert cell["spec_name"] == "C_hard_maze"
        else:
            assert cell["spec_name"] == "C_hard_rooms_large"


# ---------------------------------------------------------------------------
# Test 2: TRAIN/TEST seed disjointness, pooled across all 11 cells.
# ---------------------------------------------------------------------------

def test_train_test_seed_disjointness():
    cells = M.build_cell_grid()
    train_seeds = set()
    test_seeds = set()
    for cell in cells:
        config_idx = cell["config_idx"]
        K = cell["K"]
        for w in range(60):
            train_seeds.add(M.train_seed(w, config_idx, K))
            test_seeds.add(M.test_seed(w, config_idx, K))

    assert train_seeds.isdisjoint(test_seeds)
    # Sanity: both sets are actually populated (not accidentally identical
    # formulas collapsing to one giant coincidental overlap check).
    assert len(train_seeds) > 0
    assert len(test_seeds) > 0


# ---------------------------------------------------------------------------
# Test 3: collect_world_bundle at K=2, config A (no doors).
# ---------------------------------------------------------------------------

def _find_first_valid_test_bundle(cell, max_attempts=60):
    """Loop TEST seeds like collect_cell_dataset does, returning the first
    valid bundle. Mirrors the probe's own eval_cell skip-and-retry loop."""
    for w in range(max_attempts):
        seed = M.test_seed(w, cell["config_idx"], cell["K"])
        try:
            return M.collect_world_bundle(cell, seed)
        except RuntimeError:
            continue
    raise RuntimeError(f"no valid TEST bundle found for cell {cell} in {max_attempts} attempts")


def test_collect_world_bundle_k2():
    cells = M.build_cell_grid()
    cell_a_k2 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 2)

    bundle = _find_first_valid_test_bundle(cell_a_k2)

    assert len(bundle.wp) == 2
    assert bundle.oracle.shape == (3, 192)
    assert bundle.adj_valid is None

    targets = bundle.targets
    assert targets.ndim == 2
    assert targets.shape[1] == 3
    assert targets.shape[0] > 0
    y = targets[:, 2]
    assert np.all(y >= 0.0)
    assert np.all(y <= 4.0)
    assert bundle.preclip_min >= -1e-9

    # Determinism: two calls on the same (cell, seed) give identical targets.
    bundle2 = M.collect_world_bundle(cell_a_k2, bundle.seed)
    assert np.array_equal(bundle.targets, bundle2.targets)


# ---------------------------------------------------------------------------
# Test 4: collect_world_bundle at K=0 -- formulation-equivalence guard.
# ---------------------------------------------------------------------------

def test_collect_world_bundle_k0():
    cells = M.build_cell_grid()
    cell_a_k0 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 0)

    bundle = _find_first_valid_test_bundle(cell_a_k0)

    assert bundle.wp == ()
    assert bundle.oracle.shape == (1, 192)

    # K=0 legsum must be exactly euclid-to-goal (node 1) -- the C7-continuity
    # formulation guard. Sample up to 5 nodes with finite oracle at stage 0.
    N = bundle.rm.points.shape[0]
    finite_nodes = [i for i in range(N) if bundle.oracle[0, i] < C.INF / 10.0]
    assert len(finite_nodes) >= 1
    sample = finite_nodes[:5]
    for i in sample:
        expected = float(np.linalg.norm(bundle.rm.points[i] - bundle.rm.points[1]))
        assert bundle.hl(i, 0) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: collect_cell_dataset counts + seed-formula provenance.
# ---------------------------------------------------------------------------

def test_collect_cell_dataset_counts():
    cells = M.build_cell_grid()
    cell_a_k2 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 2)

    train_bundles = M.collect_cell_dataset(cell_a_k2, split="train", n_worlds=3)
    assert len(train_bundles) == 3
    for b in train_bundles:
        # seed = 900001 + 7919*w + 104729*config_idx + 15485863*K for SOME w.
        assert (b.seed - 900001 - 104729 * 0 - 15485863 * 2) % 7919 == 0

    test_bundles = M.collect_cell_dataset(cell_a_k2, split="test", n_worlds=2)
    assert len(test_bundles) == 2
    for b in test_bundles:
        # seed = 1234 + 7919*w + 104729*config_idx + 15485863*K for SOME w.
        assert (b.seed - 1234 - 104729 * 0 - 15485863 * 2) % 7919 == 0

    # Cross-check: no test bundle's seed satisfies the train formula and
    # vice versa (disjoint residues given the formulas' distinct constants).
    train_seed_set = {b.seed for b in train_bundles}
    test_seed_set = {b.seed for b in test_bundles}
    assert train_seed_set.isdisjoint(test_seed_set)


# ---------------------------------------------------------------------------
# Test 6: doors cell (config C) -- adj_valid present, oracle >= plain oracle.
# ---------------------------------------------------------------------------

def test_doors_cell_bundle():
    cells = M.build_cell_grid()
    cell_c_k2 = next(c for c in cells if c["config_label"] == "C" and c["K"] == 2)

    bundle = _find_first_valid_test_bundle(cell_c_k2, max_attempts=80)

    assert bundle.adj_valid is not None

    oracle_plain = C11P.product_oracle(bundle.rm, bundle.wp)
    finite_mask = (oracle_plain < C.INF / 10.0) & (bundle.oracle < C.INF / 10.0)
    assert finite_mask.any()
    assert np.all(bundle.oracle[finite_mask] >= oracle_plain[finite_mask] - 1e-9)


# ===========================================================================
# Task 2: structure-exposing encoders.
# ===========================================================================
#
# Shared fixture helpers mirror the pattern above (_find_first_valid_test_bundle):
# find the first valid TEST-seed bundle for a given cell.

def _cell(config_label, K):
    cells = M.build_cell_grid()
    return next(c for c in cells if c["config_label"] == config_label and c["K"] == K)


_RAY_DIRS = (0, 2, 4, 6, 8, 10, 12, 14)


def _expected_rays(p, world):
    return [
        C.raycast_distance(p, 2.0 * math.pi * d / 16.0, world, steps=C.FeatureConfig().ray_steps)
        / world.side_len
        for d in _RAY_DIRS
    ]


# ---------------------------------------------------------------------------
# Test 7: trace token layout (config A, K=2, no doors).
# ---------------------------------------------------------------------------

def test_trace_tokens_layout():
    assert M.TRACE_TOKEN_LAYOUT is not None
    assert len(M.TRACE_TOKEN_LAYOUT) == 12

    cell = _cell("A", 2)
    bundle = _find_first_valid_test_bundle(cell)
    side = bundle.world.side_len
    K = cell["K"]

    # --- s=0: query + 2 waypoint legs + 1 goal leg = 4 tokens.
    i, s = 0, 0
    tokens = M.encode_trace(bundle, i, s)
    assert tokens.dtype == np.float32
    assert tokens.shape == (4, 12)

    # Token 0: query layout.
    p_i = bundle.rm.points[i]
    q = tokens[0]
    assert q[0] == pytest.approx(p_i[0] / side, abs=1e-6)
    assert q[1] == pytest.approx(p_i[1] / side, abs=1e-6)
    expected_rays = _expected_rays(p_i, bundle.world)
    for k in range(8):
        assert q[2 + k] == pytest.approx(expected_rays[k], abs=1e-5)
    assert q[10] == pytest.approx(s / M.C11MissionConfig().k_max, abs=1e-6)
    assert q[10] == pytest.approx(0.0 / 8.0, abs=1e-6)
    assert q[11] == pytest.approx((K - s) / 8.0, abs=1e-6)

    # Token 1: leg to wp[0], chained from the query node's position.
    wp0 = bundle.wp[0]
    p_wp0 = bundle.rm.points[wp0]
    leg0 = tokens[1]
    exp_dx = (p_wp0[0] - p_i[0]) / side
    exp_dy = (p_wp0[1] - p_i[1]) / side
    exp_dist = math.hypot(exp_dx, exp_dy)
    assert leg0[0] == pytest.approx(exp_dx, abs=1e-6)
    assert leg0[1] == pytest.approx(exp_dy, abs=1e-6)
    assert leg0[2] == pytest.approx(exp_dist, abs=1e-6)
    assert leg0[3] == pytest.approx(0.0 / 8.0, abs=1e-6)  # (t - s) = (0 - 0)
    # Config A: no doors -- both door flags are 0.0 on every leg.
    assert leg0[4] == pytest.approx(0.0, abs=1e-9)
    assert leg0[5] == pytest.approx(0.0, abs=1e-9)
    assert leg0[6] == pytest.approx((K - 0) / (K + 1), abs=1e-6)  # remaining_frac at t=0
    assert leg0[7] == pytest.approx(0.0, abs=1e-9)  # not the goal leg
    assert np.all(leg0[8:12] == 0.0)

    # Token 2: leg to wp[1], chained from wp[0] (h_legsum geometry).
    wp1 = bundle.wp[1]
    p_wp1 = bundle.rm.points[wp1]
    leg1 = tokens[2]
    exp_dx1 = (p_wp1[0] - p_wp0[0]) / side
    exp_dy1 = (p_wp1[1] - p_wp0[1]) / side
    exp_dist1 = math.hypot(exp_dx1, exp_dy1)
    assert leg1[0] == pytest.approx(exp_dx1, abs=1e-6)
    assert leg1[1] == pytest.approx(exp_dy1, abs=1e-6)
    assert leg1[2] == pytest.approx(exp_dist1, abs=1e-6)
    assert leg1[3] == pytest.approx(1.0 / 8.0, abs=1e-6)  # (t - s) = (1 - 0)
    assert leg1[4] == pytest.approx(0.0, abs=1e-9)
    assert leg1[5] == pytest.approx(0.0, abs=1e-9)
    assert leg1[7] == pytest.approx(0.0, abs=1e-9)

    # Token 3 (last): goal leg, chained from wp[1], targets node 1.
    goal_p = bundle.rm.points[1]
    leg_goal = tokens[3]
    exp_dxg = (goal_p[0] - p_wp1[0]) / side
    exp_dyg = (goal_p[1] - p_wp1[1]) / side
    exp_distg = math.hypot(exp_dxg, exp_dyg)
    assert leg_goal[0] == pytest.approx(exp_dxg, abs=1e-6)
    assert leg_goal[1] == pytest.approx(exp_dyg, abs=1e-6)
    assert leg_goal[2] == pytest.approx(exp_distg, abs=1e-6)
    assert leg_goal[3] == pytest.approx(2.0 / 8.0, abs=1e-6)  # (t - s) = (2 - 0), t == K
    assert leg_goal[4] == pytest.approx(0.0, abs=1e-9)
    assert leg_goal[5] == pytest.approx(0.0, abs=1e-9)
    assert leg_goal[6] == pytest.approx(0.0 / (K + 1), abs=1e-6)  # remaining_frac at t=K
    assert leg_goal[7] == pytest.approx(1.0, abs=1e-9)  # is_goal_leg

    # Only the LAST token has is_goal_leg set (slot 7 means something
    # different on the query token -- a ray distance, not is_goal_leg -- so
    # only the LEG tokens before the last one are checked here).
    assert np.all(tokens[1:3, 7] == 0.0)

    # --- s=1: 3 tokens; first (only) waypoint leg targets wp[1].
    tokens_s1 = M.encode_trace(bundle, i, 1)
    assert tokens_s1.shape == (3, 12)
    leg_s1 = tokens_s1[1]
    p_i2 = bundle.rm.points[i]  # query node's own position (same node, different stage)
    exp_dx_s1 = (p_wp1[0] - p_i2[0]) / side
    exp_dy_s1 = (p_wp1[1] - p_i2[1]) / side
    assert leg_s1[0] == pytest.approx(exp_dx_s1, abs=1e-6)
    assert leg_s1[1] == pytest.approx(exp_dy_s1, abs=1e-6)
    assert leg_s1[7] == pytest.approx(0.0, abs=1e-9)
    assert tokens_s1[2, 7] == pytest.approx(1.0, abs=1e-9)  # last token is goal leg


# ---------------------------------------------------------------------------
# Test 8: door-key flags (config C, K=2).
# ---------------------------------------------------------------------------

def test_trace_tokens_doors_flags():
    cell = _cell("C", 2)
    bundle = _find_first_valid_test_bundle(cell, max_attempts=80)
    K = cell["K"]

    # --- s=0: both waypoint legs are door keys, both doors still closed.
    tokens = M.encode_trace(bundle, 0, 0)
    assert tokens.shape == (4, 12)
    leg0, leg1, leg_goal = tokens[1], tokens[2], tokens[3]
    assert leg0[4] == pytest.approx(1.0, abs=1e-9)  # is_door_key (t=0 keyed to door 0)
    assert leg0[5] == pytest.approx(0.0, abs=1e-9)  # door_open_at_s: s=0 > t=0 is False
    assert leg1[4] == pytest.approx(1.0, abs=1e-9)  # is_door_key (t=1 keyed to door 1)
    assert leg1[5] == pytest.approx(0.0, abs=1e-9)  # s=0 > t=1 is False
    assert leg_goal[4] == pytest.approx(0.0, abs=1e-9)  # goal leg is never a door key
    assert leg_goal[5] == pytest.approx(0.0, abs=1e-9)

    # --- s=1: leg t=0 is GONE (legs start at t=s=1); remaining leg t=1 is a
    # door key with door_open_at_s False (s=1 > t=1 is False); goal leg t=2.
    tokens_s1 = M.encode_trace(bundle, 0, 1)
    assert tokens_s1.shape == (3, 12)
    leg_t1, leg_goal_s1 = tokens_s1[1], tokens_s1[2]
    assert leg_t1[4] == pytest.approx(1.0, abs=1e-9)
    assert leg_t1[5] == pytest.approx(0.0, abs=1e-9)  # 1 > 1 is False
    assert leg_goal_s1[7] == pytest.approx(1.0, abs=1e-9)

    # --- s=2: only the goal leg remains.
    tokens_s2 = M.encode_trace(bundle, 0, 2)
    assert tokens_s2.shape == (2, 12)
    assert tokens_s2[1, 7] == pytest.approx(1.0, abs=1e-9)
    assert tokens_s2[1, 4] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 9: K=0 -- exactly 2 tokens (query + goal leg).
# ---------------------------------------------------------------------------

def test_trace_tokens_k0():
    cell = _cell("A", 0)
    bundle = _find_first_valid_test_bundle(cell)
    assert bundle.wp == ()

    tokens = M.encode_trace(bundle, 0, 0)
    assert tokens.shape == (2, 12)
    assert tokens[1, 7] == pytest.approx(1.0, abs=1e-9)  # goal leg

    goal_p = bundle.rm.points[1]
    p0 = bundle.rm.points[0]
    side = bundle.world.side_len
    exp_dx = (goal_p[0] - p0[0]) / side
    exp_dy = (goal_p[1] - p0[1]) / side
    assert tokens[1, 0] == pytest.approx(exp_dx, abs=1e-6)
    assert tokens[1, 1] == pytest.approx(exp_dy, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 10: padded batch.
# ---------------------------------------------------------------------------

def test_padded_batch():
    cell = _cell("A", 2)
    bundle = _find_first_valid_test_bundle(cell)
    states = [(0, 0), (0, 1), (5, 2)]

    padded, mask = M.encode_trace_padded(bundle, states)
    assert padded.shape == (3, 10, 12)
    assert padded.dtype == np.float32
    assert mask.shape == (3, 10)
    assert mask.dtype == np.bool_

    for row_idx, (i, s) in enumerate(states):
        unpadded = M.encode_trace(bundle, i, s)
        n_real = unpadded.shape[0]
        assert mask[row_idx, :n_real].all()
        assert not mask[row_idx, n_real:].any()
        assert np.array_equal(padded[row_idx, :n_real], unpadded)
        assert np.all(padded[row_idx, n_real:] == 0.0)
        # mask row sum == unpadded length.
        assert int(mask[row_idx].sum()) == n_real


# ---------------------------------------------------------------------------
# Test 11: MLP flatten == padded reshape (byte-equal).
# ---------------------------------------------------------------------------

def test_mlp_flatten():
    cell = _cell("A", 2)
    bundle = _find_first_valid_test_bundle(cell)
    states = [(0, 0), (0, 1), (5, 2)]

    flat = M.encode_mlp(bundle, states)
    padded, _mask = M.encode_trace_padded(bundle, states)
    expected = padded.reshape(len(states), -1)

    assert flat.shape == (3, 120)
    assert flat.dtype == np.float32
    assert np.array_equal(flat, expected)
    assert flat.tobytes() == expected.tobytes()


# ---------------------------------------------------------------------------
# Test 12: field grids.
# ---------------------------------------------------------------------------

def test_field_grids():
    import continuous_prm_c6_heatmap_value_field as C6

    cell = _cell("A", 2)
    bundle = _find_first_valid_test_bundle(cell)
    K = cell["K"]

    grids0 = M.encode_field_grids(bundle, 0)
    assert grids0.shape == (5, 64, 64)
    assert grids0.dtype == np.float32

    occupancy, _free, _clearance = C6.rasterize_world(bundle.world, 64)
    assert np.array_equal(grids0[0], occupancy)
    assert grids0[0].sum() > 0

    # ch3: current-target heatmap at tgt(0) == wp[0]; argmax cell matches
    # the C6 point-to-cell conversion for wp[0]'s position.
    wp0_cell = C6.point_to_cell(bundle.rm.points[bundle.wp[0]], bundle.world.side_len, 64)
    ch3_argmax = np.unravel_index(np.argmax(grids0[3]), grids0[3].shape)
    assert ch3_argmax == wp0_cell

    # ch4 (closed-door mask) all-zero for config A (no doors).
    assert np.all(grids0[4] == 0.0)

    # At s=K (goal stage), current-target heatmap argmax matches node 1 (goal).
    grids_goal = M.encode_field_grids(bundle, K)
    goal_cell = C6.point_to_cell(bundle.rm.points[1], bundle.world.side_len, 64)
    ch3_goal_argmax = np.unravel_index(np.argmax(grids_goal[3]), grids_goal[3].shape)
    assert ch3_goal_argmax == goal_cell


def test_field_grids_doors():
    import continuous_prm_c6_heatmap_value_field as C6

    cell = _cell("C", 2)
    bundle = _find_first_valid_test_bundle(cell, max_attempts=80)

    grids0 = M.encode_field_grids(bundle, 0)
    assert grids0.shape == (5, 64, 64)

    placement = bundle.adj_valid.__self__
    rect0 = placement.rects[0]
    rect1 = placement.rects[1]

    def _rect_center(rect):
        xmin, ymin, xmax, ymax = rect
        return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])

    side = bundle.world.side_len
    # One inside point (a door-rect center, both doors closed at s=0) == 1.0.
    inside_cell = C6.point_to_cell(_rect_center(rect0), side, 64)
    assert grids0[4][inside_cell] == pytest.approx(1.0, abs=1e-9)
    inside_cell1 = C6.point_to_cell(_rect_center(rect1), side, 64)
    assert grids0[4][inside_cell1] == pytest.approx(1.0, abs=1e-9)

    # A far point outside both rects == 0.0. Pick the world corner farthest
    # from both rect centers.
    corners = np.array([[0.0, 0.0], [side, 0.0], [0.0, side], [side, side]])
    c0 = _rect_center(rect0)
    c1 = _rect_center(rect1)
    dists = [min(np.linalg.norm(c - c0), np.linalg.norm(c - c1)) for c in corners]
    far_point = corners[int(np.argmax(dists))]
    far_cell = C6.point_to_cell(far_point, side, 64)
    assert grids0[4][far_cell] == pytest.approx(0.0, abs=1e-9)

    # At s=2, both doors open -- ch4 is all-zero.
    grids2 = M.encode_field_grids(bundle, 2)
    assert np.all(grids2[4] == 0.0)


# ---------------------------------------------------------------------------
# Test 13: product graph tensors.
# ---------------------------------------------------------------------------

def test_product_graph_tensors():
    cell = _cell("A", 2)
    bundle = _find_first_valid_test_bundle(cell)
    K = cell["K"]
    N = bundle.rm.points.shape[0]

    graph = M.encode_product_graph(bundle)
    assert set(graph.keys()) >= {"node_feats", "edge_index", "edge_feats"}

    node_feats = graph["node_feats"]
    edge_index = graph["edge_index"]
    edge_feats = graph["edge_feats"]

    assert node_feats.shape == ((K + 1) * N, 14)
    assert node_feats.dtype == np.float32
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == np.int64
    E = edge_index.shape[1]
    assert edge_feats.shape == (E, 3)
    assert edge_feats.dtype == np.float32

    # Flat id convention: s * 192 + i. Verify on a known node: (i=5, s=1).
    flat_5_1 = 1 * N + 5
    side = bundle.world.side_len
    p5 = bundle.rm.points[5]
    expected_rays = _expected_rays(p5, bundle.world)
    row = node_feats[flat_5_1]
    assert row[0] == pytest.approx(p5[0] / side, abs=1e-6)
    assert row[1] == pytest.approx(p5[1] / side, abs=1e-6)
    for k in range(8):
        assert row[2 + k] == pytest.approx(expected_rays[k], abs=1e-5)
    assert row[10] == pytest.approx(1.0 / 8.0, abs=1e-6)  # s/k_max
    assert row[11] == pytest.approx((K - 1) / 8.0, abs=1e-6)  # K_remaining/k_max
    tgt1 = C11P.mission_target(1, bundle.wp)
    exp_dist_tgt = float(np.linalg.norm(p5 - bundle.rm.points[tgt1])) / side
    assert row[12] == pytest.approx(exp_dist_tgt, abs=1e-6)
    assert row[13] == pytest.approx(1.0 / 8.0, abs=1e-6)  # s/k_max (dup convention slot)

    # Every roadmap (undirected) edge yields 2 product edges per stage (A has
    # no adj_valid gating): E == (K+1) * 2 * n_undirected_edges.
    n_undirected = sum(1 for i in range(N) for j, _w in bundle.rm.adj[i] if j > i)
    assert E == (K + 1) * 2 * n_undirected

    # A known arrival edge: find a roadmap neighbor j == wp[0] of some i,
    # assert edge (i@s0 -> j@s1) present with is_arrival 1.0.
    wp0 = bundle.wp[0]
    found_arrival = False
    for i in range(N):
        for j, w in bundle.rm.adj[i]:
            if j == wp0:
                s2 = C11P.transition_stage(0, j, bundle.wp)
                assert s2 == 1
                src = 0 * N + i
                dst = 1 * N + j
                matches = np.where((edge_index[0] == src) & (edge_index[1] == dst))[0]
                assert len(matches) == 1
                ef = edge_feats[matches[0]]
                assert ef[1] == pytest.approx(1.0, abs=1e-9)  # is_arrival
                assert ef[2] == pytest.approx(1.0, abs=1e-9)
                assert ef[0] == pytest.approx(w / side, abs=1e-6)
                found_arrival = True
                break
        if found_arrival:
            break
    assert found_arrival, "no roadmap edge into wp[0] found to test arrival transition"

    # Determinism: two calls give identical tensors.
    graph2 = M.encode_product_graph(bundle)
    assert np.array_equal(graph["node_feats"], graph2["node_feats"])
    assert np.array_equal(graph["edge_index"], graph2["edge_index"])
    assert np.array_equal(graph["edge_feats"], graph2["edge_feats"])


def test_product_graph_door_blocking():
    cell = _cell("C", 2)
    bundle = _find_first_valid_test_bundle(cell, max_attempts=80)
    K = cell["K"]
    N = bundle.rm.points.shape[0]

    graph = M.encode_product_graph(bundle)
    edge_index = graph["edge_index"]

    placement = bundle.adj_valid.__self__
    # Door 0 blocks its edges at s=0, opens at s>=1.
    i0, j0 = next(iter(placement.blocked[0]))
    assert not bundle.adj_valid(i0, j0, 0)
    assert bundle.adj_valid(i0, j0, 1)

    def _has_edge(i, j, s):
        src = s * N + i
        s2 = C11P.transition_stage(s, j, bundle.wp)
        dst = s2 * N + j
        return len(np.where((edge_index[0] == src) & (edge_index[1] == dst))[0]) > 0

    # Blocked at s=0: neither direction of the physical edge should produce
    # a product edge departing stage 0.
    assert not _has_edge(i0, j0, 0)
    assert not _has_edge(j0, i0, 0)
    # Present at s=2 (both doors open by then).
    assert _has_edge(i0, j0, 2) or _has_edge(j0, i0, 2)


# ---------------------------------------------------------------------------
# Test 14: stack_targets.
# ---------------------------------------------------------------------------

def test_stack_targets():
    cell = _cell("A", 2)
    bundle1 = _find_first_valid_test_bundle(cell)
    # A different world (advance past the first valid seed) for a distinct
    # second bundle.
    bundle2 = None
    seen_first = False
    for w in range(30):
        seed = M.test_seed(w, cell["config_idx"], cell["K"])
        try:
            b = M.collect_world_bundle(cell, seed)
        except RuntimeError:
            continue
        if not seen_first:
            seen_first = True
            continue
        bundle2 = b
        break
    assert bundle2 is not None

    stacked = M.stack_targets([bundle1, bundle2])
    n1 = bundle1.targets.shape[0]
    n2 = bundle2.targets.shape[0]
    assert stacked.shape == (n1 + n2, 4)
    assert np.array_equal(stacked[:n1, :3], bundle1.targets)
    assert np.array_equal(stacked[n1:, :3], bundle2.targets)
    assert np.all(stacked[:n1, 3] == 0.0)
    assert np.all(stacked[n1:, 3] == 1.0)
