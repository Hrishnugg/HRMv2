import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_dynamic_providers as P


def _toy_roadmap():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])  # 0=start,1=goal,2=mid
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    return C.Roadmap(points=points, adj=adj, dist_to_goal=dist,
                     connected_to_goal=(dist < C.INF / 10.0))


class _W:
    side_len = 1.0


def test_euclid_time_table_shape_admissible_and_constant():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.EuclidTimeProvider().h_table(_W(), rm, dyn, v_agent=1.0, dt=1.0, t_max=40, goal_idx=1)
    assert h.shape == (3, 41)
    assert np.all(np.isfinite(h)) and np.all(h >= 0)
    assert np.allclose(h, h[:, :1])  # t-independent (constant rows)
    orc = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    assert np.all(h <= orc + 1e-9)  # euclid-time is an admissible lower bound


def test_oracle_table_equals_backward_dijkstra_ttg():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn, 1.0, 1.0, 40, 1)
    ttg = ST.oracle_time_to_go(hstar, 40)
    assert np.allclose(h, ttg)
    assert np.all(h[1, :] == 0.0)  # goal node: zero time-to-go at all t


def test_oracle_makes_search_minimal():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, 5000, 1.0, 1.0, 40, 0, 1)
    assert res["found"] and res["arrival"] == 2 and res["expansions"] <= 6


def test_provider_names():
    assert P.EuclidTimeProvider().name == "euclid"
    assert P.OracleProvider().name == "oracle"


def test_scalar_temporal_features_capture_motion():
    import continuous_prm_dynamic_providers as P
    rm = _toy_roadmap()
    mc = D.MovingCircle(ax=1.0, ay=-1.0, bx=1.0, by=1.0, period=4.0, radius=0.3)
    dyn = D.Dynamics([mc])
    X = P.build_scalar_temporal_features(_W(), rm, dyn, t_max=20, dt=1.0, window_w=4, k_patrollers=4, goal_idx=1)
    assert X.shape == (3, 21, 5, 4 + 4 * 4)
    # the dynamics block (cols >=4) must change across frames for a query where the patroller moves
    node0 = X[0, 0]  # (W+1, token_dim) at node 0, t=0
    dyn_block = node0[:, 4:]
    # NOTE: this patroller has period 4.0 and the window spans 4 frames (i=0..4 at
    # dt=1), so frame 0 and frame -1 alias to the same triangle-wave endpoint. The
    # property under test ("motion captured across the window") is that the block is
    # not constant across frames -- assert against an interior frame (the wave peak).
    assert not np.allclose(dyn_block[0], dyn_block[2])  # motion captured across the window
    # static block (cols <4) constant across frames
    assert np.allclose(node0[:, :4], node0[0:1, :4])


def test_scalar_temporal_table_additive_bounded_and_temporal():
    import torch
    import continuous_prm_dynamic_providers as P
    rm = _toy_roadmap()
    mc = D.MovingCircle(ax=1.0, ay=-1.0, bx=1.0, by=1.0, period=4.0, radius=0.3)
    dyn = D.Dynamics([mc])
    prov = P.ScalarTemporalProvider.untrained_for_test(window_w=4, time_blind=False)
    prov_blind = P.ScalarTemporalProvider.untrained_for_test(window_w=4, time_blind=True)
    et = P.EuclidTimeProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 20, 1)
    h = prov.h_table(_W(), rm, dyn, 1.0, 1.0, 20, 1)
    hb = prov_blind.h_table(_W(), rm, dyn, 1.0, 1.0, 20, 1)
    assert h.shape == (3, 21)
    assert np.all(np.isfinite(h)) and np.all(h >= et - 1e-6)            # additive on euclid_time
    assert np.all((h - et) <= prov.max_norm_residual + 1e-4)           # bounded (T_scale=1 here)
    assert not np.allclose(h, hb)                                       # time-aware != time-blind
    assert prov.name == "scalar_hrm" and prov_blind.name == "scalar_hrm_blind"


def test_c6_build_model_in_channels():
    import continuous_prm_c6_heatmap_value_field as C6
    m8 = C6.build_model("unet")            # default 8
    m12 = C6.build_model("unet", in_channels=12)
    assert m8 is not None and m12 is not None


def _small_world_and_prm():
    spec = C.build_anchor_specs()["C_open"]
    for seed in range(40):
        w = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if w is None:
            continue
        rm = C.build_prm(w, C.RoadmapConfig(n_nodes=48, k_neighbors=8), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            return w, rm
    raise RuntimeError("no world")


def test_value_field_temporal_table_additive_bounded():
    import continuous_prm_dynamic_providers as P
    world, rm = _small_world_and_prm()
    mc = D.MovingCircle(ax=0.3, ay=0.3, bx=0.7, by=0.7, period=4.0, radius=0.08)
    dyn = D.Dynamics([mc])
    prov = P.ValueFieldTemporalProvider.untrained_for_test(grid_size=32, window_w=2, time_blind=False)
    h = prov.h_table(world, rm, dyn, v_agent=0.1, dt=1.0, t_max=8, goal_idx=1)
    et = P.EuclidTimeProvider().h_table(world, rm, dyn, 0.1, 1.0, 8, 1)
    N = rm.points.shape[0]
    assert h.shape == (N, 9)
    assert np.all(np.isfinite(h)) and np.all(h >= et - 1e-6)
    T_scale = world.side_len / 0.1 / 1.0
    assert np.all((h - et) <= T_scale * prov.max_norm_residual + 1e-3)
    assert prov.name == "field_unet"


def test_field_occupancy_stack_shape_and_motion():
    import continuous_prm_dynamic_providers as P
    world, rm = _small_world_and_prm()
    mc = D.MovingCircle(ax=0.2, ay=0.5, bx=0.8, by=0.5, period=4.0, radius=0.1)
    dyn = D.Dynamics([mc])
    x = P.build_field_occupancy_stack(world, dyn, grid_size=32, t=0, window_w=2, dt=1.0)
    assert x.shape == (8 + 2, 32, 32)
    # the W+1 occupancy frames (channels 0..2) should differ across frames (patroller moves)
    assert not np.allclose(x[0], x[2])


def test_field_occupancy_stack_static_base_equivalence():
    """Passing a precomputed static_base must produce a BYTE-IDENTICAL stack to the
    back-compat (static_base=None) path — the perf cache must not change behavior."""
    import continuous_prm_dynamic_providers as P
    world, rm = _small_world_and_prm()
    mc = D.MovingCircle(ax=0.2, ay=0.5, bx=0.8, by=0.5, period=4.0, radius=0.1)
    dyn = D.Dynamics([mc])
    G = 32
    static_base = P.compute_field_static_base(world, G)
    assert static_base[0].shape == (G, G) and static_base[1].shape == (7, G, G)
    for W in (0, 2):
        for t in (0, 1, 3, 5):
            a = P.build_field_occupancy_stack(world, dyn, G, t, W, dt=1.0)
            b = P.build_field_occupancy_stack(world, dyn, G, t, W, dt=1.0, static_base=static_base)
            assert np.array_equal(a, b), f"mismatch at W={W} t={t}"
            assert a.dtype == b.dtype
