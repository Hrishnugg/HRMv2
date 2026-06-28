import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C  # noqa: E402
import continuous_prm_spacetime as ST  # noqa: E402
import continuous_prm_c8_dynamic_maps as M8  # noqa: E402

DYN = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
       "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]


def test_install_registers_dyn_suites():
    M8.install_c8_dynamic_maps()
    specs = C.build_anchor_specs()
    for s in DYN:
        assert s in specs
    assert "C_hard_maze" in specs  # C7 static suites preserved


def test_dynamic_worlds_valid_and_have_patrollers():
    M8.install_c8_dynamic_maps()
    for suite in DYN:
        built = 0
        for seed in range(40):
            res = M8.build_dynamic_world(suite, seed)
            if res is None:
                continue
            world, dyn = res
            assert C.is_point_free(world.start, world.side_len, world.obstacles)
            assert C.is_point_free(world.goal, world.side_len, world.obstacles)
            assert len(dyn.circles) >= 1
            built += 1
            if built >= 5:
                break
        assert built >= 5, f"{suite}: only {built} valid worlds in 40 seeds"


def test_dynamic_worlds_space_time_solvable():
    # backward space-time Dijkstra from goal must reach the start within t_max.
    M8.install_c8_dynamic_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in DYN:
        params = M8.dynamics_params(suite)  # dict with v_agent, dt, t_max
        solved = 0
        for seed in range(60):
            res = M8.build_dynamic_world(suite, seed)
            if res is None:
                continue
            world, dyn = res
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn,
                                                   params["v_agent"], params["dt"], params["t_max"], goal=1)
            if np.isfinite(hstar[0, 0]):
                solved += 1
            if solved >= 5:
                break
        assert solved >= 5, f"{suite}: only {solved} space-time-solvable worlds"


def test_dynamic_worlds_have_timing_pressure():
    # For a fraction of worlds, the patrollers force a later arrival than with no dynamics.
    M8.install_c8_dynamic_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    import continuous_prm_dynamics as D
    for suite in DYN:
        params = M8.dynamics_params(suite)
        pressured = 0; checked = 0
        for seed in range(80):
            res = M8.build_dynamic_world(suite, seed)
            if res is None:
                continue
            world, dyn = res
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            h0 = np.zeros((rm.points.shape[0], params["t_max"] + 1))
            arr_static = ST.space_time_astar_prm(rm.adj, rm.points, D.Dynamics([]), h0, 20000,
                                                 params["v_agent"], params["dt"], params["t_max"], 0, 1)
            arr_dyn = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h0, 20000,
                                              params["v_agent"], params["dt"], params["t_max"], 0, 1)
            if arr_static["found"] and arr_dyn["found"]:
                checked += 1
                if arr_dyn["arrival"] > arr_static["arrival"]:
                    pressured += 1
            if checked >= 12:
                break
        assert checked >= 8, f"{suite}: too few solvable pairs ({checked})"
        assert pressured >= 1, f"{suite}: no timing pressure (patrollers never delay arrival)"
