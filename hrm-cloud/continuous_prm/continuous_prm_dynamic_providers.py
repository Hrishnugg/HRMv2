"""Time-aware heuristic providers returning h[node, t] tables in time-step units (C8)."""
from __future__ import annotations
import abc
import argparse

import numpy as np

import continuous_prm_common as C
import continuous_prm_spacetime as ST
from continuous_prm_spacetime import (
    space_time_astar_prm,
    space_time_focal_prm,
    backward_spacetime_dijkstra,
)


def euclid_time_row(roadmap, v_agent, goal_idx=1) -> np.ndarray:
    """Per-node straight-line time-to-go (seconds) = euclid(node,goal)/v_agent."""
    goal = roadmap.points[goal_idx]
    return np.linalg.norm(roadmap.points - goal[None, :], axis=1) / float(v_agent)


def _finite_fill(a: np.ndarray, fallback: float) -> np.ndarray:
    out = np.array(a, dtype=np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fallback
    return np.maximum(out, 0.0)


class SpaceTimeHeuristicProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx: int = 1) -> np.ndarray:
        """Return h[node, t] (shape (N, t_max+1)) = estimated remaining time-to-go in TIME-STEPS."""
        ...


class EuclidTimeProvider(SpaceTimeHeuristicProvider):
    name = "euclid"

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        row = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)  # seconds -> time-steps
        table = np.repeat(row[:, None], t_max + 1, axis=1)
        return _finite_fill(table, 0.0)


class OracleProvider(SpaceTimeHeuristicProvider):
    name = "oracle"

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        hstar = ST.backward_spacetime_dijkstra(roadmap.adj, roadmap.points, dyn, v_agent, dt, t_max, goal_idx)
        ttg = ST.oracle_time_to_go(hstar, t_max)
        finite = np.isfinite(ttg)
        fill = float(ttg[finite].max() + (t_max + 1)) if finite.any() else float(2 * (t_max + 1))
        return _finite_fill(ttg, fill)


# -----------------------------------------------------------------------------
# Scalar temporal provider (model-adaptation begins here)
# -----------------------------------------------------------------------------
def build_scalar_temporal_features(world, roadmap, dyn, t_max, dt, window_w,
                                   k_patrollers, goal_idx=1) -> np.ndarray:
    """Per (node n, time-step t) rollout-window frame-token sequence for the
    recurrent ContinuousHeuristicModel.

    Returns X of shape (N, t_max+1, W+1, token_dim) with token_dim = 4 + 4*K
    (K = k_patrollers). Each of the W+1 frames is at real time (t+i)*dt, i=0..W,
    and is a vector [static_geom(4) ++ dynamics(K*4)]:
      static_geom (constant across the window, per node n):
        goal_dx, goal_dy, euclid_to_goal, node_clearance   (all /side_len)
      dynamics (per frame i): for the K nearest patroller circles to n at time
        (t+i)*dt (ranked by center distance), [dx, dy, dist, radius] all /side_len;
        zero-padded if fewer than K circles exist.
    """
    s = float(world.side_len)
    W = int(window_w)
    K = int(k_patrollers)
    token_dim = 4 + 4 * K
    pts = np.asarray(roadmap.points, dtype=np.float64)  # (N, 2)
    N = pts.shape[0]
    goal = pts[goal_idx]
    obstacles = getattr(world, "obstacles", [])
    circles = list(dyn.circles)

    # static_geom per node (constant across t and across the window).
    static = np.zeros((N, 4), dtype=np.float64)
    diff_goal = (goal[None, :] - pts) / s          # (g - n)/s
    static[:, 0] = diff_goal[:, 0]
    static[:, 1] = diff_goal[:, 1]
    static[:, 2] = np.linalg.norm(goal[None, :] - pts, axis=1) / s
    for n in range(N):
        static[n, 3] = C.obstacle_clearance(pts[n], s, obstacles) / s

    X = np.zeros((N, t_max + 1, W + 1, token_dim), dtype=np.float64)
    X[:, :, :, :4] = static[:, None, None, :]

    if not circles:
        return X

    # Cache patroller centers per distinct real frame time. Frame real time for a
    # query (t, i) is (t+i)*dt; over t in [0,t_max] and i in [0,W] the distinct
    # frame indices f = t+i span [0, t_max+W].
    f_max = t_max + W
    centers = np.empty((f_max + 1, len(circles), 2), dtype=np.float64)
    radii = np.array([c.radius for c in circles], dtype=np.float64)
    for f in range(f_max + 1):
        tf = float(f) * float(dt)
        for j, c in enumerate(circles):
            centers[f, j] = c.center_at(tf)

    for t in range(t_max + 1):
        for i in range(W + 1):
            f = t + i
            ctr = centers[f]                       # (C, 2)
            for n in range(N):
                rel = (ctr - pts[n][None, :])      # (C, 2)
                dist = np.linalg.norm(rel, axis=1)
                order = np.argsort(dist, kind="stable")[:K]
                base = 4
                for slot, j in enumerate(order):
                    off = base + 4 * slot
                    X[n, t, i, off + 0] = rel[j, 0] / s
                    X[n, t, i, off + 1] = rel[j, 1] / s
                    X[n, t, i, off + 2] = dist[j] / s
                    X[n, t, i, off + 3] = radii[j] / s
                # remaining slots (if fewer than K circles) already zero.
    return X


class ScalarTemporalProvider(SpaceTimeHeuristicProvider):
    """Feeds the recurrent ContinuousHeuristicModel a per-(node,t) rollout-window
    frame sequence (recurrence over TIME) and adds a clamped time-to-go residual
    onto the admissible euclid-time row."""

    def __init__(self, model, device, backbone, window_w, k_patrollers, max_norm_residual, time_blind=False):
        import torch  # noqa: F401  (kept for parity with provider construction site)
        self.model = model
        self.device = device
        self.window_w = int(window_w)
        self.k_patrollers = int(k_patrollers)
        self.max_norm_residual = float(max_norm_residual)
        self.time_blind = bool(time_blind)
        self.name = f"scalar_{backbone}" + ("_blind" if time_blind else "")

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        W = 0 if self.time_blind else self.window_w
        X = build_scalar_temporal_features(world, roadmap, dyn, t_max, dt, W, self.k_patrollers, goal_idx)
        N = X.shape[0]
        flat = X.reshape(N * (t_max + 1), W + 1, X.shape[-1]).astype(np.float32)
        resid = C.predict_norm_residuals(self.model, flat, self.device)  # (N*(t_max+1),)
        resid = np.clip(np.asarray(resid, dtype=np.float64), 0.0, self.max_norm_residual).reshape(N, t_max + 1)
        euclid_t = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)  # time-steps
        T_scale = float(world.side_len) / float(v_agent) / float(dt)        # map-crossing time in steps
        h = euclid_t[:, None] + T_scale * resid
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("ScalarTemporalProvider produced non-finite h")
        return np.maximum(h, 0.0)

    @classmethod
    def untrained_for_test(cls, window_w=4, k_patrollers=4, time_blind=False, device=None, seed=0):
        import torch
        device = device or torch.device("cpu")
        token_dim = 4 + 4 * int(k_patrollers)
        # Minimal argparse.Namespace with exactly the fields build_backbone_configs
        # reads; we only use the "hrm" entry. Same pattern as
        # ScalarResidualProvider.untrained_for_test in continuous_prm_providers.py.
        ns = argparse.Namespace(
            hrm_hidden=32,
            hrm_layers=1,
            hrm_k_step=2,
            hrm_heads=4,
            head_hidden=48,
            onlstm_hidden=32,
            onlstm_layers=1,
            onlstm_chunk_size=8,
        )
        bb = C.build_backbone_configs(ns)["hrm"]
        train_cfg = C.TrainingConfig()
        model = C.ContinuousHeuristicModel(bb, token_dim=token_dim,
                                           max_norm_residual=train_cfg.max_norm_residual).to(device).eval()
        # The head inits to constant output (weight 0, bias -2) -> softplus(-2) for
        # ALL inputs. Randomize the last layer's weight so the model is
        # input-dependent (so time-aware vs time-blind can differ in tests).
        torch.manual_seed(seed)
        torch.nn.init.normal_(model.head[-1].weight, std=0.3)
        return cls(model, device, "hrm", window_w, k_patrollers, train_cfg.max_norm_residual, time_blind)


# -----------------------------------------------------------------------------
# Value-field temporal provider (occupancy-stack C6 fields, recurrence over space)
# -----------------------------------------------------------------------------
def _render_patrollers_occupancy(world, dyn, grid_size, t_real) -> np.ndarray:
    """Boolean (G, G) grid of cells whose center lies within any patroller circle
    at real time ``t_real`` (cell-center test against center_at(t_real))."""
    import continuous_prm_c6_heatmap_value_field as C6
    side = float(world.side_len)
    G = int(grid_size)
    xx, yy = C6.grid_centers(side, G)  # (G, G) cell centers (indexing="ij")
    occ = np.zeros((G, G), dtype=np.bool_)
    for c in dyn.circles:
        cx, cy = c.center_at(float(t_real))
        d2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
        occ |= d2 <= float(c.radius) ** 2
    return occ


def build_field_occupancy_stack(world, dyn, grid_size, t, window_w, dt) -> np.ndarray:
    """Render a (8 + W, G, G) occupancy-stack field input for time-step ``t``.

    Channels:
      0 .. W : (W+1) occupancy frames, frame i (i=0..W) at real time (t+i)*dt =
               static_occupancy OR patrollers_rendered_at((t+i)*dt).
      W+1 .. W+7 : the 7 static C6 channels (reachable_free, clearance, goal_g,
               start_g, x_norm, y_norm, eu_norm) unchanged from make_heatmap_example.
    W=0 -> 8 channels = the original C6 input (occupancy at time t only).
    """
    import continuous_prm_c6_heatmap_value_field as C6
    G = int(grid_size)
    W = int(window_w)
    base = C6.make_heatmap_example(world, G)["x"]  # (8, G, G); channel 0 is static occupancy
    static_occ = base[0] > 0.5                      # boolean static walls
    frames = np.empty((W + 1, G, G), dtype=np.float32)
    for i in range(W + 1):
        t_real = float(t + i) * float(dt)
        occ = static_occ | _render_patrollers_occupancy(world, dyn, G, t_real)
        frames[i] = occ.astype(np.float32)
    x = np.concatenate([frames, base[1:8]], axis=0)  # (W+1) frames ++ 7 static = (8+W, G, G)
    return x.astype(np.float32)


class ValueFieldTemporalProvider(SpaceTimeHeuristicProvider):
    """Renders a (W+1)-frame occupancy stack per time-step and runs a C6 field
    model (in_channels = 8+W) to predict a cost-to-go residual field at t, then
    samples node residuals and adds a clamped residual onto the euclid-time row."""

    def __init__(self, model, grid_size, device, backbone, window_w, max_norm_residual=4.0, time_blind=False):
        self.model = model
        self.grid_size = int(grid_size)
        self.device = device
        self.window_w = int(window_w)
        self.max_norm_residual = float(max_norm_residual)
        self.time_blind = bool(time_blind)
        self.name = f"field_{backbone}" + ("_blind" if time_blind else "")

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        import continuous_prm_c6_heatmap_value_field as C6
        W = 0 if self.time_blind else self.window_w
        euclid_t = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)
        T_scale = float(world.side_len) / float(v_agent) / float(dt)
        H = np.zeros((roadmap.points.shape[0], t_max + 1), dtype=np.float64)
        for t in range(t_max + 1):
            x_t = build_field_occupancy_stack(world, dyn, self.grid_size, t, W, dt)
            grid = C6.predict_residual_grid(self.model, x_t, self.device)
            grid = np.nan_to_num(grid, nan=0.0, posinf=10.0, neginf=0.0)
            node_resid = C6.interpolate_grid_values(np.maximum(0.0, grid), world, roadmap.points)
            node_resid = np.clip(np.nan_to_num(node_resid, nan=0.0, posinf=10.0, neginf=0.0), 0.0, self.max_norm_residual)
            H[:, t] = euclid_t + T_scale * node_resid
        if not np.all(np.isfinite(H)):
            raise FloatingPointError("ValueFieldTemporalProvider produced non-finite h")
        return np.maximum(H, 0.0)

    @classmethod
    def untrained_for_test(cls, grid_size=32, window_w=2, time_blind=False, device=None):
        import torch, continuous_prm_c6_heatmap_value_field as C6
        device = device or torch.device("cpu")
        W = 0 if time_blind else window_w
        model = C6.build_model("unet", in_channels=8 + W).to(device).eval()
        return cls(model, grid_size, device, "unet", window_w, time_blind=time_blind)


def run_world_arms_spacetime(world, roadmap, dyn, providers: dict, budgets, w_values,
                             v_agent, dt, t_max, goal_idx=1, start_idx=0):
    """Run every (provider, mode, budget[, w]) arm on one shared world+roadmap+dynamics.

    astar mode: space_time_astar_prm with the provider's h_table.
    focal mode: space_time_focal_prm with admissible euclid_time order + provider table as ranker.
    Per-provider nonfinite guard: a provider whose h_table raises FloatingPointError is
    recorded as nonfinite arms (found=False) for all its budget/w arms, not crashing the world.
    suboptimality = arrival / optimal_arrival (optimal from backward_spacetime_dijkstra[start,0]).
    Returns a flat list of record dicts.
    """
    # optimal arrival (makespan) once:
    hstar = backward_spacetime_dijkstra(
        roadmap.adj, roadmap.points, dyn, v_agent, dt, t_max, goal_idx
    )
    opt = float(hstar[start_idx, 0])  # may be inf if unsolvable

    # admissible euclid_time in time-step units, for focal ordering:
    euclid_t = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)

    # precompute each provider's table (nonfinite-robust):
    tables = {}
    nonfinite: set = set()
    for name, prov in providers.items():
        try:
            if isinstance(prov, OracleProvider):
                # Reuse the hstar already computed for opt: OracleProvider.h_table
                # would re-run the same backward DP (the per-world bottleneck). Its
                # extra _finite_fill is a no-op since oracle_time_to_go already
                # returns finite, non-negative values, so this is byte-identical.
                tables[name] = ST.oracle_time_to_go(hstar, t_max)
            else:
                tables[name] = prov.h_table(world, roadmap, dyn, v_agent, dt, t_max, goal_idx)
        except FloatingPointError:
            nonfinite.add(name)

    records = []
    for name in providers:
        if name in nonfinite:
            for b in budgets:
                records.append(_arm_record_st(name, "astar", None, b,
                    {"found": False, "arrival": -1, "expansions": 0, "closed": 0}, opt, nonfinite=1))
                for w in w_values:
                    records.append(_arm_record_st(name, "focal", float(w), b,
                        {"found": False, "arrival": -1, "expansions": 0, "closed": 0}, opt, nonfinite=1))
            continue
        h = tables[name]
        for b in budgets:
            r = space_time_astar_prm(
                roadmap.adj, roadmap.points, dyn, h, int(b),
                v_agent, dt, t_max, start_idx, goal_idx
            )
            records.append(_arm_record_st(name, "astar", None, b, r, opt))
            for w in w_values:
                rf = space_time_focal_prm(
                    roadmap.adj, roadmap.points, dyn, euclid_t, h, int(b),
                    v_agent, dt, t_max, float(w), start_idx, goal_idx
                )
                records.append(_arm_record_st(name, "focal", float(w), b, rf, opt))
    return records


def _arm_record_st(provider, mode, w, budget, res, opt, nonfinite=0):
    found = bool(res["found"])
    arrival = int(res["arrival"]) if found else -1
    sub = (arrival / opt) if (found and np.isfinite(opt) and opt > 0) else float("nan")
    return {
        "provider": provider, "mode": mode, "w": w, "budget": int(budget),
        "found": found, "expansions": int(res["expansions"]), "arrival": arrival,
        "optimal_arrival": float(opt) if np.isfinite(opt) else float("nan"),
        "suboptimality": sub, "closed": int(res["closed"]), "nonfinite": int(nonfinite),
    }
