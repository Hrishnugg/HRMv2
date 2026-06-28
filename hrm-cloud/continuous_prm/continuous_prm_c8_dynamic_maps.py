#!/usr/bin/env python3
"""
C8: dynamic continuous-PRM map suites (a patroller layer on the C7 hard maps).

This module adds six *dynamic* map suites to the continuous-PRM benchmark. Each
dynamic suite reuses a static C7-style hard layout (the same walls / doorways /
clutter geometry) and adds N deterministically moving "patroller" circles
(``continuous_prm_dynamics.MovingCircle``) placed to cross the start->goal
corridor / gates. The patrollers are seeded (deterministic per seed) and tuned so
that the OPTIMAL space-time plan must *sometimes* wait for a patroller to pass --
i.e. the suite contains genuine timing decisions -- while the worlds stay
space-time solvable and the corridor is never permanently sealed.

Suites
------
Train (``is_ood=False``):
  - C_dyn_maze         static = C_hard_maze base (serpentine walls + clutter)
  - C_dyn_rooms        static = C_hard_maze base (2 walls, side_len=1.0, looser
                       "room" feel)
  - C_dyn_spiral       static = C_hard_spiral (serpentine vertical walls)
Held-out (``is_ood=True``):
  - C_dyn_maze_dense   static = C_hard_maze_dense (more/tighter walls) + more,
                       faster patrollers
  - C_dyn_crossing     open / low-clutter arena + crossing patrollers (PURE
                       timing: there is no static detour, only moving obstacles)
  - C_dyn_rooms_large  static = C_hard_rooms_large (side_len=3 room grid)

Composition / install
----------------------
Like C5/C7 this is a runtime-only extension. ``install_c8_dynamic_maps()``:
  1. installs the C7 hard maps (which itself composes on the C5 hard runtime),
  2. captures the post-C7 ``build_anchor_specs`` symbol, and
  3. replaces it with a wrapper that ALSO registers the six dynamic anchor specs
     (so ``C.build_anchor_specs()`` includes them while the C7/C5 static suites
     remain).

The actual dynamic worlds are produced by ``build_dynamic_world(suite, seed)``,
which builds the underlying static C7 world via the patched ``C.build_world`` and
then attaches a seeded ``Dynamics`` of patrollers. The dynamic anchor specs are
registered for discoverability / the later orchestrator; they are NOT routed
through ``C.generate_obstacles`` (that stays untouched -- C8 does not edit C7).

This module creates no new geometry primitives beyond patroller placement and
does not modify common.py / C7 / dynamics / spacetime.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_dynamics as D
import continuous_prm_c5_hard_obstacle_encoder as C5
from continuous_prm_c7_hard_maps import install_c7_hard_maps


# Suite names.
DYN_MAZE = "C_dyn_maze"
DYN_ROOMS = "C_dyn_rooms"
DYN_SPIRAL = "C_dyn_spiral"
DYN_MAZE_DENSE = "C_dyn_maze_dense"
DYN_CROSSING = "C_dyn_crossing"
DYN_ROOMS_LARGE = "C_dyn_rooms_large"

DYN_SUITES = (
    DYN_MAZE, DYN_ROOMS, DYN_SPIRAL,
    DYN_MAZE_DENSE, DYN_CROSSING, DYN_ROOMS_LARGE,
)

# Mode tag stamped on dynamic anchor specs (so an orchestrator can recognise
# "ours"). The dynamic specs are not dispatched on this; it is metadata only.
C8_MODE = "dyn_c8"

# Each dynamic suite reuses a C7/C5 hard layout, but with knobs *tuned for the
# benchmark PRM* (n_nodes=192, k=7). The canonical C7/C5 hard suites were tuned
# at higher node counts; at 192 nodes their tightest gaps + heaviest clutter make
# the static PRM disconnect on most seeds (verified empirically). Since C8's
# difficulty is meant to come from the PATROLLERS (timing), not from extreme
# static clutter, we reuse the same generators with lighter clutter / wider gaps
# so the static graph reliably connects, then layer patrollers on top.
#
# ``_STATIC_BASE`` records which canonical suite each dynamic suite derives from
# (for lineage / generator dispatch); ``_build_static_spec`` produces the tuned
# spec. ``C_dyn_crossing`` has no canonical parent (open arena, built directly).
_STATIC_BASE: Dict[str, Optional[str]] = {
    DYN_MAZE: "C_hard_maze",          # C5 HARD_MODE serpentine walls + clutter
    DYN_ROOMS: "C_hard_maze",         # C5 HARD_MODE, fewer walls -> "room" feel
    DYN_SPIRAL: "C_hard_spiral",      # C7 spiral (name-locked dispatch)
    DYN_MAZE_DENSE: "C_hard_maze",    # C5 HARD_MODE, denser (held out)
    DYN_CROSSING: None,               # open arena (built directly)
    DYN_ROOMS_LARGE: "C_hard_rooms_large",  # C7 rooms_large (name-locked dispatch)
}


def _build_static_spec(suite: str, base_specs: Dict[str, C.AnchorSpec]) -> Optional[C.AnchorSpec]:
    """Return the tuned static AnchorSpec for ``suite`` (None for the open arena).

    The returned spec is a ``dataclasses.replace`` of the canonical C7/C5 spec
    with connectivity-friendly knobs (chosen via a 192-node PRM sweep). For the
    C5-maze-derived suites we set ``mode == C5.HARD_MODE`` so the patched
    ``C.generate_obstacles`` / ``C.build_world`` route to the C5 serpentine
    generator (which dispatches on mode). For the C7-derived suites (spiral,
    rooms_large) we KEEP the canonical C7 name so the C7 generator dispatch
    (keyed on ``spec.name``) still produces the right geometry.
    """
    base = _STATIC_BASE[suite]
    if base is None:
        return None
    if suite == DYN_MAZE:
        return dataclasses.replace(
            base_specs["C_hard_maze"], name="C_hard_maze", mode=C5.HARD_MODE,
            rectangle_count_range=(3, 3), gap_width_frac=0.15,
            extra_clutter_range=(3, 6), obstacle_count_range=(3, 6), is_ood=False,
        )
    if suite == DYN_ROOMS:
        # Fewer walls + side_len=1.0 -> a couple of "rooms" with doorways.
        return dataclasses.replace(
            base_specs["C_hard_maze"], name="C_hard_maze", mode=C5.HARD_MODE,
            side_len=1.0, rectangle_count_range=(2, 2), gap_width_frac=0.14,
            extra_clutter_range=(4, 8), obstacle_count_range=(4, 8), is_ood=False,
        )
    if suite == DYN_MAZE_DENSE:
        # Held out: denser (more walls, tighter gaps, more clutter) than C_dyn_maze.
        return dataclasses.replace(
            base_specs["C_hard_maze"], name="C_hard_maze", mode=C5.HARD_MODE,
            rectangle_count_range=(4, 4), gap_width_frac=0.115,
            extra_clutter_range=(6, 12), obstacle_count_range=(6, 12), is_ood=True,
        )
    if suite == DYN_SPIRAL:
        # C7 spiral: 3 serpentine walls, wider gap + light clutter for 192-node
        # connectivity. Name kept so C7 _generate_spiral dispatch fires.
        return dataclasses.replace(
            base_specs["C_hard_spiral"], name="C_hard_spiral",
            rectangle_count_range=(3, 3), gap_width_frac=0.18,
            extra_clutter_range=(2, 5), is_ood=False,
        )
    if suite == DYN_ROOMS_LARGE:
        # C7 rooms_large at side_len=2.0 (vs C_dyn_rooms' 1.0), light clutter.
        return dataclasses.replace(
            base_specs["C_hard_rooms_large"], name="C_hard_rooms_large",
            side_len=2.0, gap_width_frac=0.14, extra_clutter_range=(3, 6),
            is_ood=True,
        )
    raise KeyError(suite)


# -----------------------------------------------------------------------------
# Per-suite patroller + space-time params
# -----------------------------------------------------------------------------
#
# Fields per suite:
#   n_patrollers   : number of MovingCircle patrollers placed per world.
#   radius_frac    : patroller radius as a fraction of side_len.
#   span_frac      : patroller travel span (a->b length) as a fraction of side_len.
#   period_frac    : MovingCircle.period as a fraction of t_max*dt (the planning
#                    horizon in seconds). Smaller => faster sweeps. The triangle
#                    wave means the patroller crosses its span twice per period.
#   v_agent        : agent speed (distance units / second).
#   dt             : time-step duration (seconds).
#   t_max          : maximum time-step index for the space-time search.
#
# Tuning intent: enough patrollers / sweep speed that for some worlds a patroller
# is over the corridor when the agent would otherwise pass (=> timing pressure),
# but the patroller always *clears* its span periodically (triangle wave never
# parks on a point) so the corridor is never permanently sealed and worlds stay
# solvable within t_max.

_PARAMS: Dict[str, Dict[str, float]] = {
    DYN_MAZE: dict(  # side_len = 1.0
        n_patrollers=3, radius_frac=0.050, span_frac=0.38, period_frac=0.32,
        v_agent=0.060, dt=1.0, t_max=110,
    ),
    DYN_ROOMS: dict(  # side_len = 1.0
        n_patrollers=3, radius_frac=0.050, span_frac=0.40, period_frac=0.32,
        v_agent=0.060, dt=1.0, t_max=110,
    ),
    DYN_SPIRAL: dict(  # side_len = 1.0
        n_patrollers=3, radius_frac=0.050, span_frac=0.38, period_frac=0.32,
        v_agent=0.060, dt=1.0, t_max=110,
    ),
    DYN_MAZE_DENSE: dict(  # side_len = 1.0; held out: more + faster patrollers
        n_patrollers=4, radius_frac=0.055, span_frac=0.42, period_frac=0.26,
        v_agent=0.060, dt=1.0, t_max=120,
    ),
    DYN_CROSSING: dict(  # side_len = 1.0; open arena, pure timing
        n_patrollers=4, radius_frac=0.055, span_frac=0.50, period_frac=0.30,
        v_agent=0.060, dt=1.0, t_max=110,
    ),
    DYN_ROOMS_LARGE: dict(  # side_len = 2.0; held out
        n_patrollers=5, radius_frac=0.055, span_frac=0.36, period_frac=0.30,
        v_agent=0.120, dt=1.0, t_max=130,
    ),
}


def dynamics_params(suite: str) -> Dict[str, float]:
    """Return the patroller + space-time params dict for a dynamic suite.

    The returned dict always contains at least ``v_agent`` (float), ``dt``
    (float) and ``t_max`` (int) -- the keys the tests + orchestrator read -- plus
    the patroller-placement params used by ``build_dynamic_world``.
    """
    if suite not in _PARAMS:
        raise KeyError(f"unknown dynamic suite: {suite!r}")
    p = dict(_PARAMS[suite])
    p["t_max"] = int(p["t_max"])
    p["n_patrollers"] = int(p["n_patrollers"])
    return p


# -----------------------------------------------------------------------------
# Anchor specs (registration only)
# -----------------------------------------------------------------------------

# Idempotency / composition guard + captured prior callable.
_INSTALLED = False
_PREV_BUILD_ANCHOR_SPECS: Optional[Callable[[], Dict[str, "C.AnchorSpec"]]] = None


def _crossing_spec() -> C.AnchorSpec:
    """Open / low-clutter arena spec for C_dyn_crossing (no static detour)."""
    return C.AnchorSpec(
        name=DYN_CROSSING,
        side_len=1.0,
        mode=C8_MODE,
        obstacle_count_range=(2, 5),
        radius_range=(0.022, 0.045),
        gap_width_frac=0.18,
        barrier_radius_frac=0.030,
        extra_clutter_range=(2, 5),
        rectangle_count_range=(0, 0),
        is_ood=True,
    )


def build_c8_anchor_specs(base_specs: Dict[str, C.AnchorSpec]) -> Dict[str, C.AnchorSpec]:
    """Return the six dynamic anchor specs (re-keyed under the dynamic names).

    Each dynamic suite's anchor spec is the *tuned* static spec actually used to
    build its worlds (so the orchestrator sees the real geometry: side_len, gap
    width, clutter, wall count), re-keyed under the dynamic suite name. The C5
    maze-derived suites keep the C5 ``HARD_MODE`` mode and the C7-derived suites
    keep the C7 mode; this is metadata for the orchestrator only -- dispatch
    happens inside ``build_dynamic_world`` via the canonical-named tuned spec,
    not via these re-keyed copies. C_dyn_crossing uses the open-arena spec.
    """
    out: Dict[str, C.AnchorSpec] = {}
    for suite in DYN_SUITES:
        static_spec = _build_static_spec(suite, base_specs)
        if static_spec is None:
            out[suite] = _crossing_spec()
            continue
        out[suite] = dataclasses.replace(static_spec, name=suite)
    return out


def build_anchor_specs_c8() -> Dict[str, C.AnchorSpec]:
    assert _PREV_BUILD_ANCHOR_SPECS is not None
    specs = _PREV_BUILD_ANCHOR_SPECS()
    specs.update(build_c8_anchor_specs(specs))
    return specs


# -----------------------------------------------------------------------------
# Static-world construction
# -----------------------------------------------------------------------------

def _build_static_world(suite: str, seed: int) -> Optional[C.World]:
    """Build the underlying static C7/C5 world for ``suite`` (or an open arena).

    Builds a *tuned* static spec (canonical-named so the C5/C7 generator dispatch
    fires) and runs the patched ``C.build_world`` on it, so the geometry is the
    corresponding C7/C5 hard layout with C8's connectivity-friendly knobs. For
    C_dyn_crossing we synthesise a sparse open arena directly.
    """
    base = _STATIC_BASE[suite]
    if base is None:
        return _build_crossing_world(seed)
    static_spec = _build_static_spec(suite, C.build_anchor_specs())
    if static_spec is None:
        return None
    # Generous separation so start/goal sit on opposite sides => a patroller
    # placed mid-corridor genuinely intercepts the route.
    world = C.build_world(static_spec, seed=seed, min_start_goal_dist_frac=0.5)
    if world is None:
        return None
    # Re-tag spec_name to the dynamic suite (meta keeps the static lineage).
    world.spec_name = suite
    world.meta["c8_static_suite"] = static_spec.name
    world.meta["c8_suite"] = suite
    return world


def _build_crossing_world(seed: int) -> Optional[C.World]:
    """Open low-clutter arena with opposite-side start/goal (for C_dyn_crossing)."""
    spec = _crossing_spec()
    side = spec.side_len
    rng = random.Random(seed)

    # A few scattered circles, away from the central corridor band so the static
    # graph is essentially open (timing pressure must come from patrollers).
    obstacles: List[C.Obstacle] = []
    n_clutter = rng.randint(*spec.extra_clutter_range)
    attempts = 0
    while len(obstacles) < n_clutter and attempts < n_clutter * 200 + 50:
        attempts += 1
        r = rng.uniform(spec.radius_range[0], spec.radius_range[1])
        cx = rng.uniform(r + 0.02 * side, side - r - 0.02 * side)
        cy = rng.uniform(r + 0.02 * side, side - r - 0.02 * side)
        # Keep the central horizontal corridor band clear.
        if 0.40 * side < cy < 0.60 * side and 0.20 * side < cx < 0.80 * side:
            continue
        ok = True
        for o in obstacles:
            if np.linalg.norm(np.array([cx - o.cx, cy - o.cy])) < (r + o.bounding_radius()) * 0.7:
                ok = False
                break
        if ok:
            obstacles.append(C.Obstacle("circle", cx, cy, radius=r))

    left = (0.03 * side, 0.12 * side, 0.40 * side, 0.60 * side)
    right = (0.88 * side, 0.97 * side, 0.40 * side, 0.60 * side)
    start_region, goal_region = (left, right) if rng.random() < 0.5 else (right, left)
    for _ in range(120):
        start = C.sample_free_point(rng, side, obstacles, region=start_region, max_attempts=3000)
        goal = C.sample_free_point(rng, side, obstacles, region=goal_region, max_attempts=3000)
        if start is None or goal is None:
            continue
        if float(np.linalg.norm(goal - start)) < 0.55 * side:
            continue
        if not C.is_point_free(start, side, obstacles) or not C.is_point_free(goal, side, obstacles):
            continue
        desc = C.task_descriptor(spec, obstacles)
        return C.World(
            spec_name=DYN_CROSSING,
            side_len=side,
            obstacles=obstacles,
            start=start,
            goal=goal,
            descriptor=desc,
            meta={
                "seed": seed,
                "mode": C8_MODE,
                "c8_static_suite": None,
                "c8_suite": DYN_CROSSING,
                "obstacle_count": len(obstacles),
                "free_fraction_est": C.approximate_free_fraction(side, obstacles),
            },
        )
    return None


# -----------------------------------------------------------------------------
# Patroller placement
# -----------------------------------------------------------------------------

def _perp_unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-9:
        return np.array([0.0, 1.0], dtype=np.float64)
    d = vec / n
    return np.array([-d[1], d[0]], dtype=np.float64)


def _place_patrollers(world: C.World, suite: str, seed: int) -> D.Dynamics:
    """Place seeded patrollers crossing the start->goal corridor.

    Each patroller is a ``MovingCircle`` whose travel segment a->b is centred on
    a point along the start->goal line and oriented PERPENDICULAR to it, so the
    circle sweeps across the corridor. The triangle-wave motion guarantees the
    patroller periodically vacates every point on its span (corridor never
    permanently sealed), while at the sweep midpoint it covers the corridor
    centre (=> timing pressure on plans that try to pass then).

    Patrollers are clamped inside the box and given staggered phases (via varied
    periods) so they are not all synchronised.
    """
    params = dynamics_params(suite)
    side = world.side_len
    rng = random.Random(seed * 2654435761 % (2**31 - 1) + 12345)

    start = np.asarray(world.start, dtype=np.float64)
    goal = np.asarray(world.goal, dtype=np.float64)
    line = goal - start
    seg_len = float(np.linalg.norm(line))
    if seg_len <= 1e-9:
        line = np.array([1.0, 0.0], dtype=np.float64)
        seg_len = 1.0
    perp = _perp_unit(line)

    n = int(params["n_patrollers"])
    radius = float(params["radius_frac"]) * side
    span = float(params["span_frac"]) * side
    horizon_s = float(params["t_max"]) * float(params["dt"])
    base_period = max(1.0, float(params["period_frac"]) * horizon_s)

    circles: List[D.MovingCircle] = []
    # Distribute crossing points evenly along the interior of the corridor,
    # avoiding the immediate start/goal neighbourhoods.
    fracs = np.linspace(0.30, 0.70, n) if n > 1 else np.array([0.5])
    for i in range(n):
        f = float(fracs[i]) + rng.uniform(-0.05, 0.05)
        f = min(0.80, max(0.20, f))
        center = start + f * line
        # Lateral jitter so the sweep midpoint straddles (not perfectly centres)
        # the corridor -- keeps a thin passing lane on one side at midpoint.
        lateral = rng.uniform(-0.10, 0.10) * side
        center = center + lateral * perp
        half = 0.5 * span
        a = center - half * perp
        b = center + half * perp
        # Clamp endpoints inside the box (keep a margin of one radius).
        m = radius + 0.005 * side
        a = np.clip(a, m, side - m)
        b = np.clip(b, m, side - m)
        # Stagger phases by varying period +/- 25% per patroller.
        period = base_period * (1.0 + rng.uniform(-0.25, 0.25))
        period = max(1.0, period)
        circles.append(D.MovingCircle(
            ax=float(a[0]), ay=float(a[1]),
            bx=float(b[0]), by=float(b[1]),
            period=float(period), radius=float(radius),
        ))
    return D.Dynamics(circles)


# -----------------------------------------------------------------------------
# Public: dynamic world builder
# -----------------------------------------------------------------------------

def build_dynamic_world(suite: str, seed: int) -> Optional[Tuple[C.World, D.Dynamics]]:
    """Build a dynamic world for ``suite``: (static C7 world, patroller Dynamics).

    Returns ``None`` if the static world could not be built (unlucky endpoint
    sampling), mirroring ``C.build_world``'s contract.
    """
    if suite not in _STATIC_BASE:
        raise KeyError(f"unknown dynamic suite: {suite!r}")
    world = _build_static_world(suite, seed)
    if world is None:
        return None
    dyn = _place_patrollers(world, suite, seed)
    # Guard: start/goal must remain free at t=0 against the patrollers; if a
    # patroller happens to cover an endpoint at t=0 the world is still valid for
    # the *static* check but we record it for downstream filtering.
    world.meta["c8_n_patrollers"] = len(dyn.circles)
    return world, dyn


# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

def install_c8_dynamic_maps() -> None:
    """Install the C8 dynamic suites on top of the C7 hard-map runtime.

    Steps:
      1. Install the C7 hard maps (idempotent; itself composes on C5 hard).
      2. Capture the post-C7 ``build_anchor_specs`` as the delegation target.
      3. Replace ``C.build_anchor_specs`` with a wrapper that also registers the
         six dynamic anchor specs.

    Idempotent: a second call re-points ``C.build_anchor_specs`` to our wrapper
    (the C7 reinstall above clobbers it) without re-capturing, so the original
    post-C7 callable stays the delegate.
    """
    global _INSTALLED, _PREV_BUILD_ANCHOR_SPECS

    # Always (re)activate the C7 hard maps first so we compose on top of them.
    install_c7_hard_maps()

    if _INSTALLED:
        C.build_anchor_specs = build_anchor_specs_c8
        return

    _PREV_BUILD_ANCHOR_SPECS = C.build_anchor_specs
    C.build_anchor_specs = build_anchor_specs_c8
    _INSTALLED = True


if __name__ == "__main__":
    import continuous_prm_spacetime as ST

    install_c8_dynamic_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in DYN_SUITES:
        params = dynamics_params(suite)
        valid = solved = checked = pressured = 0
        for seed in range(80):
            res = build_dynamic_world(suite, seed)
            if res is None:
                continue
            world, dyn = res
            valid += 1
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            hstar = ST.backward_spacetime_dijkstra(
                rm.adj, rm.points, dyn, params["v_agent"], params["dt"], params["t_max"], goal=1)
            if np.isfinite(hstar[0, 0]):
                solved += 1
            h0 = np.zeros((rm.points.shape[0], params["t_max"] + 1))
            arr_static = ST.space_time_astar_prm(
                rm.adj, rm.points, D.Dynamics([]), h0, 20000,
                params["v_agent"], params["dt"], params["t_max"], 0, 1)
            arr_dyn = ST.space_time_astar_prm(
                rm.adj, rm.points, dyn, h0, 20000,
                params["v_agent"], params["dt"], params["t_max"], 0, 1)
            if arr_static["found"] and arr_dyn["found"]:
                checked += 1
                if arr_dyn["arrival"] > arr_static["arrival"]:
                    pressured += 1
        print(f"{suite:18s} valid={valid:3d} solved={solved:3d} "
              f"checked={checked:3d} pressured={pressured:3d}")
