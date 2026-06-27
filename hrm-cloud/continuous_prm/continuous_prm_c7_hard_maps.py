#!/usr/bin/env python3
"""
C7: heuristic-hostile continuous-PRM map suites.

This module adds three "heuristic-hostile" map suites to the continuous PRM
benchmark. Each suite is engineered so that the optimal (graph) path is forced
to make large detours relative to the straight-line start->goal distance, which
makes a plain Euclidean A* heuristic badly mis-calibrated:

  - C_hard_spiral       serpentine alternating walls (gap flips side per wall)
  - C_hard_bugtrap      concave U/G pocket straddling the start->goal line
  - C_hard_rooms_large  grid of rooms partitioned by walls, one doorway per wall

It deliberately leaves C1-C5 and the C5 hard suites untouched. Like C5, it
installs runtime-only extensions over ``continuous_prm_common`` instead of
editing it. The install COMPOSES on top of the C5 hard runtime: it first
activates the C5 hard runtime (via
``continuous_prm_c6_heatmap_value_field.install_c5_hard_runtime``), then captures
the post-C5 ``build_anchor_specs`` / ``generate_obstacles`` / ``build_world``
symbols and replaces them with wrappers that handle the three new suites and
delegate everything else (including the C5 ``hard_maze`` mode) to the captured
C5/base versions.

The new suites use a dedicated ``mode == C7_MODE`` so the wrappers can cheaply
decide whether a spec is "ours"; all other modes fall through to C5/base.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import continuous_prm_common as C
import continuous_prm_c5_hard_obstacle_encoder as C5
from continuous_prm_c6_heatmap_value_field import install_c5_hard_runtime


C7_MODE = "hard_c7"

# Suite names.
SPIRAL = "C_hard_spiral"
BUGTRAP = "C_hard_bugtrap"
ROOMS_LARGE = "C_hard_rooms_large"
NEW_SUITES = (SPIRAL, BUGTRAP, ROOMS_LARGE)

# Idempotency / composition guard. Holds the captured pre-C7 callables.
_INSTALLED = False
_PREV_BUILD_ANCHOR_SPECS: Optional[Callable[[], Dict[str, "C.AnchorSpec"]]] = None
_PREV_GENERATE_OBSTACLES: Optional[Callable] = None
_PREV_BUILD_WORLD: Optional[Callable] = None


# -----------------------------------------------------------------------------
# Anchor specs
# -----------------------------------------------------------------------------

def build_c7_anchor_specs() -> Dict[str, C.AnchorSpec]:
    """Return the three new C7 anchor specs (keyed by suite name).

    Every required ``AnchorSpec`` field is set explicitly, mirroring the way
    ``C5.build_hard_anchor_specs`` populates its specs. ``mode`` is ``C7_MODE``
    so the runtime wrappers can route these suites to the C7 geometry.

    ``rectangle_count_range`` is overloaded (as in C5) to mean "wall count"
    where applicable; ``gap_width_frac`` is the doorway width as a fraction of
    ``side_len``. Spiral is the trained suite (``is_ood=False``); bugtrap and
    rooms_large are held out (``is_ood=True``).
    """
    return {
        SPIRAL: C.AnchorSpec(
            name=SPIRAL,
            side_len=1.0,
            mode=C7_MODE,
            obstacle_count_range=(0, 0),
            radius_range=(0.014, 0.030),
            gap_width_frac=0.17,
            barrier_radius_frac=0.022,
            extra_clutter_range=(4, 8),
            rectangle_count_range=(4, 4),  # number of serpentine walls
            is_ood=False,
        ),
        BUGTRAP: C.AnchorSpec(
            name=BUGTRAP,
            side_len=1.0,
            mode=C7_MODE,
            obstacle_count_range=(0, 0),
            radius_range=(0.014, 0.030),
            gap_width_frac=0.16,
            barrier_radius_frac=0.022,
            extra_clutter_range=(3, 7),
            rectangle_count_range=(3, 3),  # three walls form the U pocket
            is_ood=True,
        ),
        ROOMS_LARGE: C.AnchorSpec(
            name=ROOMS_LARGE,
            side_len=3.0,
            mode=C7_MODE,
            obstacle_count_range=(0, 0),
            radius_range=(0.020, 0.040),
            gap_width_frac=0.24,
            barrier_radius_frac=0.016,
            extra_clutter_range=(2, 4),
            rectangle_count_range=(1, 1),  # interior grid lines per axis (2x2 rooms)
            is_ood=True,
        ),
    }


# -----------------------------------------------------------------------------
# Wall geometry helpers (thin rect chains with doorway gaps)
# -----------------------------------------------------------------------------

def _wall_segment(
    obstacles: List[C.Obstacle],
    p0: Sequence[float],
    p1: Sequence[float],
    thickness: float,
    gaps: Sequence[float] = (),
    gap_width: float = 0.0,
) -> None:
    """Emit a chain of thin rect obstacles approximating the wall p0->p1.

    The wall is built as a row of small rects laid along the segment so the C5
    feature/encoder code (which understands axis-aligned rects) keeps working.
    ``gaps`` are positions along the wall (as fractions in [0, 1]) where a
    doorway of width ``gap_width`` is carved out (no rects placed there).

    Walls are axis-aligned in practice (we only build horizontal / vertical
    walls), but this works for any segment by approximating it with small
    square-ish rects.
    """
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    if length <= 1e-9:
        return
    half = 0.5 * thickness
    # Step so adjacent rects overlap a little -> no pinholes the planner sneaks
    # through. Each rect has half-extent `half` along the wall too.
    step = max(1e-4, thickness * 0.9)
    n = max(1, int(math.ceil(length / step)))
    direction = (b - a) / length
    for i in range(n + 1):
        t = i / n  # fraction along [0, 1]
        # Skip if this position falls inside any doorway gap.
        in_gap = False
        for g in gaps:
            gap_lo = g - 0.5 * gap_width / length
            gap_hi = g + 0.5 * gap_width / length
            if gap_lo <= t <= gap_hi:
                in_gap = True
                break
        if in_gap:
            continue
        center = a + direction * (t * length)
        obstacles.append(C.Obstacle("rect", float(center[0]), float(center[1]), hw=half, hh=half))


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


# -----------------------------------------------------------------------------
# Per-suite obstacle generators
# -----------------------------------------------------------------------------

def _generate_spiral(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    """Serpentine: vertical walls each leaving a gap on the opposite side."""
    side = spec.side_len
    obstacles: List[C.Obstacle] = []
    n_walls = max(3, int(spec.rectangle_count_range[0]))
    thick = max(0.012 * side, spec.barrier_radius_frac * side)
    gap_w = max(0.060 * side, spec.gap_width_frac * side)
    xs = np.linspace(0.22 * side, 0.78 * side, n_walls)
    for wi, x in enumerate(xs):
        xj = float(x + rng.uniform(-0.010 * side, 0.010 * side))
        # Alternate the open end: even walls open near the top, odd near bottom.
        if wi % 2 == 0:
            gap_center = rng.uniform(0.82 * side, 0.92 * side)
        else:
            gap_center = rng.uniform(0.08 * side, 0.18 * side)
        gap_center = _clip(gap_center, 0.5 * gap_w, side - 0.5 * gap_w)
        g_frac = gap_center / side
        _wall_segment(
            obstacles,
            (xj, 0.0),
            (xj, side),
            thickness=thick,
            gaps=(g_frac,),
            gap_width=gap_w,
        )
    C5._add_random_circles(spec, rng, obstacles, rng.randint(*spec.extra_clutter_range))
    return obstacles


def _generate_bugtrap(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    """Concave bug-trap straddling start->goal, closed back facing the start.

    Classic bug-trap: the goal sits just behind a solid vertical back wall whose
    *closed* face points at the start (left). Two horizontal arms wrap back from
    the back wall toward the start, enclosing the goal in a C/U pocket whose
    mouth opens to the LEFT (toward the start) but only through narrow openings
    at the arm tips. A straight shot from the start hits the closed back wall,
    so the planner must detour up/down to clear an arm tip and curl back in.

    Geometry (start on left, goal in pocket interior):
        x_back  : solid vertical wall (no gap), goal just left of it
        arms    : horizontal walls at y_top / y_bot from x_mouth..x_back
        x_mouth : left ends of arms; gaps left between arm tips and the box edge
                  are the only ways in, far above/below the straight-line route.
    """
    side = spec.side_len
    obstacles: List[C.Obstacle] = []
    thick = max(0.013 * side, spec.barrier_radius_frac * side)

    cx = 0.55 * side + rng.uniform(-0.03 * side, 0.03 * side)
    half_h = rng.uniform(0.26 * side, 0.32 * side)   # vertical half-height of pocket
    depth = rng.uniform(0.28 * side, 0.34 * side)    # how far arms reach toward start
    y_mid = 0.5 * side + rng.uniform(-0.03 * side, 0.03 * side)
    y_top = _clip(y_mid + half_h, 0.06 * side, 0.94 * side)
    y_bot = _clip(y_mid - half_h, 0.06 * side, 0.94 * side)
    x_back = _clip(cx + 0.12 * side, 0.45 * side, 0.80 * side)   # closed back (right)
    x_mouth = _clip(cx - depth, 0.10 * side, x_back - 0.22 * side)  # arm tips (left)

    # Solid back wall (no gap) -> the trap. Goal will sit just left of it.
    _wall_segment(obstacles, (x_back, y_bot), (x_back, y_top), thickness=thick)
    # Top + bottom arms reaching back toward the start.
    _wall_segment(obstacles, (x_mouth, y_top), (x_back, y_top), thickness=thick)
    _wall_segment(obstacles, (x_mouth, y_bot), (x_back, y_bot), thickness=thick)
    # Mouth "lip": a vertical wall at the mouth that closes the CENTRE of the
    # opening (covering y_mid), leaving only narrow lanes hugging each arm tip.
    # This is the crux of the trap -- a straight horizontal shot from the start
    # hits the lip, so the planner must climb to a tip lane and curl back in.
    lip_gap = max(0.06 * side, 0.10 * side)  # opening height at each tip
    _wall_segment(obstacles, (x_mouth, y_bot + lip_gap), (x_mouth, y_top - lip_gap), thickness=thick)

    C5._add_random_circles(spec, rng, obstacles, rng.randint(*spec.extra_clutter_range))
    return obstacles


def _generate_rooms_large(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    """Grid of rooms: vertical + horizontal partition walls, one doorway each.

    The doorways are deliberately staggered (not aligned) so traversing the box
    means weaving between rooms rather than going straight.
    """
    side = spec.side_len
    obstacles: List[C.Obstacle] = []
    thick = max(0.014 * side, spec.barrier_radius_frac * side)
    gap_w = max(0.060 * side, spec.gap_width_frac * side)
    n_interior = max(2, int(spec.rectangle_count_range[0]))  # interior lines per axis

    xs = np.linspace(0.0, side, n_interior + 2)[1:-1]
    ys = np.linspace(0.0, side, n_interior + 2)[1:-1]

    # Vertical interior walls (each spans full height, one staggered doorway).
    for vi, x in enumerate(xs):
        xj = float(x + rng.uniform(-0.012 * side, 0.012 * side))
        base = 0.30 if vi % 2 == 0 else 0.70
        g = _clip(base + rng.uniform(-0.10, 0.10), gap_w / side, 1.0 - gap_w / side)
        _wall_segment(obstacles, (xj, 0.0), (xj, side), thickness=thick, gaps=(g,), gap_width=gap_w)

    # Horizontal interior walls (each spans full width, one staggered doorway).
    for hi, y in enumerate(ys):
        yj = float(y + rng.uniform(-0.012 * side, 0.012 * side))
        base = 0.70 if hi % 2 == 0 else 0.30
        g = _clip(base + rng.uniform(-0.10, 0.10), gap_w / side, 1.0 - gap_w / side)
        _wall_segment(obstacles, (0.0, yj), (side, yj), thickness=thick, gaps=(g,), gap_width=gap_w)

    C5._add_random_circles(spec, rng, obstacles, rng.randint(*spec.extra_clutter_range))
    return obstacles


_SUITE_GENERATORS: Dict[str, Callable[[C.AnchorSpec, random.Random], List[C.Obstacle]]] = {
    SPIRAL: _generate_spiral,
    BUGTRAP: _generate_bugtrap,
    ROOMS_LARGE: _generate_rooms_large,
}


def generate_c7_obstacles(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    """Dispatch to the per-suite generator, else delegate to the prior chain."""
    gen = _SUITE_GENERATORS.get(spec.name) if spec.mode == C7_MODE else None
    if gen is None:
        assert _PREV_GENERATE_OBSTACLES is not None
        return _PREV_GENERATE_OBSTACLES(spec, rng)
    return gen(spec, rng)


# -----------------------------------------------------------------------------
# World construction (opposite-side start/goal, free placement)
# -----------------------------------------------------------------------------

def build_c7_world(spec: C.AnchorSpec, seed: int, min_start_goal_dist_frac: float) -> Optional[C.World]:
    if spec.mode != C7_MODE or spec.name not in _SUITE_GENERATORS:
        assert _PREV_BUILD_WORLD is not None
        return _PREV_BUILD_WORLD(spec, seed, min_start_goal_dist_frac)

    rng = random.Random(seed)
    side = spec.side_len
    # Generate via the patched C.generate_obstacles so the dispatch is uniform.
    obstacles = C.generate_obstacles(spec, rng)

    # Opposite-side endpoint regions: force left<->right traversal.
    left = (0.03 * side, 0.14 * side, 0.10 * side, 0.90 * side)
    right = (0.86 * side, 0.97 * side, 0.10 * side, 0.90 * side)
    if spec.name == BUGTRAP:
        # Start on the far left; goal inside the pocket just left of the closed
        # back wall (centre band). The straight line then runs into the back
        # wall, forcing a detour around an arm tip.
        start_region = (0.03 * side, 0.13 * side, 0.30 * side, 0.70 * side)
        goal_region = (0.52 * side, 0.62 * side, 0.42 * side, 0.58 * side)
    else:
        start_region, goal_region = (left, right) if rng.random() < 0.5 else (right, left)

    for _ in range(220):
        start = C.sample_free_point(rng, side, obstacles, region=start_region, max_attempts=3000)
        goal = C.sample_free_point(rng, side, obstacles, region=goal_region, max_attempts=3000)
        if start is None or goal is None:
            continue
        if float(np.linalg.norm(goal - start)) < min_start_goal_dist_frac * side:
            continue
        if not C.is_point_free(start, side, obstacles) or not C.is_point_free(goal, side, obstacles):
            continue
        desc = C.task_descriptor(spec, obstacles)
        return C.World(
            spec_name=spec.name,
            side_len=side,
            obstacles=obstacles,
            start=start,
            goal=goal,
            descriptor=desc,
            meta={
                "seed": seed,
                "mode": spec.mode,
                "suite": spec.name,
                "obstacle_count": len(obstacles),
                "free_fraction_est": C.approximate_free_fraction(side, obstacles),
                "c7_wall_count": int(spec.rectangle_count_range[0]),
                "c7_gap_width_frac": float(spec.gap_width_frac),
            },
        )
    return None


# -----------------------------------------------------------------------------
# Anchor-spec wrapper (compose new suites over the captured prior specs)
# -----------------------------------------------------------------------------

def build_anchor_specs_c7() -> Dict[str, C.AnchorSpec]:
    assert _PREV_BUILD_ANCHOR_SPECS is not None
    specs = _PREV_BUILD_ANCHOR_SPECS()
    specs.update(build_c7_anchor_specs())
    return specs


# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

def install_c7_hard_maps(sector_tokens: int = 16) -> None:
    """Install the C7 hard-map suites on top of the C5 hard runtime.

    Steps:
      1. Activate the C5 hard runtime (idempotent in effect; it re-patches the
         same symbols from the same originals).
      2. Capture the *post-C5* ``build_anchor_specs`` / ``generate_obstacles`` /
         ``build_world`` symbols.
      3. Replace them with wrappers that handle the three C7 suites and delegate
         all other specs/modes to the captured C5/base callables.

    Idempotent: a second call is a no-op (the captured prior callables are
    preserved from the first install so composition does not stack our wrappers
    on themselves).
    """
    global _INSTALLED
    global _PREV_BUILD_ANCHOR_SPECS, _PREV_GENERATE_OBSTACLES, _PREV_BUILD_WORLD

    # Always (re)activate the C5 hard runtime first so we compose on top of it.
    install_c5_hard_runtime(sector_tokens)

    if _INSTALLED:
        # Already composed once. Re-point C to our wrappers (the C5 reinstall
        # above clobbered them) but do NOT recapture -- keep the original C5
        # callables as our delegates.
        C.build_anchor_specs = build_anchor_specs_c7
        C.generate_obstacles = generate_c7_obstacles
        C.build_world = build_c7_world
        return

    # First install: capture the post-C5 symbols as our delegation targets.
    _PREV_BUILD_ANCHOR_SPECS = C.build_anchor_specs
    _PREV_GENERATE_OBSTACLES = C.generate_obstacles
    _PREV_BUILD_WORLD = C.build_world

    C.build_anchor_specs = build_anchor_specs_c7
    C.generate_obstacles = generate_c7_obstacles
    C.build_world = build_c7_world
    _INSTALLED = True


if __name__ == "__main__":
    install_c7_hard_maps()
    specs = C.build_anchor_specs()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in NEW_SUITES:
        spec = specs[suite]
        ratios = []
        built = 0
        for seed in range(80):
            world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if world is None:
                continue
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            built += 1
            straight = float(np.linalg.norm(world.start - world.goal))
            ratios.append(float(rm.dist_to_goal[0]) / max(1e-6, straight))
        med = float(np.median(ratios)) if ratios else float("nan")
        print(f"{suite:20s} built={built:3d} n_ratio={len(ratios):3d} median_detour={med:.3f}")
