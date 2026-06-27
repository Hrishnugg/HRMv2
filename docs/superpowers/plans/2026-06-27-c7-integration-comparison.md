# C7 Integration Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a matched, multi-arm comparison of heuristic-integration strategies (Euclid baseline, additive-scalar, value-field, focal-ranker) × backbones (HRM, ON-LSTM, U-Net) + oracle ceiling, on six hard continuous-PRM suites, with expansions-on-matched-solved as the primary metric and path-suboptimality first-class — producing a publication-grade result.

**Architecture:** A `HeuristicProvider` strategy interface decouples "how the learned signal becomes a per-node `h` array" from the planner. The planner has two modes: existing `astar_search` (`f = g + h`) and a new `focal_astar_search` (A*ε: admissible Euclid ordering + bounded focal band ranked by the provider's signal). New heuristic-hostile suites are added via the C5 runtime-install pattern (no `common.py` map edits). One orchestrator script (`continuous_prm_c7_integration_compare.py`) reuses C6's field training/eval infra and C5/common's scalar training infra, enumerates arms, runs sharded eval, and computes the six pre-registered comparisons.

**Tech Stack:** Python 3, NumPy, PyTorch (RTX 5090, CUDA), pytest. Reuses `continuous_prm_common.py` (PRM/A*/scalar models), `continuous_prm_c5_hard_obstacle_encoder.py` (hard suites + encoder, runtime-install pattern), `continuous_prm_c6_heatmap_value_field.py` (field models, oracle, McNemar/BH stats).

**Spec:** `docs/superpowers/specs/2026-06-27-c7-integration-comparison-design.md`

---

## Preconditions (read before Task 1)

1. **Uncommitted WIP in `continuous_prm_common.py`.** The working tree has user edits to `hrm-cloud/continuous_prm/continuous_prm_common.py` (and `hrm-cloud/transfer_astar_heuristic_clean_parallel_fixed.py`). Task 1 edits `common.py`. **Before starting, ask the user to commit or stash their `continuous_prm_common.py` WIP** so our commits stay clean. Do not bundle their WIP into our commits. (Tasks 2–13 create new files and do not touch `common.py`, so only Task 1 is affected.)
2. **Run location.** All commands assume CWD `C:/Users/hrish/Code Projects/HRMv2`. Tests live in `hrm-cloud/tests/`; mirror the import bootstrap used by `hrm-cloud/tests/test_c6_heatmap_value_field.py` (it inserts `hrm-cloud/continuous_prm` on `sys.path`). Check that file's first ~15 lines and copy the pattern verbatim into each new test.
3. **GPU.** Provider/model tests run on CPU (tiny). The training + eval runs (Tasks 8–12) use the GPU; pass `--cpu` only for smoke.

## Deviations from spec (deliberate, faithful to existing patterns)

- **New maps via runtime-install, not `common.py` edits.** The spec said "extend the C5 machinery in `common.py`"; the *actual* C5 mechanism (`continuous_prm_c5_hard_obstacle_encoder.py::install_runtime_extensions`) monkeypatches `common.py` at runtime from a separate module. We follow that real pattern (Task 3) — it matches the codebase and avoids the `common.py` WIP entanglement. `focal_astar_search` still lives in `common.py` per spec (Task 1), since it is a generic graph primitive beside `astar_search`.

## File structure

| File | New/Mod | Responsibility |
|---|---|---|
| `hrm-cloud/continuous_prm/continuous_prm_common.py` | **Mod** | Add `focal_astar_search()` beside `astar_search()`. |
| `hrm-cloud/continuous_prm/continuous_prm_c7_hard_maps.py` | **New** | Spiral / bugtrap / rooms_large specs + obstacle generators + `install_c7_hard_maps()` runtime extension. |
| `hrm-cloud/continuous_prm/continuous_prm_providers.py` | **New** | `HeuristicProvider` ABC + `EuclidProvider`, `OracleProvider`, `ScalarResidualProvider`, `ValueFieldProvider`; `build_arms()` enumeration. |
| `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py` | **New** | Orchestrator: CLI/config/scale presets; modes `collect/train/eval/analyze/full/calibrate`; unified sharded eval; metrics. |
| `hrm-cloud/tests/test_c7_focal_prm.py` | **New** | Focal search correctness. |
| `hrm-cloud/tests/test_c7_providers.py` | **New** | Provider correctness. |
| `hrm-cloud/tests/test_c7_hard_maps.py` | **New** | New-suite validity. |
| `hrm-cloud/tests/test_c7_matched_integrity.py` | **New** | Matched-comparison integrity + arm enumeration. |
| `hrm-cloud/continuous_prm/C7_RESULTS.md` | **New** | Publication-grade writeup (Task 13). |

---

## Task 1: `focal_astar_search` in `common.py`

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_common.py` (add after `astar_search`, ~line 754)
- Test: `hrm-cloud/tests/test_c7_focal_prm.py`

Context: `astar_search(adj, heuristic, budget, start_idx=0, goal_idx=1)` (common.py:729) uses heap tuples `(f, g, idx)`, returns `{"found", "cost", "expansions", "closed"}`. `dijkstra_to_goal(adj, goal_idx=1)` (common.py:712) gives exact cost-to-go. Euclid heuristic on a PRM is admissible+consistent (edge weights are straight-line lengths).

- [ ] **Step 1: Write the failing tests**

Create `hrm-cloud/tests/test_c7_focal_prm.py` (copy the sys.path bootstrap from `test_c6_heatmap_value_field.py` first):

```python
import math
import numpy as np
import continuous_prm_common as C


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
    res = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.0)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"] and opt["found"]
    assert math.isclose(res["cost"], opt["cost"], rel_tol=1e-9)


def test_focal_bound_never_violated():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rng = np.random.default_rng(0)
    rank = rng.random(len(adj))  # adversarial-ish ranker
    w = 2.0
    res = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=w)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"]
    assert res["cost"] <= w * opt["cost"] + 1e-9


def test_focal_completeness_and_budget():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rank = np.zeros(len(adj))
    # Budget too small to reach goal -> not found, expansions capped.
    res = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1, w=1.5)
    assert res["expansions"] <= 1
    assert res["found"] is False
    # Ample budget -> found.
    res2 = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.5)
    assert res2["found"]


def test_focal_determinism():
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    rank = np.linspace(0, 1, len(adj))
    a = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.3)
    b = C.focal_astar_search(adj, euclid_h=h, rank_h=rank, budget=1000, w=1.3)
    assert a == b


def test_focal_constant_rank_degrades_to_astar_expansions():
    # Collapsed ranker (constant) -> selection falls through to f -> behaves like A*.
    adj, _ = _line_graph()
    h = _euclid_like_admissible(adj)
    const_rank = np.full(len(adj), 3.14)
    res = C.focal_astar_search(adj, euclid_h=h, rank_h=const_rank, budget=1000, w=1.0)
    opt = C.astar_search(adj, h, budget=1000)
    assert res["found"]
    assert math.isclose(res["cost"], opt["cost"], rel_tol=1e-9)
    assert res["expansions"] <= opt["expansions"] + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest hrm-cloud/tests/test_c7_focal_prm.py -v`
Expected: FAIL with `AttributeError: module 'continuous_prm_common' has no attribute 'focal_astar_search'`.

- [ ] **Step 3: Implement `focal_astar_search`**

Add to `continuous_prm_common.py` immediately after `astar_search` (after line 754):

```python
def focal_astar_search(
    adj: List[List[Tuple[int, float]]],
    euclid_h: np.ndarray,
    rank_h: np.ndarray,
    budget: int,
    w: float = 1.0,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """Bounded-suboptimal focal A* (A*epsilon) on a static weighted graph.

    OPEN is ordered by the admissible f = g + euclid_h (euclid_h must be
    admissible+consistent, e.g. straight-line distance on a PRM). The FOCAL
    set {n in OPEN : f(n) <= w * f_min} is expanded by minimum rank_h (the
    learned cost-to-go estimate), tie-broken by (rank_h, f, insertion_counter).
    Returns the same dict shape as astar_search; cost <= w * optimal.
    """
    if w < 1.0:
        raise ValueError(f"focal w must be >= 1.0, got {w}")
    n = len(adj)
    g = np.full(n, INF, dtype=np.float64)
    g[start_idx] = 0.0
    # OPEN entries: (f, g, node, counter). counter breaks ties deterministically.
    counter = 0
    open_entries: List[Tuple[float, float, int, int]] = [
        (float(euclid_h[start_idx]), 0.0, start_idx, counter)
    ]
    closed = np.zeros(n, dtype=np.bool_)
    expansions = 0
    while open_entries and expansions < budget:
        # Drop stale entries (node closed, or g superseded) from the front-set view.
        live = [e for e in open_entries if not closed[e[2]] and e[1] == g[e[2]]]
        if not live:
            break
        open_entries = live
        f_min = min(e[0] for e in open_entries)
        threshold = w * f_min
        focal = [e for e in open_entries if e[0] <= threshold + 1e-12]
        # Select by (rank_h, f, counter).
        best = min(focal, key=lambda e: (float(rank_h[e[2]]), e[0], e[3]))
        open_entries.remove(best)
        _, cur_g, u, _ = best
        if closed[u] or cur_g != g[u]:
            continue
        closed[u] = True
        expansions += 1
        if u == goal_idx:
            return {"found": True, "cost": float(g[u]), "expansions": expansions, "closed": int(closed.sum())}
        for v, ew in adj[u]:
            if closed[v]:
                continue
            ng = g[u] + ew
            if ng < g[v]:
                g[v] = ng
                counter += 1
                open_entries.append((ng + float(euclid_h[v]), ng, v, counter))
    return {"found": False, "cost": float("nan"), "expansions": expansions, "closed": int(closed.sum())}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest hrm-cloud/tests/test_c7_focal_prm.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_common.py hrm-cloud/tests/test_c7_focal_prm.py
git commit -m "feat(c7): focal A*epsilon search on the PRM graph + tests"
```

---

## Task 2: `HeuristicProvider` ABC + `Euclid` / `Oracle` providers

**Files:**
- Create: `hrm-cloud/continuous_prm/continuous_prm_providers.py`
- Test: `hrm-cloud/tests/test_c7_providers.py`

Context: `Roadmap` (common.py:659) has `.points [N,2]`, `.adj`, `.dist_to_goal`, `.connected_to_goal`. `World` has `.start`, `.goal`, `.side_len`, `.obstacles`. Index 0 = start, 1 = goal. Euclid-to-goal per node = `norm(points - goal, axis=1)`. Exact graph cost-to-go = `dist_to_goal` (or `dijkstra_to_goal(adj)`).

- [ ] **Step 1: Write the failing tests**

Create `hrm-cloud/tests/test_c7_providers.py` (copy sys.path bootstrap):

```python
import numpy as np
import continuous_prm_common as C
import continuous_prm_providers as P


def _tiny_world_and_prm():
    spec = C.build_anchor_specs()["C_open"]
    for seed in range(50):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=64, k_neighbors=8), seed=seed)
        if rm is not None:
            return world, rm
    raise RuntimeError("could not build a tiny world")


def test_euclid_provider_admissible():
    world, rm = _tiny_world_and_prm()
    h = P.EuclidProvider().node_h(world, rm, goal_idx=1)
    assert h.shape == (rm.points.shape[0],)
    assert np.all(np.isfinite(h)) and np.all(h >= -1e-9)
    # Euclid <= exact graph cost-to-go on connected nodes (admissible).
    conn = rm.connected_to_goal
    assert np.all(h[conn] <= rm.dist_to_goal[conn] + 1e-6) if (conn := rm.connected_to_goal).any() else True


def test_oracle_provider_equals_dijkstra():
    world, rm = _tiny_world_and_prm()
    h = P.OracleProvider().node_h(world, rm, goal_idx=1)
    dij = C.dijkstra_to_goal(rm.adj, goal_idx=1)
    conn = np.isfinite(dij)
    assert np.allclose(h[conn], dij[conn], atol=1e-9)
    assert np.all(np.isfinite(h))  # disconnected nodes filled finite, not inf/nan


def test_oracle_makes_astar_optimal_and_minimal():
    world, rm = _tiny_world_and_prm()
    h = P.OracleProvider().node_h(world, rm, goal_idx=1)
    res = C.astar_search(rm.adj, h, budget=10_000)
    assert res["found"]
    assert np.isclose(res["cost"], rm.dist_to_goal[0], rtol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'continuous_prm_providers'`.

- [ ] **Step 3: Implement the ABC + two providers**

Create `hrm-cloud/continuous_prm/continuous_prm_providers.py`:

```python
"""Heuristic providers for the C7 integration comparison.

A provider maps (world, roadmap, goal_idx) -> a finite, non-negative per-node
heuristic array h[N]. The planner consumes h directly (astar) or as the focal
ranker alongside the admissible Euclid array (focal_astar).
"""
from __future__ import annotations

import abc
from typing import Any

import numpy as np

import continuous_prm_common as C


def _finite_fill(vals: np.ndarray, fallback: float) -> np.ndarray:
    out = np.array(vals, dtype=np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fallback
    return np.maximum(out, 0.0)


class HeuristicProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def node_h(self, world: "C.World", roadmap: "C.Roadmap", goal_idx: int = 1) -> np.ndarray:
        ...


class EuclidProvider(HeuristicProvider):
    name = "euclid"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        goal = roadmap.points[goal_idx]
        d = np.linalg.norm(roadmap.points - goal[None, :], axis=1)
        return _finite_fill(d, fallback=0.0)


class OracleProvider(HeuristicProvider):
    """Exact graph cost-to-go (the minimal-expansion ceiling for A* on this graph)."""
    name = "oracle"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        dij = C.dijkstra_to_goal(roadmap.adj, goal_idx=goal_idx)
        finite = np.isfinite(dij)
        fill = float(dij[finite].max() + world.side_len) if finite.any() else 10.0 * world.side_len
        return _finite_fill(dij, fallback=fill)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_providers.py hrm-cloud/tests/test_c7_providers.py
git commit -m "feat(c7): HeuristicProvider ABC + Euclid/Oracle providers + tests"
```

---

## Task 3: New heuristic-hostile suites (runtime-install)

**Files:**
- Create: `hrm-cloud/continuous_prm/continuous_prm_c7_hard_maps.py`
- Test: `hrm-cloud/tests/test_c7_hard_maps.py`

Context: study `continuous_prm_c5_hard_obstacle_encoder.py` first — `build_hard_anchor_specs()` (:120), `generate_hard_obstacles()` (:179), `build_hard_world()` (:205), `install_runtime_extensions()` (:412). The install pattern monkeypatches `C.build_anchor_specs`, the obstacle generator, and `C.build_world` so the rest of the pipeline transparently sees the new suites. `AnchorSpec` (common.py:173) fields and `Obstacle` (common.py:153) (`kind` in {"circle","rect"}, `cx,cy,radius` or rect extents) — read both dataclasses before writing generators. `is_point_free`/`is_segment_free` (common.py:441/452) validate geometry.

Mirror C5's structure. Define three specs and a generator that produces forced-detour geometry:
- `C_hard_spiral` — concatenated arc/segment walls forming a serpentine corridor; start/goal at opposite ends.
- `C_hard_bugtrap` — a concave U/G pocket straddling the start→goal line + clutter.
- `C_hard_rooms_large` — side ≈ 3.0; a grid of rooms separated by walls with single doorways; clutter inside rooms.

The C7 install must compose with C5's (we need the C5 hard *encoder* for the scalar arm's features). So `install_c7_hard_maps()` calls C5's install first, then extends the suite registry with the three new specs and routes their obstacle generation to the C7 generators (delegating other suites back to the C5/base generator).

- [ ] **Step 1: Write the failing tests**

Create `hrm-cloud/tests/test_c7_hard_maps.py` (copy sys.path bootstrap):

```python
import numpy as np
import continuous_prm_common as C
import continuous_prm_c7_hard_maps as M

NEW_SUITES = ["C_hard_spiral", "C_hard_bugtrap", "C_hard_rooms_large"]


def test_install_registers_new_suites():
    M.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    for s in NEW_SUITES:
        assert s in specs
    # C5 hard suites still present (composition preserved).
    assert "C_hard_maze" in specs


def test_new_suites_build_valid_connected_worlds():
    M.install_c7_hard_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in NEW_SUITES:
        spec = C.build_anchor_specs()[suite]
        built = 0
        for seed in range(40):
            world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if world is None:
                continue
            # start/goal must be free and obstacles within bounds
            assert C.is_point_free(world.start, world.side_len, world.obstacles)
            assert C.is_point_free(world.goal, world.side_len, world.obstacles)
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is not None and rm.connected_to_goal[0]:
                built += 1
            if built >= 5:
                break
        assert built >= 5, f"{suite}: only built {built} connected worlds in 40 seeds"


def test_new_suites_force_detours():
    # On these maps the optimal graph path should be meaningfully longer than
    # the straight-line distance (heuristic-hostile geometry).
    M.install_c7_hard_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in NEW_SUITES:
        spec = C.build_anchor_specs()[suite]
        ratios = []
        for seed in range(60):
            world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if world is None:
                continue
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            straight = float(np.linalg.norm(world.start - world.goal))
            ratios.append(rm.dist_to_goal[0] / max(1e-6, straight))
            if len(ratios) >= 8:
                break
        assert len(ratios) >= 8
        assert float(np.median(ratios)) >= 1.15, f"{suite}: detour ratio {np.median(ratios):.2f} too low"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest hrm-cloud/tests/test_c7_hard_maps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'continuous_prm_c7_hard_maps'`.

- [ ] **Step 3: Implement the maps module**

Create `hrm-cloud/continuous_prm/continuous_prm_c7_hard_maps.py`. Read `AnchorSpec`/`Obstacle`/`build_hard_world` first and reuse their fields exactly. Skeleton (fill obstacle geometry to match the dataclass field names you observe — do not invent fields):

```python
"""C7 heuristic-hostile suites, installed via the C5 runtime-extension pattern.

Adds C_hard_spiral, C_hard_bugtrap, C_hard_rooms_large by composing on top of
the C5 hard install (which we still need for the scalar arm's feature encoder).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

import continuous_prm_common as C
import continuous_prm_c5_hard_obstacle_encoder as C5

NEW_SUITES = ("C_hard_spiral", "C_hard_bugtrap", "C_hard_rooms_large")
_INSTALLED = False


def build_c7_anchor_specs() -> Dict[str, C.AnchorSpec]:
    """Return the three new specs. Match AnchorSpec field names from common.py:173.
    Use spec.mode='narrow' so build_world forces start/goal to opposite sides
    (the detour-forcing path used by C5 hard maps); set is_ood per the split."""
    specs: Dict[str, C.AnchorSpec] = {}
    # NOTE: copy the construction style from C5.build_hard_anchor_specs() (:120);
    # set name, side_len, mode, obstacle-count ranges, gap params, is_ood.
    specs["C_hard_spiral"] = C.AnchorSpec(name="C_hard_spiral", side_len=1.0, mode="narrow", is_ood=False)  # + count/gap fields
    specs["C_hard_bugtrap"] = C.AnchorSpec(name="C_hard_bugtrap", side_len=1.0, mode="narrow", is_ood=True)
    specs["C_hard_rooms_large"] = C.AnchorSpec(name="C_hard_rooms_large", side_len=3.0, mode="narrow", is_ood=True)
    return specs


def _wall_segment_as_rects(x0, y0, x1, y1, thickness, gaps: Sequence[tuple]) -> List[C.Obstacle]:
    """Build a wall from (x0,y0)->(x1,y1) as a chain of thin rect obstacles,
    leaving 'gaps' (list of (frac_start, frac_end)) open as doorways.
    Construct C.Obstacle(kind='rect', ...) using the rect field names from
    common.py:153 (read them; do NOT guess)."""
    raise NotImplementedError  # implement per Obstacle rect fields


def generate_c7_obstacles(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    if spec.name == "C_hard_spiral":
        return _gen_spiral(spec, rng)
    if spec.name == "C_hard_bugtrap":
        return _gen_bugtrap(spec, rng)
    if spec.name == "C_hard_rooms_large":
        return _gen_rooms_large(spec, rng)
    raise KeyError(spec.name)


def _gen_spiral(spec, rng) -> List[C.Obstacle]:
    """Serpentine: alternating horizontal walls each leaving a gap on opposite
    sides, so the path must weave up/down the box. Add a few clutter circles
    via C5._add_random_circles(spec, rng, obstacles, n)."""
    raise NotImplementedError


def _gen_bugtrap(spec, rng) -> List[C.Obstacle]:
    """Concave pocket: three walls forming a U whose mouth faces the start, placed
    across the start->goal line; clutter around it."""
    raise NotImplementedError


def _gen_rooms_large(spec, rng) -> List[C.Obstacle]:
    """Grid of rooms: vertical+horizontal walls partitioning the 3.0 box into a
    2x3 (or 3x3) grid, each interior wall with one doorway; clutter inside cells."""
    raise NotImplementedError


def install_c7_hard_maps(sector_tokens: int = 16) -> None:
    """Compose on top of the C5 hard install, then register the C7 suites and
    route their obstacle generation to generate_c7_obstacles (delegating other
    suites to the existing generator)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from continuous_prm_c6_heatmap_value_field import install_c5_hard_runtime
    install_c5_hard_runtime(sector_tokens=sector_tokens)

    base_specs_fn = C.build_anchor_specs
    base_world_fn = C.build_world
    # Capture whatever obstacle generator C5 installed.
    base_obs_fn = C.generate_obstacles

    def patched_specs() -> Dict[str, C.AnchorSpec]:
        specs = dict(base_specs_fn())
        specs.update(build_c7_anchor_specs())
        return specs

    def patched_generate_obstacles(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
        if spec.name in NEW_SUITES:
            return generate_c7_obstacles(spec, rng)
        return base_obs_fn(spec, rng)

    def patched_build_world(spec: C.AnchorSpec, seed: int, min_start_goal_dist_frac: float):
        # The C7 suites use the same narrow/opposite-side endpoint logic as C5
        # hard maps. If C5's build_hard_world handles 'narrow' detours, route to it.
        if spec.name in NEW_SUITES:
            return C5.build_hard_world(spec, seed, min_start_goal_dist_frac)
        return base_world_fn(spec, seed, min_start_goal_dist_frac)

    C.build_anchor_specs = patched_specs
    C.generate_obstacles = patched_generate_obstacles
    C.build_world = patched_build_world
    _INSTALLED = True
```

Implementation notes for Step 3:
- Read `Obstacle` (common.py:153) and `AnchorSpec` (common.py:173) and use their **exact** field names. The `AnchorSpec(...)` calls above are illustrative — set every required field (count ranges, gaps) as C5 does.
- Implement the three `_gen_*` and `_wall_segment_as_rects` with real geometry (no `NotImplementedError` left). Validate by Step 4.
- If `C5.build_hard_world` is too specialized, fall back to `base_world_fn` and instead force start/goal placement inside the generator by reserving free endpoints; the detour test will catch a bad placement.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest hrm-cloud/tests/test_c7_hard_maps.py -v`
Expected: PASS (3 tests). If `test_new_suites_force_detours` fails, increase wall coverage / reduce doorway count until median detour ratio ≥ 1.15.

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_hard_maps.py hrm-cloud/tests/test_c7_hard_maps.py
git commit -m "feat(c7): heuristic-hostile suites (spiral/bugtrap/rooms_large) via runtime install"
```

---

## Task 4: `ScalarResidualProvider`

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_providers.py`
- Modify: `hrm-cloud/tests/test_c7_providers.py`

Context: the scalar (C5-style) heuristic is `h = euclid + side_len * clip(yhat, 0, max_norm_residual)`. Reuse: `make_hard_features_for_roadmap(world, roadmap, feature_cfg)` (C5 :375), `predict_norm_residuals(model, features, device, max_norm_residual)` (common.py:1527), and model loaders `load_base_model(...)` (common.py:1543). The provider holds a loaded `ContinuousHeuristicModel` + `FeatureConfig` + device + `max_norm_residual`.

- [ ] **Step 1: Add the failing test**

Append to `hrm-cloud/tests/test_c7_providers.py`:

```python
def test_scalar_provider_matches_additive_formula():
    import torch
    import continuous_prm_c7_hard_maps as M
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    world = None
    for seed in range(60):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is not None:
            rm = C.build_prm(world, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
            if rm is not None and rm.connected_to_goal[0]:
                break
    assert world is not None and rm is not None

    prov = P.ScalarResidualProvider.untrained_for_test(world)  # tiny random model on CPU
    h = prov.node_h(world, rm, goal_idx=1)
    euclid = P.EuclidProvider().node_h(world, rm, goal_idx=1)
    assert h.shape == euclid.shape
    assert np.all(np.isfinite(h))
    # additive & >= euclid (non-negative residual)
    assert np.all(h >= euclid - 1e-6)
    # residual bounded: h - euclid <= side_len * max_norm_residual + eps
    assert np.all((h - euclid) <= world.side_len * prov.max_norm_residual + 1e-4)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py::test_scalar_provider_matches_additive_formula -v`
Expected: FAIL (`AttributeError: ... ScalarResidualProvider`).

- [ ] **Step 3: Implement `ScalarResidualProvider`**

Append to `continuous_prm_providers.py` (read the exact signatures of `make_hard_features_for_roadmap`, `predict_norm_residuals`, `build_model`/`load_base_model`, `FeatureConfig`, `BackboneConfig`, `TrainingConfig` before finalizing):

```python
import torch
import continuous_prm_c5_hard_obstacle_encoder as C5


class ScalarResidualProvider(HeuristicProvider):
    """Additive per-node scalar residual: h = euclid + side_len * clip(yhat,0,B)."""

    def __init__(self, model, feature_cfg, device, backbone: str, max_norm_residual: float):
        self.model = model
        self.feature_cfg = feature_cfg
        self.device = device
        self.max_norm_residual = float(max_norm_residual)
        self.name = f"scalar_{backbone}"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        feats = C5.make_hard_features_for_roadmap(world, roadmap, self.feature_cfg)
        yhat = C.predict_norm_residuals(self.model, feats, self.device, self.max_norm_residual)
        yhat = np.clip(np.asarray(yhat, dtype=np.float64), 0.0, self.max_norm_residual)
        goal = roadmap.points[goal_idx]
        euclid = np.linalg.norm(roadmap.points - goal[None, :], axis=1)
        h = euclid + world.side_len * yhat
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("ScalarResidualProvider produced non-finite h")
        return np.maximum(h, 0.0)

    @classmethod
    def untrained_for_test(cls, world, device=None):
        """Tiny random model on CPU for unit tests (no checkpoint)."""
        device = device or torch.device("cpu")
        feature_cfg = C.FeatureConfig()  # defaults match the installed C5 encoder
        bb = C.build_backbone_configs(_dummy_args())["hrm"]
        train_cfg = C.TrainingConfig()
        model = C.build_model(bb, feature_cfg, train_cfg, device).eval()
        return cls(model, feature_cfg, device, backbone="hrm", max_norm_residual=train_cfg.max_norm_residual)
```

Add a small `_dummy_args()` helper that returns an `argparse.Namespace` with the fields `build_backbone_configs` reads (inspect common.py:265). If `FeatureConfig()` defaults don't match the installed C5 encoder dims, construct it the way `continuous_prm_c5_hard_obstacle_encoder` / C6 builds it (look at how C6 builds the eval bundle / feature cfg) and reuse that.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py::test_scalar_provider_matches_additive_formula -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_providers.py hrm-cloud/tests/test_c7_providers.py
git commit -m "feat(c7): ScalarResidualProvider (additive C5-style heuristic) + test"
```

---

## Task 5: `ValueFieldProvider`

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_providers.py`
- Modify: `hrm-cloud/tests/test_c7_providers.py`

Context: the C6 field heuristic is built inside `evaluate_shard` (c6 :965) — read it to see exactly how a model's predicted grid becomes a per-node `h`. The relevant helpers: `predict_residual_grid(model, x, device)` (c6 :901), `make_heatmap_example` / occupancy `x` construction (c6 :333), `interpolate_grid_values(grid, world, points)` (c6 :371), and `make_oracle_heuristic(world, roadmap, grid_distance, euclid_h)` (c6 :927) as the reference shape. `ValueFieldProvider` must reproduce C6's field→node-h mapping exactly.

- [ ] **Step 1: Add the failing test**

Append to `hrm-cloud/tests/test_c7_providers.py`:

```python
def test_value_field_provider_reproduces_c6_oracle_path():
    import continuous_prm_c7_hard_maps as M
    import continuous_prm_c6_heatmap_value_field as C6
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    for seed in range(60):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            break
    # Oracle field provider should equal C6.make_oracle_heuristic exactly.
    grid_size = 64
    free, _, _ = C6.rasterize_world(world, grid_size)
    grid_dist = C6.grid_dijkstra_to_goal(world, free)
    euclid = P.EuclidProvider().node_h(world, rm, goal_idx=1)
    ref = C6.make_oracle_heuristic(world, rm, grid_dist, euclid)
    prov = P.ValueFieldProvider.oracle_for_test(grid_size=grid_size)
    got = prov.node_h(world, rm, goal_idx=1)
    assert np.allclose(got, ref, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py::test_value_field_provider_reproduces_c6_oracle_path -v`
Expected: FAIL (`AttributeError: ... ValueFieldProvider`).

- [ ] **Step 3: Implement `ValueFieldProvider`**

Append to `continuous_prm_providers.py`:

```python
class ValueFieldProvider(HeuristicProvider):
    """Cost-to-go field sampled at PRM nodes (mirrors C6's field->h mapping)."""

    def __init__(self, model, grid_size: int, device, backbone: str, is_oracle: bool = False):
        self.model = model
        self.grid_size = int(grid_size)
        self.device = device
        self.is_oracle = is_oracle
        self.name = "grid_oracle" if is_oracle else f"field_{backbone}"

    def _grid_distance(self, world):
        import continuous_prm_c6_heatmap_value_field as C6
        if self.is_oracle:
            free, _, _ = C6.rasterize_world(world, self.grid_size)
            return C6.grid_dijkstra_to_goal(world, free)
        # Learned: build occupancy input x, predict residual grid, convert to a
        # cost-to-go grid EXACTLY as evaluate_shard does. Factor that conversion
        # into a shared helper if it is inline in evaluate_shard.
        x = C6.make_heatmap_example(world, self.grid_size)["x"]  # confirm key name
        return C6.predict_residual_grid(self.model, x, self.device)

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        import continuous_prm_c6_heatmap_value_field as C6
        goal = roadmap.points[goal_idx]
        euclid = np.linalg.norm(roadmap.points - goal[None, :], axis=1)
        grid = self._grid_distance(world)
        h = C6.make_oracle_heuristic(world, roadmap, grid, euclid)
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("ValueFieldProvider produced non-finite h")
        return np.maximum(h, 0.0)

    @classmethod
    def oracle_for_test(cls, grid_size=64):
        return cls(model=None, grid_size=grid_size, device=None, backbone="oracle", is_oracle=True)
```

Important: the learned branch must match `evaluate_shard`'s field→h pipeline **exactly** (same occupancy construction, same residual→distance conversion, same `make_oracle_heuristic` wrap). If `evaluate_shard` does the conversion inline, extract it into a small function in the C6 module (e.g. `field_node_heuristic(model, world, roadmap, grid_size, device, euclid)`) and call it from both `evaluate_shard` and here — DRY, and guarantees parity. Add that extraction as part of this step and re-run the C6 tests (`python -m pytest hrm-cloud/tests/test_c6_heatmap_value_field.py -v`) to confirm no regression.

- [ ] **Step 4: Run to verify pass (both C7 and C6 tests)**

Run: `python -m pytest hrm-cloud/tests/test_c7_providers.py hrm-cloud/tests/test_c6_heatmap_value_field.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_providers.py hrm-cloud/continuous_prm/continuous_prm_c6_heatmap_value_field.py hrm-cloud/tests/test_c7_providers.py
git commit -m "feat(c7): ValueFieldProvider (C6 field->node-h parity) + DRY extraction"
```

---

## Task 6: Arm enumeration + matched-integrity test

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_providers.py`
- Test: `hrm-cloud/tests/test_c7_matched_integrity.py`

Define the canonical arm list and a helper that, given a world+roadmap and the loaded providers, runs every arm and returns per-arm records — used by the orchestrator and pinned by tests.

- [ ] **Step 1: Write the failing test**

Create `hrm-cloud/tests/test_c7_matched_integrity.py` (copy bootstrap):

```python
import numpy as np
import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_hard_maps as M


def _world_prm():
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    for seed in range(60):
        w = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if w is None:
            continue
        rm = C.build_prm(w, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            return w, rm
    raise RuntimeError("no world")


def test_run_arm_records_shape_and_suboptimality():
    world, rm = _world_prm()
    providers = {"euclid": P.EuclidProvider(), "oracle": P.OracleProvider()}
    recs = P.run_world_arms(world, rm, providers, budgets=[200], w_values=[1.0, 1.5], goal_idx=1)
    # euclid: 1 astar arm; oracle: 1 astar arm; each provider also gets focal arms per w.
    names = {(r["provider"], r["mode"], r.get("w")) for r in recs}
    assert ("euclid", "astar", None) in names
    assert ("oracle", "astar", None) in names
    assert ("euclid", "focal", 1.0) in names
    for r in recs:
        assert set(["provider", "mode", "budget", "found", "expansions", "cost", "suboptimality"]).issubset(r)
        if r["found"]:
            assert r["suboptimality"] >= 1.0 - 1e-6
            if r["mode"] == "focal":
                assert r["suboptimality"] <= r["w"] + 1e-6


def test_matched_worlds_identical_across_providers():
    # Same seed -> identical roadmap geometry (the matched guarantee).
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    w1 = C.build_world(spec, seed=7, min_start_goal_dist_frac=0.5)
    w2 = C.build_world(spec, seed=7, min_start_goal_dist_frac=0.5)
    rm1 = C.build_prm(w1, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=7)
    rm2 = C.build_prm(w2, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=7)
    assert np.array_equal(rm1.points, rm2.points)
    assert rm1.adj == rm2.adj
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest hrm-cloud/tests/test_c7_matched_integrity.py -v`
Expected: FAIL (`AttributeError: ... run_world_arms`).

- [ ] **Step 3: Implement `run_world_arms` + arm spec**

Append to `continuous_prm_providers.py`:

```python
def run_world_arms(world, roadmap, providers: dict, budgets, w_values, goal_idx: int = 1, start_idx: int = 0):
    """Run every (provider, mode, budget[, w]) arm on one shared world+roadmap.

    - astar mode for all providers (additive/direct).
    - focal mode for all providers, for each w in w_values (Euclid+collapsed
      signals degrade to A* automatically).
    Returns a flat list of record dicts with matched suboptimality.
    """
    opt = float(roadmap.dist_to_goal[start_idx])  # graph optimal cost from start
    euclid_h = providers["euclid"].node_h(world, roadmap, goal_idx) if "euclid" in providers \
        else np.linalg.norm(roadmap.points - roadmap.points[goal_idx][None, :], axis=1)
    # Precompute each provider's node_h once (matched + efficient).
    node_h = {name: prov.node_h(world, roadmap, goal_idx) for name, prov in providers.items()}
    records = []
    for name in providers:
        h = node_h[name]
        for b in budgets:
            r = C.astar_search(roadmap.adj, h, budget=int(b), start_idx=start_idx, goal_idx=goal_idx)
            records.append(_arm_record(name, "astar", None, b, r, opt))
            for w in w_values:
                rf = C.focal_astar_search(roadmap.adj, euclid_h=euclid_h, rank_h=h,
                                          budget=int(b), w=float(w), start_idx=start_idx, goal_idx=goal_idx)
                records.append(_arm_record(name, "focal", float(w), b, rf, opt))
    return records


def _arm_record(provider, mode, w, budget, res, opt):
    found = bool(res["found"])
    cost = float(res["cost"]) if found else float("nan")
    sub = (cost / opt) if (found and opt > 0) else float("nan")
    return {
        "provider": provider, "mode": mode, "w": w, "budget": int(budget),
        "found": found, "expansions": int(res["expansions"]), "cost": cost,
        "optimal": opt, "suboptimality": sub,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest hrm-cloud/tests/test_c7_matched_integrity.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_providers.py hrm-cloud/tests/test_c7_matched_integrity.py
git commit -m "feat(c7): run_world_arms enumeration + matched-integrity tests"
```

**GATE 0 COMPLETE** when Tasks 1–6 tests all pass:
`python -m pytest hrm-cloud/tests/test_c7_*.py -v`

---

## Task 7: Orchestrator skeleton — CLI, config, scale presets

**Files:**
- Create: `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py`

Mirror C6's CLI/config style (`continuous_prm_c6_heatmap_value_field.py`: `C6Config` :122, `config_from_args` :149, `parse_args` :1289, `main` :1322, modes `collect/train/eval/analyze/full`). Add a `calibrate` mode and a `--scale {local,cluster}` preset that sets defaults.

- [ ] **Step 1: Implement skeleton with `--help` working**

Create the file with: `C7Config` dataclass (fields: grid_size=64, roadmap_nodes=192, roadmap_k=7, train_tasks="C_hard_maze,C_hard_rooms,C_hard_spiral", eval_suites="C_hard_maze,C_hard_maze_dense,C_hard_rooms,C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large", scalar_backbones="hrm,onlstm", field_backbones="unet,onlstm,hrm", budgets per-suite (filled by calibrate), w_values="1.0,1.05,1.1,1.25", eval_worlds, train_worlds, epochs, seed=1234, out_dir, mode, scale); `apply_scale_preset(cfg)`; `parse_args()`; `main()` dispatching on mode (stubs raising `NotImplementedError("task N")` for now). At module import, call `continuous_prm_c7_hard_maps.install_c7_hard_maps()` so all suites exist.

Scale presets:
```python
def apply_scale_preset(cfg):
    if cfg.scale == "local":
        cfg.eval_worlds = cfg.eval_worlds or 24
        cfg.train_worlds = cfg.train_worlds or 96
        cfg.epochs = cfg.epochs or 16
        cfg.w_values = cfg.w_values or "1.0,1.1"
        cfg.budget_grid_size = 2
    else:  # cluster
        cfg.eval_worlds = cfg.eval_worlds or 120
        cfg.train_worlds = cfg.train_worlds or 160
        cfg.epochs = cfg.epochs or 24
        cfg.w_values = cfg.w_values or "1.0,1.05,1.1,1.25"
        cfg.budget_grid_size = 3
    return cfg
```

- [ ] **Step 2: Verify CLI loads**

Run: `python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --help`
Expected: argparse help prints with all flags incl. `--mode`, `--scale`, `--out-dir`; no import errors.

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py
git commit -m "feat(c7): orchestrator skeleton (CLI, C7Config, scale presets, suite install)"
```

---

## Task 8: `train` mode — both model families on the C7 split

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py`

Train (a) field models {unet, onlstm, hrm} via the C6 path and (b) scalar avgbase models {hrm, onlstm} via the C5/common path — both on `train_tasks` (maze/rooms/spiral). Reuse, do not reimplement:
- Field: `C6.collect_dataset(...)` (:398) per train task → merged dataset; `C6.train_model(...)` (:832) per backbone; checkpoints via `C6.checkpoint_path(out_dir, name)`.
- Scalar: `C.collect_task_dataset(...)` (:911) per train task; `C.train_avgbase(...)` (:1337) per backbone on the pooled dataset; checkpoints via `C.model_checkpoint_path(out_dir, backbone, kind="avgbase")`.

- [ ] **Step 1: Implement `run_train(out_dir, cfg, device)`**

Wire the two paths. Keep field checkpoints under `out_dir/checkpoints/field_*` and scalar under `out_dir/checkpoints/scalar_*`. Write a `train_manifest.json` recording which checkpoints exist. Reference the exact arg objects each function needs (build `RoadmapConfig/FeatureConfig/TrainingConfig/BackboneConfig` via `C.build_configs_from_args`-style construction; for C6, build `C6Config` from the C7 config fields). Log per-epoch losses to `out_dir/logs/`.

- [ ] **Step 2: Smoke-run train on CPU (tiny)**

Run:
```bash
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode train --scale local --cpu \
  --train-tasks C_hard_spiral --train-worlds 4 --epochs 1 --out-dir hrm-cloud/continuous_prm/runs/c7_smoke
```
Expected: completes; `runs/c7_smoke/checkpoints/` contains both a field and a scalar checkpoint; no nonfinite-loss errors.

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py
git commit -m "feat(c7): train mode (field via C6 path + scalar avgbase via C5 path)"
```

---

## Task 9: `eval` mode — unified sharded arm evaluation

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py`

For each eval suite, build the per-suite providers (load trained checkpoints into `ScalarResidualProvider`/`ValueFieldProvider` + `EuclidProvider`/`OracleProvider`), generate seeded worlds+PRMs, and call `P.run_world_arms(...)` per world. Write per-shard raw CSV (`out_dir/results/_shards/c7/<suite>/shard_XXXX.csv`) and merge to `out_dir/results/continuous_prm_c7_eval_raw.csv` — reuse C6's shard/merge helpers (`merge_eval_shards` :1227) or mirror them. Records carry: suite, world_idx, provider, mode, w, budget, found, expansions, cost, optimal, suboptimality.

- [ ] **Step 1: Implement `run_eval(out_dir, cfg, device)`**

Use the per-suite calibrated budgets (from Task 10's `calibration.json` if present; else `cfg`'s default budgets). Per suite: loop world seeds via `C.build_world` + `C.build_prm` (skip invalid worlds, same retry policy as C6's `make_eval_world_bundle`); build providers once per suite (checkpoints loaded once); accumulate records; write shard CSV incrementally (survives interruption). Add the nonfinite guard: any provider raising `FloatingPointError` records the arm as `found=False, nonfinite=1` and increments a per-suite counter logged at the end (do not silently zero-fill).

- [ ] **Step 2: Smoke-run eval on CPU after the Task 8 smoke train**

Run:
```bash
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode eval --scale local --cpu \
  --eval-suites C_hard_spiral --eval-worlds 4 --budgets 200 --w-values 1.0 \
  --out-dir hrm-cloud/continuous_prm/runs/c7_smoke
```
Expected: `runs/c7_smoke/results/continuous_prm_c7_eval_raw.csv` exists with rows for euclid/oracle/scalar/field × {astar, focal}; `suboptimality` finite for found rows; focal rows obey `suboptimality <= w`.

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py
git commit -m "feat(c7): eval mode (unified sharded arm eval, suboptimality + nonfinite guard)"
```

---

## Task 10: `calibrate` mode — binding-budget band (GATE 1)

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py`

For each eval suite, run **Euclid + Oracle only** across a budget grid and pick the in-band budgets (Euclid success ∈ ~[0.40, 0.60] for the low/mid points; verify oracle success < ~0.95 so there is headroom). Write `out_dir/calibration.json` mapping suite → chosen budgets (`budget_grid_size` of them) + the measured Euclid/oracle success at each.

- [ ] **Step 1: Implement `run_calibrate(out_dir, cfg, device)`**

Sweep budgets (e.g. a coarse grid `[64, 96, 128, 144, 168, 200, 256, 320]`); for each suite run Euclid-astar + Oracle-astar over `cfg.eval_worlds` seeded worlds; record success per budget; select the budgets whose Euclid success is closest to {0.45, 0.55, 0.65} (taking `budget_grid_size` of them); flag (log a WARNING) any suite where oracle ≥ 0.95 at the chosen budgets (insufficient headroom) — that suite needs harder generation (Task 3 tuning) or denser sampling.

- [ ] **Step 2: Run calibration locally (real GPU, all suites)**

Run:
```bash
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode calibrate --scale local \
  --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms,C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large \
  --out-dir hrm-cloud/continuous_prm/runs/c7_local
```
Expected: `runs/c7_local/calibration.json` written; each suite has Euclid in ~[0.40,0.60] at its chosen budgets and oracle < 0.95. **If any suite is out of band, fix Task 3 generation or roadmap density and re-run before proceeding.**

- [ ] **Step 3: Commit (code + calibration artifact)**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py hrm-cloud/continuous_prm/runs/c7_local/calibration.json
git commit -m "feat(c7): calibrate mode + local binding-budget bands (Gate 1)"
```

---

## Task 11: `analyze` mode — stats + pre-registered comparisons + figures

**Files:**
- Modify: `hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py`

From the raw CSV: build a per-`(suite,arm,budget,w)` summary CSV; run McNemar+BH on success (each arm vs euclid) reusing `C6.mcnemar_exact_p` (:1099) + `C6.bh_q_values` (:1108) + the structure of `C6.analyze_significance` (:1123); add paired Wilcoxon signed-rank + bootstrap CI on expansion ratios over the matched set (instances solved by Euclid); emit the six pre-registered comparisons to a markdown report; render figures.

- [ ] **Step 1: Implement `run_analyze(out_dir, cfg)`**

Outputs:
- `results/continuous_prm_c7_eval_summary.csv` — per `(suite,provider,mode,w,budget)`: success_rate, expansions_mean/median, suboptimality_mean/p95, spearman_to_oracle, gap_to_oracle.
- `results/continuous_prm_c7_significance.md` — McNemar/BH success table + Wilcoxon/CI expansion table.
- `results/continuous_prm_c7_preregistered.md` — the six comparisons (Section 6.9 of the spec): (1) field-HRM vs euclid; (2) scalar-HRM-additive vs field-HRM; (3) scalar-HRM-focal vs scalar-HRM-additive; (4) field-focal vs field-additive; (5) learned vs oracle gap; (6) in-dist vs each held-out axis. Each as a small table with effect + p/q + CI.
- Figures via `matplotlib` (guard import like C6): expansion-ratio bars, gap-to-ceiling, suboptimality-vs-w curve, saved under `figures/`.

Matched-set expansion stat helper:
```python
from scipy.stats import wilcoxon  # if scipy unavailable, implement signed-rank manually
def matched_expansion_stats(euclid_rows, arm_rows):
    # align by (suite, world_idx, budget); keep instances where euclid found.
    pairs = [(e["expansions"], a["expansions"]) for e, a in _align(euclid_rows, arm_rows) if e["found"]]
    ...
```
If `scipy` is not installed, implement a manual Wilcoxon signed-rank + a percentile bootstrap CI on the median ratio (no new heavy dependency required).

- [ ] **Step 2: Run analyze on the smoke results (shape check)**

Run:
```bash
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode analyze \
  --out-dir hrm-cloud/continuous_prm/runs/c7_smoke
```
Expected: summary CSV + significance MD + preregistered MD written without error (numbers meaningless at smoke scale; we are checking the pipeline).

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py
git commit -m "feat(c7): analyze mode (McNemar/BH + Wilcoxon/CI + 6 pre-registered comparisons + figures)"
```

---

## Task 12: Local validation run (GATE 2)

**Files:** none (produces artifacts under `runs/c7_local`)

Full pipeline at `local` scale on the real GPU, using the calibrated budgets.

- [ ] **Step 1: Run `full` locally**

Run (tee to a persistent repo-dir log so it survives session restarts):
```bash
python -u hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode full --scale local \
  --train-tasks C_hard_maze,C_hard_rooms,C_hard_spiral \
  --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms,C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large \
  --out-dir hrm-cloud/continuous_prm/runs/c7_local 2>&1 | tee hrm-cloud/continuous_prm/runs/c7_local_console.log
```
Expected: completes end-to-end; raw + summary + significance + preregistered MD + figures produced.

- [ ] **Step 2: Directional sanity gate** (verify before claiming success)

Confirm in `runs/c7_local/results/continuous_prm_c7_preregistered.md`:
- `field-HRM` beats `euclid` on expansions on ≥1 in-dist suite (expected from C6).
- `scalar-HRM-additive` does **not** beat `field-HRM` (the representation lever; expected from C5→C6).
- focal arms never violate the bound (all `suboptimality ≤ w`).
- oracle is at/above all learned arms (ceiling sane).
If any of these is wildly off (e.g., focal violates the bound, oracle worse than euclid), STOP and debug — it indicates a wiring bug, not a scientific result.

- [ ] **Step 3: Commit the local run's text artifacts** (binaries excluded by existing `runs/.gitignore`)

```bash
git add hrm-cloud/continuous_prm/runs/c7_local/results hrm-cloud/continuous_prm/runs/c7_local/logs hrm-cloud/continuous_prm/runs/c7_local/calibration.json
git commit -m "results(c7): local validation run (Gate 2) — text artifacts"
```

**GATE 3 (cluster scale-up)** is conditional and user-decided. When approved, the same command with `--scale cluster` and more shards produces headline numbers — no code change.

---

## Task 13: Writeup — `C7_RESULTS.md`

**Files:**
- Create: `hrm-cloud/continuous_prm/C7_RESULTS.md`

- [ ] **Step 1: Write the results doc**

Follow the `C6_RESULTS.md` structure. Sections: context + the integration matrix; methodology (providers, two planner modes, focal bound, suites, calibration); the per-suite results tables (success + expansions + suboptimality) pulled from `continuous_prm_c7_eval_summary.csv`; the six pre-registered comparison outcomes (from `continuous_prm_c7_preregistered.md`); gap-to-ceiling; the focal-rescues-scalar-HRM verdict; generalization across the three OOD axes; threats-to-validity; and bidirectional links to `C6_RESULTS.md`, `../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md`, `../EXPERIMENT_RESULTS_COMPENDIUM.md`, and the spec. Quote actual numbers from the local run; mark them "local validation" and note the conditional cluster confirmation.

- [ ] **Step 2: Commit**

```bash
git add hrm-cloud/continuous_prm/C7_RESULTS.md
git commit -m "docs(c7): integration-comparison results writeup (local validation)"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** provider interface (T2,4,5) ✓; two planner modes incl. focal+bound (T1) ✓; new suites + calibration (T3,T10) ✓; train/held-out split (T7 config, T8 train, T9 eval) ✓; arms matrix (T6) ✓; metrics incl. suboptimality + gap-to-ceiling + spearman (T6,T9,T11) ✓; binding-budget sweep + expansions-primary (T10,T11) ✓; McNemar/BH + Wilcoxon/CI (T11) ✓; six pre-registered comparisons (T11) ✓; scale presets local→cluster (T7,T12) ✓; staged gates (T6/T10/T12) ✓; tests incl. matched-integrity + nonfinite guard (T1–T6,T9) ✓; writeup (T13) ✓.
- **Placeholder scan:** the only `NotImplementedError`/`raise` markers are in Task 3's skeleton, with an explicit Step-3 instruction to implement real geometry before Step 4 (and the detour test enforces it) and Task 7's mode stubs filled by Tasks 8–11. No vague "add error handling" steps.
- **Type consistency:** `node_h(world, roadmap, goal_idx)` signature consistent across all providers (T2,4,5,6); `run_world_arms`/`_arm_record` record keys consistent with the matched-integrity test and the eval writer (T6,T9); `focal_astar_search(adj, euclid_h, rank_h, budget, w, start_idx, goal_idx)` signature consistent T1↔T6.
- **Open dependency to verify during execution:** exact field names of `AnchorSpec`/`Obstacle`/`FeatureConfig` and the inline field→h logic in `evaluate_shard` — each task step says to read the source before finalizing; tests pin the behavior.
</content>
