# Learned Focal Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-integrate the existing learned heuristic (0.99-correlated ranker, miscalibrated magnitude) as a *focal-search ordering signal* so it reduces A* expansions instead of inflating the heuristic.

**Architecture:** Add a drop-in `space_time_focal_astar` planner (admissible Manhattan defines `f`; learned `manhattan+δ` orders the focal band within `w·f_min`; returns the same `PlanResult`). Select it via `PLANNER`/`FOCAL_W` env knobs in the live `run_policy_episode`. Validate locally (CPU unit tests + a local-GPU matched-expansion benchmark) — no Modal/cloud spend. Reuses existing trained models; no retraining.

**Tech Stack:** Python 3.10, NumPy, PyTorch (CPU + local CUDA), `heapq`, pytest. Modal only for the deferred Phase-B at-scale run.

**Spec:** `docs/superpowers/specs/2026-06-23-learned-focal-search-design.md`

---

## Context the implementer needs

- **Dual-copy file gotcha:** `hrm-cloud/residual_tasklora_v2.py` has a v1 base (lines 1–~3200) and a "Residual task LoRA v2 overrides" block (~3200–end). The **live** `run_policy_episode` is the later copy (~`:4501`, planner call at `:4710`); the copy at `:2561` is DEAD. `space_time_astar` (`:725`), `PlanResult` (`:706`), `_reconstruct_path_states` (`:714`), `manhattan` (`:477`), `simulate_occupancy` (`:656`), `make_episode` (`:812`), `compute_true_cost_to_goal`, `ACTIONS`, `WAIT_ACTION`, `INF` exist once (shared).
- Run pytest from inside `hrm-cloud`: `cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python -m pytest tests/test_focal.py -v`.
- The module imports cleanly locally (~3.6s); `modal` is installed. On Windows, run the Modal CLI as `python -m modal` with `PYTHONIOENCODING=utf-8`.
- Existing tests live in `hrm-cloud/tests/`. Commit only the named files (the repo has unrelated WIP — never `git add -A`).
- Work continues on the current branch `perf/eval-speedup` (it carries the `EVAL_DIAG`/forwarding changes focal builds on).

## File structure

- **Modify** `hrm-cloud/residual_tasklora_v2.py`:
  - Add `space_time_focal_astar` in the shared planner section (right after `space_time_astar`, ~`:809`).
  - Add `PLANNER` / `FOCAL_W` globals after `EVAL_DIAG` (~`:3389`), and add `"PLANNER"`, `"FOCAL_W"` to `_EVAL_FORWARD_VARS` (~`:65`).
  - Branch the planner call in the **live** `run_policy_episode` (~`:4710`).
- **Create** `hrm-cloud/tests/test_focal.py` — planner unit tests (CPU, dummy heuristics, no model).
- **Create** `hrm-cloud/tests/test_focal_wiring.py` — env-knob + planner-selection tests.
- **Create** `hrm-cloud/bench_focal.py` — local matched-expansion benchmark (CPU/GPU, real model).

---

## Task 1: `space_time_focal_astar` planner

**Files:**
- Modify: `hrm-cloud/residual_tasklora_v2.py` (add function after `space_time_astar`, ~`:809`)
- Test: `hrm-cloud/tests/test_focal.py`

- [ ] **Step 1: Write the failing tests**

Create `hrm-cloud/tests/test_focal.py`:

```python
"""Unit tests for space_time_focal_astar (CPU, dummy heuristics, no model)."""
import residual_tasklora_v2 as R
from residual_tasklora_v2 import PlanResult


def _small_static():
    # 16x16 static map, no dynamics -> deterministic, solvable, fast.
    ep = R.make_episode(7, "A", 16, 60, 0, 0, 0)
    occ = R.simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    return ep, occ


def _zero_delta(states):
    return [0.0 for _ in states]


def _optimal(ep, occ, horizon=40, budget=20000):
    # space_time_astar with zero residual == optimal A* with admissible Manhattan.
    return R.space_time_astar(ep.start, ep.goal, 0, horizon, budget, occ, _zero_delta, alpha=1.0)


def test_w1_matches_optimal_cost():
    ep, occ = _small_static()
    opt = _optimal(ep, occ)
    assert opt.found
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=1.0)
    assert foc.found
    assert len(foc.actions) == len(opt.actions)  # w=1 is optimal


def test_w2_respects_suboptimality_bound():
    ep, occ = _small_static()
    opt = _optimal(ep, occ)
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert foc.found
    assert len(foc.actions) <= 2 * len(opt.actions)  # bounded suboptimal


def test_completeness_and_interface():
    ep, occ = _small_static()
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert isinstance(foc, PlanResult)
    assert foc.found
    assert len(foc.actions) >= 1
    assert foc.expansions > 0


def test_determinism():
    ep, occ = _small_static()
    a = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    b = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert a.actions == b.actions
    assert a.expansions == b.expansions


def test_perfect_signal_reduces_expansions():
    # A perfect focal signal (true cost-to-go residual) should expand no more
    # nodes than plain Manhattan A* on the same instance.
    ep, occ = _small_static()
    dist = R.compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
    gx, gy = ep.goal
    maxt = occ["blocked"].shape[0] - 1

    def perfect_delta(states):
        out = []
        for x, y, t_rel in states:
            d = int(dist[min(t_rel, maxt), x, y])
            if d >= R.INF:
                out.append(1e9)  # unreachable -> deprioritize
            else:
                out.append(float(max(0, d - R.manhattan(x, y, gx, gy))))
        return out

    astar = R.space_time_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, alpha=1.0)
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, perfect_delta, w=5.0)
    assert foc.found
    assert foc.expansions <= astar.expansions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python -m pytest tests/test_focal.py -v`
Expected: FAIL with `AttributeError: module 'residual_tasklora_v2' has no attribute 'space_time_focal_astar'`.

- [ ] **Step 3: Implement `space_time_focal_astar`**

In `hrm-cloud/residual_tasklora_v2.py`, immediately AFTER the end of `space_time_astar` (the `return PlanResult(...)` at ~`:809`, before `def make_episode`), add:

```python
def space_time_focal_astar(
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    t0_abs: int,
    plan_horizon: int,
    max_expansions: int,
    occ: Dict[str, np.ndarray],
    heuristic_delta_batch_fn,
    w: float = 2.0,
) -> PlanResult:
    # Focal search (A*_eps): OPEN is ordered by the admissible f = g + manhattan, which
    # bounds suboptimality by w. Among OPEN nodes with f <= w * f_min (the focal band) we
    # expand the one minimizing the learned focal key hf = manhattan + delta. The learned
    # signal only orders within the bounded band -> it can never break admissibility or
    # misdirect the search the way the additive heuristic did; a bad signal degrades to
    # Manhattan ordering. Entry layout: (f, counter, g, state, hf).
    gx, gy = goal_xy
    max_t_abs = occ["blocked"].shape[0] - 1
    n = occ["blocked"].shape[1]
    w = max(1.0, float(w))
    start_state = (start_xy[0], start_xy[1], 0)
    start_h = manhattan(start_xy[0], start_xy[1], gx, gy)
    counter = 0
    open_heap: List[Tuple[float, int, int, Tuple[int, int, int], float]] = []
    heapq.heappush(open_heap, (float(start_h), counter, 0, start_state, float(start_h)))
    counter += 1
    g_cost = {start_state: 0}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_state: None}
    closed: List[Tuple[int, int, int]] = []
    best_goal_state = start_state
    best_goal_score = start_h
    expansions = 0
    while open_heap and expansions < max_expansions:
        # drop stale entries (superseded by a cheaper path) from the OPEN top
        while open_heap and g_cost.get(open_heap[0][3], INF) != open_heap[0][2]:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        f_min = open_heap[0][0]
        thresh = w * f_min
        # extract the focal band: all valid OPEN entries with f <= thresh
        band: List[Tuple[float, int, int, Tuple[int, int, int], float]] = []
        while open_heap and open_heap[0][0] <= thresh:
            e = heapq.heappop(open_heap)
            if g_cost.get(e[3], INF) == e[2]:
                band.append(e)
        if not band:
            break
        # expand the band node with the smallest learned focal key (deterministic tiebreak)
        pick_idx = min(range(len(band)), key=lambda i: (band[i][4], band[i][1]))
        pick = band[pick_idx]
        for i, e in enumerate(band):
            if i != pick_idx:
                heapq.heappush(open_heap, e)
        _, _, g, s, _ = pick
        x, y, t_rel = s
        t_abs = min(t0_abs + t_rel, max_t_abs)
        closed.append(s)
        expansions += 1
        h_base = manhattan(x, y, gx, gy)
        if h_base < best_goal_score:
            best_goal_score = h_base
            best_goal_state = s
        if (x, y) == (gx, gy) and occ["blocked"][t_abs, x, y] == 0:
            best_goal_state = s
            break
        if t_rel >= plan_horizon:
            continue
        next_states: List[Tuple[int, int, int]] = []
        next_gs: List[int] = []
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            nt_rel = t_rel + 1
            nt_abs = min(t0_abs + nt_rel, max_t_abs)
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if occ["blocked"][nt_abs, nx, ny] != 0:
                continue
            ns = (nx, ny, nt_rel)
            ng = g + 1
            if ng < g_cost.get(ns, INF):
                next_states.append(ns)
                next_gs.append(ng)
        if not next_states:
            continue
        deltas = heuristic_delta_batch_fn(next_states)
        for ns, ng, delta in zip(next_states, next_gs, deltas):
            x2, y2, _ = ns
            h_base2 = manhattan(x2, y2, gx, gy)
            if ng < g_cost.get(ns, INF):
                g_cost[ns] = ng
                parent[ns] = s
                f = float(ng) + float(h_base2)                 # admissible bound (no delta)
                hf = float(h_base2) + max(0.0, float(delta))   # learned focal ordering key
                heapq.heappush(open_heap, (f, counter, ng, ns, hf))
                counter += 1
                if h_base2 < best_goal_score:
                    best_goal_score = h_base2
                    best_goal_state = ns
    path_states = _reconstruct_path_states(parent, best_goal_state)
    actions: List[Action] = []
    for a, b in zip(path_states[:-1], path_states[1:]):
        ax, ay, _ = a
        bx, by, _ = b
        dx, dy = bx - ax, by - ay
        try:
            actions.append(ACTIONS.index((dx, dy)))
        except ValueError:
            actions.append(WAIT_ACTION)
    if not actions:
        actions = [WAIT_ACTION]
    found = (best_goal_state[0], best_goal_state[1]) == (gx, gy) and occ["blocked"][min(t0_abs + best_goal_state[2], max_t_abs), gx, gy] == 0
    return PlanResult(found, actions, expansions, closed, path_states)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python -m pytest tests/test_focal.py -v`
Expected: all 5 PASS. If `test_w1_matches_optimal_cost` fails, the band/`f_min` logic diverged from optimal A* — debug the focal band extraction (do not weaken the test).

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/residual_tasklora_v2.py hrm-cloud/tests/test_focal.py
git commit -m "feat(planner): add space_time_focal_astar (learned focal search)"
```

---

## Task 2: `PLANNER`/`FOCAL_W` knobs + live planner branch + forwarding

**Files:**
- Modify: `hrm-cloud/residual_tasklora_v2.py` (`:65` allowlist, `:3389` flags, `:4710` branch)
- Test: `hrm-cloud/tests/test_focal_wiring.py`

- [ ] **Step 1: Write the failing tests**

Create `hrm-cloud/tests/test_focal_wiring.py`:

```python
"""Planner selection + env-forwarding wiring for focal search."""
import residual_tasklora_v2 as R


def test_forward_env_includes_planner(monkeypatch):
    monkeypatch.setenv("PLANNER", "focal")
    monkeypatch.setenv("FOCAL_W", "2.5")
    d = R._eval_forward_env()
    assert d.get("PLANNER") == "focal"
    assert d.get("FOCAL_W") == "2.5"


def test_run_policy_episode_routes_to_focal(monkeypatch):
    suite = [s for s in R.build_eval_suites(False, 10) if s.suite_id == "ID_A32_static"][0]
    calls = {"focal": 0, "astar": 0}
    real_focal = R.space_time_focal_astar
    real_astar = R.space_time_astar

    def spy_focal(*a, **k):
        calls["focal"] += 1
        return real_focal(*a, **k)

    def spy_astar(*a, **k):
        calls["astar"] += 1
        return real_astar(*a, **k)

    monkeypatch.setattr(R, "space_time_focal_astar", spy_focal)
    monkeypatch.setattr(R, "space_time_astar", spy_astar)

    monkeypatch.setattr(R, "PLANNER", "focal")
    monkeypatch.setattr(R, "FOCAL_W", 2.0)
    R.run_policy_episode(suite, seed=0, model=None, alpha=1.0, max_expansions=80, device="cpu")
    assert calls["focal"] > 0 and calls["astar"] == 0

    calls["focal"] = calls["astar"] = 0
    monkeypatch.setattr(R, "PLANNER", "astar")
    R.run_policy_episode(suite, seed=0, model=None, alpha=1.0, max_expansions=80, device="cpu")
    assert calls["astar"] > 0 and calls["focal"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python -m pytest tests/test_focal_wiring.py -v`
Expected: FAIL — `R.PLANNER`/`R.FOCAL_W` don't exist yet (AttributeError on `monkeypatch.setattr`), and `_eval_forward_env` lacks the keys.

- [ ] **Step 3a: Add the flags**

In `hrm-cloud/residual_tasklora_v2.py`, immediately AFTER the `EVAL_DIAG = (_env_int("EVAL_DIAG", 1) == 1)` line (~`:3389`), add:

```python
# Planner selection. PLANNER="focal" uses bounded-suboptimal focal search, where the
# learned signal orders the focal band (robust to magnitude miscalibration) instead of
# inflating the heuristic. FOCAL_W is the suboptimality factor (w>=1; larger = more
# reliance on the learned ranking, fewer expansions, bounded-longer paths).
PLANNER = (os.environ.get("PLANNER", "astar").strip().lower() or "astar")
FOCAL_W = _env_float("FOCAL_W", 2.0)
```

- [ ] **Step 3b: Forward the knobs to remote workers**

In the `_EVAL_FORWARD_VARS` list (~`:65`), add `"PLANNER"` and `"FOCAL_W"` (e.g., right after `"EVAL_DIAG",`):

```python
    "EVAL_DIAG",
    "PLANNER", "FOCAL_W",
```

- [ ] **Step 3c: Branch the live planner call**

In the **live** `run_policy_episode` (~`:4710`), replace:

```python
        plan = space_time_astar(agent_xy, ep.goal, t_abs, suite.plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, alpha=alpha)
```
with:
```python
        if PLANNER == "focal":
            plan = space_time_focal_astar(agent_xy, ep.goal, t_abs, suite.plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, w=FOCAL_W)
        else:
            plan = space_time_astar(agent_xy, ep.goal, t_abs, suite.plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, alpha=alpha)
```
(Edit the copy at ~`:4710`, the one inside the live `run_policy_episode` below the `# Residual task LoRA v2 overrides` comment — NOT the dead copy at `:2561`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python -m pytest tests/test_focal_wiring.py tests/test_focal.py -v`
Expected: all PASS. Also confirm import: `python -c "import residual_tasklora_v2"`.

- [ ] **Step 5: Commit**

```bash
git add hrm-cloud/residual_tasklora_v2.py hrm-cloud/tests/test_focal_wiring.py
git commit -m "feat(eval): PLANNER/FOCAL_W knobs route run_policy_episode to focal search (forwarded)"
```

---

## Task 3: local matched-expansion benchmark (the go/no-go gate)

**Files:**
- Create: `hrm-cloud/bench_focal.py`

This is the headline validation: does the *learned* focal signal expand fewer nodes than Manhattan A* on the *same* instances? Runs on local GPU (model forward) / CPU; no Modal compute. Requires one checkpoint downloaded locally.

- [ ] **Step 1: Create the benchmark script**

Create `hrm-cloud/bench_focal.py`:

```python
#!/usr/bin/env python3
"""Local matched-expansion benchmark for learned focal search (no Modal compute).

Compares, on identical instances (same seeds):
  - baseline: Manhattan A* (model=None, PLANNER=astar)
  - focal-learned: focal search ordered by the model's signal (PLANNER=focal)
across map scales and a sweep of w. Reports per-(suite,w) median expansion ratio
(focal/baseline) and success rates. Headline metric: ratio < 1 means fewer expansions.

Setup (one-time, ~free read): download a checkpoint locally, e.g.
  python -m modal volume get residual-tasklora-v2-vol \
    residual_tasklora_v2/runs/residual_tasklora_v2/models/avgbase__hrm__ALL_TASKS.pt ./ckpts/

Usage:
  python hrm-cloud/bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --suites ID_A64_static,OOD_A128_static,OOD_A192_static --seeds 5 --budget 500 \
    --w 1.0,1.5,2.0,3.0 --device cuda
"""
import argparse, statistics as st
import torch
import residual_tasklora_v2 as R
from residual_tasklora_v2 import BackboneConfig, CleanHeuristicModel


def load_model(ckpt, device):
    payload = torch.load(ckpt, map_location="cpu")
    cfg = BackboneConfig(**payload["cfg"])
    m = CleanHeuristicModel(cfg)
    m.load_state_dict(payload["model_state"], strict=False)
    m.to(device).eval()
    m.arm = "avgbase"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suites", default="ID_A64_static,OOD_A128_static,OOD_A192_static")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--w", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    R.EVAL_DIAG = False  # fast path; we only need outcomes
    suites = {s.suite_id: s for s in R.build_eval_suites(True, 100)}
    model = load_model(args.ckpt, args.device)
    ws = [float(x) for x in args.w.split(",")]
    print(f"device={args.device} budget={args.budget} seeds={args.seeds}")
    print(f"{'suite':20s} {'w':>4s} {'exp_ratio(med)':>14s} {'succ_base':>9s} {'succ_focal':>10s}")
    for sid in args.suites.split(","):
        s = suites[sid]
        # baseline: Manhattan A* (no model), per seed
        R.PLANNER = "astar"
        base = [R.run_policy_episode(s, seed=i, model=None, alpha=1.0, max_expansions=args.budget, device="cpu")
                for i in range(args.seeds)]
        for w in ws:
            R.PLANNER = "focal"; R.FOCAL_W = w
            foc = [R.run_policy_episode(s, seed=i, model=model, alpha=1.0, max_expansions=args.budget, device=args.device)
                   for i in range(args.seeds)]
            ratios = [f["expansions"] / max(1, b["expansions"]) for f, b in zip(foc, base)]
            sb = sum(b["success"] for b in base) / len(base)
            sf = sum(f["success"] for f in foc) / len(foc)
            print(f"{sid:20s} {w:4.1f} {st.median(ratios):14.2f} {sb:9.2f} {sf:10.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download a checkpoint locally (one-time)**

Run:
```bash
cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && mkdir -p ckpts && \
PYTHONIOENCODING=utf-8 python -m modal volume get residual-tasklora-v2-vol \
  residual_tasklora_v2/runs/residual_tasklora_v2/models/avgbase__hrm__ALL_TASKS.pt ckpts/
```
Expected: `ckpts/avgbase__hrm__ALL_TASKS.pt` exists (~17 MB).

- [ ] **Step 3: Smoke-run on CPU (small/fast) to verify the script works**

Run:
```bash
cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python bench_focal.py \
  --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt --suites ID_A64_static --seeds 2 --budget 150 \
  --w 1.0,2.0 --device cpu
```
Expected: a table with `exp_ratio(med)` and success columns, no errors.

- [ ] **Step 4: Real validation run on local GPU**

Run:
```bash
cd "C:/Users/hrish/Code Projects/HRMv2/hrm-cloud" && python bench_focal.py \
  --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
  --suites ID_A64_static,OOD_A96_static,OOD_A128_static,OOD_A192_static,OOD_A256_static,OOD_A128_moderateDyn \
  --seeds 5 --budget 500 --w 1.0,1.5,2.0,3.0 --device cuda
```
**Gate / acceptance:** focal-learned shows `exp_ratio(med) < 1` (materially, e.g. ≤ ~0.85 at some `w`) at matched-or-better success, especially at larger scales. If ratios are ≥ 1 everywhere, STOP and report — the ranking isn't reducing expansions in practice and we revisit the spec (do not proceed to Phase B).

- [ ] **Step 5: Commit (script only; do not commit `ckpts/`)**

```bash
git add hrm-cloud/bench_focal.py
git commit -m "feat(eval): local matched-expansion benchmark for learned focal search"
```
(Ensure `ckpts/` is untracked — add to `.gitignore` if needed; never `git add -A`.)

---

## Deferred — Phase B (Modal, only if Task 3 passes AND billing unblocked)

Out of scope for this plan (gated on the local gate + billing). When ready: run the full-suite confirmation via the existing durable `resume_spawn` with `PLANNER=focal FOCAL_W=<chosen> EVAL_DIAG=0`, comparing baseline / avgbase-focal / expert-focal; tag focal eval cells by `w` (not `α`) in `eval_shard_path`/`eval_agg_path` so they don't collide with existing aggregates. This will be its own plan once Phase A confirms the effect.

---

## Self-Review

**Spec coverage:**
- §3 algorithm (f=g+manhattan bound, focal band ≤ w·f_min, h_focal=manhattan+δ, guarantees) → Task 1 (`space_time_focal_astar` + w=1-optimality / w=2-bound / completeness / determinism / perfect-signal tests). ✔
- §4 integration (drop-in PlanResult, PLANNER/FOCAL_W knobs forwarded, live-copy branch, edit sites) → Task 2. ✔
- §5 metrics (matched expansion ratio by scale, success) + §6 tests → Task 1 tests + Task 3 benchmark. ✔
- §7 compute plan (Phase A local CPU+GPU gate; Phase B deferred) → Tasks 1–3 + deferred section. ✔
- §8 risks (heap-maintenance correctness via optimality/bound/completeness tests; dynamic-suite success measured in bench; expert-load deferred) → covered. ✔

**Placeholder scan:** No TBD/TODO; all code blocks complete; the deferred Phase-B is explicitly out of scope, not a placeholder within committed tasks.

**Type/name consistency:** `space_time_focal_astar(start_xy, goal_xy, t0_abs, plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, w)` used identically in the planner, the wiring branch, the wiring test, and the benchmark. Entry tuple `(f, counter, g, state, hf)` consistent throughout. `PLANNER`/`FOCAL_W` names consistent across flags, allowlist, branch, tests, and bench. Returns `PlanResult` (same five fields as `space_time_astar`).

**Note on planner efficiency:** the band-extract-and-repush loop is intentionally simple (correct, metric-exact for the expansion count). It is not the maximally-optimized A*_ε; that only affects wall-clock, not the reported expansions. A persistent-FOCAL optimization is a possible later tweak if local runs are slow.
