# Residual TaskLoRA v2 Eval Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut residual_tasklora_v2 eval wall-clock from "weeks" to hours by removing the dominant CPU costs in the per-episode planning loop, without changing any reported metric.

**Architecture:** The eval bottleneck is CPU-bound A* that calls a neural net mid-search. We attack it in measured priority order: (0) measure the real split, (1) gate the diagnostics-only exact-cost DP that Fable missed, (2) cache the NN heuristic per state within a replan, (3) prune the budget sweep using existing data + tune sharding env vars. A numba rewrite (Fable's #2) is deferred behind measurement because it is high-risk and likely unnecessary once 1–3 land.

**Tech Stack:** Python 3.10, NumPy, PyTorch (CPU), Modal. No new runtime deps for Phases 0–3.

---

## CRITICAL CONTEXT: read before touching code

This file is **not** what Fable assumed. Two structural facts dominate the whole plan:

1. **The file has TWO copies of the eval machinery.** Lines 1–3170 are the "v1 base"; lines 3172–4801 are an explicit **"Residual task LoRA v2 overrides"** block (see the comment at `hrm-cloud/residual_tasklora_v2.py:3172`). Later top-level defs shadow earlier ones at import time, so the **live** functions are the v2 copies:
   - `run_policy_episode` → **`residual_tasklora_v2.py:4501`** (the copy at 2472 is DEAD)
   - `_evaluate_pair_chunk_impl` → **`:4639`** (the copy at 2579 is DEAD)
   - `_diagnostics_update` → **`:4418`**, `_new_diag_accumulator` → **`:4224`**
   - live model class with `predict_components_from_ctx` → **`:4162`**
   - **All edits to the eval loop / heuristic closure / diagnostics MUST target the 4162–4700 region.** Editing the dead copies will compile and run but change nothing.
   - `space_time_astar` (`:692`), `compute_true_cost_to_goal` (`:649`), `compute_target_delta_from_dist` (`:1325`), `extract_local_patch_2ch` (`:1297`), `build_node_meta` (`:1313`), `simulate_occupancy` (`:623`) have only ONE copy each (shared) — edit those in place.

2. **`compute_true_cost_to_goal` is diagnostics-only in eval — and it is a pure-Python O(max_steps · n² · 4) triple loop** (`:655`–`:668`). In `run_policy_episode` (`:4505`) its output `dist_abs` feeds ONLY `compute_target_delta_from_dist` → `target_deltas` → `_diagnostics_update`. The `pred` that actually drives A* comes from the model (`:4574`), never from `dist_abs`. For an `OOD_A256_*` suite (n=256, max_steps≈660) that loop is ≈660·256²·4 ≈ **170M Python iterations per episode × 100 episodes**. Fable did not mention this; it is very likely the single largest sink on large maps.

### Verdict on Fable's five suggestions (evidence-based)

| # | Fable's suggestion | Verdict | Why |
|---|---|---|---|
| 1 | Precompute a dense `h_table[t,x,y]` once per replan | **Partially right; do the cache form, not the dense form.** | The delta does depend only on `(x,y,t_rel)` within a replan (ctx fixed; `dynamic_cur` fixed; patch depends on `(x,y)` only — `:4548`). But a *dense* `(H+1, n, n)` precompute evaluates the whole grid; budget-limited A* (200–2000 expansions) on a 256² map touches far fewer cells than 65k, so dense precompute is wasteful there. A **lazy per-state cache** captures the same redundancy and is always a win → **Phase 2**. |
| 2 | numba-JIT the whole `space_time_astar` | **Defer; high-risk.** | numba is **not in the Modal image** (`:54` installs only torch/numpy/tqdm); needs image change + cold-start JIT on ephemeral containers. numba can't call torch, so it requires fully decoupling the NN first; `heapq` and the dict parent-map need manual reimplementation. Only worth it if Phase 0 shows the pure-Python search loop dominates *after* Phases 1–2. → **Phase 4 (gated).** |
| 3 | Stop reallocating `np.full(...)` per replan | **Based on a false premise here.** | Those big arrays are allocated **once per episode** (`:4504`–`:4505`), not per replan/env-step. No win available as described. (The real per-call allocs are tiny `patches`/`metas`; Phase 2 removes most of them via caching.) |
| 4 | Skip budget 2000 where success is flat across budgets | **Right instinct; claim is overstated — verify per-suite.** | The compendium shows budget DOES help some hard OOD suites (`OOD_A192_static` 0.56→0.67 from B500→B2000, lines 376–377) but "often increased expansions without improving success" elsewhere (line 433). So drop B2000 **only on suites the data shows are flat**, don't blanket-drop. → **Phase 3 (data-driven).** |
| 5 | `torch.set_num_threads(1)` per worker | **Already done.** | `_configure_eval_torch_threads()` (`:266`) defaults `EVAL_TORCH_THREADS=1` and is called in the live chunk impl. Remaining lever is `EVAL_SHARD_SIZE` (default 10) / `MAX_PARALLEL_EVAL` (default 48) → **Phase 3**, zero code. |

**Net:** the biggest, lowest-risk, exact win (Phase 1, diagnostics gate) is one Fable missed. Phase 2 is the corrected form of #1. Phase 3 is #4 + #5 done safely. Fable's #2/#3 are deferred/dropped.

### Design decision that keeps metrics byte-identical
Phases 1 & 2 are unified behind a single flag `EVAL_DIAG` (default **1 = current behavior exactly**):
- `EVAL_DIAG=1`: unchanged path. `dist_abs` computed, diagnostics recorded, **no cache** (so diag aggregates are bit-identical to today).
- `EVAL_DIAG=0`: skip `compute_true_cost_to_goal`, skip `target_deltas`, skip `_diagnostics_update`, **enable** the `(x,y,t_rel)`→pred cache. Diagnostics come back as a blank accumulator.

Because the cache returns the *same* delta the NN would have returned, A* decisions are unchanged → `success`/`steps`/`expansions` are identical between the two modes for any deterministic (static) suite. That equivalence is the correctness test.

### No test harness exists for hrm-cloud
There is no `pytest` setup under `hrm-cloud/` (the only tests live in the unrelated `HRM-v2/` subproject). This plan **creates** `hrm-cloud/tests/test_eval_speedup.py`. It imports functions directly from the module. Importing requires `modal`, `torch`, `numpy` to be importable locally (they are — the user runs `modal run` from this machine). All tests use CPU + tiny static maps and a deterministic dummy model, so they run in seconds with no GPU and no Modal calls.

---

## File Structure

- **Modify** `hrm-cloud/residual_tasklora_v2.py`:
  - Add `EVAL_DIAG` flag next to `SANITIZE_NONFINITE_EVAL` (`:3325`).
  - Gate `dist_abs` computation in live `run_policy_episode` (`:4505`).
  - Replace the live heuristic closure (`:4533`–`:4583`) with a diag-gated, cache-enabled version.
- **Create** `hrm-cloud/tests/test_eval_speedup.py` — equivalence + "DP skipped when diag off" + cache-correctness tests.
- **Create** `hrm-cloud/bench_eval_episode.py` — Phase 0 local timing harness (throwaway-ish; keep it, it's useful for regression).
- **Create** `hrm-cloud/analyze_budget_invariance.py` — Phase 3 analysis over existing `eval_agg` JSONs to pick which suites can drop B2000.
- **No change** to the dead v1 copies, to `space_time_astar`, or to the Modal image (Phases 0–3).

---

## Task 1: Phase 0 — Measure the real cost split (do this first)

**Files:**
- Create: `hrm-cloud/bench_eval_episode.py`

This is Fable's one unambiguously-correct instinct ("profile before optimizing"), adapted to run locally on CPU without Modal. It confirms the Phase-1/2 hypotheses and gives a baseline to prove the speedup against.

- [ ] **Step 1: Write the benchmark harness**

```python
#!/usr/bin/env python3
"""Local CPU timing harness for residual_tasklora_v2 eval (no Modal, no GPU).

Usage:
  python hrm-cloud/bench_eval_episode.py --suite ID_A64_static --seeds 3 --budget 500
  python hrm-cloud/bench_eval_episode.py --suite OOD_A256_static --seeds 1 --budget 2000

Reports per-episode wall-clock and the share spent in compute_true_cost_to_goal
(the diagnostics-only DP), which is the Phase-1 hypothesis.
"""
import argparse, time, cProfile, pstats, io
import residual_tasklora_v2 as R


def _suite(suite_id):
    for s in R.build_eval_suites(include_stretch=True, eval_episodes=100):
        if s.suite_id == suite_id:
            return s
    raise SystemExit(f"unknown suite {suite_id}; pick from build_eval_suites()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ID_A64_static")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    suite = _suite(args.suite)

    # model=None isolates pure search + diagnostics DP (no NN); good enough for the
    # Phase-1 hypothesis (compute_true_cost_to_goal dominates large maps).
    def run():
        for i in range(args.seeds):
            t0 = time.time()
            res = R.run_policy_episode(suite, seed=i, model=None, alpha=1.0,
                                       max_expansions=args.budget, device="cpu")
            print(f"  seed={i} steps={res['steps']} exp={res['expansions']} "
                  f"wall={time.time()-t0:.2f}s")

    print(f"[bench] suite={suite.suite_id} n={suite.size} max_steps={suite.max_steps} "
          f"budget={args.budget} seeds={args.seeds}")
    if args.profile:
        pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
        print(s.getvalue())
    else:
        t0 = time.time(); run()
        print(f"[bench] total {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on a small and a large static suite, with the profiler**

Run:
```bash
python hrm-cloud/bench_eval_episode.py --suite ID_A64_static --seeds 3 --budget 500 --profile
python hrm-cloud/bench_eval_episode.py --suite OOD_A256_static --seeds 1 --budget 2000 --profile
```
Expected: on `OOD_A256_static`, the cumulative-time profile shows `compute_true_cost_to_goal` as a dominant line (hypothesis: a large fraction of per-episode wall-clock). Record the numbers — they justify Phase 1 and are the before/after baseline.

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/bench_eval_episode.py
git commit -m "perf(eval): add local CPU timing/profiling harness for run_policy_episode"
```

---

## Task 2: Phase 1+2 — `EVAL_DIAG` flag: skip the diagnostics DP and cache the NN heuristic

**Files:**
- Modify: `hrm-cloud/residual_tasklora_v2.py:3325` (add flag)
- Modify: `hrm-cloud/residual_tasklora_v2.py:4505` (gate `dist_abs`)
- Modify: `hrm-cloud/residual_tasklora_v2.py:4530`–`4583` (gate diag + add cache in the live closure)
- Test: `hrm-cloud/tests/test_eval_speedup.py`

- [ ] **Step 1: Write the failing equivalence test**

Create `hrm-cloud/tests/test_eval_speedup.py`:

```python
"""Phase 1/2 correctness: EVAL_DIAG=0 must not change A* outcomes, and must
skip the diagnostics-only cost DP. Static suite => fully deterministic."""
import importlib
import numpy as np
import pytest

import residual_tasklora_v2 as R


def _static_suite():
    # ID_A32_static: no gates/patrollers/drifters => deterministic, fast.
    for s in R.build_eval_suites(include_stretch=False, eval_episodes=10):
        if s.suite_id == "ID_A32_static":
            return s
    raise AssertionError("ID_A32_static not found")


def _run(seed, budget):
    return R.run_policy_episode(_static_suite(), seed=seed, model=None,
                                alpha=1.0, max_expansions=budget, device="cpu")


def test_diag_off_matches_diag_on_baseline():
    suite = _static_suite()
    for seed in range(4):
        R.EVAL_DIAG = True
        on = _run(seed, 200)
        R.EVAL_DIAG = False
        off = _run(seed, 200)
        assert (on["success"], on["steps"], on["expansions"]) == \
               (off["success"], off["steps"], off["expansions"]), \
               f"seed={seed} diverged on/off: {on} vs {off}"
    R.EVAL_DIAG = True  # restore default


def test_diag_off_skips_cost_dp(monkeypatch):
    # When diag is off, compute_true_cost_to_goal must not be called at all.
    def boom(*a, **k):
        raise AssertionError("compute_true_cost_to_goal called with EVAL_DIAG=0")
    monkeypatch.setattr(R, "compute_true_cost_to_goal", boom)
    R.EVAL_DIAG = False
    try:
        res = _run(seed=0, budget=200)
        assert "steps" in res
    finally:
        R.EVAL_DIAG = True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd hrm-cloud && python -m pytest tests/test_eval_speedup.py -v`
Expected: `test_diag_off_skips_cost_dp` FAILS with the `AssertionError("compute_true_cost_to_goal called...")` because today `dist_abs` is computed unconditionally at `:4505`. (`test_diag_off_matches...` may pass trivially since the flag doesn't exist yet / both branches identical — that's fine; it locks behavior once the flag exists.)

- [ ] **Step 3: Add the `EVAL_DIAG` flag**

In `hrm-cloud/residual_tasklora_v2.py`, immediately after the line `SANITIZE_NONFINITE_EVAL = (_env_int("SANITIZE_NONFINITE_EVAL", 1) == 1)` (`:3325`), add:

```python
# Diagnostics (correction-saturation / ordering metrics) require an O(max_steps*n^2)
# pure-Python exact-cost DP per episode that does NOT affect A* decisions. Default ON
# (preserves headline metrics + diag exactly). Set EVAL_DIAG=0 for fast re-eval runs
# where only success/expansions are needed; that path also enables the per-replan
# heuristic cache.
EVAL_DIAG = (_env_int("EVAL_DIAG", 1) == 1)
```

- [ ] **Step 4: Gate the cost DP in the live `run_policy_episode`**

In the **live** copy, change `hrm-cloud/residual_tasklora_v2.py:4505` from:

```python
    dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
```
to:
```python
    dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps) if EVAL_DIAG else None
```

- [ ] **Step 5: Replace the live heuristic closure with the diag-gated + cached version**

Replace the closure body in the live copy — the block from `dynamic_cur = ...` / `def heuristic_delta_batch_fn` down to its `return pred` (`hrm-cloud/residual_tasklora_v2.py:4530`–`4583`) — with the following. Keep indentation at the `for t_abs` loop body level. The per-replan cache `delta_cache` is defined fresh each `t_abs` (so it is correctly scoped to one replan, where `ctx`/`dynamic_cur` are constant):

```python
        dynamic_cur = np.clip(occ["gate"][t_abs] + occ["pat"][t_abs] + occ["drift"][t_abs], 0, 1).astype(np.uint8)
        gx, gy = ep.goal
        delta_cache: Dict[Tuple[int, int, int], float] = {}

        def heuristic_delta_batch_fn(states: List[Tuple[int, int, int]]) -> List[float]:
            if model is None:
                pred = [0.0 for _ in states]
                if EVAL_DIAG:
                    h_bases = [manhattan(x, y, gx, gy) for x, y, _ in states]
                    target_deltas = []
                    for x, y, t_rel in states:
                        tgt = compute_target_delta_from_dist(dist_abs, min(t_abs + t_rel, ep.max_steps), x, y, gx, gy)
                        target_deltas.append(0.0 if tgt is None else float(tgt))
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
                return pred

            # ---- Fast path: cache NN delta per (x,y,t_rel) within this replan ----
            if not EVAL_DIAG:
                out: List[Optional[float]] = [None] * len(states)
                todo_idx: List[int] = []
                todo_states: List[Tuple[int, int, int]] = []
                for i, s in enumerate(states):
                    c = delta_cache.get(s)
                    if c is None:
                        todo_idx.append(i)
                        todo_states.append(s)
                    else:
                        out[i] = c
                if todo_states:
                    p = 2 * PATCH_RADIUS + 1
                    patches = np.zeros((1, len(todo_states), PATCH_CHANNELS, p, p), dtype=np.float32)
                    metas = np.zeros((1, len(todo_states), NODE_META_DIM), dtype=np.float32)
                    for j, (x, y, t_rel) in enumerate(todo_states):
                        patches[0, j] = extract_local_patch_2ch(ep.walls, dynamic_cur, x, y, PATCH_RADIUS).astype(np.float32)
                        metas[0, j] = build_node_meta(x, y, gx, gy, t_rel, ep.walls.shape[0])
                    patch_t = torch.from_numpy(patches).to(device)
                    meta_t = torch.from_numpy(metas).to(device)
                    with torch.no_grad():
                        if hasattr(model, "predict_components_from_ctx"):
                            parts = model.predict_components_from_ctx(ctx, patch_t, meta_t)
                            if SANITIZE_NONFINITE_EVAL:
                                parts, _ = _sanitize_residual_parts_for_eval(eval_tag, parts)
                            pred_t = parts["final_delta"]
                        else:
                            pred_t = model.predict_delta_from_ctx(ctx, patch_t, meta_t)
                            if SANITIZE_NONFINITE_EVAL:
                                pred_t, _ = _sanitize_eval_delta_tensor(eval_tag, pred_t)
                        vals = [float(v) for v in pred_t[0].detach().float().cpu().numpy().tolist()]
                    for j, s in enumerate(todo_states):
                        delta_cache[s] = vals[j]
                        out[todo_idx[j]] = vals[j]
                return [float(v) for v in out]

            # ---- Diagnostics-on path: unchanged behavior (no cache) ----
            h_bases = [manhattan(x, y, gx, gy) for x, y, _ in states]
            target_deltas = []
            for x, y, t_rel in states:
                tgt = compute_target_delta_from_dist(dist_abs, min(t_abs + t_rel, ep.max_steps), x, y, gx, gy)
                target_deltas.append(0.0 if tgt is None else float(tgt))
            p = 2 * PATCH_RADIUS + 1
            patches = np.zeros((1, len(states), PATCH_CHANNELS, p, p), dtype=np.float32)
            metas = np.zeros((1, len(states), NODE_META_DIM), dtype=np.float32)
            for i, (x, y, t_rel) in enumerate(states):
                patches[0, i] = extract_local_patch_2ch(ep.walls, dynamic_cur, x, y, PATCH_RADIUS).astype(np.float32)
                metas[0, i] = build_node_meta(x, y, gx, gy, t_rel, ep.walls.shape[0])
            patch_t = torch.from_numpy(patches).to(device)
            meta_t = torch.from_numpy(metas).to(device)
            _assert_finite_tensor(f"{eval_tag}/patch_t", patch_t)
            _assert_finite_tensor(f"{eval_tag}/meta_t", meta_t)
            with torch.no_grad():
                if hasattr(model, "predict_components_from_ctx"):
                    parts = model.predict_components_from_ctx(ctx, patch_t, meta_t)
                    if SANITIZE_NONFINITE_EVAL:
                        parts, nonfinite_component_count = _sanitize_residual_parts_for_eval(eval_tag, parts)
                        diag_acc["nonfinite_pred_count"] += int(nonfinite_component_count)
                    else:
                        _assert_finite_eval_value(f"{eval_tag}/parts", parts)
                    pred_t = parts["final_delta"][0].detach().float().cpu().numpy().tolist()
                    base_t = parts["base_delta"][0].detach().float().cpu().numpy().tolist()
                    corr_t = parts["correction"][0].detach().float().cpu().numpy().tolist()
                    uncorr_t = parts["uncorrected_residual"][0].detach().float().cpu().numpy().tolist()
                    bound_B = float(parts["bound_B"].detach().float().mean().cpu().item()) if isinstance(parts["bound_B"], torch.Tensor) else float(parts["bound_B"])
                    bound_B = _require_finite_scalar(f"{eval_tag}/bound_B", bound_B)
                    pred = [float(v) for v in pred_t]
                    base_vals = [float(v) for v in base_t]
                    corr_vals = [float(v) for v in corr_t]
                    uncorr_vals = [float(v) for v in uncorr_t]
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases, base_vals, corr_vals, uncorr_vals, bound_B)
                else:
                    pred_delta_t = model.predict_delta_from_ctx(ctx, patch_t, meta_t)
                    if SANITIZE_NONFINITE_EVAL:
                        pred_delta_t, nonfinite_pred_count = _sanitize_eval_delta_tensor(eval_tag, pred_delta_t)
                        diag_acc["nonfinite_pred_count"] += int(nonfinite_pred_count)
                    else:
                        _assert_finite_tensor(f"{eval_tag}/pred_delta", pred_delta_t)
                    pred_t = pred_delta_t[0].detach().float().cpu().numpy().tolist()
                    pred = [float(v) for v in pred_t]
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
            return pred
```

Note: the "diagnostics-on path" block above is copied verbatim from the current closure (`:4544`–`4583`) so that `EVAL_DIAG=1` behavior is byte-identical. Only the `model is None` branch was lifted above it and the fast path was inserted.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd hrm-cloud && python -m pytest tests/test_eval_speedup.py -v`
Expected: both tests PASS. `test_diag_off_matches_diag_on_baseline` proves A* outcomes are identical; `test_diag_off_skips_cost_dp` proves the DP is skipped.

- [ ] **Step 7: Re-run the Phase 0 benchmark with diag off to quantify the win**

Run:
```bash
EVAL_DIAG=0 python hrm-cloud/bench_eval_episode.py --suite OOD_A256_static --seeds 1 --budget 2000
python hrm-cloud/bench_eval_episode.py --suite OOD_A256_static --seeds 1 --budget 2000
```
Expected: the `EVAL_DIAG=0` run is dramatically faster on the large suite (the DP is gone). Record the ratio.

- [ ] **Step 8: Commit**

```bash
git add hrm-cloud/residual_tasklora_v2.py hrm-cloud/tests/test_eval_speedup.py
git commit -m "perf(eval): EVAL_DIAG flag skips diagnostics-only cost DP + caches NN heuristic per replan"
```

---

## Task 3: Phase 2 verification — cache-correctness with a deterministic model

**Files:**
- Test: `hrm-cloud/tests/test_eval_speedup.py` (extend)

The Phase 1 test used `model=None`. This adds a tiny deterministic dummy model so the **cache path itself** is exercised and proven equivalent to the uncached path.

- [ ] **Step 1: Add a dummy-model cache-equivalence test**

Append to `hrm-cloud/tests/test_eval_speedup.py`:

```python
class _DummyDeterministicModel:
    """Returns a deterministic delta per (x,y,t_rel)-ish input; no torch params.
    Mimics the minimal interface run_policy_episode uses: encode_obs_sequence +
    predict_delta_from_ctx. No predict_components_from_ctx (exercises simple path)."""
    arm = "avgbase"

    def encode_obs_sequence(self, obs_seq):
        import torch
        return torch.zeros((obs_seq.shape[0], 8), dtype=torch.float32)

    def predict_delta_from_ctx(self, ctx, node_patch, node_meta):
        import torch
        # delta = sum of meta channels => deterministic function of node meta only.
        return node_meta.sum(dim=-1)


def test_cache_path_matches_uncached_for_dummy_model():
    suite = _static_suite()
    m = _DummyDeterministicModel()
    for seed in range(3):
        R.EVAL_DIAG = True   # uncached path
        on = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        R.EVAL_DIAG = False  # cached path
        off = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        assert (on["success"], on["steps"], on["expansions"]) == \
               (off["success"], off["steps"], off["expansions"]), f"seed={seed}: {on} vs {off}"
    R.EVAL_DIAG = True
```

- [ ] **Step 2: Run it**

Run: `cd hrm-cloud && python -m pytest tests/test_eval_speedup.py::test_cache_path_matches_uncached_for_dummy_model -v`
Expected: PASS — the cached heuristic produces identical A* trajectories to the uncached one.

- [ ] **Step 3: Commit**

```bash
git add hrm-cloud/tests/test_eval_speedup.py
git commit -m "test(eval): prove per-replan heuristic cache matches uncached path"
```

---

## Task 4: Phase 3a — Data-driven budget pruning (Fable's #4, done safely)

**Files:**
- Create: `hrm-cloud/analyze_budget_invariance.py`

Do NOT blanket-drop budget 2000. Read the existing aggregated eval results and drop B2000 **only** on suites where success does not improve over B500 (within noise). Output a ready-to-use `EVAL_BUDGETS` recommendation per suite.

- [ ] **Step 1: Write the analyzer**

```python
#!/usr/bin/env python3
"""Scan existing eval_agg JSONs and report, per (model, suite), whether success
improves from B500 -> B2000. Suites that are flat are candidates for dropping
B2000. Reads local mirror of the Modal volume's results/eval_agg directory.

Usage:
  python hrm-cloud/analyze_budget_invariance.py --agg-dir /path/to/results/eval_agg
"""
import argparse, glob, json, os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-dir", required=True, help="dir of eval_agg/*.json")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="success delta below which budgets are 'flat'")
    args = ap.parse_args()

    # key (model_eval_id, suite_id) -> {budget: success}
    rows = defaultdict(dict)
    for path in glob.glob(os.path.join(args.agg_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        try:
            key = (d["model_eval_id"], d["suite_id"])
            succ = d.get("success_rate")
            if succ is None and d.get("episodes"):
                succ = d.get("successes", 0) / max(1, d["episodes"])
            rows[key][int(d["budget"])] = float(succ)
        except (KeyError, TypeError):
            continue

    flat, helps = [], []
    for (model, suite), by_b in sorted(rows.items()):
        if 500 in by_b and 2000 in by_b:
            delta = by_b[2000] - by_b[500]
            (flat if delta <= args.eps else helps).append((model, suite, by_b[500], by_b[2000], delta))

    print("=== B2000 HELPS (keep) ===")
    for m, s, a, b, d in sorted(helps, key=lambda r: -r[4]):
        print(f"  {s:28s} {m:30s} B500={a:.3f} B2000={b:.3f} (+{d:.3f})")
    print("\n=== FLAT: B2000 ~= B500 (candidate to drop B2000) ===")
    for m, s, a, b, d in flat:
        print(f"  {s:28s} {m:30s} B500={a:.3f} B2000={b:.3f} ({d:+.3f})")
    drop_suites = sorted({s for _, s, *_ in flat} - {s for _, s, *_ in helps})
    print(f"\nSuites flat for ALL models (safe to drop B2000): {drop_suites}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Locate a local copy of the eval_agg results and run it**

The results live on the Modal volume `residual-tasklora-v2-vol` under `runs/<RUN_TAG>/results/eval_agg`. If a local mirror exists (the survey in `EXPERIMENT_RESULTS_COMPENDIUM.md` downloaded 13,671 JSONs), point `--agg-dir` at it; otherwise pull it with `modal volume get`. Run:
```bash
python hrm-cloud/analyze_budget_invariance.py --agg-dir <local eval_agg dir>
```
Expected: a list of suites where B2000 is flat vs where it helps. **Decision artifact** — no code merges yet.

- [ ] **Step 3: Apply via env var (no source change), then commit the analyzer**

Budgets are already env-configurable: `EVAL_BUDGETS` (`:1622`, default `200,500,2000`). For the next eval run, set `EVAL_BUDGETS=200,500` and use `FORCE_REEVAL_SUITE_IDS` only for the suites the analyzer says still need B2000. (No source edit needed — this is the intended knob.)

```bash
git add hrm-cloud/analyze_budget_invariance.py
git commit -m "perf(eval): add budget-invariance analyzer to prune redundant B2000 sweeps"
```

---

## Task 5: Phase 3b — Sharding / parallelism env tuning (Fable's #5, the part not already done)

**Files:** none (all env-var driven; verified against `:1625`, `:1630`, `:266`).

- [ ] **Step 1: Confirm current knobs**

`EVAL_TORCH_THREADS=1` is already the default and applied (`:266`). The remaining levers are `EVAL_SHARD_SIZE` (default 10, `:1625`) and `MAX_PARALLEL_EVAL` (default 48, `:1630`). With per-episode cost crushed by Phase 1–2, smaller shards = better tail latency and recovery.

- [ ] **Step 2: Set tuned env for the next run**

For the next eval launch, export:
```bash
EVAL_DIAG=0 EVAL_SHARD_SIZE=3 MAX_PARALLEL_EVAL=96 EVAL_BUDGETS=200,500 \
  modal run hrm-cloud/residual_tasklora_v2.py
```
(Adjust `MAX_PARALLEL_EVAL` to your Modal concurrency limit.) Expected: more, smaller, parallel shards finishing fast; total wall-clock dominated by the slowest single shard rather than coarse 10-episode blocks.

- [ ] **Step 3 (optional): Persist the recommended defaults in a README note**

If you want these discoverable, add a short "Fast eval" section to the run notes documenting `EVAL_DIAG=0`, `EVAL_SHARD_SIZE=3`, `EVAL_BUDGETS=200,500`. No code change.

---

## Task 6 (CONDITIONAL — Phase 4): only if Phase 0 shows the Python search loop still dominates AFTER Phases 1–3

Do NOT start this unless the re-profiled `bench_eval_episode.py --profile` (with `EVAL_DIAG=0`) shows `space_time_astar`'s own Python overhead (heap ops, tuple unpacking, dict writes) — not the NN, not the DP — as the top cumulative cost. If the NN forward dominates, the higher-value move is the **patch-embedding-by-(x,y) cache** (split the model into `encode_patches` + `predict_from_patch_emb` so the conv runs once per cell per replan instead of once per (cell,t_rel)); scope that as its own plan after reading the live model class at `:4162`.

This task is intentionally an investigation spike, not pre-written code, because the right shape depends on Phase-0 data and a numba dep is a real cost.

- [ ] **Step 1:** Add `numba` to the Modal image (`:52`–`:55`) and confirm cold-start JIT cost on a throwaway Modal run (measure first-call vs warm-call latency).
- [ ] **Step 2:** Write a numba-`@njit` `space_time_astar_core(blocked, t0_abs, plan_horizon, max_expansions, delta_table, alpha, ...)` that takes a **precomputed** `delta_table` array (deltas already materialized by the NN for the visited frontier) — numba cannot call torch, so the NN must be fully decoupled first. Reimplement the heap as a typed-array binary heap and the parent map as a flat `int32` array indexed by `t*n*n + x*n + y`.
- [ ] **Step 3:** Keep the `PlanResult` return type identical so `run_policy_episode` is unchanged. Add an equivalence test (same pattern as Task 2) asserting the numba planner returns identical `actions`/`expansions` to the Python planner on static maps.
- [ ] **Acceptance:** numba planner passes the equivalence test AND the re-profiled benchmark shows a net speedup that justifies the dependency. If not, abandon and keep the pure-Python planner.

---

## Self-Review

**Spec coverage (Fable's 5 + the missed win):**
- #1 dense h-table → Task 2 (corrected to per-replan cache). ✔
- #2 numba → Task 6 (gated). ✔
- #3 per-replan alloc → addressed (debunked; alloc is per-episode; tiny per-call allocs removed by Task 2 cache). ✔
- #4 budget pruning → Task 4. ✔
- #5 thread/shard hygiene → Task 5. ✔
- Missed win (diagnostics DP) → Task 1 (measure) + Task 2 (gate). ✔
- "Profile first" → Task 1. ✔

**Placeholder scan:** No "TODO/implement later" in committed tasks (1–5). Task 6 is explicitly a conditional spike with acceptance criteria, not in committed scope.

**Type/name consistency:** `EVAL_DIAG` (bool) used consistently; `delta_cache: Dict[Tuple[int,int,int], float]` keys match the `(x,y,t_rel)` state tuples produced by `space_time_astar`; the diag-on block is verbatim from the current source so `_diagnostics_update` signatures (5-arg simple / 9-arg components) match `:4572` and `:4582`; `_sanitize_residual_parts_for_eval`/`_sanitize_eval_delta_tensor` return `(value, count)` per `:3662`/`:3705` and the fast path discards the count.

**Risk notes:** Phases 1–3 cannot change any reported metric when `EVAL_DIAG=1` (default) because the diag-on path is byte-identical and the cache is off there. The speed comes from running future evals with `EVAL_DIAG=0`, where the equivalence tests prove A* outcomes are unchanged.
