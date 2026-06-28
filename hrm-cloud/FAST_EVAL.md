# Fast eval for `residual_tasklora_v2.py`

This documents the eval speedups added on branch `perf/eval-speedup` and how to use them.
The goal: turn full-suite re-evaluation from days/weeks into hours, without changing any
reported metric in the default configuration.

## TL;DR — fast re-eval invocation

```bash
EVAL_DIAG=0 EVAL_SHARD_SIZE=3 EVAL_BUDGETS=200,500 MAX_PARALLEL_EVAL=96 \
  modal run hrm-cloud/residual_tasklora_v2.py
```

`EVAL_DIAG=0` is the big one. The rest are tuning. See caveat below about env propagation.

## What changed

1. **`EVAL_DIAG` flag (default `1`).** Default behavior is byte-identical to before
   (verified: full result dict including the diagnostics accumulator is identical for the
   default path). With `EVAL_DIAG=0`:
   - Skips `compute_true_cost_to_goal`, an O(`max_steps`·n²) pure-Python dynamic-program run
     **once per episode** whose output (`dist_abs`) only ever feeds diagnostics — it does
     **not** affect A\* decisions. On large maps (e.g. `OOD_A256_*`, n=256) this DP dominates
     per-episode wall-clock; profiling showed it was ~61% of episode time even on a small
     64² map.
   - Enables a per-replan cache of the learned-heuristic prediction keyed by `(x, y, t_rel)`.
     Within one replan `ctx` and the dynamic-obstacle layer are constant, so the model's delta
     is a pure function of node identity; caching removes redundant neural-net calls on
     re-generated search states. The cache returns the same value the net would, so A\*
     outcomes (`success`/`steps`/`expansions`) are unchanged — proven by
     `tests/test_eval_speedup.py`.

   **What `EVAL_DIAG=0` trades away:** the diagnostics block (correction-saturation, ordering
   change, rank displacement, prediction/target histograms, `nonfinite_pred_count`, etc.).
   Headline metrics — `success_rate`, `avg_expansions`, `avg_steps`, collision/timeout rates —
   are **fully preserved**; only the `diag` fields become blank. Use `EVAL_DIAG=1` (default)
   for runs where you need the diagnostics for the writeup; use `EVAL_DIAG=0` for fast
   re-eval where you only need success/expansions.

2. **Budget-invariance analyzer** (`analyze_budget_invariance.py`). Decides, per
   `(model, suite)`, whether the expensive `budget=2000` sweep actually improves success vs
   `budget=500`, so you can drop it only where it's wasted (the data shows it helps some hard
   OOD suites but is flat on others — do not blanket-drop).

3. **Profiling harness** (`bench_eval_episode.py`). Local CPU timing/profiling of
   `run_policy_episode`, no Modal/GPU, for before/after measurement.

## Env-var propagation (now handled automatically)

The eval work runs in remote Modal containers, which by default get the container's
environment — **not** your shell's. Modal 1.4.0 does not auto-forward local env vars, so
originally a shell-set `EVAL_DIAG=0` (and every other `EVAL_*` knob) was silently ignored
remotely. This was confirmed: a baseline shard on `OOD_A256_static` with `EVAL_DIAG=0` ran at
`eps_s=0.01` (~100s/ep — the diagnostics DP still ran).

This is now fixed. An allowlist of eval-control vars is forwarded to the remote functions via
`eval_env_secret = modal.Secret.from_dict(_eval_forward_env())`, built from your **local**
environment at deploy time and attached to `run_pipeline` and the three `evaluate_pair_chunk*`
functions. `EVAL_DIAG` is additionally re-read at the start of each eval shard for robustness.
After the fix, the same shard ran at `eps_s=0.39` (~2.6s/ep) — a ~39× remote speedup,
confirming `EVAL_DIAG=0` now takes effect on the workers.

Forwarded (set them in your shell before `modal run` and they reach the workers):
`EVAL_DIAG`, `EVAL_TORCH_THREADS`, `EVAL_TORCH_INTEROP_THREADS`, `EVAL_BUDGETS`,
`EVAL_SHARD_SIZE`, `EVAL_CHECKPOINT_EVERY`, `EVAL_ONLY_SUITES`, `EVAL_SKIP_SUITES`,
`FORCE_REEVAL_SUITES`, `FORCE_REEVAL`, `FORCE_REEVAL_MODELED`, `EVAL_EPISODES`,
`VALIDATION_EPISODES`, `ALPHA_CANDIDATES`, `ALPHA_TUNE_BUDGET`, `SKIP_COLLECT`, `SKIP_TRAIN`,
`SKIP_ALPHA_TUNE`, `SKIP_EVAL`, `EVAL_ARMS`, `EVAL_MODELS`. (Only vars you actually set are
forwarded; `EVAL_DIAG` always is, defaulting to `1`.)

**Intentionally NOT forwarded:** path/identity vars `RUN_TAG`, `MODEL_RUN_TAG`, `VOLUME_NAME` —
forwarding these could silently relocate where results are read/written. Set those by your
existing mechanism if you need them remotely. `MAX_PARALLEL_EVAL` is already passed as an
explicit pipeline argument, so it needs no forwarding.

## Budget pruning workflow

```bash
# 1. Pull a local mirror of the eval_agg results if you don't have one:
#    modal volume get residual-tasklora-v2-vol <run>/results/eval_agg ./eval_agg_local
# 2. Find suites where budget 2000 is wasted:
python hrm-cloud/analyze_budget_invariance.py --agg-dir ./eval_agg_local
#    (or compare 200 vs 500 with --lo 200 --hi 500)
# 3. Re-run eval with the trimmed sweep, keeping 2000 only where the analyzer says it helps:
EVAL_DIAG=0 EVAL_BUDGETS=200,500 modal run hrm-cloud/residual_tasklora_v2.py
#    For the few suites that still need 2000, run them separately with FORCE_REEVAL_SUITES.
```

## Relevant env knobs (all pre-existing unless noted)

| Var | Default | Effect |
|-----|---------|--------|
| `EVAL_DIAG` | `1` | **(new)** `0` skips diagnostics DP + enables heuristic cache (headline metrics unchanged) |
| `EVAL_BUDGETS` | `200,500,2000` | A\* expansion budgets swept per replan |
| `EVAL_SHARD_SIZE` | `10` | episodes per eval shard; smaller = better tail latency/recovery |
| `MAX_PARALLEL_EVAL` | `48` | max concurrent eval shards (raise toward your Modal limit) |
| `EVAL_TORCH_THREADS` | `1` | torch threads per worker (keep at 1 when packing many shards/box) |
| `FORCE_REEVAL_SUITES` | — | comma-separated suite ids to force re-eval |
