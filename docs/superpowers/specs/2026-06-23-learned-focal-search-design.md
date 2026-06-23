# Learned Focal Search — Design Spec

**Date:** 2026-06-23
**Status:** Design approved (pending written-spec review)
**Component:** `hrm-cloud/residual_tasklora_v2.py` eval planner
**Supersedes (integration only):** the additive-residual heuristic (`f = g + manhattan + α·δ`) used by `space_time_astar`. Models, data, and training are **unchanged** — this is a re-integration of the existing learned signal.

---

## 1. Motivation & diagnosis

The thesis: **a hierarchical model (HRM) + transfer/specialization (pooled base → LoRA experts) beats a purely algorithmic planner** on dynamic-grid A* planning. Headline metric: **fewer A\* node expansions on the same instances** (efficiency), with success rate as the secondary lift.

The original integration failed. Investigation established:

1. **The learned signal is an excellent ranker.** Probing `avgbase__hrm__ALL_TASKS` locally (predicted residual vs. true cost-to-go residual): **Pearson r ≈ 0.99 at every map scale (n=64→256).** The model knows *which* nodes are closer to the goal almost perfectly.
2. **Its magnitude is scale-miscalibrated.** Predicted residual is ~flat (~90–105) across scales while the true residual grows (121 → 493); `pred/true` falls from 0.73 (n=64) to 0.19 (n=256). It under-predicts, worsening with scale (trained on n≤64). It does **not** over-inflate (admissible on average).
3. **A single global α, tuned on small maps, throttled it.** Validation α-tuning ran on n≤64 maps where `pred≈true`, so α>1 pushes `manhattan+α·δ` above true cost (inadmissible) → search misdirects → success collapses (val success 0.86→0.13 as α:0.5→2.0). The tuner correctly pinned α to the floor (0.5) on all 10 models. Applied globally at eval, α=0.5 keeps the heuristic ≈ Manhattan on large maps (where δ is already 5× under-scaled), so `avgbase ≈ Manhattan + 0.01`.

**Conclusion:** the failure is *integration*, not the model. The signal is a near-perfect ranker used wrongly as a miscalibrated additive magnitude under one global α. The fix is to consume the signal as a **ranking**, where magnitude miscalibration is structurally irrelevant.

---

## 2. Goals & success criteria

**Goal:** integrate the existing learned signal as a *ranking* via focal search, so it reduces A\* expansions on matched instances without the inadmissibility cliff.

**Success criteria (in priority order):**
1. **Primary:** focal-learned expands **materially fewer nodes than Manhattan A\*** on the same instances, at matched-or-better success — especially at larger scales.
2. **Secondary:** focal-learned **solves more** instances within a fixed expansion budget.
3. **Thesis:** expert-focal ranks better (fewer expansions) than avgbase-focal on matched cells (transfer/specialization shows up via ranking).

**Non-goals (this spec):** retraining, fixing the magnitude calibration, model/task changes. Those are deferred (see §9).

**Pillars preserved:** HRM (produces the ranking signal) · transfer (base-vs-expert ranking comparison) · beats the algorithmic planner (fewer expansions than Manhattan A\*).

---

## 3. Algorithm: Learned Focal Search (A\*_ε)

Replace the single-queue magnitude-weighted A\* with two-queue focal search; the learned signal orders only the focal band.

- **Admissible bound (defines `f`):** `OPEN` ordered by `f(n) = g(n) + manhattan(n, goal)`. Manhattan is an admissible lower bound on remaining moves on a 4-connected unit-cost grid, valid even with dynamic obstacles. `f_min = min f over OPEN`.
- **Focal set:** `FOCAL = { n ∈ OPEN : f(n) ≤ w · f_min }`, suboptimality factor `w ≥ 1`.
- **Learned ordering:** among `FOCAL`, expand `argmin_n h_focal(n)` where `h_focal(n) = manhattan(n, goal) + δ_model(n)` (the model's predicted cost-to-go). The learned signal never enters `f`.

**Guarantees:**
- *Completeness:* the `f_min` node is always in `FOCAL`, so `FOCAL` is non-empty whenever `OPEN` is.
- *Bounded suboptimality:* returned solution cost ≤ `w ·` optimal.
- *Robustness:* a bad ranking cannot misdirect into inadmissible regions — worst case it degenerates to Manhattan ordering. This is the structural fix for the α cliff.

**The knob `w` replaces `α`:**
- `w = 1` → optimal A\*; learned signal used only to break `f`-ties (**zero-risk floor** — never worse in success; usually fewer expansions on tie-heavy grids).
- larger `w` → wider band → more reliance on the (near-perfect) learned ranking → fewer expansions, bounded-longer paths.
- No inadmissibility cliff; degrades gracefully. Candidate sweep: `w ∈ {1.0, 1.5, 2.0, 3.0}`.

**Why this matches the diagnosis:** focal uses only the *order* of `manhattan+δ` (argmin within the band), so the 0.99 ranking is exploited and the magnitude miscalibration is irrelevant.

---

## 4. Components & integration

### 4.1 New planner (shared planner section, beside `space_time_astar` ~`:692`)
`space_time_focal_astar(start_xy, goal_xy, t0_abs, plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, w)` → **identical `PlanResult`** (drop-in; nothing downstream changes).

- Two heaps (standard A\*_ε):
  - `OPEN`: min-heap keyed `(f, tie, state)`, `f = g + manhattan`.
  - `FOCAL`: min-heap keyed `(h_focal, tie, state)`, holding nodes with `f ≤ w·f_min`.
- As `f_min` rises, migrate newly-eligible `OPEN` nodes into `FOCAL`. Pop best-`h_focal` from `FOCAL`.
- Reuse the existing patterns from `space_time_astar`: **stale-entry skip** via the `g_cost` check, expansion counter, parent map, path reconstruction, and the **same partial-path fallback** on budget exhaustion (track best-so-far node) so budget-limited behavior is comparable to the A\* baseline.
- `tie` is a stable, deterministic secondary key (e.g., insertion counter or `t*n*n + x*n + y`) so results are reproducible.

### 4.2 Heuristic closure — unchanged
`heuristic_delta_batch_fn(states)` already returns per-node δ. Focal consumes `manhattan + δ` as the focal key. Same batched model inference, same per-replan cache, same `EVAL_DIAG` behavior. No model-path changes.

### 4.3 Selection knobs (env globals, forwarded)
- `PLANNER ∈ {astar, focal}` — default `astar` (current behavior unchanged).
- `FOCAL_W` (float) — the suboptimality weight.
- Both added to the `_EVAL_FORWARD_VARS` allowlist (next to `EVAL_DIAG`) so they reach remote workers identically; both read as module globals.

### 4.4 Planner branch (live `run_policy_episode` ~`:4501`)
`plan = space_time_focal_astar(..., w=FOCAL_W) if PLANNER=="focal" else space_time_astar(..., alpha=alpha)`. Episode loop, stepping, and metrics are otherwise untouched.

### 4.5 Edit sites (respecting the dual-copy structure — see memory note)
- `space_time_focal_astar`: NEW, single shared planner section (~`:692`).
- planner branch: **live** `run_policy_episode` (~`:4501`), not the dead copy (~`:2472`).
- `PLANNER` / `FOCAL_W` flags: near `EVAL_DIAG` (~`:3331`); allowlist entries near the forwarding block (~`:59`).

### 4.6 Data flow (one replan)
ctx encoded once per env-step (model) → closure captures it → focal loop: compute `f_min`; refill `FOCAL` by `f ≤ w·f_min`; pop min-`h_focal`; expand; batch-score successors' δ; push (`f=g+manhattan`, focal-key=`manhattan+δ`); until goal or budget → `PlanResult`.

---

## 5. Metrics & comparison methodology

Matched, paired comparison on identical instances (same seeds/suites/budget):
1. **Expansions-to-solve ratio** (focal / Manhattan-A\*), by map scale. **Primary** (< 1 = win). Already recorded per episode (`expansions`).
2. **Success within budget** (secondary).
3. **Path cost vs optimal** (the suboptimality "spent" for the expansion savings; bounded by `w`).
4. **Transfer:** expert-focal vs avgbase-focal expansions on matched cells.

Arms compared: baseline Manhattan A\* · `avgbase`-focal · `expert`-focal — all `PLANNER=focal` except baseline (`model=None`), across `w` values.

---

## 6. Testing strategy (TDD, local, free — `hrm-cloud/tests/`)

- **`w=1` optimality:** on a tiny hand-built map with Dijkstra ground truth, focal at `w=1` returns optimal-cost path (learned signal only tie-breaks).
- **Bounded suboptimality:** at `w=2`, solution cost ≤ `2×` optimal.
- **Completeness:** finds a path whenever one exists (solvable small map).
- **Determinism:** static suite + fixed seed → identical result (stable tiebreaker).
- **Interface parity:** `PlanResult` fields valid (actions reach goal, `found`/`expansions` correct).
- **Heap-maintenance correctness** is covered transitively by the optimality / bound / completeness tests.

---

## 7. Compute & validation plan (cheapest first; you have local CPU + GPU)

- **Phase A — local, free (the go/no-go gate):**
  - Unit tests on CPU.
  - **`bench_focal.py`** matched-expansion benchmark: load real model(s) (`avgbase`, and an expert if its base path is resolvable locally) on **local GPU** (`device='cuda'`), run Manhattan-A\* vs focal-learned on the *same* instances across multiple seeds × scales × static **and** dynamic suites, sweeping `w ∈ {1,1.5,2,3}`. Report the §5 metrics. GPU runs the model forward passes (the expensive part), so this can be substantial — many suites/seeds — at **zero cloud cost**.
  - **Gate:** if focal-learned does not expand materially fewer nodes than Manhattan locally, stop and reconsider — nothing spent.
- **Phase B — Modal, only if A passes *and* billing is unblocked:** full-matrix at-scale confirmation (all suites, baseline vs avgbase-focal vs expert-focal, 1–2 `w`), `EVAL_DIAG=0`, via the durable `resume_spawn` + high parallelism already built; focal cells tagged by `w` (not `α`) to avoid colliding with existing aggregates.

Local GPU likely covers the scientific demonstration on a representative subset; Modal is for full-matrix completeness only.

---

## 8. Risks & mitigations

- **Bounded-suboptimal paths × dynamic obstacles** could dent success at high `w` → measure success (not just expansions); the `w`-sweep picks the safe knee; the suboptimality bound caps the damage.
- **Two-heap sync bugs** (the trickiest code) → caught by the `w=1`-optimality, `w=2`-bound, and completeness tests.
- **Local expert load needs the base model path** (`/data/...` in expert metadata) → Phase-A gate uses `avgbase` (sufficient); expert validation can fall to Phase B, or download both base+expert and patch the path locally.
- **Probe was static/t=0/one-seed** → the benchmark uses multiple seeds and includes dynamic suites to confirm the ranking holds under dynamics.
- **Ranking could be weaker on experts/dynamics than on the static avgbase probe** → the benchmark measures it directly before any conclusion.

---

## 9. Out of scope / future work

- **Magnitude recalibration** (so `manhattan+δ` is usable as a true admissible/near-admissible heuristic): per-scale α, a learned scale head, or a training target that scales with map size. Would enable bounded-suboptimal *weighted* A\* on top of focal.
- **Retraining** the heuristic with a ranking/ordering loss (directly optimize what focal uses) or imitation of oracle expansion order.
- **onlstm backbone** evaluation (entirely missing) and completing the expert matrix.

---

## 10. Open questions

- `h_focal` = `manhattan + δ` vs `δ` alone vs a learned distance head — default `manhattan + δ` (predicted cost-to-go); revisit if the benchmark suggests otherwise.
- Default `FOCAL_W` for the headline run (pick from the Phase-A `w`-sweep knee).
- Whether to also report an *anytime* variant (lower `w` after first solution) — deferred unless the benchmark motivates it.
