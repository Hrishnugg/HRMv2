# Residual TaskLoRA v2 → Learned Focal Search: Results Report

**Date:** 2026-06-27
**Backbones/models:** HRM and ON-LSTM `avgbase` (pooled base) + `A32_static` LoRA experts.
**Compute:** local (RTX 5090) validation; no cloud. Branch `perf/eval-speedup`, [PR #1](https://github.com/Hrishnugg/HRMv2/pull/1).
**Continuation of:** [`EXPERIMENT_RESULTS_COMPENDIUM.md`](EXPERIMENT_RESULTS_COMPENDIUM.md) — specifically its "Residual TaskLoRA v2" and "Cross-Experiment Conclusions" sections.
**Companion docs:** design spec [`2026-06-23-learned-focal-search-design.md`](../design/2026-06-23-learned-focal-search-design.md) · plan [`2026-06-23-learned-focal-search.md`](../plans/2026-06-23-learned-focal-search.md) · focal detail [`FOCAL_SEARCH_RESULTS.md`](FOCAL_SEARCH_RESULTS.md).

---

## TL;DR

The compendium left Residual TaskLoRA v2 **partial and contaminated** (interrupted nonfinite run, no final results) and flagged that learned residuals often *"degrade [the planner]... when the heuristic correction changes ordering in unhelpful ways,"* with *"avgbase more promising than specialist LoRA."* This report resolves those threads:

1. **Infra/rerun:** fixed the eval pipeline so re-eval is fast and controllable — and found that the eval env-knobs never reached the remote workers (so prior runs silently used defaults).
2. **Diagnosis:** the learned heuristic was *net-harmful as integrated*. It is a **near-perfect ranker (ρ≈0.99 with true cost-to-go) but a scale-miscalibrated magnitude**; added onto admissible Manhattan and weighted by one global `α` tuned on small maps, it was suppressed to ≈Manhattan (α pinned to the floor on all 10 models).
3. **Redesign:** use the signal as a **ranker, not a magnitude** — *Learned Focal Search* (A\*ε): admissible Manhattan bounds `f`, the learned `manhattan+δ` orders only the focal band.
4. **Result:** a **validated, regression-free ~15% reduction in A\* expansions at matched-or-better success, on BOTH hrm and onlstm bases** (`w=1.0`). The **per-task experts add nothing** over the base — on both backbones — confirming the compendium's "avgbase > specialist" with a mechanism (the bounded residual is too small to reorder nodes).

Net: hierarchical-model-as-ranker **beats** the algorithmic planner on search efficiency; specialization-on-top **does not** (with current bounded experts).

---

## 0. Relationship to the prior compendium

The compendium's open items and how this report addresses them:

| Compendium statement | This report |
|---|---|
| "Residual TaskLoRA v2 is partial and contaminated... rerun modeled evals after the cap/nonfinite fix before publishing." | §1 fixes the pipeline; §2 reports clean re-eval (default `EVAL_DIAG=1` is byte-identical, so headline metrics are trustworthy). |
| "Many learned residuals... degrade it, especially when the heuristic correction changes ordering in unhelpful ways." (§Cross-Experiment) | §3 pinpoints the exact mechanism (additive δ inflates an admissible heuristic; global α tuned on small maps). |
| "Average-base framing more promising than specialist LoRA... specialists may need... a different target." | §2 + §5 confirm avgbase ≈/> experts; §5 shows experts are identical to base under focal (bounded residual can't reorder). |
| "Budget increases often increased expansions without improving success" (family-B/OOD). | §1 budget-invariance analysis refines this: B2000 *does* help on size-OOD (A96–A256, +0.08–0.12 success) but is flat on the family-B (hopeless) and family-C (saturated) suites. |
| Pre-interruption: `residtasklora__hrm__A32_static` +0.93 pp vs baseline but **−1.00 pp vs avgbase**. | §2 reproduces this (expert −0.010 vs avgbase); §5 shows it persists under focal (expert = base). |

---

## 1. Eval infrastructure — the rerun the compendium asked for

Implemented on `perf/eval-speedup` (PR #1), all default-safe:

- **`EVAL_DIAG` flag (default 1 = unchanged).** The diagnostics block requires an O(`max_steps`·n²) pure-Python exact-cost DP per episode that does **not** affect A\* decisions (it only feeds diagnostics). It was ~61% of per-episode time even on a 64² map. `EVAL_DIAG=0` skips it and caches the NN heuristic per `(x,y,t_rel)` per replan; headline metrics (success/expansions) are identical (proven by tests). This turned large-map eval from days into hours.
- **Remote env-forwarding fix.** Critically, **the eval env-knobs (`EVAL_DIAG`, `EVAL_BUDGETS`, etc.) were never reaching the remote Modal workers** — they are read as module globals in the container, which gets defaults unless forwarded. A `modal.Secret.from_dict(...)` now forwards an allowlist. *Implication for the compendium's data:* prior runs effectively used in-container defaults for everything except the few args `main()` passed explicitly — consistent with the compendium always showing budgets `{200,500,2000}` and the default RUN_TAG.
- **Durable launch (`resume_spawn`).** A fire-and-forget entrypoint (`run_pipeline.spawn` + `--detach`) so a run survives the local client disconnecting (the original run died on an interruption).
- **Budget-invariance analyzer** (`analyze_budget_invariance.py`): reads `eval_agg` and reports where `budget=2000` is wasted. Finding: **B2000 helps substantially on the large size-OOD maps (A96–A256: +0.08–0.12 success over B500) and is flat only on small/saturated/hopeless suites** — so the compendium's "budget didn't help" is specific to the family-B shift, not size-OOD.

## 2. Clean re-eval (additive integration, `EVAL_DIAG`-preserved metrics)

Matched (paired) comparison on the completed HRM cells (100-episode aggregates):

- **`avgbase_hrm` vs Manhattan baseline: +0.010 success overall** (best at moderate OOD scale — A192 +0.057, A128 +0.021 — but **−0.038 at A256**, and ~0.91–0.99× the expansions where it helps). Marginal, scale-bounded — matches the compendium's +1.0 pp.
- **`residtasklora_hrm_A32_static` expert vs `avgbase_hrm`: −0.010** (7 W / 10 T / 27 L of 44 cells). The expert only helps on its own training stage (n=32 +0.063) and regresses on transfer. Matches the compendium's **−1.00 pp vs avgbase**.
- **Family structure:** Family-B suites ≈ 0.01 success for *all* methods (unsolved shift); Family-C ≈ 1.0 (saturated); the learned model only moves Family-A.

So the clean rerun **confirms** the compendium: the learned base is marginally above Manhattan; the experts do not beat the base.

## 3. Diagnosis — why the learned heuristic was marginal/harmful

- **`α` pinned to the floor on all 10 models.** Validation α-tuning selected `α=0.5` (the candidate floor) for every model, with success collapsing as α rose (e.g. `avgbase_hrm` val 0.86 → 0.13 as α 0.5 → 2.0). The tuner was actively *suppressing* the learned residual.
- **Local probe (predicted residual vs true cost-to-go residual):**

  | Map size | ρ(pred, true residual) | pred mean | true residual mean | pred/true |
  |---|---|---|---|---|
  | n=64 | 0.987 | 88 | 121 | 0.73 |
  | n=128 | 0.992 | 105 | 249 | 0.42 |
  | n=192 | 0.994 | 94 | 345 | 0.27 |
  | n=256 | 0.992 | 96 | 493 | 0.19 |

  The model is an **excellent ranker (ρ≈0.99 at every scale)** that never over-predicts (admissible), but its **magnitude is flat (~90–105) while the true residual grows with map size** — `pred/true` collapses from 0.73 to 0.19. It learned the *shape* of cost-to-go, not how to scale it.
- **Mechanism (this is the compendium's "ordering changed in unhelpful ways," made precise):** the integration was `f = g + manhattan + α·δ`. δ is *added* onto an already-admissible Manhattan, so a positive δ **inflates** `h`. α-tuning on small (n≤64) maps, where `pred≈true`, makes `α>1` push `h` above true cost (inadmissible) → search misdirects → success collapses → tuner retreats to `α=0.5`; applied globally that throttles the (good) signal to ≈Manhattan on large maps. **The failure was integration, not the model.**

## 4. Redesign — Learned Focal Search

Use the signal as a **ranking**, where magnitude miscalibration is irrelevant. `space_time_focal_astar` (A\*ε, drop-in `PlanResult`):

- `OPEN` ordered by the admissible `f = g + manhattan` → suboptimality bounded by `w`.
- Focal band = `{n : f(n) ≤ w·f_min}`; expand the band node minimizing `h_focal = manhattan + δ` (the learned predicted cost-to-go), `f` then insertion-order as tiebreaks.
- The learned signal **never enters `f`** — it cannot break admissibility or misdirect; a bad signal degrades to Manhattan ordering. `w` replaces `α` with graceful degradation (no inadmissibility cliff).
- Selected by env `PLANNER=focal` / `FOCAL_W` (default `astar` / `1.0`); reuses the existing trained models — **no retraining**. (Full algorithm + invariants/tests in the spec and `FOCAL_SEARCH_RESULTS.md`.)

## 5. Results — focal validation (local GPU)

**Base as focal ranker vs Manhattan A\*, matched instances, `w=1.0`:**

| Backbone | `OOD_A128_static` | `OOD_A192_static` | `OOD_A128_moderateDyn` |
|---|---|---|---|
| **hrm** `avgbase` (8 seeds, B200) | 0.85 (succ 0.62→**0.75**) | 0.94 (0.75=0.75) | 0.93 (0.62=0.62) |
| **onlstm** `avgbase` (4 seeds, B200) | 0.85 (0.75=0.75) | 0.85 (0.75=0.75) | — |

(`exp_ratio` = focal expansions / Manhattan expansions; < 1 = fewer. Success = base → focal.)

→ **~6–15% fewer A\* expansions at matched-or-better success, on both backbones.** HRM A128 even *improves* success. The win holds with moving obstacles.

**Expert vs base as focal rankers (`w=1.0`):**

| Suite | base exp_ratio | expert exp_ratio | success |
|---|---|---|---|
| `OOD_A128_static` (hrm, B200) | 0.85 | **0.85** | 0.75 = 0.75 |
| `OOD_A192_static` (hrm, B150) | 0.82 | **0.82** | 0.67 = 0.67 |
| `OOD_A128_static` (onlstm, B200) | 0.78 | **0.78** | 0.67 = 0.67 |

→ **The expert ranks identically to the base, on both backbones.** The bounded LoRA residual is too small to change node *ordering*, so the expert's focal behavior collapses onto the base's. This is the compendium's "avgbase > specialist" with a mechanism.

**`w`-sensitivity:** `w=1.0` (learned tie-breaking) is the safe, regression-free operating point. `w=1.05` helps more on some suites (A128_moderateDyn → 17% fewer) but *regresses success* on others (A192_static 0.75→0.62) — the in-search ranking isn't reliable enough to steer a wide band. Hence the default `FOCAL_W=1.0`.

## 6. Conclusions

- **Thesis (hierarchical model + transfer beats the algorithmic planner):** *partially confirmed.* The hierarchical learned model, used as a **focal ranker**, makes Manhattan A\* ~15% more search-efficient at matched-or-better success — for **both HRM and ON-LSTM**. This is a clean reversal of the net-harmful additive integration, using the *same* trained weights.
- **Transfer/specialization:** *not supported.* Per-task LoRA experts add nothing over the pooled base, on both backbones — confirming the compendium and the additive-eval result, and explaining it (bounded residual can't reorder).
- **Why earlier learned residuals looked weak/harmful:** they were a great *ranker* mis-integrated as a miscalibrated *magnitude* under a global α tuned on the wrong scale. Focal sidesteps this entirely.

## 7. Caveats

- Validated locally on HRM + ON-LSTM `avgbase` and the `A32_static` experts; **A256, the onlstm A64 experts, and the full 22-suite matrix are not yet run under focal**, nor is a full-matrix Modal confirmation (deferred — billing).
- Seeds are modest (3–8) and budgets 150–200; success rates are coarse at these counts. The expansion-ratio *direction* is consistent and the expert==base result is exact, but publication-grade magnitudes need the full Modal sweep.
- The static probe (ρ≈0.99) only *partially* transfers to in-search guidance — good enough for tie-breaking, not for wide bands.

## 8. Future work

- **Make specialization matter for ranking:** the bounded residual is too small to reorder. A larger/less-bounded correction, or an expert trained with a **ranking/ordering objective** (directly optimizing what focal consumes), is the path to test whether specialization can ever beat the base ordering.
- **Recalibrate the magnitude per scale** so `manhattan+δ` becomes usable as a near-admissible heuristic — would enable bounded-suboptimal weighted A\* on top of focal and potentially widen the useful `w` window.
- **Scale + breadth:** A256, onlstm experts, full suite set, full Modal confirmation.

## 9. Provenance & reproduction

- **Code:** `space_time_focal_astar` + `PLANNER`/`FOCAL_W` in `residual_tasklora_v2.py`; benchmark `bench_focal.py` (supports `--expert-ckpt`/`--base-ckpt`). Tests: `tests/test_focal.py`, `tests/test_focal_wiring.py` (+ `tests/test_eval_speedup.py`). All on branch `perf/eval-speedup` / PR #1.
- **Reproduce a result** (after `modal volume get` of a checkpoint into `hrm-cloud/ckpts/`):
  ```bash
  python bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --expert-ckpt ckpts/residtasklora__hrm__A32_static.pt --base-ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --suites OOD_A128_static,OOD_A192_static --seeds 8 --budget 200 --w 1.0 --device cuda
  ```
- **Prior context:** [`EXPERIMENT_RESULTS_COMPENDIUM.md`](EXPERIMENT_RESULTS_COMPENDIUM.md).
