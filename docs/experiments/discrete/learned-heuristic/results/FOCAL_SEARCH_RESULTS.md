# Learned Focal Search — Results

**Date:** 2026-06-23
**Scope:** Residual TaskLoRA v2, HRM backbone, `avgbase` (pooled base) model. Local validation (RTX 5090), no cloud.
**Design spec:** `docs/experiments/discrete/learned-heuristic/design/2026-06-23-learned-focal-search-design.md`

## TL;DR

The learned A\* heuristic was failing as originally integrated (added on top of Manhattan, weighted by a global `α`) — it was **net-harmful**. Diagnosis showed the model is a **near-perfect ranker (ρ≈0.99 with true cost-to-go) but a badly scale-miscalibrated magnitude**, and the global `α` (tuned on small maps) suppressed it to ~Manhattan. Re-integrating the same model as a **focal-search ranking signal** — where only the *order* matters, not the magnitude — turns it into a net positive: **~6–15% fewer A\* expansions at matched-or-better success** on large OOD maps (static and dynamic), with **zero success regressions at the safe setting `w=1.0`**. The thesis (hierarchical learned model beats the algorithmic planner on search efficiency) holds, conservatively.

---

## 1. The original failure

Arms (HRM): `baseline` = Manhattan A\* (no model); `avgbase` = pooled base heuristic; `residtasklora` = base + bounded LoRA residual per stage. Heuristic used as `f = g + manhattan + α·δ` in budget-limited space-time A\*.

From the completed eval cells:
- **`avgbase` vs Manhattan: +0.010 success overall** (peak +0.057 at A192, **−0.038 at A256**). Barely better than no model, and *worse* at the largest scale.
- **Task-LoRA experts did not beat the pooled base** (A32 expert vs avgbase: **−0.010**; only helped on its own training stage). The central specialization hypothesis was unsupported.
- **`α`-tuning pinned `α` to the floor (0.5) on all 10 models**, with validation success collapsing as `α` rose (e.g. avgbase 0.86→0.13 as α 0.5→2.0). The tuner was actively *suppressing* the learned residual.

## 2. Diagnosis (local probe of `avgbase__hrm`)

Comparing the model's predicted residual to the true cost-to-go residual on sampled states:

| Map size | ρ(pred, true residual) | pred mean | true residual mean | pred/true |
|---|---|---|---|---|
| n=64 | 0.987 | 88 | 121 | 0.73 |
| n=128 | 0.992 | 105 | 249 | 0.42 |
| n=192 | 0.994 | 94 | 345 | 0.27 |
| n=256 | 0.992 | 96 | 493 | 0.19 |

- **The model is an excellent *ranker*** (ρ≈0.99 at every scale) and never over-predicts (admissible).
- **Its magnitude is flat (~90–105) regardless of map size** while the true residual grows with scale — so `pred/true` collapses from 0.73 (n=64) to 0.19 (n=256). It learned the *shape* of cost-to-go but not how to scale it.
- This explains everything: `α`-tuning on small (n≤64) maps, where `pred≈true`, pushes `manhattan+α·δ` above true cost (inadmissible) at `α>1` → search misdirects → success collapses → tuner retreats to `α=0.5`; applied globally, that throttles the (good) signal to ≈Manhattan on large maps.

**Conclusion:** the failure was *integration*, not the model. A near-perfect ranker was being used as a miscalibrated additive magnitude.

## 3. The redesign: Learned Focal Search (A\*_ε)

Use the signal as a **ranking**, where magnitude miscalibration is irrelevant.
- `OPEN` ordered by the admissible `f = g + manhattan` → suboptimality bounded by `w`.
- Focal band = `{n : f(n) ≤ w·f_min}`; expand the band node minimizing `h_focal = manhattan + δ` (primary), `f` then insertion-order as tiebreaks.
- The learned signal only orders within the bounded band — it can never break admissibility or misdirect into bad regions; a bad signal degrades to Manhattan ordering.
- `w` replaces `α`. `w=1` = optimal A\* with learned tie-breaking (zero-risk floor); larger `w` = more reliance on the ranking, fewer expansions, bounded-longer paths.

Implemented as a drop-in `space_time_focal_astar` (same `PlanResult`), selected by env `PLANNER=focal` / `FOCAL_W`. No retraining — reuses the existing models.

## 4. Results (local, RTX 5090, 8 seeds, budget 200)

Matched comparison vs Manhattan A\* on identical instances. `exp_ratio` = focal expansions / baseline expansions (median); **< 1 = fewer expansions**.

| Suite | `w=1.0` exp_ratio / succ(base→focal) | `w=1.05` exp_ratio / succ |
|---|---|---|
| `OOD_A128_static` | **0.85** / 0.62 → **0.75** | 0.85 / 0.62 → 0.75 |
| `OOD_A192_static` | **0.94** / 0.75 → 0.75 | 0.94 / 0.75 → **0.62** ⚠ |
| `OOD_A128_moderateDyn` | **0.93** / 0.62 → 0.62 | **0.83** / 0.62 → 0.62 |

(An earlier 3-seed gate showed larger ratios — 0.78/0.83 — but that was noise; the 8-seed numbers above are the firmed-up result.)

**Findings:**
- **`w=1.0` (learned tie-breaking) is a consistent, regression-free win:** 6–15% fewer expansions at matched-or-better success across static *and* dynamic large OOD maps. A128 even *improves* success (0.62→0.75).
- **`w=1.05` is an unreliable dial:** bigger win on some suites (A128_moderateDyn → 17% fewer) but *regresses success* on others (A192_static 0.75→0.62). The in-search ranking isn't reliable enough to safely widen the band.
- So the operating point is **`w=1.0`** (the default), where the benefit is real and safe.

## 4b. Expert vs base — the transfer thesis under focal

The task-LoRA expert (`residtasklora__hrm__A32_static`) was compared head-to-head against the pooled base as the focal ranker, both matched to the same Manhattan baseline (`w=1.0`):

| Suite | base exp_ratio | expert exp_ratio | success (base / expert) |
|---|---|---|---|
| `OOD_A128_static` (budget 200, hrm) | 0.85 | **0.85** | 0.75 / 0.75 |
| `OOD_A192_static` (budget 150, hrm) | 0.82 | **0.82** | 0.67 / 0.67 |
| `OOD_A128_static` (budget 200, onlstm) | 0.78 | **0.78** | 0.67 / 0.67 |

**The expert ranks identically to the base** — no expansion advantage, matched success — on every suite tested, for **both the hrm and onlstm experts**. Mechanistically, the **bounded LoRA residual is too small to change the node *ordering***, so the expert's focal behavior collapses onto the base's. This corroborates — now under the focal integration — the earlier additive-eval result that the experts did not beat the pooled base (−0.010). **The expansion win is entirely the base model's ranking; the per-task specialization adds nothing.**

## 4c. onlstm backbone — the win generalizes

Same focal integration, the onlstm `avgbase` model as the focal ranker (4 seeds, budget 200, `w=1.0`):

| Suite | exp_ratio (focal/Manhattan) | success (base / focal) |
|---|---|---|
| `OOD_A128_static` | 0.85 | 0.75 / 0.75 |
| `OOD_A192_static` | 0.85 | 0.75 / 0.75 |

The onlstm base reduces expansions ~15% at matched success — comparable to (slightly better than) the hrm base. **The learned-ranker win is not hrm-specific; it holds across both hierarchical backbones.**

## 5. Interpretation

The model's static-probe ranking (ρ≈0.99) only *partially* translates to in-search guidance — good enough to help as a tie-breaker, not good enough to steer a wide focal band. The win is therefore modest but genuine and regression-free, and it is a **clean reversal** of the original additive approach (net-harmful → net-positive) using the *same* trained weights.

## 6. Caveats

- Both **hrm and onlstm** `avgbase` bases *and* A32 experts validated under focal (§4 / §4b / §4c). Not yet run: A256 scale and a full-suite Modal confirmation.
- Budget 200 (base sweep, 8 seeds) / budget 150–200 (expert comparison, 3–4 seeds), large OOD suites. Success rates are coarse at these seed counts.
- Local-only; no full-matrix Modal confirmation (deferred — billing).

## 7. Future work

- **Widen the useful `w` window** by improving the *in-search* ranking: retrain with a ranking/ordering loss (directly optimize what focal consumes), or recalibrate the magnitude per scale so `manhattan+δ` is usable as a near-admissible heuristic.
- **Make specialization matter for *ranking*:** experts currently don't (the bounded residual is too small to reorder nodes — §4b). A larger/less-bounded correction, or an expert trained with a ranking/ordering objective, would be needed to test whether per-task specialization can ever improve the focal ordering over the base.
- **Scale + breadth:** A256, onlstm, full suite set; budget sweep.
