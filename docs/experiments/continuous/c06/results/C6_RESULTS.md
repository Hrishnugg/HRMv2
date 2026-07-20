# C6 Heatmap / Value-Field — Results & Findings

**Date:** 2026-06-27 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c6_heatmap_value_field.py` (spec: `continuous_prm_c6_heatmap_value_field_spec.md`)
**Context:** continuous-PRM direction; see `../EXPERIMENT_RESULTS_COMPENDIUM.md` (continuous-PRM framing) and the discrete focal redesign `../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md` (referenced in §Focal below).

C6 replaces C5's per-node scalar residual with a **goal-conditioned spatial value field** (cost-to-go heatmap, bilinearly sampled at PRM nodes). Motivation: C5's HRM residual collapsed to a constant; a field is a better match for gates/bottlenecks (field-level structure).

---

## Run 1 — first local run (diagnostic, undersized) — `runs/c6_local_run2/`

Config: grid 64, models `oracle,unet,onlstm,hrm`, train 24 / eval 16 worlds, **roadmap 128/k6**, **6 epochs**, budgets 128/144, train `C_hard_maze`, eval `C_hard_maze` + `C_hard_maze_dense`.

**All methods reached 100% success** (Euclidean included; budget 128 vs 144 identical → budget not binding). No success headroom → significance table empty. Read on **expansions** only:

| Method | C_hard_maze exp | dense exp | heatmap_std | spearman→oracle |
|---|---|---|---|---|
| euclidean | 96.2 | 95.4 | 0.24 | 0.87 |
| **grid_oracle** (ceiling) | **70.2 (−27%)** | **74.2 (−22%)** | 0.59 | **0.95** |
| unet | 88.1 (−8%) | 93.6 (−2%) | 0.27 | 0.86 |
| onlstm | 96.2 (~0%) | 95.4 (0%) | **0.0015** | 0.87 |
| hrm | 95.4 (−1%) | 95.2 (~0%) | 0.056 | 0.86 |

Train loss: U-Net 0.30→0.18 (learning); **ON-LSTM 0.288→0.285 (flat, didn't learn)**; HRM 0.29→0.26.

### Findings (Run 1)
1. **The value-field framing has a real ceiling:** the oracle cost-to-go heatmap cuts expansions **22–27%** (spearman 0.95). Gate 1 confirmed.
2. **Learned field models are undertrained at this scale:** U-Net learned a partly-useful field (−8% maze); **ON-LSTM collapsed to a near-constant heatmap** (std 0.0015); HRM nearly so. Their rank-correlation to the oracle (~0.86) is **no better than Euclidean's 0.87.** 6 epochs / 24 worlds is too little.
3. **Eval band wrong:** Euclidean at 100% (roadmap 128/k6 too sparse → budget never binds). Need denser roadmap / lower budget so Euclidean sits at 50–70% (the spec's target band) to measure *success*.

---

## Methodology gotchas (record these)

- **Grid size must down/up-sample cleanly for the U-Net.** `--grid-size 56` crashes training with a decoder skip-connection size mismatch (`Expected size 6 but got size 7`, `continuous_prm_c6_heatmap_value_field.py:580` — it `cat`s mismatched skips). Use **48 or 64**. (Latent robustness gap: the U-Net should pad/crop skips; default 64 sidesteps it.)
- **Roadmap density sets the difficulty band.** C5's Euclidean ~52–60% on `C_hard_maze` (B144) came from **roadmap 192/k7**. With 128/k6 the search solves within budget → Euclidean 100% → no measurable success gap. Use **192/k7** (the spec default) for meaningful eval.
- **`--mode full` does run significance** (collect→train→eval→significance written to `results/continuous_prm_c6_significance.md`); `--mode analyze` re-runs just the significance pass on an existing raw CSV.
- C6 is a **standalone local script** (no top-level `modal` import); GPU auto-detected; dataset generated on-the-fly. Results land in `--out-dir` incrementally (survive restarts).

---

## On wiring focal search into the PRM planner (corrected analysis)

The discrete work fixed a net-harmful additive heuristic by switching to **focal search** (use the learned signal as a *ranker* within an admissible band, not an additive magnitude). The PRM planner (`../continuous_prm/continuous_prm_common.py:729` `astar_search`) has the same `f = g + euclidean + residual` structure, so focal ports structurally. **But the C6 data sharpens when focal actually helps:**

- Focal helps a model whose signal **ranks well but is mis-integrated by magnitude** (the discrete `avgbase` case: ρ≈0.99 ranking wasted by inflation → focal recovers it).
- It does **not** help a **flat / no-better-than-baseline** signal (C5 HRM constant residual; C6 ON-LSTM/HRM near-constant fields, spearman ≈ Euclidean) — there is no ranking to exploit.
- And the **oracle field works fine under the existing *additive* integration** (−27% expansions), so for value-fields the integration is **not** the bottleneck — **model/field quality is.**

**Verdict:** focal remains a sound general robustness fix for the PRM planner (and would matter once a learned model genuinely out-ranks Euclidean but is mis-calibrated), but it is **not** the immediate lever for C6. Priority order: (1) train the field models enough to produce informative fields; (2) fix the eval difficulty band; (3) revisit focal only if a good-ranking field is being wasted by additive magnitude.

---

## Run 2 — bigger run (IN PROGRESS) — `runs/c6_local_big1/`

Config: grid 64, models `oracle,unet,onlstm,hrm`, **train 96 / eval 40** worlds, **roadmap 192/k7** (proper difficulty band), **16 epochs**, budgets 128/144/168, train `C_hard_maze`, eval `C_hard_maze` + `C_hard_maze_dense` + `C_hard_rooms`.

Tests the two Run-1 limitations: (a) does more training make any learned field approach the oracle's ranking (non-constant, spearman > Euclidean)? (b) with the denser roadmap, does Euclidean drop into the 50–70% band so success gains are measurable?

**Results** — both Run-1 limitations fixed: Euclidean dropped into the band, and the field models learned (train loss U-Net 0.26→0.10, ON-LSTM 0.30→0.16, HRM 0.28→0.13; heatmap_std now **0.33–0.38** with spearman→oracle **0.90–0.97**, vs Run-1's near-zero / 0.86).

Success rate by suite × budget:

| Suite | B128 euclid | B128 oracle / unet / onlstm / hrm | B144 euclid | B144 oracle / unet / onlstm / hrm |
|---|---|---|---|---|
| C_hard_maze | 0.175 | 0.825 / 0.75 / 0.725 / **0.80** | 0.625 | 0.95 / 0.95 / 0.90 / **0.975** |
| C_hard_maze_dense | 0.075 | 0.975 / 0.40 / 0.375 / 0.50 | 0.45 | 1.0 / 0.85 / 0.85 / 0.825 |
| C_hard_rooms (OOD) | 0.0 | 0.775 / 0.10 / 0.05 / 0.20 | 0.375 | 0.90 / 0.70 / 0.675 / 0.70 |

(B168: all methods 1.0 — saturated.)

**Significant claims (McNemar, BH-corrected) — C_hard_maze B144 (Euclid 0.625):**
- **HRM: 0.975 (+0.350, p=1.2e-4, q=3.4e-4, −42 expansions)**
- U-Net: 0.950 (+0.325, p=2.4e-4, −38 exp)
- ON-LSTM: 0.900 (+0.275, p=3.4e-3, −31 exp)
- (grid_oracle ceiling: 0.950, −45 exp)

### Findings (Run 2) — the field framing works, and rescues HRM
1. With adequate training + the proper difficulty band, **all three learned field models significantly beat Euclidean** on the in-distribution suite (C_hard_maze B144: +0.28 to +0.35 success, p<0.01, ~30–42 fewer expansions).
2. **HRM is rescued by the value-field framing** — it is the *best* learned model on C_hard_maze (0.975, spearman 0.95, near the 0.984 oracle), a complete reversal of its C5 per-node-residual collapse and its Run-1 flat field. This validates the C6 hypothesis: the field formulation fixes the C5 HRM failure.
3. **Transfer degrades with distribution shift:** strong on C_hard_maze (in-dist); partial on the dense variant (B144 ~0.83–0.85 vs Euclid 0.45, but oracle is 0.975 at B128 where learned models only reach ~0.4–0.5); weaker on C_hard_rooms at low budget (B128 learned 0.05–0.20 vs oracle 0.775) — trained on maze, so rooms is OOD.
4. **Focal not needed for C6 (confirmed):** the learned fields now rank well (spearman 0.90–0.97) and the existing *additive* integration already captures it (significant wins). The C6 bottleneck was model/training quality — which the field framing + scale solved — not the integration.

## Status

C6 value-field framing is **validated locally**: it produces learned heuristics (including HRM) that **significantly beat Euclidean** on in-distribution hard maps (+0.35 success / −42 expansions at the binding budget), with HRM near the oracle ceiling. This is the key reversal of the C5 residual-per-node failure.

## Run 3 — multi-suite training (train `C_hard_maze` + `C_hard_rooms`) — `runs/c6_local_multi1/`

Same scale (96 worlds/task, 16 epochs, roadmap 192/k7, B128/144/168), now training on **both** maze and rooms; `C_hard_maze_dense` held out as a maze-variant transfer test.

Success @ B144 (Euclid in-band):

| Suite | euclid | oracle | unet | onlstm | hrm | Run-2 (maze-only) hrm |
|---|---|---|---|---|---|---|
| C_hard_maze (in-dist) | 0.625 | 0.95 | 0.975 | 0.95 | **0.975** | 0.975 (held) |
| C_hard_maze_dense (held-out) | 0.45 | 1.0 | 0.925 | 0.925 | **1.0** | 0.825 → **1.0** |
| C_hard_rooms (now trained) | 0.375 | 0.90 | 0.80 | 0.75 | **0.875** | 0.70 → **0.875** |

### Findings (Run 3) — multi-suite training closes the OOD gap
1. **Adding rooms to training lifts rooms** from 0.70 → **0.875** (near the 0.90 oracle); U-Net 0.70→0.80, ON-LSTM 0.675→0.75 too.
2. **Transfer to the held-out dense variant improves:** HRM 0.825 → **1.0** (= oracle) — broader training generalizes to an *unseen* maze variant, not only the trained suites.
3. **In-distribution maze holds** (0.975) — no capacity-splitting penalty from training on two layouts.
4. **HRM is the best learned model on every suite** (maze 0.975, dense 1.0, rooms 0.875), at/near the oracle ceiling, spearman 0.92–0.96 — fully reversing its C5 collapse and extending Run-2 across map types.
- Significant claims (analyzer band-gates to maze B144): HRM/U-Net +0.35 (p=1.2e-4), ON-LSTM +0.325. Dense/rooms deltas are even larger (HRM +0.55 on dense, +0.50 on rooms vs Euclid) but Euclid there (0.45/0.375) sits just below the analyzer's in-band claim filter.

## Status (consolidated)

C6 value-field framing is **validated locally and generalizes**: learned heuristics (HRM best, near the oracle ceiling) **significantly beat Euclidean** on in-distribution hard maps, and **multi-suite training closes the OOD gap** (rooms 0.70→0.875; held-out dense variant 0.825→1.0). This is the decisive reversal of the C5 per-node-residual failure (where HRM collapsed). Focal integration is not needed here — the field framing + adequate, diverse training is the fix.

> **Continuation (2026-06-27) → [`C7_RESULTS.md`](../../c07/results/C7_RESULTS.md).** The integration comparison (additive scalar vs value field vs focal ranker, × HRM/ON-LSTM/U-Net, matched). Findings that refine the C6 story: additive learned heuristics cut A\* expansions ~15–48% over Euclidean across six hard suites and generalize OOD; **the per-node scalar (C5-style) does *not* reproduce its C5 collapse when trained properly** (so C5's failure was training-specific, not a representation limit); and **additive beats focal** here because Euclidean is a weak admissible baseline (the opposite of the discrete-grid focal result — the better integration depends on baseline tightness).

## Open questions / next steps

- **Scale + confirm:** full 160/80-world, more-epoch, all-suite run; eventually a Modal full-suite confirmation for publication-grade numbers (current runs are 40 eval-worlds, local).
- **Harder band:** push budgets lower / add harder suites so even the trained models leave headroom vs the oracle (at B144 several already hit ≥0.95).
- **Breadth:** more map families in training; test transfer to genuinely novel topologies.
- **HRM vs ON-LSTM:** HRM consistently edges ON-LSTM under the field framing (reverses the C5 ranking) — worth confirming at scale.
