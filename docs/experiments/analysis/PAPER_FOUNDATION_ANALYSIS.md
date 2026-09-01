# Paper Foundation Analysis: Recurrent Ideas, Innovations, and Key Results

**Date:** 2026-07-20
**Purpose:** the analytical bridge between the [master evidence synthesis](../MASTER_EXPERIMENT_SYNTHESIS.md) and the AAAI 2027 paper draft. This document extracts what is *recurrent*, what is *new*, and what is *claimable*, and commits to a single paper narrative. Numbers cited here are taken from the synthesis's evidence-safe wording; the paper must not go beyond them.

---

## 1. The program in one paragraph

Across roughly six months, the program ran two coupled experiment ladders — a discrete grid-world program (dynamic-obstacle forecasting, learned A* heuristics, an HRM-v2 direct solver port) and a continuous PRM program (C1–C13) — asking one implicit question: **do hierarchical/recurrent architectures (HRM in particular) provide an advantage for learned planning guidance?** The answer that emerged is not the intended one, but it is more useful: learned guidance produces large, reproducible search-efficiency gains, yet *every* factor that governed those gains sat in the harness — representation and training objective, planner integration, information access, task formulation, and supervision regime — and *none* sat in architectural hierarchy. The program closes with a constructive demonstration: after restructuring the harness (bounded observations, behavior-derived supervision, one inference-time local Bellman backup), a small flat MLP beats the best complete-map hierarchical provider by ~16% expansions at matched empirical path quality on 144 held-out worlds (C13-M).

## 2. Recurrent ideas (the paper's connective tissue)

### R1. Calibrate headroom before comparing methods

The single most repeated methodological move: before testing a hypothesis, prove the benchmark can express the effect.

- C1–C4 saturated (Euclid success 0.925–1.000 at B100); differences were invisible → motivated C5's calibrated hard maps (Euclid banded to ~0.25–0.62).
- C11's G0-H oracle probe (oracle/leg-sum ratios 0.082–0.225, monotone in mission length K) *authorized* the architecture test before any learned arm was trained.
- C12-A's G0 established memory-relevant headroom (71.3% decision-relevant aliasing; privileged mode/history diagnostic +0.467 completion, −65.1% collision-adjusted regret) before testing learned memory.
- C13-C/D established oracle integration ceilings before learned providers were inserted (a privileged oracle that *loses* 0/6 under a bad integration is a harness verdict, not a model verdict).

**Paper use:** present "headroom-gated experiment design" as an explicit methodological pattern. A null result under a passed headroom gate is informative; without the gate it is noise.

### R2. Three harness layers can each manufacture an apparent architecture result

The program audit's organizing scheme, now with completed evidence at every layer:

| Layer | Failure observed | Controlled recovery | Stage |
| --- | --- | --- | --- |
| Representation/training | Scalar HRM collapses to the residual cap (constant predictions; MAE 3.112; delta_mean=4.0) | Goal-conditioned value-field target rescues HRM: success 0.625→0.975 (q=0.000338), −42.4 mean expansions | C5→C6 |
| Planner integration | Discrete ranker is near-perfect in order (rho 0.987–0.994) but scale-flat (pred/true 0.19–0.73); additive insertion is inert/harmful | Same frozen weights as focal tie-breaker at w=1.0: 6–15% expansion cut, no observed success regression | discrete focal |
| Task formulation | One-shot local scalar regression suspected MLP-saturable | C11 gives every arm global inputs and compositional missions: real shallow-K separation appears (U-Net/GNN 0.67–0.87 vs MLP) — but still no depth dose-response | C11 |

Corollary result: C7 later showed a properly trained *scalar* model performs comparably to fields (ratios 0.427–0.822), so C6's lesson is "training/objective," not "fields are necessary." Apparent architecture failures should not be read at face value; neither should apparent wins.

### R3. Integration choice is governed by baseline tightness (cross-domain principle)

- Discrete grids, Manhattan baseline (tight, admissible): additive learned magnitude is inert or harmful (clean-v3: every learned arm ≤ Manhattan); focal *ranking* inside the admissible band rescues the same weights (6–15%).
- Continuous hard PRMs, Euclidean baseline (loose): additive learned magnitude wins big (C7 field-HRM 15–48%; C8 65–95% on dynamics); focal variants add little (0.789–0.977 vs additive 0.427–0.822 in C7).
- C13 replays the principle at a finer grain: a *fresh* certifier search discards phase-1 work (C13-C loses 6/6 with an oracle), a *shared-queue* certifier repairs it (C13-D wins 6/6 with the same oracle); a static insertion of a well-trained local model fails (+16.17) while one radius-bounded Bellman backup at inference flips it (−1.21 → calibrated −13.04).

**Paper use:** this is the closest thing the program has to a transferable design rule: *diagnose the baseline and the search regime first; the same learned signal can be useless or decisive depending on where it enters the planner.*

### R4. Pooled priors beat specialists; supervision scarcity — not method — sets the transfer regime

- Specialist adapters repeatedly fail to beat pooled bases (C3: 1.1–3.0% expansion gains; discrete TaskLoRA experts mostly regress, −0.42 to −20.25 pp; focal experts exactly match their base).
- C9/C9h: at K=1, rank-8 LoRA preserves the strong pooled base while full fine-tuning is high-variance; by K=8–16 full FT reaches lower ratios (0.571 maze-dense; 0.500 rooms-large). C9h's dissection: bounded-vs-unbounded LoRA delta is 0.000 ± 0.008 across 27 cells — the plateau is **low-rank capacity, not the output clamp**.
- C9b: under dynamics one world supplies ~25k (node,t) labels; the low-data crossover disappears — full FT is no longer catastrophic at K=1 and LoRA keeps improving. The regime variable is labels, not "number of worlds."
- C10: zero-label descriptor-weighted adapter interpolation is a clean null (deltas −0.024 to +0.116 vs zero-shot; uniform often matches RBF) even though the routing machinery works almost perfectly (own-axis mass 0.986–0.998). Composition cannot exceed a plateaued expert family sitting on a strong base.

### R5. The hierarchy hypothesis fails every controlled dose-response test — from many directions

This is the paper's central negative, and its strength is *convergence across formulations*:

1. **Scalar regression (C5):** HRM collapses (optimization pathology, not hierarchy per se).
2. **Field regression (C6/C7/C8):** HRM works but never separates from ON-LSTM/U-Net; U-Net (global view) is often strongest.
3. **Forecasting (discrete):** the one regime with real separation — and the sign flips by task (HRM 71/100 vs LSTM 68–69 on one protocol; ON-LSTM 0.3875 vs HRM 0.2650 on Preset M+). Architecture separation exists on forecasting but is task-dependent, not uniformly hierarchical.
4. **Compositional missions (C11):** with proven headroom (0.082–0.225), global inputs, three seeds, and 198 checkpoints: G1 negative (no structured arm beats MLP with a non-decreasing gap over depth), forced ACT compute flat (G2a), learned halting *anti-correlates* with mission depth (rho=−0.407, p≈0.0005 — G2b inverted), 5/33 recurrent cells collapse, and the 12-run scaled addendum preserves U-Net > HRM (completion 0.804 vs 0.736 at K=2; 0.413 vs 0.307 at K=8, where HRM equals the leg-sum baseline exactly — persistent collapse).
5. **Persistent hidden-regime memory (C12-A):** memory headroom is real (G0 pass) yet the matched temporal-hierarchy pilot is `strong_negative` — fails forecast, planning, and carry gates.
6. **Iterative refinement depth (C12-B):** recurrent cycles genuinely help bounded search *monotonically in every cell* (e.g., C/K8 0.576→0.517) — but the gain is no larger at K=8 than K=2 (the preregistered dose-response fails), and weight tying wins only one config (C/K8, q=0.000067) while losing on A/K8.
7. **Architecture substitution under a working method (C13-N/O):** replacing the successful MLP with trimmed HRM keeps useful pooled signal but fails suite robustness and path-quality gates; summary-last readout alignment helps transiently (iteration 6) and fails at the endpoint.
8. **Direct-solver fidelity (HRM-v2):** the celebrated pre-fix ACT metrics were a validity artifact (74.60% q-halt accuracy = 100 − 25.40% exact: a frozen constant-false halt head); the faithful port trains its mechanisms but has not reproduced the paper result.

What survives *for* structure: global-input architectures (U-Net/GNN) at shallow K in C11; ON-LSTM on structured dynamics; C12-B's monotone cycle gains; C6's field-HRM strength. The honest summary: **architecture matters at specific harness configurations, but no tested formulation converts nominal hierarchy or added depth-of-compute into a growing advantage on deeper/more compositional problems.**

### R6. Negatives are made publishable by mechanism isolation and twin controls

- Aware/blind *twins* differing only in the future window (C8): 7 uncorrected cells favor blind vs 1 aware; MAE nearly split (11/24 vs 13/24) → "no systematic future-window benefit," with a calibration-vs-search-utility mechanism note. Adaptation cannot unlock it either (C9b: 0/9 aware wins at full-FT K16).
- C10's null is paired with a *working* routing mechanism (selectivity ~99%), separating "composition fails" from "routing fails."
- C11's negative is paired with collapse forensics (the 1.1397 loss signature; padding experiments showing models exploit pad steps as extra compute) so optimization pathology is not misread as capacity evidence.
- C13's ladder is a chain of single-variable falsifications: exact behavior target fails at exact value (C13-E/F), shallow local constructions too weak (C13-G), one-suite training fails distribution (C13-I), balanced training fails static insertion (C13-J), integration flips it (C13-K), scale calibration (C13-L), then confirmation (C13-M).

### R7. Preregistration, hard stops, and integrity manifests as first-class scientific machinery

From C7 onward: frozen designs, binding budgets, matched-solved paired statistics, BH correction, world-clustered bootstrap plans, untouched confirmation cohorts (C13-M's 144 worlds generated only after freezing model/integration), refusal to retune failed development blocks (C13-N/O explicitly do not reopen confirmation), forbidden-information boundaries with leakage tests (C13-P), hash-bound artifacts, and deterministic duplicate-evaluation checks. Two validity catches justify the machinery: the HRM-v2 frozen-halt metric and the C11 K=0 continuity diagnosis (premise failure, not measurement bug).

### R8. Information boundaries as an instrument, not a handicap (C13's inversion)

The professor's constraint — current-state, bounded-radius observation, no shortest-path supervision — initially looks like a straitjacket. The ladder converts it into the paper's constructive result: **a model that sees less (bounded local observations) but computes correctly at inference (one radius-bounded Bellman backup) beats models that see everything (complete-map field providers).** 68.31 vs 81.26 expansions; delta −12.96, CI [−16.30, −9.74]; 109/3/32 W/T/L; all six suites negative; better empirical mean/max cost ratios (1.0235/1.1624 vs 1.0311/1.3346). Caveats that must travel with it: single model seed and cohort; the direct arm is not formally bounded (a separate w=1.10 FOCAL control passes all certificates); the unoptimized Python feature builder is ~14× slower per world than field-HRM inference; the observation simulator uses a known PRM (not map-free sensing).

## 3. Main innovations (what a reviewer can take away)

1. **A calibrated continuous-PRM benchmark ladder with headroom gates.** Hard static suites (C5), space-time dynamic suites with exact space-time Dijkstra labels (C8), compositional multi-leg missions with exact oracle headroom (C11), and bounded-observation current-state planning (C13) — each stage authorized by an explicit oracle/baseline calibration gate, evaluated with matched-solved paired ratios rather than raw means.
2. **The baseline-tightness integration principle** (R3): a cross-domain, twice-replicated design rule for *where* a learned signal should enter search (additive magnitude vs within-band ranking vs inference-time local backup), including the shared-queue certifier repair (C13-C→D: same oracle, 0/6→6/6).
3. **A capacity-vs-bound dissection of LoRA transfer** (R4): the plateau attribution experiment (bounded vs unbounded rank-8: 0.000 ± 0.008) plus the supervision-density account that predicts when the LoRA/full-FT crossover exists (C9 vs C9b).
4. **A multi-formulation hierarchy audit with favorable-condition design** (R5): headroom-authorized compositional missions, memory-authorized persistent dynamics, matched-compute forced-depth arms, learned-halting correlation tests, collapse forensics, and architecture-substitution controls under a working method — jointly the strongest controlled negative on hierarchical planning advantage we are aware of at this substrate scale.
5. **The bounded-observation local-Bellman result** (R8): less information + correct inference-time local computation beats complete-map learned providers at matched empirical path quality — with the mechanism isolated by the C13 ladder (integration, not distribution or capacity, is decisive).
6. **Validity forensics patterns**: the frozen-halt signature (74.60 = 100 − 25.40), the residual-cap collapse signature, pad-steps-as-compute, and the K=0 continuity premise analysis — reusable diagnostics for "too good/too flat to be true" learned-planning metrics.

## 4. Key results inventory (paper-safe numbers, by claim)

| # | Claim (evidence-safe) | Headline numbers | Tier/caveats |
| --- | --- | --- | --- |
| K1 | Learned heuristics cut static PRM search effort | C7 field-HRM ratios 0.521–0.850 (15–48%); corrected success 4/6 suites; scalar HRM 0.427–0.822; suboptimality 1.02–1.14 | Local, 1 seed, 24 worlds/suite, matched n as low as 6 |
| K2 | Learned heuristics cut dynamic space-time search effort | C8 heavy: ratios 0.046–0.380 on strong cells (65–95%); success +0.2 to +0.7; corrected success 5/6 | Local, 1 seed; dense-maze matched n=1 |
| K3 | No systematic future-window benefit | 7 blind vs 1 aware significant cells (uncorrected); MAE 11/24 vs 13/24; C9b 0/9 aware wins at full-FT K16 | Failure-to-observe, not equivalence |
| K4 | Pooled prior + LoRA preserves at low K; full FT wins with labels; plateau is capacity | LoRA K1 ratios ~0.65–0.77 at high success; full-FT K8–16 0.490–0.571; bound delta 0.000±0.008 | Needs world-clustered reanalysis before formal CIs |
| K5 | Dense supervision removes the crossover | ~25k labels/world in dynamics; full-FT K1 no longer catastrophic | Mechanistic account, descriptive |
| K6 | Zero-label adapter interpolation does not beat zero-shot | Deltas −0.024…+0.116; routing selectivity 0.986–0.998 | Clean null, transductive descriptors |
| K7 | Discrete integration inversion | rho 0.987–0.994 but pred/true 0.19–0.73; focal w=1.0 cuts 6–15%, no observed regression; additive arms ≤ Manhattan | Local pilot, 3–8 seeds |
| K8 | Compositional headroom is real; no depth dose-response | Oracle ratios 0.082–0.225; G1 negative; forced ACT flat; halting rho=−0.407 (p≈0.0005); U-Net/GNN shallow-K 0.67–0.87; 5/33 collapse; scaled addendum preserves ordering | World-clustered reanalysis needed for CIs |
| K9 | Memory headroom real; learned temporal hierarchy adds nothing | Aliasing 71.3%; privileged +0.467 completion, −65.1% regret; pilot strong_negative | One-seed development pilot |
| K10 | Refinement cycles help bounded search but without K-dose response | Monotone in all 4 cells (e.g., 0.576→0.517); K8−K2 gain +0.0038/−0.0305 (CIs cross 0); C/K8 tied win q=0.000067 | Bounded-search-efficiency under inadmissible heuristic; Bellman residual worsens |
| K11 | Bounded-observation local-Bellman beats complete-map providers | 68.31 vs 81.26 expansions; delta −12.96 CI [−16.30, −9.74]; 109/3/32; 6/6 suites; cost ratios 1.0235/1.1624 vs 1.0311/1.3346; w=1.10 control zero violations | 1 model seed, 1 cohort; not formally bounded; feature build 5.14 s/world vs 0.371 |
| K12 | Literal HRM substitution / readout alignment do not preserve K11 | C13-N pooled −8.625 (CI [−16.667, −1.208]) but 3/6 suites; flat comparison CI crosses 0; C13-O endpoint +2.083 vs trimmed | Development-only diagnostics |
| K13 | Validity artifacts masquerade as capability | 74.60 = 100 − 25.40 frozen-halt; 1.1397 cap signature; pad-steps-as-compute | Forensic evidence |

## 5. The contribution decision

**One-sentence contribution (the paper's thesis):**

> In learned guidance for search-based planning, the harness — representation, planner integration, information access, and supervision regime — determines success or failure; architectural hierarchy does not: across a preregistered thirteen-stage program, correctly harnessed simple models repeatedly matched or beat hierarchical ones, no depth/hierarchy dose-response survived controlled tests, and restructuring the harness (bounded observations + one inference-time local Bellman backup) beat the best complete-map hierarchical provider by ~16% expansions at matched empirical path quality.

**Framing rationale.** Three candidate framings were considered: (a) a C13-M method paper (too thin alone: one seed/cohort, modest method novelty, wastes the audit corpus); (b) a pure negative-results paper on HRM (invites "you tuned it wrong" — answerable only with the full harness evidence anyway); (c) the harness-dominance thesis with the hierarchy audit as the running case study and C13-M as the constructive demonstration. Option (c) is chosen: it matches the synthesis's own evidence-safe working thesis, gives reviewers a positive takeaway (design rules + benchmark + constructive result), and makes every negative load-bearing.

**Why AAAI fits.** Heuristic search, bounded-suboptimal search, and search+learning are core AAAI topics (HSDIP/SoCS community); the paper speaks to both the search audience (integration principle, certificates, matched-quality evaluation) and the ML audience (architecture claims, transfer regimes, preregistration).

## 6. Narrative outline mapped to evidence

1. **Introduction** — hierarchy hype vs. harness reality; contribution bullets C1–C4 (benchmark+methodology; harness-dominance evidence; transfer-regime account; hierarchy audit + constructive C13-M result).
2. **Benchmark and methodology** — PRM substrate, exact labels, budget calibration, headroom gates, matched-solved paired statistics, preregistration discipline. (R1, R7)
3. **Harness layer I: representation/training** — C5 collapse → C6 rescue → C7 scalar viability. (R2)
4. **Harness layer II: planner integration** — C7/C8 additive-over-focal on loose baselines; discrete inversion; the baseline-tightness principle; C13-C/D shared-queue repair as a microcosm. (R3)
5. **Harness layer III: information and supervision** — C8 future-window null; C9/C9h/C9b transfer regimes; C10 interpolation null. (R4, R6)
6. **The hierarchy audit under favorable conditions** — C11 (headroom, gates, halting inversion, collapse forensics, scaled addendum), C12-A/B, C13-N/O, HRM-v2 fidelity note. (R5, K8–K10, K12–K13)
7. **Constructive result: bounded observation + local Bellman** — the C13 ladder and C13-M confirmation with explicit caveats. (R8, K11)
8. **Related work** — learned heuristics (SaIL, Neural A*, TransPath, bootstrapped heuristics), bounded-suboptimal/focal search, real-time search (RTAA*/LRTA*), VIN/neural algorithmic reasoning, HRM and its independent analyses, LoRA/merging, preregistration+statistics in ML.
9. **Limitations** — single-GPU/seed locality, repeated-world pooling pending clustered reanalysis, one-cohort C13-M, no formal bound on the direct arm, wall-time gap, known-PRM observation model, C13-P unlaunched.
10. **Conclusion** — the harness-first checklist for learned search guidance; C13-P as preregistered future work.

## 7. Overreach guard (claims the paper must NOT make)

- Not "hierarchy is useless" — only: no tested formulation produced a robust hierarchy/depth advantage; several favorable-condition tests failed it. C6 field-HRM and C12-B cycles are real structure-favorable cells.
- Not "value fields are necessary" (C7 scalar refutes), not "fields beat scalars."
- Not "time-awareness hurts" — only "no systematic benefit observed."
- Not "LoRA never catastrophic" (bugtrap/rooms-large higher-K degradations exist).
- Not "all suites significant" for C7/C8/C10 success (respect the per-suite q-values).
- Not "C13-M is faster" (wall-time is currently worse) and not "formally bounded" (direct arm) and not "map-free."
- No C13-P outcome claims (harness verified; run not launched).
- Uncertainty language: world-clustered reanalysis is still pending for C9/C9h/C9b/C10/C11 — the paper must label those CIs as record-level and flag the clustering caveat, or recompute before camera-ready.

## 8. Figure and table plan (paper)

| Item | Content | Source |
| --- | --- | --- |
| Fig. 1 | Program map / harness-layer schematic: 3 layers × where each stage intervened; timeline C1→C13-P with verdict icons | This document §2 |
| Fig. 2 | Integration principle: discrete (tight baseline: additive fails, focal wins) vs continuous (loose baseline: additive wins) grouped bars | K1/K7 tables |
| Fig. 3 | Transfer curves: ratio vs K for LoRA/full-FT/scratch (maze-dense + rooms-large), with bound-vs-capacity inset | C9/C9h curves |
| Fig. 4 | C11 gates: headroom ratios by K; learned-vs-MLP ratios by K with collapse cells marked; halting-vs-depth scatter (rho=−0.407) | C11 results |
| Fig. 5 | C13 ladder waterfall: per-rung verdict + delta vs matched control, ending in C13-M per-suite paired deltas | C13 results |
| Tab. 1 | Master evidence matrix: stage, claim, headline number, status tier | §4 |
| Tab. 2 | C13-M confirmation detail: expansions, W/T/L, cost ratios, safety control | K11 |
| Appendix | Full per-suite tables, discrete program, HRM-v2 fidelity case, collapse forensics, preregistration index | synthesis |

## 9. Open items that constrain the paper

1. **Statistics:** C9/C9h/C9b/C10/C11 inference pools repeated worlds; the paper labels these as record-level intervals with an explicit clustering caveat (or they are recomputed world-clustered before submission — preferred if time allows).
2. **Single-seed locality:** most continuous headline results are one training seed on one RTX 5090; phrased as "local validation," replication listed in limitations.
3. **C13-M hardening:** feature-time optimization and cohort/seed replication are named future work; the wall-time number is reported honestly.
4. **C13-P:** cited only as preregistered future work with a verified harness.
5. **Anonymity:** AAAI double-blind; no repo links or acknowledgment of the professor by name; "our earlier stages" phrasing avoided.
