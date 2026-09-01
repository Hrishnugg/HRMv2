# E1 + E2 Structural Experiments: Methodology and Results (final)

**Status:** complete (2026-07-28), including the review-driven repair and
sensitivity rounds. Both experiments ran on Modal under frozen
preregistered designs with dated amendments; every deviation is disclosed
inline. Both studies are integrated into the v14 AAAI-27 submission (main
paper and supplement Section L); this report is the standalone
methodology-and-results record.

**Why these two experiments.** The reviews identified the paper's two
structural weaknesses: (1) the dynamic result lives at one graph scale
(192 nodes) and loses on wall time; (2) the external MovingAI test failed
on a 5-map cohort with no map-level inference. E1 scales the identical
substrate to 2048 nodes with a preregistered wall-time crossover rule. E2
rebuilds the external study at map scale with held-out-map adaptation
transfer. A subsequent deep review of the first results produced three
repair tiers, all executed: analysis upgrades (multiplicity, slope CIs,
effort/path/fidelity tables), a balanced-draw repair of E2's coverage
contrast, and a sensitivity recalibration of E1's degenerate cells.

| | E1 (C8-S) | E2 (C8-X) |
|---|---|---|
| Question | Does the expansion advantage persist with graph size, and convert to wall time? | Does the external zero-shot negative persist at map scale, and does adaptation transfer to unseen maps? |
| Design | `c08/design/2026-07-26-c8-scale-walltime.md` + Amendments 1–2 | `c08/design/2026-07-26-c8-movingai-scale-transfer.md` + Amendments 1–3 |
| Result doc | `c08/results/C8S_SCALE_WALLTIME_RESULT.md` | `c08/results/C8X_SCALE_TRANSFER_RESULT.md` |
| Analysis | `analysis/c8s2_analysis.py`, `c8s2_sens_analysis.py` | `analysis/c8x2_analysis.py` |

Both freeze the paper's exact artifact: the blind field U-Net checkpoint
(SHA-256 `b8378950…dc17fb6f`, asserted at every load), additive
integration, no future-motion input.

---

## E1 (C8-S): scale and wall time, 192 → 2048 nodes

### Methodology

**Cohorts.** One shared fresh cohort per suite (10 dev + 30 eval worlds,
six dynamic suites); a world is accepted only if its roadmap builds and
connects at **all** of N ∈ {192, 512, 1024, 2048} (k=7; per-size roadmap
seed = world seed). The paired unit is the world. Scope: the cohort is
conditioned on connectivity at every tested density, and PRMs are
independently rebuilt per size (not nested prefixes).

**Calibration/tuning.** Budget grid scaled by N/192; anchor-only
calibration to the paper's realized anchor operating points; WA\* re-tuned
per (size, suite) on dev worlds under the frozen rule.

**Arms and timing.** euclid, tuned WA\*, learned_cpu, learned_gpu
(identical code path, device argument, synchronized), SIPP; three repeats
per world, arm order randomized per (world, repeat); first eval world per
shard = warmup (timing-excluded); roadmap construction excluded (common);
L4 containers. A density probe records predicted-vs-true residual
correlation/MAE/bias on five worlds per (size, suite).

**Preregistered readouts.** R1 success persistence (exact McNemar + BH
within each N; pass = paired CI excludes zero in ≥5/6 suites); R2 matched
expansion ratios < 1 everywhere (descriptive); R3 crossover — smallest N
where the paired learned_gpu−WA\* total-time 95% CI < 0 AND success
noninferior within 0.05 AND path suboptimality within +0.02 (CPU
alongside; log-log slope contrast secondary); R4 SIPP reference; R5 GPU
table-build component.

### Results

**R1.** PASS at 192 and 512 (5/6 CIs exclude zero; BH q ≤ 0.0024/0.0094);
FAIL under the frozen bar at 1024 (4/6) and 2048 (3/6). Every failure is a
floor or ceiling cell, not a reversal: dense maze degenerates at all sizes
(the closest-to-0.06 rule picks the grid floor on the harder shared-world
cohort — a **calibration failure mode**: the 0.06/0.16 targets sit below
the 10-world dev resolution of 0.10), spiral@2048 is anchor-unreachable at
every scaled budget, rooms-large saturates (0.97–1.00). Every
non-degenerate cell at every size is individually significant (q < 0.01;
deltas +0.27 to +1.00; maze +0.93 and crossing +0.80 at N=2048).

**Amendment 2 sensitivity (post hoc, descriptive).** At discriminative
operating points (frozen rule: smallest ladder budget with dev anchor in
[0.30, 0.70], ladder = scaled grid ∪ {2×, 4× max}), four of the seven
degenerate cells un-degenerate **in the learned method's favor**, none
reverses: dense@192 +0.57 [+0.40, +0.73] (17/0 discordant, p=1.5e-5),
dense@512 +0.70 [+0.53, +0.87] (21/0, p=9.5e-7), dense@1024 +0.50
[+0.33, +0.67] (15/0, p=6.1e-5) — each tying re-tuned WA\* — and
rooms-large@1024 +0.53 [+0.37, +0.70] (16/0, p=3.1e-5), which also beats
the original-binding-tuned WA\* by +0.23 [+0.10, +0.40]. Three
2048-size cells stay uninformative: the anchor's success-vs-budget curve
is steep enough there that even the doubling ladder jumps the
discriminative range (deltas +0.07/+0.00/+0.10, none excluding zero). The
frozen R1 verdicts are unchanged by this study.

**R2: PASS at every size.** Every defined matched expansion-ratio median
is 0.062–0.285 with map-bootstrap CIs below 1. The expansion advantage
remains large through 2048 nodes in every estimable suite-size cell.

**R3: the preregistered crossover criterion is met on spiral** — GPU at
N=512 (Δt −1.19 s [−1.82, −0.57], success +0.17 *in the learned arm's
favor*, path within rule), with the negative difference remaining at
N=1024 on the same paired cohort (−5.64 s [−6.88, −4.41]); CPU at N=1024.
**Multiplicity companion:** BH across all 24 suite×size GPU time contrasts
gives q = 0.0002 for both spiral cells — not a multiple-comparisons
artifact. No other suite crosses by 2048 (the registered bounded
negative), but the scaling structure is uniform: estimated log-log
wall-time slopes are 0.62–0.91 (learned GPU) and 0.43–0.73 (CPU) versus
1.27–1.66 (WA\*) and 1.02–1.35 (anchor) on the non-degenerate suites, and
the paired world-bootstrap **slope contrasts exclude zero in every case**
(WA\*−GPU: maze +0.36 [+0.30, +0.43], rooms +0.38 [+0.32, +0.45], crossing
+1.02 [+0.92, +1.12], rooms-large +0.76 [+0.67, +0.86]). The maze
learned/WA\* time ratio collapses from ~3.3× (192) to ~1.2× (2048).
Caveats: four size points; cross-size slopes are estimates (paired
fixed-size contrasts are robust to host variation).

**R4.** SIPP solves 0.97–1.00 everywhere (feasibility ceiling, expected);
wall time grows steeply, 1.7–3.7 s (192) → 17.4–39.0 s (2048), exceeding
learned_gpu in every suite at 2048 — different success semantics, never
merged. **R5.** GPU table build 1.40 → 7.61 s (192 → 2048), ~5.4× for
10.7× nodes. **Probe.** Prediction–truth correlation is size-stable (e.g.
spiral 0.74→0.79, maze 0.64→0.68) while under-prediction bias grows: the
ranking signal survives a 10× node extrapolation; magnitude calibration
drifts.

**Integrity.** CPU/GPU agree on success in all 720 paired evaluations;
100 differ by ≤10 expansions from device float tie-breaking. Success and
expansions are repeat-identical. All 48 eval/probe shards plus the 20
sensitivity shards completed.

---

## E2 (C8-X): MovingAI at map scale + adaptation transfer to unseen maps

### Methodology

**Maps.** 40 selected by frozen rule (20 street + 20 dao); 37 usable
(dao's brc100d/brc300d/brc502d fail conversion); pool = first 8 usable per
category (labels + calibration only); confirmatory inference on the
remaining **21 held-out maps** (12 street + 9 dao), 12 frozen eval
instances each. One frozen source-map split; no split resampling.
Conversion = the submitted study's pipeline unchanged (majority-coarsen to
64×64, rectangle-decompose, trained maze-family patrollers).

**Calibration.** Every arm runs once at BIGB = 14,000; success at any
budget by expansion thresholding. Binding budget (canonical
closest-target rule) and WA\* weight selected **only from pool-map dev
rows**: realized binding 400 and w=3 for both categories (the submitted
5-map study realized 600/900, w 1.5/2 — qualitative comparison only).

**Adaptation.** K=8 labeled dev instances per cell from M ∈ {1, 2, 4, 8}
pool maps; methods conv-LoRA r8 / full FT / scratch; 2 optimizer seeds;
exact-N thinning to N_target = 65,536 labels (computed
outcome-independently, manifest-recorded). Draws (post-Amendment 3,
balanced): M=1 all eight one-map sets, M=2 the four disjoint pairs, M=4
the two disjoint halves (+ a legacy subsample replicate, excluded from
balanced summaries), M=8 one draw — **192 cells**. Amendment-3 cells use
stable SHA-256-digest thinning seeds.

**Primary family** (8 tests, source-map-clustered bootstrap, two-sided
add-one bootstrap p, BH): B1 adapted(M=8) − zero-shot > 0 and B2
adapted(M=8) − adapted(M=1) > 0, per category × {LoRA, full FT}. Scratch
is descriptive (added by amendment without amending the family). The
original three-draw B2 was found to compare against an unbalanced M=1
sample; the **repaired family** (B1 + B2 over all eight M=1 draws) is the
corrected primary — a post-outcome repair motivated by the structural
imbalance, not by effect direction, and labeled as not inheriting
preregistered status.

### Results

**Repaired primary family (balanced draws):**

| Test | Δ success | 95% CI | q (BH/8) |
|---|---:|---|---:|
| B1 street LoRA | +0.076 | [+0.017, +0.132] | **0.035** |
| B1 street full | +0.125 | [+0.059, +0.191] | **0.0016** |
| B1 dao LoRA / full | −0.019 / −0.000 | | 0.857 / 0.989 |
| B2-bal street LoRA | +0.023 | [−0.027, +0.076] | 0.738 |
| B2-bal street full | +0.037 | [+0.011, +0.063] | **0.026** |
| B2-bal dao LoRA / full | +0.005 / −0.008 | | 0.857 |

**The street full-FT coverage effect survives the balanced repair** at the
same magnitude as the frozen version (+0.037, q=.026): with all eight
one-map sets, adapting on eight maps beats adapting on one for full
fine-tuning on unseen street maps. Street LoRA's coverage contrast is
null; dao is null throughout — the registered bounded negative. B1 is the
strongest adaptation result: on street, both methods transfer to the 12
unseen maps (q=.0016 full, .035 LoRA); on dao neither does.

**A1 — the submitted external negative does not generalize to map scale**
(descriptive, map-clustered CIs): zero-shot beats the anchor on held-out
maps in both categories — street +0.097 [+0.042, +0.160] (0.715 vs
0.618), dao +0.204 [+0.093, +0.324] (0.694 vs 0.491) — while clearly
losing to tuned WA\* (street −0.222 [−0.285, −0.174]; dao −0.148
[−0.278, −0.019]). The submitted 5-map verdict stands on its own frozen
cohort; the durable external boundary is tuned WA\*.

**Effort, path, fidelity (descriptive).** On held-out maps the learned
arms expand fewer nodes than the anchor (matched median ratios: street
zero-shot 0.449 [0.379, 0.618], adapted-full 0.441; dao adapted-full
0.397 [0.281, 0.463]) but 1.7–2.5× more than tuned WA\* — WA\* wins
external effort as well as success. Path suboptimality: learned 1.03–1.11
vs the anchor's optimal 1.000, comparable to WA\* (1.02–1.10 vs
1.04–1.05). Conversion fidelity explains the category split's context:
street pool and held-out maps are tightly matched (mean free-space 0.750
vs 0.755) while dao spans 0.128–0.964 with fragmented geometry (pool mean
components 3.5 vs held-out 1.8).

**B3 — the dao null is localized:** dao pool maps (which supplied labels)
gain ~+0.10 over zero-shot while held-out dao maps gain nothing — the
submitted within-group dao "full rescue" was **map-specific adaptation**,
not transferable structure. Street shows no pool/held-out gap.

**Scratch (descriptive).** Street scratch is the strongest arm (0.868 vs
full 0.840, LoRA 0.792; balanced dose .849/.838/.859/.868 across M); dao
scratch sits at zero-shot parity. At ~65k exact labels, external street
maps are label-dense enough that source pretraining shows no advantage
over from-scratch training — consistent with the program's
supervision-density story. Dose curves are endpoint contrasts, not
monotone dose responses.

**Integrity (all disclosed).** (1) Dataset-name assert crash repaired
before any adaptation outcome existed. (2) Pre-amendment thinning RNG was
salted-hash-seeded; Amendment-3 cells use stable digests, and a SHA-256
manifest of every cell dataset and checkpoint archives the exact
materialized subsets for all cells. (3) Heavy Modal preemption (48 kills
in ~3 h) collapsed whole-cell evaluation throughput; evaluation was
resharded per (cell, held-out map) with a merge step — repair cells
evaluate held-out maps only, and no readout consumes their pool rows.
(4) Five cells killed mid-write left partial CSVs that resume logic
counted as complete; a per-cell completeness scan caught them and the
cells were re-evaluated. (5) All 192 cells verified complete.

---

## What the pair changes for the paper

1. **Scale:** "fewer expansions do not translate into lower wall time" is
   now a statement about N=192 only. The expansion and success advantages
   persist wherever the operating point is discriminative (including the
   re-calibrated dense cells), estimated learned wall-time slopes are
   lower than every classical arm's (contrast CIs exclude zero), and the
   preregistered crossover criterion is met on spiral by N=512
   (multiplicity-robust, q=0.0002).
2. **External validity:** the 5-map anchor-relative failure was a
   small-cohort artifact — at 21 held-out maps, zero-shot beats the anchor
   in both categories. The durable boundary is tuned WA\* (success and
   effort). Adaptation transfers across street maps (both methods), the
   coverage effect survives the balanced repair for full FT, and the dao
   rescue is honestly re-scoped as map-specific.
3. **Honest negatives preserved:** five suites without wall-time crossover
   by 2048; three uninformative 2048 sensitivity cells; dao transfer null;
   street LoRA coverage null; scratch strongest on external street maps;
   WA\* external dominance.
