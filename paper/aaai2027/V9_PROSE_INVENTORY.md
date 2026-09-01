# v9 prose inventory — long/complex sentences and paragraphs

Generated from `submission_v9.tex` / `supplementary_v9.tex` (LaTeX-stripped; flags: >=40 words, em-dash pairs in long sentences, >=3 semicolons, >=3 parentheticals).

**Main paper: 52 flagged** — Tier A (split required): 20; Tier B (simplify): 17; Tier C (dense-but-conventional, recommend keep): 14; Abstract (professor-gated): 1.
**Supplement: 77 flagged** — top offenders listed at the end; propose handling after the main paper.

## Main paper: Tier A — split required

### M-04  (Introduction | para | 57w, 0 dash, 3 semi, 0 paren)
> Changing the input--target--output formulation rescues a collapsed recurrent model; changing only the planner integration turns an inert discrete residual into useful guidance; and sharing one search state turns an exact oracle from a six-map loss into a six-map win; a negative planning result therefore does not isolate the model unless target and search interface are held fixed

**Plan:** Contribution 2 bullet, 57w, three semicolon clauses + trailing inference. Split: sentence 1 = the three controls (keep semicolons); sentence 2 = 'A negative planning result therefore...' (already separate clause; promote to sentence).

### M-05  (Introduction | para | 86w, 0 dash, 2 semi, 0 paren)
> Low-rank adaptation shows the least frequent and smallest degradation across the tested cells, though one redraw produces a significant LoRA degradation. itemize Additional controls define the scope: under the tested weighted-A* grid the learned heuristic keeps a success advantage only on spiral and lower matched expansions on four suites, at higher path cost on two; unbudgeted safe-interval planning solves every feasible-labeled instance optimally under the implemented collision model; and the zero-shot success advantage does not reproduce in a small MovingAI-derived boundary test, analyzed post hoc below

**Plan:** Tokenizer merged contribution-3 tail with the post-list controls paragraph. Real targets: (a) contribution 3's factorial sentence -- split at ': distributing the same labels...'; (b) the 'Additional controls define the scope:' paragraph -- currently one 3-clause mega-sentence; re-expand into the 3 short sentences the v8 review proposed (WA*/SIPP/MovingAI one sentence each).

### M-08  (Problem Formulation and Learned-Heuristic Interfaces | para | 69w, 1 dash, 3 semi, 2 paren)
> With [m] the exact cost-to-go (graph Dijkstra on the static roadmap; backward space--time Dijkstra over [m] in the dynamic setting), the regression target is the normalized anchor residual [m], where [m] is the normalization scale ([m] statically; [m], the map-crossing time in steps, dynamically---the same value for every suite by construction) and the cap is [m] in normalized units, applied after normalization; unreachable states are masked from the loss

**Plan:** Residual-target definition 69w. Split into three: (1) define h* sources; (2) define the target formula; (3) 'T is the normalization scale... the cap is c=4.0, applied after normalization; unreachable states are masked.'

### M-11  (Experimental Setup | para | 59w, 2 dash, 1 semi, 1 paren)
> Dynamic training draws worlds only from the dynamic maze, rooms, and spiral families; dynamic dense maze, crossing, and large rooms are held out---crossing a distinct open-arena topology, dense maze narrower corridors and adjusted patrollers, large rooms doubled scale with a faster agent---so three of six test suites share a generator with training (disjoint maps) and three are suite-level shifts

**Plan:** Dynamic held-out sentence 59w with double dash + trailing consequence. Split at '---so three of six...': new sentence 'Three of six test suites therefore share a generator with training (disjoint maps); three are suite-level shifts.'

### M-18  (Dynamic Zero-Shot Transfer | para | 61w, 0 dash, 2 semi, 0 paren)
> The original 20-map-per-suite cohort served as development data: based on it, we selected the field U-Net blind twin and froze its checkpoint, additive integration, and per-suite budgets; the 50-map-per-suite confirmation cohort was then generated from a pre-specified seed range, map-disjoint by fingerprint check, with no further choices; selection still happened, on development data, and the protocol protects only the confirmation cohort

**Plan:** Chronology 61w. Split at '; selection still happened...': 'Selection still happened, on development data. The protocol protects only the confirmation cohort.'

### M-19  (Dynamic Zero-Shot Transfer | para | 63w, 1 dash, 1 semi, 2 paren)
> On the confirmation cohort, the fixed blind U-Net raises success on all six suites: deltas [m] to [m], with every paired map-level 95\% CI excluding zero (exact McNemar [m] on all six suites, every discordant map favoring the learned heuristic), and matched expansion-ratio medians of 0.047--0.273---a 73--95\% reduction in search effort on jointly solved maps (Figure [r]; jointly solved counts 3--41 per suite)

**Plan:** Confirmation result 63w. Split after the McNemar parenthetical: sentence 2 starts 'Matched expansion-ratio medians are 0.047--0.273---a 73--95% reduction...'

### M-20  (Dynamic Zero-Shot Transfer | para | 61w, 2 dash, 2 semi, 2 paren)
> Two additional full training pipelines with independent seeds---run, we disclose, at a larger world-collection budget (139/149 usable worlds versus 53; appendix), varying seed and data scale together---reproduce the success pattern on the same held-out cohort: across the three pipelines, 17 of 18 pipeline--suite paired CIs exclude zero (deltas [m] to [m]; the exception is seed 2001's near-ceiling large-rooms cell, anchor 0.82)

**Plan:** Retraining 61w. Split disclosure from result: sentence 1 ends '...varying seed and data scale together.' Sentence 2: 'On the same held-out cohort they reproduce the success pattern: 17 of 18...'

### M-26  (Dynamic Zero-Shot Transfer | para | 77w, 2 dash, 1 semi, 4 paren)
> Because three suites selected the grid maximum [m], we also ran a disclosed post-hoc extension ([m]; design frozen before execution): five selections are unchanged---spiral's development success stays 0.80, so its learned advantage is grid-robust---and dense maze's re-selection to [m] (development 19/20 versus 18/20) changes the dense-maze point estimates but no inferential conclusion: success 0.74 versus 0.70 against both the frozen rows and the learned arm (exact [m]), effort at parity (median 1.40 [m]), path quality within [m]

**Plan:** Extended-grid 77w. Three sentences: (1) why + design; (2) five unchanged incl. spiral robustness; (3) dense-maze re-selection + no inferential change (stats).

### M-29  (Dynamic Zero-Shot Transfer | para | 56w, 0 dash, 1 semi, 3 paren)
> To test the claim beyond our own generators, six MovingAI maps [r] (three street, three dungeon) were converted to the canonical substrate (64[m]64 occupancy, rectangle-decomposed obstacles), given the trained maze-family patroller dynamics, and evaluated with the same frozen heuristic under the frozen calibration/tuning protocol; one dungeon map yields no usable instances, so five maps contribute (appendix)

**Plan:** MovingAI conversion 56w. Split: (1) six maps converted (conversion detail); (2) patrollers + frozen protocol; (3) brc202d zero instances.

### M-30  (Dynamic Zero-Shot Transfer | para | 70w, 0 dash, 2 semi, 4 paren)
> The success advantage does not transfer: the learned heuristic never significantly beats the anchor (deltas [m] on both groups) and loses to tuned weighted A* on dungeon instances ([m], 0/7 discordant, exact [m]; instance-level, unadjusted, nested in two maps) while expanding more nodes there (matched ratio 2.20 [m]) and paying path-quality premiums with intervals excluding zero on both groups; only the street-map matched-effort advantage over the anchor survives (0.48 [m])

**Plan:** MovingAI results 70w. Three sentences: anchor result; WA* dungeon loss (with labels); the street effort survivor.

### M-32  (Dynamic Zero-Shot Transfer | para | 69w, 1 dash, 2 semi, 3 paren)
> On the held-out cohort the contrast is suite-dependent in both directions and unstable across pipelines: of 18 contrasts the paired interval excludes zero for the aware twin in two, the blind twin in two, and neither in fourteen---and those cells fall in different suites for different pipelines (large rooms flips sign); the canonical pipeline also trained on fewer worlds, but the two collection-matched retrains still disagree (Figure [r](c); appendix)

**Plan:** Twin contrast 69w. Split: (1) the 2/2/14 counts; (2) cells fall in different suites (large rooms flips sign); (3) the collection-matched-retrain point.

### M-33  (Effects of Supervision Target and Planner Integration | para | 61w, 0 dash, 1 semi, 0 paren)
> On the six calibrated static suites, the transferred field HRM solves 0.75--1.00 of test maps against anchor success of 0.25--0.58, with BH-adjusted [m] values from below 0.001 to 0.025 on four suites and matched expansion ratios of 0.521--0.850; three of six suites are held out as above, and pooled source-family models improve success on every held-out family before any target labels

**Plan:** Static transfer 61w. Split at '; three of six suites are held out as above, and pooled...' -> two sentences.

### M-36  (Effects of Supervision Target and Planner Integration | para | 53w, 1 dash, 0 semi, 3 paren)
> An oracle control removes model-estimation error: the same exact value ranker loses all six development-map comparisons when paired with a fresh certifier search that discards the incumbent's work (141.2 vs.\ 129.7 expansions) and wins all six when sharing one search queue and [m]-state (122.8)---identical values, only search-state reuse changed (a development mechanism test)

**Plan:** Oracle control 53w+3 parens. Split: (1) loses all six under fresh certifier (141.2 vs 129.7); (2) wins all six sharing queue/g-state (122.8); (3) identical values, only reuse changed.

### M-38  (Few-Shot Adaptation | para | 59w, 0 dash, 1 semi, 3 paren)
> At [m] the map-paired full-FT[m]LoRA success difference is negative in all six target/backbone cells, with the 95\% interval excluding zero in four (worst [m] [m] on large rooms/HRM); LoRA retains the zero-shot gain (dense maze ratio 0.686, success [m]) while full fine-tuning reaches ratio 1.293 on large rooms and scratch is harmful on dense maze ([m], interval excluding zero)

**Plan:** K=1 contrast 59w. Split after the worst-cell parenthetical; LoRA-retains / full-FT-reaches / scratch-harmful as its own sentence.

### M-41  (Few-Shot Adaptation | para | 58w, 0 dash, 1 semi, 4 paren)
> A preregistered factorial therefore holds the number of supervised states [m] fixed (256 to 65,536 by factors of 4) and varies only whether those states are drawn from the fewest worlds that can supply them (concentrated) or from eight times as many (distributed), in both domains, under matched optimizer steps (Figure [r]; protocol and tables in the appendix)

**Plan:** Factorial design 58w. Split: (1) the factors sentence; (2) 'Sources are frozen... the target is maze-dense in both domains.' (already separate) -- so split factor sentence at ', in both domains, under matched optimizer steps' -> new sentence 'Both domains run under matched optimizer steps (Figure 2; protocol and tables in the appendix).'

### M-42  (Few-Shot Adaptation | para | 101w, 2 dash, 3 semi, 4 paren)
> Concentrated supervision causes a held-out success drop for full fine-tuning and scratch in both domains---[m] in every seed at dynamic concentrated [m], still negative in every seed at [m] from a single world (significantly in one of three)---while distributing the same states across more worlds prevents the drop ([m] to [m]; Figure [r]); direct distributed[m]concentrated contrasts for all 30 cells (appendix; BH-corrected sign-flip tests, added post hoc during review response): [m] for full fine-tuning and scratch in all 10 cells where concentration forces at most two distinct worlds and none of the 10 where it forces four or more (threshold descriptive)

**Plan:** THE WORST: 101w. Four sentences: (1) concentrated drop for FT+scratch w/ dynamic numbers; (2) still negative at N=16,384 single-world (one of three seeds excludes zero); (3) distributing the same states prevents it (+0.15..+0.20); (4) the 30-cell BH dichotomy sentence.

### M-43  (Few-Shot Adaptation | para | 96w, 2 dash, 2 semi, 3 paren)
> No significant LoRA success degradation appears in the original 60 cells (worst [m] [m]); absent a noninferiority margin this is a failure to observe degradation, not equivalence, and targeted independent world-draw replications at the decisive cells make the caveat concrete: the replicated coverage contrasts are positive in all 18 tested cases---excluding zero for scratch in all six and for full fine-tuning in five of six (the second draw's static contrast touches zero)---but one redraw LoRA cell does degrade, with its interval excluding zero (static concentrated [m]: [m] [m], still the least-degraded method in that cell; appendix)

**Plan:** Second worst: 96w. Three sentences: (1) no significant LoRA degradation in 60 cells + noninferiority caveat; (2) targeted redraws: contrasts positive in all 18, scratch 6/6, FT 5/6 (draw-2 static touches); (3) the one degraded LoRA redraw cell.

### M-44  (Few-Shot Adaptation | para | 63w, 1 dash, 1 semi, 2 paren)
> On the static matched expansion ratios (the domain with adequate jointly solved counts), full fine-tuning is at or below LoRA at every tested [m] and the full-fine-tuning[m] interaction is null ([m] [m]); dynamic effort cells contain one jointly solved map and support no inference, and because ratios condition on solved maps, degraded-success cells can look spuriously efficient---so success is the factorial's primary endpoint

**Plan:** Static-ratio + interaction 63w. Split at '; dynamic effort cells contain one jointly solved map...' and make 'so success is the factorial's primary endpoint' the closing sentence.

### M-47  (Related Work | para | 63w, 2 dash, 1 semi, 3 paren)
> Under a frozen comparison on a disjoint 144-map cohort---whose path-quality gate is the comparator-relative criterion adopted after the original absolute ceiling failed in development (appendix)---this local MLP reduces pooled expansions against the complete-map field HRM by 12.96 (95\% CI [m]; 109/3/32 W/T/L) at better empirical path quality (mean/maximum cost ratios 1.024/1.162 versus 1.031/1.335), and a separate [m] bounded control passes all 144 certificates

**Plan:** C13 confirmation 63w. Split: (1) frozen comparison + gate chronology; (2) the result numbers; (3) the bounded-control sentence (already separate).

### M-51  (Limitations | para | 65w, 0 dash, 5 semi, 2 paren)
> The factorial covers one target family per domain, three adaptation seeds, and one sampled world set per cell (targeted redraws in the appendix); coverage counts worlds, not measured geometric diversity; matching covers labels and steps, not world-generation cost; per-cell readouts are marginal (adjusted inference covers the 30 coverage contrasts; the two-world threshold is descriptive); the extended grid and probe/rescue are post-hoc sensitivity and diagnostic evidence

**Plan:** Factorial limitations 65w/5 semis. Split into two sentences at '; matching covers labels and steps...'

## Main paper: Tier B — simplify / shorten

### M-03  (Introduction | para | 43w, 2 dash, 1 semi, 1 paren)
> A single U-Net heuristic---development-selected, frozen before the confirmation cohort was generated, and given no future-motion input---improves success on all six dynamic suites of a 50-map-per-suite confirmation cohort ([m] to [m]; every paired map-level CI excludes zero) and cuts matched search effort by 73--95\%

**Plan:** Contribution 1 core. Move the appositive out: 'A single U-Net heuristic improves success on all six dynamic suites... The model was development-selected, frozen before the confirmation cohort was generated, and given no future-motion input.'

### M-07  (Problem Formulation and Learned-Heuristic Interfaces | para | 54w, 2 dash, 1 semi, 1 paren)
> Each roadmap has 192 nodes in total---the start and goal at indices 0 and 1 plus 190 sampled free-space nodes---connected by [m] nearest-neighbor candidate edges per node (doubled for the start and goal, then symmetrized into an undirected graph); edges are validated by sampled collision checks at resolution [m], and unconnected worlds are rejected

**Plan:** Roadmap spec 54w. Split at '; edges are validated by sampled collision checks...' into its own sentence.

### M-09  (Problem Formulation and Learned-Heuristic Interfaces | para | 49w, 0 dash, 1 semi, 2 paren)
> Few-shot adaptation draws [m] labeled target maps and compares low-rank adaptation []hu2021lora (rank 8 for the HRM, rank 24 for the wider ON-LSTM), full fine-tuning, and training from scratch (LoRA and full fine-tuning start from the same trained checkpoint; scratch uses the same architecture and schedule from random initialization)

**Plan:** Transfer protocol 49w. Promote the parenthetical to a sentence: 'LoRA and full fine-tuning start from the same trained checkpoint; scratch uses the same architecture and schedule from random initialization.'

### M-15  (Experimental Setup | para | 42w, 2 dash, 1 semi, 1 paren)
> Each suite is evaluated at a per-suite expansion budget selected by anchor-only calibration: the grid budget whose anchor success is closest to pre-specified targets (evenly spaced in [m]; ties smaller) is fixed---the smallest selected budget is binding---before any learned method is compared

**Plan:** Calibration sentence 42w. Split: '...is fixed before any learned method is compared. The smallest selected budget is binding.'

### M-22  (Dynamic Zero-Shot Transfer | para | 52w, 0 dash, 1 semi, 3 paren)
> Because the anchor is deliberately simple, we also compare against weighted A* ([m]), success-tuned per suite over [m] on the development cohort (rule frozen in advance: highest success, ties smaller), evaluated once on the confirmation cohort; the control was specified after the learned results were known, tuning touching only development data (appendix)

**Plan:** WA* setup 52w. Split at '; the control was specified after the learned results were known...' into its own sentence.

### M-23  (Dynamic Zero-Shot Transfer | para | 44w, 2 dash, 1 semi, 2 paren)
> Success: no significant difference is detected on five suites---at most one discordant map on four, and crossing's four-map weighted advantage is not significant under the exact McNemar policy ([m])---while the learned heuristic holds the one BH-significant advantage, on spiral ([m] [m]; 15/0 discordant, [m])

**Plan:** Success clause 44w. Pull the crossing exception into its own sentence: 'Crossing's four-map weighted advantage is not significant under the exact McNemar policy (q=0.375).'

### M-25  (Dynamic Zero-Shot Transfer | para | 44w, 0 dash, 1 semi, 2 paren)
> Path quality: on jointly solved maps, the additive learned heuristic pays a premium over tuned inflation whose paired 95\% interval excludes zero on crossing (per-map difference [m] [m]) and large rooms ([m] [m]); the other four suites stay within [m] with intervals covering zero

**Plan:** Path-quality 44w. Split the two-suite listing from the four-suite null.

### M-27  (Dynamic Zero-Shot Transfer | para | 39w, 2 dash, 1 semi, 1 paren)
> Under the implemented sampled-collision model, unbudgeted SIPP solves every instance labeled feasible at the backward-Dijkstra optimal arrival---per-suite success 0.94--1.00, exactly the substrate's feasibility ceiling---with small expansion counts in its own interval-state unit (medians 80--290; never merged with [m] expansions)

**Plan:** SIPP result 39w+dashes. Split the feasibility-ceiling appositive into its own sentence: 'Its per-suite success, 0.94--1.00, equals the substrate's feasibility ceiling exactly.'

### M-31  (Dynamic Zero-Shot Transfer | para | 37w, 1 dash, 0 semi, 0 paren)
> Exploratory few-shot adaptation on 2--8 development instances from the same source-map groups restores dungeon success to 0.90--1.00 and street to classical parity, while single-instance adaptation degrades street---qualitatively consistent with the factorial's narrow-coverage degradation, not a matched replication

**Plan:** Rescue sentence 37w+dash. Split the qualitative-consistency clause into its own sentence.

### M-34  (Effects of Supervision Target and Planner Integration | para | 52w, 0 dash, 0 semi, 2 paren)
> On the calibrated hard static maps, a pooled ON-LSTM scalar residual model raises hard-maze success from 0.525 to 1.000 over the anchor ([m]), while the parameter-matched scalar HRM collapses to a constant prediction at the residual cap and merely matches the baseline (lower learning rates and a soft cap reproduce the collapse)

**Plan:** C5 collapse 52w. Split at 'while the parameter-matched scalar HRM collapses...'

### M-35  (Effects of Supervision Target and Planner Integration | para | 47w, 2 dash, 0 semi, 1 paren)
> On discrete grids the learned signal is demonstrably informative---the pooled residual correlates 0.987--0.994 with the true residual across map sizes---but its magnitude is scale-flat (predicted/true mean 0.73 to 0.19 as maps grow), and every additive configuration in the 22-suite run is at or below its matched baseline

**Plan:** Discrete signal 47w. Split at 'but its magnitude is scale-flat...'

### M-39  (Few-Shot Adaptation | para | 40w, 0 dash, 0 semi, 2 paren)
> At [m] the contrast reverses on effort: the paired expansion-ratio difference favors full fine-tuning in all six cells, with the map-bootstrap interval excluding zero in five (up to [m] [m]), at near-parity success (one cell's success interval favors full fine-tuning)

**Plan:** K=16 contrast 40w. Acceptable, or split the near-parity parenthetical out.

### M-40  (Few-Shot Adaptation | para | 42w, 0 dash, 1 semi, 2 paren)
> A 27-cell matched-compute ablation finds bounded and unbounded LoRA nearly identical (median map-level ratio delta 0.000, maximum 0.037), so the output clamp is not the primary cause of the LoRA plateau; LoRA is not universally protective either (bugtrap ratio 1.153 by [m])

**Plan:** 27-cell ablation 42w. Split at '; LoRA is not universally protective either...'

### M-45  (Scope and Boundary Results | para | 41w, 0 dash, 2 semi, 2 paren)
> The pooled HRM base improves mean grid success 0.591[m]0.612 with lower mean expansions (descriptive); adapters do not improve broadly; a 3--8-seed rank-only pilot on selected maps finds 6--15\% expansion reductions at [m] with equal or higher observed success (no full-suite confirmation)

**Plan:** Discrete scope list 41w. Semicolon list; acceptable, or split the pilot clause.

### M-46  (Scope and Boundary Results | para | 50w, 0 dash, 2 semi, 2 paren)
> In compositional missions with verified oracle headroom (three-seed grids; appendix), no structured model beats the MLP at two depths with a non-decreasing gap, and learned halting moves opposite to the depth hypothesis (map-level [m], permutation [m]); five of 33 recurrent cells suffer constant-prediction collapse and are excluded from capacity conclusions

**Plan:** Missions 50w. Split at '; five of 33 recurrent cells...'

### M-49  (Limitations | para | 44w, 0 dash, 2 semi, 1 paren)
> A suite-indexing defect made the original SIPP, extended-grid, wall-time, and probe-reference runs evaluate sibling cohorts; all were re-executed on the frozen cohort under unchanged rules, identity verified by a serialized-world hash manifest (erratum in the appendix; the weighted-A* baseline always used the correct cohort)

**Plan:** Erratum 44w. Split at '; all were re-executed...'

### M-52  (Conclusion | para | 43w, 0 dash, 0 semi, 0 paren)
> A frozen blind U-Net transferred zero-shot on the procedural substrate: success improved on all six confirmation suites, matched effort fell 73--95\%, and the advantage over success-tuned weighted A* is decisive on spiral, effort-favorable on four suites, and robust to the tested grid extension

**Plan:** Conclusion sentence 1, 43w, 3-part list. Acceptable rhythm; optional split after '73--95%'.

## Main paper: Abstract (professor-gated)

### M-01  (preamble | para | 44w, 2 dash, 0 semi, 0 paren)
> We show that, within budgeted time-expanded A* and with the right training objective and planner integration, a frozen heuristic transfers zero-shot across six procedural continuous dynamic suites---even given no information about future obstacle motion---improving success over a geometric anchor while substantially reducing search effort

**Plan:** Abstract's core transfer sentence (44w, two dash pairs). Split option: end sentence after 'six procedural continuous dynamic suites'; start new sentence 'Even with no information about future obstacle motion, it improves success over a geometric anchor while substantially reducing search effort.' PROFESSOR-GATED.

## Main paper: Tier C — dense but conventional (recommend keep)

### M-02  (Introduction | para | 35w, 2 dash, 0 semi, 0 paren)
**Plan:** Intro three-factor sentence; the dash pair is doing real work. Keep.

### M-06  (Introduction | caption | 22w, 0 dash, 1 semi, 3 paren)
**Plan:** Fig 1 caption panel-(b) clause; caption idiom. Keep.

### M-10  (Experimental Setup | para | 44w, 1 dash, 3 semi, 3 paren)
**Plan:** Family list; enumerative by nature. Keep.

### M-12  (Experimental Setup | para | 37w, 0 dash, 1 semi, 3 paren)
**Plan:** Scalar model list. Keep.

### M-13  (Experimental Setup | para | 40w, 0 dash, 2 semi, 4 paren)
**Plan:** Field model list. Keep.

### M-14  (Experimental Setup | para | 48w, 0 dash, 2 semi, 1 paren)
**Plan:** Optimizer recipe; spec idiom. Keep.

### M-16  (Dynamic Zero-Shot Transfer | para | 36w, 1 dash, 0 semi, 2 paren)
**Plan:** Dynamic setting opener. Keep.

### M-17  (Dynamic Zero-Shot Transfer | para | 33w, 0 dash, 1 semi, 3 paren)
**Plan:** World-count sentence; numbers list. Keep.

### M-21  (Dynamic Zero-Shot Transfer | para | 38w, 0 dash, 1 semi, 3 paren)
**Plan:** Effort-stability sentence. Keep.

### M-24  (Dynamic Zero-Shot Transfer | para | 35w, 1 dash, 1 semi, 2 paren)
**Plan:** Effort clause; parenthetical policy note is fine. Keep.

### M-28  (Dynamic Zero-Shot Transfer | para | 27w, 0 dash, 2 semi, 3 paren)
**Plan:** Wall-time list. Keep.

### M-37  (Few-Shot Adaptation | caption | 29w, 0 dash, 3 semi, 2 paren)
**Plan:** Fig 2 caption legend clause; caption idiom. Keep.

### M-48  (Limitations | para | 29w, 1 dash, 3 semi, 1 paren)
**Plan:** Limitations list idiom. Keep.

### M-50  (Limitations | para | 30w, 0 dash, 3 semi, 2 paren)
**Plan:** Limitations list idiom. Keep.

## Supplement: top offenders (by severity)

### S-77  (K. Reproducibility and Artifact Inventory | para | 306w, 0 dash, 5 semi, 11 paren)
> The anonymized artifact archive submitted with this paper contains: (i) the analysis scripts that regenerate the quoted map-level statistics from raw rows (the fixed-provider dynamic reanalysis, the map-clustered reanalysis, the weighted-A* analysis with its exact McNemar, ratio-CI, and matched path-quality computations, the budget-curve derivation, the reachable-label recount, the factorial analy...

### S-33  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 132w, 1 dash, 6 semi, 6 paren)
> The rescue then adapts the frozen model on the first [m] recorded development instances per group (conv-LoRA r8 and full fine-tuning, exact C14 recipe, two adaptation seeds; evaluation instances untouched by training) and evaluates once on the frozen evaluation instances: on dao, [m] already reaches 0.90--0.96 success (the first [m] covering both contributing base maps) and [m] full fine-tuning so...

### S-42  (D. Adaptation: [m]-Indexed Transfer, Composition, and the Matched-Label Factorial | para | 67w, 1 dash, 4 semi, 11 paren)
> Representative C9 HRM curves (ratio at success): maze-dense LoRA [m] (1.00) at [m] to [m] (0.97) at [m]; maze-dense full FT [m] (0.76) to [m] (0.99) at [m] and [m] at [m]; scratch [m] (0.37) to [m] (0.80); rooms-large full FT [m] (0.37) at [m] to [m] (0.92) at [m]; bugtrap LoRA degrades from [m]0.696 (0.83) to [m] (0.48) at [m]---LoRA is usually but not universally base-preserving

### S-52  (D. Adaptation: [m]-Indexed Transfer, Composition, and the Matched-Label Factorial | para | 79w, 2 dash, 2 semi, 2 paren)
> The coverage contrasts replicate in every draw: distributed[m]concentrated is positive in all 18 replicated contrasts, with CIs excluding zero for scratch in all six (up to [m]) and for full fine-tuning in five of six---draw 2 dynamic [m] [m] and [m] [m]; draw 3 dynamic [m] [m] and [m] [m], static [m] [m]; the one exception is draw 2's static contrast, [m] [m], whose lower bound touches zero---and...

### S-32  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 56w, 2 dash, 4 semi, 3 paren)
> Median Pearson correlations degrade gradedly with distribution shift---trained procedural suites 0.95--0.98; the held-out parameter shift 0.94; the held-out topology and scale shifts 0.73--0.76; MovingAI street 0.385; dao 0.240---while MAE roughly doubles (0.68 versus 0.14--0.35) and the bias flips sign at the shift boundary: in-family suites under-predict ([m] to [m]), every shifted set over-pred...

### S-12  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 51w, 3 dash, 2 semi, 5 paren)
> Preregistered readouts: R1 (success CI excludes zero in [m]5/6 suites) 5/6 and 6/6---pass; R2 (deltas within [m] of canonical in [m]5 suites) 5/6 and 6/6---pass; R3 (no suite significantly aware-over-blind) fails as stated for seed 2001 (large rooms [m] [m])---the same suite that is significantly blind-better under seed 1234 ([m] [m])

### S-14  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 86w, 1 dash, 2 semi, 2 paren)
> The headline result is not an artifact of the calibrated operating point: Euclidean search needs 1.7--2.8[m] the binding budget to reach the fixed blind provider's binding-budget success (crossing 2.49[m], maze 2.29[m], dense maze 1.95[m], rooms 1.73[m], large rooms 1.94[m], spiral 2.80[m]), and by [m] the binding budget every provider converges to the same feasibility ceiling (0.94--1.00; on dens...

### S-29  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 65w, 2 dash, 3 semi, 2 paren)
> Six MovingAI maps [r]---street: Berlin\_0\_256, Boston\_0\_256, Paris\_0\_256; dungeon (dao): den312d, den520d, brc202d---are majority-coarsened to the canonical [m] occupancy over the unit square and rectangle-decomposed into standard obstacles; the trained maze-family patroller configuration is applied on top; start/goal sampling, usability rules, the 192-node roadmap, budget calibration (anchor...

### S-54  (D. Adaptation: [m]-Indexed Transfer, Composition, and the Matched-Label Factorial | para | 68w, 1 dash, 3 semi, 2 paren)
> The harm's magnitude depends on the drawn worlds: draw 3's static concentrated cells are significantly harmful for all three methods (full FT [m]; scratch [m]; LoRA [m] [m], failing R4---the first significant LoRA degradation in the program, still the least-degraded method in its cell), while draw 2's static concentrated worlds happen to be benign (full FT [m], n.s., failing R3; LoRA never degrade...

### S-71  (J. Statistics and Map-Clustered Reanalysis | para | 59w, 0 dash, 5 semi, 1 paren)
> Units are test maps; adaptation/model seeds are averaged within maps before inference; success deltas use 10,000-resample map bootstraps; expansion ratios are medians of per-map ratios over matched maps at the binding budget in additive mode; the C11 halting test aggregates model seeds within maps and permutes mission length within config (maps are independent across [m] cells; 20,000 stratified p...

### S-59  (F. C12 | para | 40w, 1 dash, 4 semi, 4 paren)
> C12-B (tied refiner): cycle-1[m]8 normalized burden improves monotonically in every cell (A/[m] 0.859[m]0.815; A/[m] 0.624[m]0.576; C/[m] 0.867[m]0.777; C/[m] 0.576[m]0.517; all cycle-1/8 map-bootstrap intervals separate), but the [m] gain is [m] (CI [m]) on A and [m] ([m]) on C---no dose--response

### S-17  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 67w, 0 dash, 3 semi, 2 paren)
> The control was designed and frozen after the learned confirmation results were known (its purpose is to answer the ``weak anchor'' objection those results raise); its tuning touched only the development cohort, and the tuned weights were evaluated exactly once on the confirmation cohort under the frozen rule (candidate grid [m]; highest development success, ties toward smaller [m]; the anchor its...

### S-46  (D. Adaptation: [m]-Indexed Transfer, Composition, and the Matched-Label Factorial | para | 67w, 0 dash, 3 semi, 2 paren)
> Concentrated low-world cells degrade full fine-tuning and scratch in both domains (success vs.\ the frozen zero-shot source: [m] [m] in all three seeds at dynamic concentrated [m] and [m]; [m] to [m], all seeds significant, at static concentrated [m]), while the distributed cells at the same [m] rescue them (dynamic [m] to [m]; static [m]; paired dist[m]conc full-FT deltas [m] to [m] dynamic, [m] ...

### S-09  (B. Exact Environment, Model, and Data Specifications | para | 62w, 1 dash, 1 semi, 4 paren)
> Collection draws a fixed number of candidate worlds per family from a deterministic per-(family, index) seed stream and keeps the usable ones: the canonical run drew 24 per family and kept 53 (maze 17 / rooms 19 / spiral 17); the seed-2001 and seed-2002 retrains drew 64 per family and kept 139 (41/50/48) and 149 (40/54/55)---the collection-budget divergence disclosed in Section C

### S-25  (C. Zero-Shot Transfer: Tables, Replications, Weighted A * | para | 70w, 2 dash, 0 semi, 2 paren)
> Instance identity for the corrected runs holds by construction---they share the canonical enumeration code path verbatim---and is independently checkable: a serialized-world hash manifest for all 300 confirmation instances (generation seed, start, goal, static obstacle geometry, patroller parameters, roadmap nodes and adjacency) ships in the artifact package, with optimal-arrival equality (186/186...

(Plus 62 further supplement flags — full dump in the scratch files; most are table captions and the erratum/probe/rescue paragraphs already known to be dense.)
