# Complete Critical Review: `submission_v5` and `supplementary_v5`

**Review date:** 2026-07-25  
**Artifacts reviewed:**

- `submission_v5.tex`
- `submission_v5.pdf`
- `supplementary_v5.tex`
- `supplementary_v5.pdf`
- `references.bib` and generated bibliography materials
- all rendered figure assets
- LaTeX compilation logs and auxiliary evidence
- the supplied AAAI-27 author kit
- the submitted artifact package, its README, raw rows, scripts, manifests, and preregistration documents

**Intervention:** Review only. The manuscript, supplement, PDFs, figures, bibliography, and experiment artifacts were not edited. This Markdown report is a separate review document.

---

## Executive assessment

**Recommendation: major scientific revision and not submission-ready in its current form.**

There is a potentially strong empirical paper inside this submission. The best-supported result is narrow but useful:

> On a fresh 50-map-per-suite cohort and at selected tight expansion budgets, one development-selected U-Net heuristic substantially improves success over ordinary Euclidean A* and often uses fewer expansions than a subsequently tuned weighted-A* control.

The manuscript deserves credit for disclosing adverse results, post-hoc decisions, low matched-solved sample sizes, a replication-protocol divergence, and invalid early analyses. Independent recomputation found no evidence of fabricated rows, cohort leakage, or a wrong map-level experimental unit that reverses a headline result. The central C8, C13-M, and C14 observations are supported at a fixed-checkpoint/fixed-protocol descriptive level.

However, several of the manuscript's strongest formulations are materially stronger than the evidence:

- Not all six “zero-shot transfer” suites are held-out families.
- The model called “fixed” was selected after seeing the original target results.
- The two purported training-seed replications also change the number of training worlds and labels.
- The matched-label factorial does not replicate sampled training-world sets, so it does not establish that distinct-world coverage generally “governs” stability.
- The tuned classical control eliminates most of the learned method's success advantage, leaving a mixed effort/path-quality result.
- The C13 result confirms a criterion introduced only after the original preregistered quality ceiling failed.
- Several reproduction and artifact-inventory statements conflict directly with the submitted package.
- The paper accommodates too many experimental programs at the cost of a coherent scientific narrative.
- The main table and every supplementary table violate the AAAI minimum table-font requirement.

The central empirical observations may survive, but the paper must distinguish:

1. observed results in the tested synthetic setting;
2. post-hoc interpretations;
3. prospectively confirmed claims;
4. broad causal or architectural conclusions.

At present, these levels are often blended.

---

# I. Submission-blocking venue and rendering findings

## 1. Main Table 1 violates the AAAI minimum table-font size

**Location:** main PDF p. 4; `submission_v5.tex:99–117`, especially line 101.

The table is set in `\scriptsize`, approximately **7 pt** in this 10-pt document. The AAAI-27 author kit permits table text at 10 pt, or **9 pt if necessary**, but not 7 pt.

This is a hard venue-compliance failure, not merely a visual preference.

## 2. Every supplementary table is below the permitted minimum

The supplement contains 18 tables:

- 16 use `\scriptsize`, approximately 7 pt;
- 2 use `\footnotesize`, approximately 8 pt.

Affected source lines include:

- `\scriptsize`: 43, 72, 91, 112, 140, 195, 215, 235, 255, 284, 317, 353, 387, 454, 480, and 540;
- `\footnotesize`: 175 and 422.

This affects all supplementary tables rendered on pp. 2–7 and 9. Several are visibly difficult to read at normal size.

## 3. Figure text is also substantially undersized

The embedded figure labels are generally far below the AAAI figure-text expectation. Examples include approximately:

- 5–7 pt in the main dynamic and factorial figures;
- roughly 3–4 pt in the one-column six-panel budget figure after scaling;
- 5–7 pt in the integration, transfer, and C11 figures.

The budget figure is especially problematic: a wide source figure is scaled to one column, reducing already small Matplotlib fonts to an unreadable final size.

## 4. Supplement overflows and layout warnings

The supplied `supplementary_v5.log` reports:

- an **overfull `\hbox` of 1.39908 pt** in the architecture-table region (`supplementary_v5.tex:114–132`);
- an **overfull `\vbox` of 38.62646 pt**;
- numerous underfull boxes, including several with badness 10000.

The rendered supplement also shows margin/gutter pressure, dense float packing, and poor correspondence between section prose and its delayed tables.

## 5. Custom float packing worsens the supplement's readability

**Location:** `supplementary_v5.tex:18–24`.

The supplement raises float limits and changes `\topfraction`, `\textfraction`, and `\floatpagefraction`. This produces pages containing several small tables and figures while relevant section headings or prose appear on earlier pages. The resulting density and float order materially damage readability and may also conflict with the venue's prohibition on modifying the prescribed layout.

## 6. Compilation evidence

The supplied main-paper log reports only two minor underfull `\hbox` warnings. The PDFs show no visible `??` references, unresolved citations, missing figures, or malformed mathematical glyphs.

A fully independent clean compilation was attempted in a temporary copy, but the local MiKTeX installation stalled on its update-check warning and timed out. Therefore, the supplied logs and PDFs were audited, but source-to-PDF identity was not independently re-proved through a successful clean rebuild.

---

# II. Highest-priority scientific and methodological concerns

## 1. “Zero-shot transfer” mixes source-family generalization with held-out-family transfer

**Locations:** main Figure 1, p. 3; supplement §B, especially `supplementary_v5.tex:136`; protocol table lines 52–54.

The dynamic model is trained on maze, rooms, and spiral worlds. Figure 1 evaluates crossing, maze, dense maze, rooms, large rooms, and spiral.

The rows therefore have different meanings:

- **Maze, rooms, spiral:** new-world generalization inside source-family generators.
- **Dense maze, large rooms:** transfer to parameterized relatives of source families.
- **Crossing:** a more distinct timing-control family.

Calling all six rows “dynamic zero-shot transfer” obscures this distinction. The paper should mark source-family versus held-out-family rows explicitly and report breadth claims separately.

Dense maze and large rooms are meaningful shifts, but they do not demonstrate unrestricted transfer to unrelated planning domains.

## 2. The “fixed” primary model was selected after target results were known

**Location:** `supplementary_v5.tex:302`.

The supplement correctly states that the original 20-map evaluation results were known before the field U-Net blind model was designated as the primary model. The model, integration mode, and budgets were fixed only before generation of the fresh 50-map confirmation cohort.

The accurate evidential status is:

- development-target results informed model selection;
- the selected system was then prospectively evaluated on a disjoint confirmation cohort;
- the confirmation validates that selected system, not an a priori architecture-selection hypothesis.

The manuscript should consistently say **development-selected, confirmation-tested model** rather than relying on “fixed learned heuristic.”

## 3. The multi-seed “replication” is confounded by a major training-data change

**Locations:** main pp. 4–5; `supplementary_v5.tex:136`, 162, and 302.

The canonical pipeline used:

- 24 candidate worlds per family;
- 53 usable worlds total;
- approximately 0.82 million supervised states.

The two retraining pipelines used:

- 64 candidate worlds per family;
- 139 and 149 usable worlds;
- approximately 2.12 and 2.26 million supervised states.

These are not clean training-seed replications. They vary:

- initialization and stochastic training order;
- sampled worlds;
- number of worlds;
- number of supervised states.

The success advantage over Euclid appears robust across the three trained pipelines, which is useful. However, future-aware-versus-blind sign changes cannot be attributed specifically to “training-seed noise.” They could reflect data-collection noise, data-volume differences, or interactions between these factors.

This confound is especially important because the later C14 narrative treats distinct-world count as a governing variable.

## 4. The tuned weighted-A* control materially changes the headline conclusion

**Locations:** main Table 1, p. 4; supplement Table 11 and `supplementary_v5.tex:280–299`.

Against ordinary Euclidean A*, the fixed learned model shows large success gains. Against tuned weighted A*:

- weighted A* has higher observed success on crossing;
- maze and dense maze are essentially tied in success;
- learned is only 0.02 higher on rooms and large rooms;
- only spiral retains a BH-corrected learned success advantage;
- the learned system shows effort advantages on four of six suites, not all six.

The central interpretation therefore changes from a broad success rescue over classical planning to a narrower result:

> Relative to a tuned classical operating point, the learned system often reduces expansions, while success differences are usually small and one suite retains a strong success advantage.

### Path-quality tradeoff

The tuned baseline exposes meaningful quality differences:

- Crossing: weighted-A* arrival/optimal ≈ 1.008 versus learned ≈ 1.164.
- Large rooms: ≈ 1.003 versus learned ≈ 1.082.

Some learned expansion savings therefore coexist with worse paths. The abstract and conclusion should not foreground effort reduction without an equally prominent quality qualification.

### Post-hoc status

The weighted-A* control was designed after the learned confirmation result was known. Its weights were selected on the development cohort and evaluated once on the confirmation cohort, which is internally reasonable, but the baseline-family intervention remains a post-hoc robustness analysis rather than part of the original confirmatory design.

## 5. The budget calibration creates unusually weak Euclidean operating points

**Locations:** supplement Table 5, `supplementary_v5.tex:138–155`; supplement budget Figure 1.

The stated target-success band is `[0.45, 0.70]`, yet confirmation-cohort Euclidean success at the binding budget is:

- 0.12 on crossing;
- 0.12 on maze;
- 0.06 on dense maze;
- 0.42 on rooms;
- 0.82 on large rooms;
- 0.16 on spiral.

The manuscript discloses coarse grids and cohort drift, but “calibrated toward pre-specified target success rates” sounds stronger than these operating points justify.

The budget curves help by showing that Euclid needs approximately 1.7–2.8 times the binding budget to reach the learned model's binding-budget success. They also show that all providers converge at larger budgets. The correct framing is therefore **search acceleration under tight expansion budgets**, not a general ability to solve problems ordinary A* cannot solve.

Budget-selection uncertainty is not propagated into later inference.

## 6. Matched-solved effort ratios are selected and sometimes based on tiny samples

**Locations:** main Figure 1(b); supplement Tables 7–8; `supplementary_v5.tex:528`.

Dynamic matched counts include approximately:

- crossing: 6;
- maze: 6;
- dense maze: 3 on confirmation and 1 on the earlier cohort;
- spiral: 8;
- rooms: 21;
- large rooms: 41.

Expansion ratios are conditioned on both methods solving the map. This can favor a lower-success method by retaining only easier maps. The paper acknowledges this, but the figure still gives these ratios major visual prominence.

Consequences:

- dense-maze effort supports essentially no robust inference;
- crossing and maze medians remain fragile;
- ratios based on 6 maps are not commensurate with ratios based on 41 maps;
- paired success should remain primary wherever matched counts are small.

## 7. “Statistically indistinguishable” is not supported by an equivalence test

**Location:** main line 97; compare main line 172 and `supplementary_v5.tex:528`.

The manuscript uses “statistically indistinguishable,” while explicitly stating elsewhere that no equivalence tests were run. Failure to reject a difference is not evidence of equivalence or indistinguishability.

The defensible claim is only that no statistically significant difference was detected under the specified test and sample size.

## 8. The future-window conclusion is too broad

**Locations:** main Figure 1(c), pp. 4–5; `supplementary_v5.tex:162` and 229.

Across 18 suite-by-pipeline contrasts:

- 2 significantly favor aware;
- 2 significantly favor blind;
- 14 are null;
- signs vary across pipelines.

This supports:

> No reproducible advantage of this particular `W=8` future-occupancy encoding was observed across three trained pipelines.

It does not show that future-motion information generally fails to help dynamic planning. Limitations include:

- one encoding;
- one horizon;
- one input/model family;
- three confounded training pipelines;
- no equivalence or noninferiority margin;
- no model of between-training-run variance.

## 9. The matched-label factorial does not establish that world coverage “governs” stability

**Locations:** main Figure 2 and `submission_v5.tex:146`; supplement §D, especially lines 345–409.

### Design strengths

The factorial usefully provides:

- exact state counts;
- matched optimizer steps;
- concentrated versus distributed sampling;
- direct LoRA/full-FT/scratch comparisons;
- explicit rejection of the preregistered ratio-crossover hypothesis;
- a correct warning that absence of LoRA collapse is not equivalence.

### Missing replication at the manipulated level

`Supplementary_v5.tex:407` states that adaptation seeds vary initialization and batch order, while sampled state indices are shared across methods and seeds within a cell.

Thus, there are not three independent draws of concentrated and distributed world sets. There is one selected training dataset per cell, followed by repeated optimization runs.

Inference is over evaluation maps, not sampled training-world sets. The result can show that the selected distributed datasets outperform the selected concentrated datasets. It does not quantify robustness to which training worlds happen to be selected.

A general world-coverage conclusion requires independent world-set replicates at each `N × coverage` condition, with training dataset represented as an inferential level.

### Distinct worlds are not measured diversity

The supplement concedes that the factor is the number of procedural worlds, not measured geometric diversity. Increasing world count may simultaneously alter:

- geometry;
- state-distribution diversity;
- label correlation;
- start-goal variation;
- revisit frequency;
- world usability and rejection characteristics.

The experiment identifies a world-count treatment, not the mechanism by which that treatment helps.

### Domain and architecture are confounded

The static and dynamic conditions also change:

- environment dynamics;
- architecture;
- representation;
- source model;
- states per world;
- required world counts.

The cross-domain result is descriptive, not a clean domain factorial.

### One target family per domain

Both conditions use maze-dense targets. The observed pattern may be specific to that source-target relationship.

### “Exactly where” and “governed by” are too strong

Recovery in the dynamic concentrated arm occurs when one selected deterministic stream moves from one to four worlds at `N=65,536`. This is evocative but not a replicated mechanism. “Exactly” and “governed by” overstate one threshold in one target and one sampled world stream.

### Multiplicity and noninferiority

The 60 per-cell LoRA readouts have no multiplicity adjustment and no noninferiority margin. The paper correctly states that this is a failure to observe collapse, not equivalence, but still uses stronger language such as “the protection low-rank adaptation provides.”

### The direct factor contrast lacks matching inferential support

The central coverage claim requires a direct distributed-minus-concentrated contrast with uncertainty, clustered appropriately over evaluation maps and independent training-dataset replicates. Comparing one significant cell with one nonsignificant cell is not itself a significance test of their difference.

## 10. Adaptation method fairness is not fully established

**Locations:** supplement Tables 4 and 12–14; §D.

Open alternatives include:

- method-specific learning-rate and regularization effects;
- unequal convergence requirements;
- a fixed 2,560-step schedule that is compute-equal but not convergence-equal;
- scratch models receiving a schedule designed around pretrained methods;
- different LoRA ranks across architectures;
- absent optimization-curve evidence;
- no decisive demonstration that full-FT collapse is not an optimization artifact.

The bounded-versus-unbounded LoRA ablation excludes one proposed clamp explanation, but it does not uniquely identify low-rank capacity as the mechanism.

## 11. C13 confirms a revised post-hoc criterion, not the original preregistered gate

**Locations:** main Scope and Boundary Results, p. 6; `supplementary_v5.tex:448–504`, especially line 500.

The chronology is:

1. An absolute maximum-cost ceiling of 1.10 was preregistered.
2. Every scale setting failed that ceiling on calibration data.
3. A comparator-relative quality criterion was introduced after observing the failure.
4. `α=1.50` was selected.
5. The revised criterion and model were frozen.
6. A fresh 144-map confirmation cohort was generated.
7. The revised criterion passed.

This is a legitimate exploratory revision followed by prospective confirmation, but it is not an uninterrupted preregistered success. The original criterion failed.

Additional limitations:

- one model seed;
- one static cohort;
- no formal quality bound for the direct arm;
- comparator-relative rather than absolute quality;
- prototype feature construction around 5.138 s per map;
- search itself around 0.0002 s;
- complete-map field-HRM inference around 0.371 s.

The method reduces node expansions but is currently much slower end-to-end. “Efficiency” must therefore be qualified as **search-effort efficiency, not runtime efficiency**.

## 12. C11/C12 hierarchy claims are diagnostic, not broad architectural evidence

**Locations:** main p. 6; supplement §§E–F.

The evidence is complicated by:

- recurrent/ACT cell collapse;
- padding tokens being co-opted as recurrent compute;
- a false `K=0` continuity premise;
- unequal inputs in some comparisons;
- abandoned `K=16` construction due to too few valid missions;
- a one-seed C12-A hierarchy pilot.

These studies show that the tested implementations did not produce the intended hierarchy effects. They do not show that hierarchical or recurrent models generally lack the relevant capability.

“Overall strong negative” is undefined and too strong for a one-seed pilot.

## 13. Practical relevance is not justified relative to graph scale and runtime

The roadmaps contain 192 nodes. Search takes tiny fractions of a second, while learned feature construction may take seconds. The paper needs stronger justification for learned heuristics at this scale and comparisons with classical alternatives such as:

- landmark/ALT heuristics;
- differential heuristics;
- reverse-Dijkstra or reusable preprocessing;
- greedy best-first variants;
- additional weighted/focal policies;
- end-to-end latency;
- amortization over repeated queries per map.

Reducing expansion count is scientifically interesting, but it is not automatically a practical planner improvement.

---

# III. Claim-by-claim evidential assessment

| Claim | Assessment | Main reason |
|---|---|---|
| Fixed learned heuristic beats Euclidean A* on the fresh cohort | Supported within the chosen budget regime | Large paired success gains on all six suites |
| The result generalizes to unseen planning families | Partially supported | Three suites are source-family generators; two are close variants |
| Learned guidance broadly beats tuned classical search | Mixed | Only spiral has a corrected success advantage; effort improves on four suites |
| Future occupancy information does not help | Not established generally | One encoding, three confounded pipelines, no equivalence test |
| LoRA protects performance under scarce supervision | Moderately supported in tested cells | Direct low-`K` contrasts favor LoRA, but optimization and dataset uncertainty remain |
| Full fine-tuning overtakes LoRA at larger `K` | Reasonably supported for the six reported static cells | Direct paired contrasts; five of six effort effects significant at `K=16` |
| Label count is not relevant | Not established as stated | Ratio endpoint is selected by success; one training dataset per condition |
| Distinct-world coverage governs stability | Suggestive but overclaimed | No independent world-set replicates; one target family per domain |
| Low-rank capacity is the mechanism | Insufficiently identified | Clamp ablation eliminates one explanation, not all alternatives |
| Local bounded-observation method beats complete-map models | Supported for expansion count under the revised criterion | Fresh cohort, but one seed, revised criterion, and no runtime advantage |
| Hierarchical depth does not help | Boundary evidence only | Collapse, unequal inputs, one-seed pilot, limited valid mission depth |
| Discrete focal ranking improves search | Exploratory pilot | Selected cells, 3–8 seeds, no full-suite confirmation |
| Discrete pooled transfer is beneficial | Descriptive only | Small aggregate difference without complete uncertainty analysis |

---

# IV. Statistical and reporting issues

## 1. Training-run uncertainty is generally underrepresented

Map-clustered intervals appropriately measure variation across evaluation maps conditional on trained models. They do not capture variability due to:

- initialization;
- selected training worlds;
- collection size;
- checkpoint choice;
- training instability;
- hyperparameter sensitivity.

The paper should distinguish map uncertainty, training-pipeline variability, and training-dataset variability.

## 2. Multiplicity families are not reconstructible

The paper says BH correction occurs “within stage families,” but does not clearly specify which hypotheses belong to each family across suites, methods, backbones, `K` levels, and endpoints.

The exact multiplicity family and primary/secondary status should accompany each inferential table.

## 3. Bootstrap intervals and McNemar tests need one explicit policy

Success is described through paired bootstrap intervals and corrected exact McNemar tests. The paper should state:

- which result determines significance;
- whether bootstrap intervals are descriptive;
- what happens when a marginal interval excludes zero but corrected McNemar does not;
- which tests are primary.

## 4. Earlier pseudoreplication was corrected but competing estimators remain

`Supplementary_v5.tex:306` retains record-level summaries “for continuity,” while Section J says map-level values govern. An earlier pass also mixed additive and focal modes and changed effect magnitudes substantially.

The paper should provide one sensitivity table with:

- original estimator;
- corrected estimator;
- point-estimate change;
- interval change;
- verdict change.

“The question is verdict-level agreement” (`supplementary_v5.tex:532`) is too dismissive because effect-size changes matter even when a binary verdict does not flip.

## 5. Sequential stages create substantial researcher degrees of freedom

C1–C14 is an adaptive program in which failed stages motivate new methods, criteria, and analyses. Local frozen Markdown records help with provenance but do not turn the complete program into one preregistered experiment.

The paper should distinguish:

- internally frozen design;
- externally verifiable preregistration;
- exploratory revision;
- fresh-cohort confirmation.

## 6. Sample-size and power rationale is absent

There is no clear justification for cohorts of 20, 24, 30, or 50 maps. This is especially relevant for:

- null future-window claims;
- LoRA non-collapse rhetoric;
- C14 interactions;
- suite-level C13 confirmation;
- hierarchy dose-response failures.

Failure to detect an effect cannot support equivalence without an equivalence margin or minimum-detectable-effect analysis.

---

# V. Prose, vagueness, repetition, and AI-like register

Text alone cannot establish AI authorship. The paper nevertheless exhibits a highly standardized, heavily polished academic register associated with templated or AI-assisted prose.

## 1. Systemic patterns

### Assurance saturation

Repeated phrases include:

- “design frozen before launch”;
- “reported rather than repaired”;
- “reported rather than suppress”;
- “no retraining, reselection, or recalibration”;
- “integrity machinery functioning as designed.”

These are often useful disclosures, but repeated assurance becomes self-validating rhetoric rather than scientific explanation.

### Repeated antithetical constructions

Examples include:

- “failure to observe collapse, not an equivalence claim”;
- “persistent collapse, not recovered hierarchy”;
- “search effort and path quality, not latency”;
- “not label count”;
- “not as models in isolation.”

The pattern is rhetorically polished but repetitive.

### Internal scorecard vocabulary

The manuscript relies heavily on:

- gate;
- rung;
- verdict;
- readout;
- pass/fail;
- frozen;
- claim supported.

This makes the paper read like an internal experiment ledger rather than a question-oriented scientific article.

### Causal slogans stronger than the design

Examples include:

- “localizes the real effect”;
- “the protective variable is distinct-world coverage”;
- “governed by”;
- “exactly where.”

These phrases compress uncertain interpretations into definitive mechanism statements.

## 2. Quote-level examples

- **Main abstract, p. 1:** “We study the dynamic path planning problem…” is a generic opener that contributes little specific positioning.
- **`submission_v5.tex:140`:** the paragraph restates its `K=1` LoRA conclusion at the end after already presenting it in detail.
- **`submission_v5.tex:146`:** “raw state count does not govern the failure” is too causal for a null interaction on a selected matched-solved endpoint.
- **Same line:** “recovery occurring exactly where…” overstates one observed threshold.
- **Same line:** “is governed by distinct-world coverage rather than label count” is the manuscript's clearest evidential overreach.
- **`supplementary_v5.tex:35`:** the appendix meta-outline is long, code-heavy, and difficult to parse before the reader knows the stages.
- **`supplementary_v5.tex:162`:** “reported rather than repaired” is part of a recurring defensive pattern.
- **`supplementary_v5.tex:302`:** “we disclose rather than suppress” repeats the same assurance.
- **`supplementary_v5.tex:306`:** retaining weaker record-level summaries “for continuity with the run reports” prioritizes internal history over scientific clarity.
- **`supplementary_v5.tex:347`:** “the hypothesized crossover point is degenerate” is loaded and unclear.
- **`supplementary_v5.tex:349`:** “localizes the real effect” implies alternative mechanisms have been ruled out.
- **`supplementary_v5.tex:448`:** “Claim supported.” reads like an internal verdict file.
- **`supplementary_v5.tex:444`:** “overall strong negative” is undefined and disproportionate to a one-seed pilot.
- **`supplementary_v5.tex:500`:** “Chronology, stated explicitly:” is necessary in substance but legalistic in style.
- **`supplementary_v5.tex:532`:** “the question is verdict-level agreement” improperly minimizes estimator changes.
- **`supplementary_v5.tex:566`:** “No preregistered verdict flips…” reassures instead of presenting a compact sensitivity analysis.
- **`supplementary_v5.tex:570`:** “the integrity machinery functioning as designed” is self-congratulatory and unnecessary.
- **Main conclusion, p. 7:** the field-level recommendation is broader than the synthetic program establishes.

## 3. The worst prose problem is compression

Several main-paper paragraphs combine protocol, results, interpretation, caveats, and appendix directions in one unit. Particularly difficult source lines include approximately:

- 71;
- 81;
- 85;
- 146;
- 168.

`Submission_v5.tex:146` is the most severe example: one paragraph contains the factorial design, null ratio interaction, multiple success effects, per-seed significance, LoRA non-collapse, noninferiority caveat, matched-solved warning, and final causal conclusion.

## 4. Malformed “single crossing interval” wording

**Locations:** main line 95; supplement line 249.

The text calls the one CI containing zero “the single crossing interval.” Because **Crossing** is a named suite, this is lexically ambiguous. The exception is actually the **large-rooms/seed-2001 interval that crosses zero**.

## 5. Internal stage codes displace semantic organization

C1–C14, rungs D–M, and labels such as C13-N/O/P require readers to decode a lab history. Semantic names would be clearer:

- dynamic confirmation;
- tuned classical control;
- scarce-label adaptation;
- matched-label coverage study;
- compositional missions;
- bounded-observation confirmation.

## 6. Repeated caveats should be consolidated

The manuscript repeatedly states that:

- ratios condition on solved maps;
- no equivalence test was run;
- designs were frozen;
- results were not suppressed;
- direct arms lack formal bounds.

These should be centralized in a concise inference-and-claim policy and repeated only where a claim-specific exception matters.

---

# VI. Vague or insufficiently specified details

## 1. Terms defined too late or incompletely

The following need earlier, operational definitions:

- collapse;
- trained source;
- fixed provider;
- complete-map provider;
- binding budget;
- focal mode;
- burden;
- headroom;
- strong negative;
- live evaluation;
- fresh/confirmation/untouched cohort;
- matched empirical path quality;
- would-halt steps.

## 2. “Blind” is easy to misread

“Blind U-Net” means blind to future-motion channels, not blind to map geometry or current-state information. Headline figures and tables should say this explicitly.

## 3. Search accounting is not sufficiently accessible

The paper needs precise definitions of:

- what counts as an expansion;
- stale priority-queue treatment;
- reopening policy;
- tie-breaking;
- inconsistent-heuristic handling;
- termination condition;
- time-state discretization;
- collision-check accounting;
- whether feature construction is amortized.

These choices materially affect expansion counts.

## 4. Evaluation populations are conditioned by world-build filters

Worlds can be rejected for build failure, roadmap disconnection, or dynamic unsolvability at `t=0`. Retained and rejected counts should be reported for every evaluation cohort, not only selected training collections.

## 5. Source/target/cohort lineage needs one explicit table

Readers currently have to infer which suites and cohorts are:

- source-family;
- held-out-family;
- parameter variants;
- calibration;
- development;
- original evaluation;
- confirmation;
- reused across studies.

## 6. Architecture details remain incomplete

Supplement Table 4 reports broad configurations, but mission models have wide 0.5–3.5M parameter bands, several counts are approximate, and constructor details are deferred to code that is not included in the examined package.

## 7. C14 training-world selection needs clearer description

The paper should state:

- how candidate worlds are ordered;
- whether world sets are nested across `N`;
- whether sampling is with replacement;
- how distributed worlds are selected;
- whether concentrated and distributed cells share any worlds;
- how rejected worlds affect the stream;
- whether methods receive exactly identical sampled rows.

## 8. Hyperparameter selection is unclear

Missing or unclear points include:

- how LoRA rank was selected;
- why ranks differ across architectures;
- whether full-FT and scratch learning rates were tuned separately;
- checkpoint-selection policy;
- use of target validation data;
- evidence that each method converged under the fixed schedule.

## 9. Notation and naming drift

Examples:

- `K\in\{0..32\}` is not proper set notation and hides actual values.
- `N\in\{256,\ldots,65,536\}` hides the factor-of-four grid.
- `n=80/40` does not identify train versus evaluation.
- “Patr.,” “Disc.,” “Sub. W/B,” “A@cal.,” and “O@cal.” impose unnecessary decoding.
- Dataset names drift among dense maze, maze-dense, and dense; and among large rooms, rooms-large, and rl.
- ON-LSTM abbreviations are inconsistent.
- “single crossing interval” collides with the Crossing suite name.

---

# VII. Structure and narrative flow

## 1. The paper contains several partially connected papers

The manuscript covers:

1. dynamic learned-heuristic generalization;
2. future-aware versus blind inputs;
3. weighted-A* control;
4. static/dynamic adaptation;
5. LoRA versus full fine-tuning;
6. matched-label coverage;
7. discrete-grid transfer;
8. compositional missions;
9. temporal hierarchy;
10. bounded-observation static search;
11. persistent planning state.

The first six can form one coherent paper. Items 7–11 feel like a second research program or an extended negative-results report.

## 2. The title and content breadth do not align

“Transfer for Dynamic Path Planning” suggests a focused dynamic-planning study, but substantial space is devoted to static roadmaps, discrete grids, compositional missions, hierarchy failures, and a static bounded-observation method.

The paper should either narrow to dynamic transfer/adaptation or explicitly reframe itself as a broad empirical study of learned search heuristics.

## 3. Related Work comes too late

Related Work begins on main p. 7, after all results. Readers therefore encounter the baseline and novelty claims before understanding how the study relates to Neural A*, TransPath, classical dynamic planning, and learned heuristic literature.

A compact related-work/background section should precede the method.

## 4. The tuned classical control belongs beside the headline result

The weighted-A* result changes the interpretation materially. The dynamic result should be organized as:

1. Euclidean comparison;
2. tuned weighted-A* comparison;
3. path quality;
4. budget sensitivity;
5. training-pipeline variability;
6. conclusion.

## 5. Adaptation should separate observation from mechanism

A clearer sequence is:

- observed `K`-indexed crossover;
- preregistered state-count hypothesis and its rejection;
- secondary coverage manipulation;
- limitations of the coverage interpretation.

## 6. C13 should be secondary or separate

C13 introduces a different method, assumptions, comparator chain, criterion chronology, and confirmation cohort. It could be a separate paper. If retained, it should be a clearly bounded discussion result rather than a coequal headline contribution.

## 7. Recommended high-level organization

1. Introduction with narrowly stated claims.
2. Related work and classical baselines.
3. Task, source/target splits, data lineage, and inference policy.
4. Dynamic confirmation:
   - Euclidean A*;
   - tuned weighted A*;
   - path quality;
   - budget curves;
   - pipeline variability.
5. Scarce-label adaptation:
   - `K`-indexed result;
   - matched-label study;
   - limitations of the world-coverage interpretation.
6. Discussion and external validity.
7. Short boundary-results paragraph.
8. Conclusion.

The supplement should be organized semantically rather than by C-stage chronology.

---

# VIII. Figure and table review

## Main Figure 1: dynamic headline

### Strengths

- Vector rendering.
- Paired success display is preferable to independent bars.
- Matched sample sizes are shown.
- Marker shapes supplement color in panel C.
- Confidence intervals are present.

### Problems

- Interval and legend text is approximately 5–6 pt.
- Panels B and C omit suite labels and depend on exact row alignment with panel A.
- Estimates with `n=3` or `n=6` receive similar visual weight to `n=41`.
- Source-family and held-out-family suites are not marked.
- Marginal intervals are not visually identified as unadjusted.
- “Blind” is not self-explanatory.
- The lower end of at least one displayed ratio interval is visually clipped by the panel's x-axis limit.

## Main Figure 2: matched-label factorial

- Sparse `N` values are connected, implying a continuous trend.
- World-count annotations are difficult to read.
- No uncertainty over training-world-set selection exists.
- Evaluation-map uncertainty and optimization-seed variation are not clearly distinguished.
- “Success delta versus source” requires domain-specific background knowledge.
- The visible dynamic threshold can be mistaken for replicated mechanism evidence.
- Domain also changes architecture, representation, and states-per-world regime.

The caption should state that each cell uses one selected sampled world/state set.

## Supplement Figure 1: budget curves

### Strengths

- Curves are derived exactly by thresholding recorded solve expansions.
- Normalization by `B*` assists comparison.
- The binding budget is marked.

### Problems

- Six panels at one-column width make labels and legends extremely small.
- Curves overlap heavily near saturation.
- The learned target success used for the 1.7–2.8× claims is not marked.
- Path-quality tradeoffs are absent.
- “Every provider converges” describes this finite cohort, not a general asymptotic property.

## Supplement Figure 2: integration summary

This figure combines heterogeneous experiments and estimators:

- discrete additive;
- selected discrete focal pilots;
- continuous additive;
- exploratory post-hoc continuous focal results.

Individual points lack suite/model identity, sample size, and uncertainty. The shared axis suggests stronger comparability than warranted. This figure adds little beyond a qualitative slogan and could be removed.

## Supplement Figure 3: `K`-indexed curves

- Only two or three `K` values are connected.
- Connected lines imply unmeasured interpolation.
- Panels use independent y scales.
- Confidence intervals and raw seed variation are absent.
- Expansion ratio is shown without simultaneous success.
- Methods have nonmatching x grids.

## Supplement Figure 4: compositional missions

- Neither panel shows uncertainty.
- The right panel reduces the result to three means.
- The correlation annotation is too small and overcrowded.
- Mission-depth cells use different maps, but the visual suggests a simple within-population trend.
- The figure uses 6.79 for one mean while `supplementary_v5.tex:534` reports 6.78.

## Main Table 1

Abbreviations such as `Disc.`, `Bl./WA*`, `WA*/anch.`, and `Sub. W/B` are difficult to decode. Success, effort, and path quality may require two tables or a table plus plot.

The 7-pt table font violates venue requirements.

## Supplement Table 13

Ranges across three seeds hide which seed produced each value and suppress the corresponding intervals. Star notation inside ranges is difficult to interpret.

## Supplement Table 16 header mismatch

**Location:** `supplementary_v5.tex:452–473`.

The last column is labeled `Δ expansions [95% CI]`, but early rows C–G contain parenthetical win counts such as `(0/6 wins)` rather than confidence intervals. The heading is factually inaccurate for those rows.

## All supplementary tables

All 18 use impermissibly small fonts. Several captions function as full result paragraphs, and dense abbreviations make the tables difficult to read even when zoomed.

---

# IX. PDF page-level flow

## Main paper

| PDF page | Main content and observations |
|---:|---|
| 1 | Title, abstract, introduction, beginning of contribution list. Final contribution bullet is split across the page boundary. |
| 2 | Contribution continuation, problem formulation/method, environment setup. |
| 3 | Dynamic Figure 1, model/training, inference policy. Dense but generally clean. |
| 4 | Weighted-A* Table 1, dynamic interpretation, seed discussion. |
| 5 | C14 Figure 2 and beginning of few-shot adaptation. Figure and prose compete for attention. |
| 6 | Adaptation continuation and overloaded “Scope and Boundary Results.” |
| 7 | Related Work, Limitations, and Conclusion all share one page; limitations are too compressed for the study's breadth. |
| 8 | References. |
| 9 | References continuation. |

## Supplement

| PDF page | Main content and observations |
|---:|---|
| 1 | Title; prose for A, B, and beginning of C appears before supporting A/B tables. |
| 2 | Protocol/environment Tables 1–3; section C prose is already underway. |
| 3 | Architecture/calibration Tables 4–7 and budget Figure 1; extremely dense. |
| 4 | Dynamic Tables 8–11 and integration Figure 2; overcrowded. |
| 5 | Adaptation Figure 3, direct-contrast Table 12, and D text. |
| 6 | C14 Tables 13–14, C11 Figure 4, and start of E. |
| 7 | C11 Table 15, C12/C13 text, Tables 16–17. |
| 8 | C13 continuation, discrete program, validity forensics, statistics. |
| 9 | C9 reanalysis Table 18 and reproducibility inventory. |

The supplement's float order is a material reading problem because sections frequently begin before their supporting tables appear.

---

# X. Reproducibility and artifact-package findings

## 1. Direct contradiction about checkpoints

**Supplement, `supplementary_v5.tex:572`:** the README supposedly lists included fixed dynamic and frozen adaptation checkpoints.

**Artifact README, lines 41–43:** “Checkpoints are not included for size reasons.”

The examined package contains zero `.pt`, `.pth`, or `.ckpt` files. One statement is factually wrong.

## 2. Claimed constructors and generators are absent

The supplement says full generator ranges and constructor details ship with the code. The examined package lacks the complete:

- environment generator;
- PRM builder;
- model constructors;
- trainer;
- planner/search implementation;
- collision checker;
- checkpoint files;
- focused test suite.

This prevents end-to-end reproduction of training and evaluation.

## 3. “Every quoted statistic” is not supported

The package does not provide complete raw/reproduction support for all reported programs, including substantial parts of:

- the discrete program;
- C12;
- C13 architecture substitutions and persistent-state work;
- early C1–C6 claims;
- end-to-end label recount inputs.

## 4. Documented commands fail as shipped

The README commands were run in a temporary package copy. The following failed because they resolve data paths outside the archive:

- `analysis/world_clustered_reanalysis.py`;
- `analysis/c8_fixed_provider_reanalysis.py`;
- its seed-2001 and seed-2002 variants;
- `figures/make_fig_dynamic.py`;
- `figures/make_fig_budget_curves.py`;
- `figures/make_fig_c14.py`;
- `analysis/c14_analysis.py`.

They attempt to access stale repository-level `hrm-cloud/...` or `docs/experiments/analysis/...` paths.

The README warns users to edit path constants, but that means the displayed commands are not standalone reproduction commands.

## 5. One figure generator succeeds because values are hand-entered

`figures/make_figures.py` executes, but corresponding generator code contains hard-coded arrays for integration, adaptation, C11, and C13 figures. This conflicts with the README statement that every plotted number is read from raw rows or analysis outputs and that no value is hand-entered.

## 6. Generated figure outputs do not match authoritative submitted assets

In clean temporary execution, the generic figure script produced PDF hashes that did not match the authoritative submitted figure assets, except for the C14 factorial figure already present in the package. This indicates stale or divergent figure lineage.

## 7. Unused and stale figure lineage

Files such as `fig1_program_map.pdf` and `fig5_c13.pdf` exist in generator lineages but are not used in the current TeX. Multiple differing generator copies make the authoritative source unclear.

## 8. Hash manifests do not independently prove chronology

Integrity hashes are useful only if a root hash is immutably anchored before execution. A manifest stored inside the same final archive proves present consistency, not independently verifiable preregistration timing.

---

# XI. Bibliography findings

The bibliography renders, but its metadata is inconsistent and often incomplete.

Problems include:

- journal entries missing volume, issue, or page ranges;
- conference papers represented as generic articles or arXiv preprints;
- venue fields and publication types used inconsistently;
- several older arXiv records likely having canonical publication versions;
- inconsistent author-name rendering for accented or non-ASCII names;
- incomplete proceedings metadata.

The bibliography needs a systematic entry-by-entry metadata pass rather than isolated cosmetic edits.

---

# XII. Unnecessary or low-value material

The following can be removed or heavily reduced without weakening the central paper:

1. C1–C6 history when those stages are not central claims.
2. Discrete forecasting values without uncertainty.
3. The “fifteen experiment families” inventory.
4. The frozen-halt anecdote unless it motivates a formal central validation procedure.
5. Repeated statements that results were not suppressed.
6. Obsolete record-level summaries retained for run-report continuity.
7. The long artifact-content paragraph while its claims remain inaccurate.
8. The heterogeneous integration summary figure.
9. Detailed C11/C12 material in the main paper unless hierarchy becomes a clearly defined contribution.
10. Numerical restatements already available in adjacent figures or tables.

The validity-forensics material may be useful as a separate artifact report, but it interrupts the paper's main argument.

---

# XIII. Concrete internal inconsistencies and editorial defects

1. Supplement line 572 says checkpoints are included; the README says no checkpoints are included.
2. The README says no figure values are hand-entered; plotting code contains hard-coded arrays.
3. Supplement Table 16 promises CIs where early rows contain win counts.
4. C11 mean is 6.79 in one place and 6.78 in another.
5. Dataset naming drifts among dense maze/maze-dense/dense and large rooms/rooms-large/rl.
6. `K\in\{0..32\}` is malformed and hides actual evaluated levels.
7. “Single crossing interval” is ambiguous because Crossing is a suite.
8. The architecture table claims exactness but gives broad approximate parameter bands.
9. “Preregistered” is used without always distinguishing internal freezing from externally verifiable registration.
10. “Replication” is used despite a known training-data intervention.
11. “Fixed” is used without always stating that initial target results informed selection.
12. Figure 1 labels all rows transfer without marking source-family rows.
13. Dense captions hide substantive reasoning.
14. The supplement claims to be organized by claim but largely follows run chronology.
15. “Statistically indistinguishable” appears despite no equivalence test.

---

# XIV. Important strengths

Several aspects are genuinely strong and should be preserved:

- The authors disclose that primary model selection followed original target results.
- The collection-budget replication divergence is explicitly reported.
- A fresh map-disjoint confirmation cohort is used.
- Success and effort are reported separately rather than collapsed into one score.
- The paper repeatedly warns about conditioning on jointly solved maps.
- Direct method-versus-method contrasts are used.
- The weak-anchor objection is addressed with tuned weighted A*.
- Exact budget curves are derived from deterministic traces.
- C9–C11 pseudoreplication is revisited at map level.
- The preregistered C14 crossover hypothesis is reported as rejected.
- The paper explicitly avoids claiming equivalence from nonsignificance in several places.
- C13's original quality-gate failure and revised criterion chronology are disclosed.
- The main PDF is generally clean outside the undersized table and figure text.
- Independent empirical checks found no evidence of fabricated central rows, cohort leakage, or an inference-grain error that reverses the main descriptive results.
- The fresh-cohort learned-versus-Euclidean success improvement is substantial within the tested operating point.

The core problem is not that every reported observation is unreliable. It is that careful local observations are often converted into broad causal and field-level conclusions too quickly.

---

# XV. Prioritized revision plan

## Blocking

1. Correct artifact/package claims and make all documented commands run as shipped.
2. Resolve the checkpoint and code-inventory contradictions.
3. Raise main and supplement table text to the AAAI-compliant minimum.
4. Enlarge all figure labels to venue-compliant, readable sizes.
5. Resolve supplement overflow and float-order failures.
6. Complete and standardize bibliography metadata.
7. Separate source-family generalization from held-out-family transfer.
8. Reframe the seed replications as pipelines confounded by collection size.
9. Replace “world coverage governs” with a claim proportional to the design, or add independent world-set replicates.
10. Present weighted-A* and path-quality findings as part of the headline result.
11. Reframe C13 as confirmation of a revised post-hoc criterion.
12. Narrow the paper to a coherent central question.

## Major

13. Separate conditional map uncertainty from training-run and training-dataset uncertainty.
14. Specify all multiplicity families and significance policies.
15. Define search accounting and implementation details.
16. Provide a complete source/target/cohort lineage table.
17. Clarify optimization fairness across LoRA, full FT, and scratch.
18. Reduce causal language around capacity, future information, hierarchy, and diversity.
19. Redesign or remove the integration figure and sparse adaptation curves.
20. Consolidate original versus map-clustered estimators in a sensitivity table.
21. Add sample-size or minimum-detectable-effect rationale for negative claims.

## Editorial

22. Remove repeated self-audit rhetoric.
23. Replace C-stage codes with semantic study names.
24. Split overloaded result paragraphs, especially main line 146.
25. Standardize dataset names, abbreviations, and rounding.
26. Correct table-header and notation defects.
27. Move Related Work earlier.
28. Reduce historical run-report material and stale artifact references.
29. Replace “statistically indistinguishable” with non-equivalence language appropriate to the actual test.
30. Rewrite “single crossing interval” to name the large-rooms/seed-2001 exception directly.

---

# Bottom line

The strongest defensible conclusion is narrower than the current abstract and conclusion:

> In this synthetic PRM setup, a development-selected learned heuristic generalizes to a fresh cohort and strongly outperforms ordinary Euclidean A* under tight expansion budgets. Against tuned weighted A*, its advantage is primarily reduced expansions on several suites, with mixed success and path-quality tradeoffs. Low-data adaptation results suggest—but do not yet establish—that distributing labels across more procedural worlds can stabilize full fine-tuning and scratch relative to concentrated supervision.

That is a credible empirical contribution. The current manuscript reaches beyond it.

In addition to the scientific major revisions, the undersized tables and figures, supplement overflow, bibliography quality, and artifact contradictions are concrete submission blockers.
