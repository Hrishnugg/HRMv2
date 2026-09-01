# Complete Deep-Pass Review: `submission_v6` and `supplementary_v6`

**Review date:** 2026-07-25  
**Review type:** Scientific, empirical, editorial, structural, rendered-PDF, LaTeX, AAAI-27 compliance, bibliography, anonymity, and artifact-reproducibility audit  
**Comparison baseline:** Version 5 and `submission_v5_complete_review.md`  
**Intervention:** Read-only review. No authoritative manuscript, supplement, figure, raw-data, or artifact files were modified.

---

## Executive verdict

**Major revision; not submission-ready in its current form.**

Version 6 is substantially stronger than version 5. It now presents a much more defensible scientific story:

- the fixed-model chronology is disclosed;
- the tuned weighted-A* comparison is framed as a tradeoff rather than dominance;
- SIPP establishes that the learned method is not the best planner for this deterministic substrate;
- end-to-end timing is reported;
- failure on MovingAI-derived geometry bounds external validity;
- the retraining/data-scale confound is explicit;
- the C14 primary-hypothesis rejection and independent world draws are reported;
- the earlier “statistically indistinguishable” and “single crossing interval” defects are gone;
- the version-5 table-font problem is largely fixed.

However, the current package still contains:

1. hard AAAI formatting violations;
2. direct internal and factual inconsistencies;
3. a serious experimental-unit limitation in the MovingAI benchmark;
4. an over-promoted interpretation of C14;
5. stale and non-turnkey artifact/figure provenance;
6. an anonymity leak in auxiliary artifact files;
7. dense, audit-like prose that obscures the scientific narrative.

The strongest defensible result is narrower than the paper’s current headline:

> A development-selected blind U-Net, frozen before a 300-instance confirmation cohort, strongly improved success and expansion effort over Euclidean A* under tight budgets on related deterministic procedural PRM suites. Against tuned weighted A*, its advantage was primarily lower matched-solved expansion effort rather than broad success dominance; SIPP solved the substrate outright, end-to-end latency did not favor the learned method, and the success advantage did not extend to the tested MovingAI-derived geometry.

That is a credible and potentially valuable contribution. The paper should center that bounded claim.

---

# 1. Audit basis and evidence scope

## 1.1 Authoritative files reviewed

- `submission_v6.tex`
- `submission_v6.pdf`
- `supplementary_v6.tex`
- `supplementary_v6.pdf`
- `submission_v6.log`
- `supplementary_v6.log`
- `submission_v6.bbl`
- `supplementary_v6.bbl`
- `submission_v6.blg`
- `supplementary_v6.blg`
- `references.bib`
- all six figures used by the v6 PDFs;
- relevant AAAI-27 author-kit instructions;
- the supplied `artifact_package/`;
- version-5 source/PDF and the complete version-5 review.

## 1.2 Coverage

The independent audits checked:

- all **9 main-paper pages**;
- all **16 supplementary pages**;
- all **25 physical PDF pages** in total;
- all **305 substantive source units**:
  - 58 main-paper units;
  - 247 supplementary units;
- every paragraph, contribution bullet, caption, main table row, and supplementary table row;
- all six used figures at native resolution;
- final embedded font sizes, PDF geometry, margin boundaries, vector stroke widths, font embedding, metadata, and page/float order.

The independent v6 editorial ledger was locked before the v5 comparison:

`205d681d4eed782d46eb48e4e3546b9e84bb22c92be8cf4db6f74669179a249a`

## 1.3 Empirical reproduction performed

In a disposable artifact copy, after changing only hard-coded path constants, the following analyses regenerated successfully:

- SIPP;
- MovingAI;
- C14-R.

Their regenerated Markdown outputs were **byte-for-byte identical** to the committed outputs. This supports the authenticity of those reported summaries.

Several other advertised commands did not run directly because they reference external repository paths. That is an artifact-packaging defect, not evidence that the experiments did not exist.

---

# 2. Submission blockers

## 2.1 Figure text remains below AAAI’s nine-point minimum

**Severity: submission blocker**

The AAAI author kit requires figure labels and internal text at least 9 pt:

- `kit_extract/AuthorKit27/AnonymousSubmission2027.tex:581`.

Measured final sizes in the submitted PDFs are approximately 6–7 pt in several figures:

- main Figure 1, PDF page 3;
- main Figure 2, PDF page 6;
- supplementary Figures 1–4, PDF pages 3–6.

The plotting code also declares undersized text:

- `artifact_package/figures/make_fig_dynamic.py:22–25`;
- `artifact_package/figures/make_fig_c14.py:31–34`;
- `artifact_package/figures/make_fig_budget_curves.py:24–27`.

Typical declared values are only 5.8–7.6 pt before or at final inclusion.

### Consequence

The figures can be read when magnified, but that does not satisfy the venue’s final-print-size rule.

### Required revision

Regenerate all figures with:

- no internal label below 9 pt after embedding;
- larger panels or less annotation;
- simplified legends;
- final inspection at 100% PDF size.

If necessary, move one Figure 1 panel or the full C14 display to the supplement.

---

## 2.2 Main figures contain sub-0.5-point strokes

**Severity: submission blocker**

The AAAI kit requires vector strokes of at least 0.5 pt:

- `AnonymousSubmission2027.tex:605`.

Measured minimum positive strokes include approximately:

- 0.48 pt on main page 3;
- 0.39 pt on main page 6.

The latter is materially below the rule. Grid lines, uncertainty bars, zero-reference lines, and some marker edges are vulnerable in print.

### Required revision

Regenerate the main figures with final embedded strokes safely above the boundary—preferably at least 0.7–0.8 pt before final export—and verify the compiled PDF rather than only the source asset.

---

## 2.3 Figure colors fail the author kit’s contrast requirement

**Severity: likely submission blocker**

The author kit requires color contrast above 4.5:1 and figures that remain decipherable without color:

- `AnonymousSubmission2027.tex:601–602`.

Against white, the figure palette yields approximately:

| Color | Contrast |
|---|---:|
| Blue `#0072B2` | 5.19:1 |
| Orange `#D55E00` | 3.87:1 |
| Green `#009E73` | 3.42:1 |
| Light blue `#56B4E9` | 2.31:1 |
| Gray `#5f5f5c` | 6.41:1 |

Marker shapes and line styles help, but the orange, green, and light-blue elements still fail the literal stated threshold. Figure 1(c) is especially exposed because seed identity uses those colors.

### Required revision

- darken failing colors;
- preserve marker/line-style redundancy;
- inspect grayscale conversion;
- verify contrast numerically after final export.

---

## 2.4 The supplementary title page overflows the bottom margin

**Severity: submission blocker**

`supplementary_v6.log:441` reports:

> `Overfull \vbox (38.62646pt too high)`

The anonymous-submission notice on supplementary page 1 extends to roughly 740 pt, well below the normal text boundary near 704 pt. The author kit explicitly requires fixing all margin intrusion:

- `AnonymousSubmission2027.tex:597–598`.

This is visible in the rendered PDF and is not merely a benign log warning.

### Required revision

Reduce or move title-page content so the notice returns to its standard position. Rebuild until the overfull-vbox warning disappears, then visually inspect page 1.

---

## 2.5 Supplementary floats form a seven-page table dump

**Severity: high structural/compliance issue**

Supplementary pages 1–8 contain nearly all narrative prose. Pages 9–15 then contain Tables 1–18 in a largely continuous deferred-float block, followed by a nearly empty one-reference bibliography on page 16.

This conflicts with the author-kit instruction to place figures and tables near their first discussion and not group them at the end:

- `AnonymousSubmission2027.tex:573`.

Examples:

- page 1 refers to Tables 5 and 9, which appear on pages 10 and 11;
- page 2 refers to Table 11, which appears on page 12;
- pages 9–15 contain almost nothing except deferred tables;
- page 13 is sparse;
- page 14 is partially occupied;
- page 16 contains only one short reference.

The custom float settings at `supplementary_v6.tex:18–24` did not solve the problem.

### Required revision

- interleave tables with Sections A–J;
- split oversized tables where needed;
- retain only review-critical tables in the PDF;
- move exhaustive machine-readable result tables to the artifact;
- use barriers or placement changes only after restructuring the float load.

---

# 3. Direct factual and internal inconsistencies

## 3.1 The Related Work section says SIPP was not evaluated

**Severity: high**

`submission_v6.tex:159` states:

> “safe-interval planning … is a related, typically faster alternative that we do not evaluate.”

But SIPP is a prominent evaluated baseline:

- main contribution: `submission_v6.tex:54`;
- main result: `submission_v6.tex:116`;
- conclusion: `submission_v6.tex:171`;
- supplement: `supplementary_v6.tex:298`;
- raw rows and analysis in the artifact.

This is a direct contradiction. It is particularly damaging because SIPP is one of the most important improvements in version 6.

### Required revision

Replace the Related Work statement with an accurate summary of the evaluated SIPP result and its implication for scope.

---

## 3.2 The manuscript says six MovingAI maps, but only five contribute rows

**Severity: high**

The manuscript says six MovingAI maps were converted and evaluated:

- `submission_v6.tex:118`;
- `supplementary_v6.tex:300`.

The supplement lists three street maps and three dungeon maps, including `brc202d`.

The supplied raw file

`artifact_package/raw/c8r_movingai/raw.csv`

contains rows from:

- Berlin;
- Paris;
- Boston;
- `den312d`;
- `den520d`;

but no `brc202d` rows.

Contributing-map counts are:

| Group | Development instances | Evaluation instances | Contributing base maps |
|---|---:|---:|---:|
| Street | 10 | 25 | 3 |
| Dungeon | 10 | 25 | 2 |

If `brc202d` produced no usable instance, disclose that exclusion/failure and say five base maps contributed data. If its absence was accidental, correct or rerun the benchmark.

---

## 3.3 Main Table 1 disagrees with the supplement and artifact

**Severity: medium-high factual inconsistency**

`submission_v6.tex:104–109` gives learned path-cost ratios that differ from the supplement/artifact:

| Suite | Main Table 1 | Supplement/artifact |
|---|---:|---:|
| Maze | 1.013 | 1.012 |
| Large rooms | 1.083 | 1.082 |
| Spiral | 1.014 | 1.011 |

The supplement says the main table contains the three-decimal artifact values (`supplementary_v6.tex:292`), so these are not merely different rounding conventions.

### Required revision

Use one generated analysis file as the sole source of truth and regenerate both tables.

---

## 3.4 C14-R is misstated as having all relevant intervals exclude zero

**Severity: high**

`submission_v6.tex:145` says the replicated distributed-minus-concentrated contrasts for full fine-tuning and scratch have:

> “all CIs excluding zero”

That is inconsistent with `supplementary_v6.tex:418` and the committed C14-R output:

- scratch contrasts exclude zero consistently;
- full-fine-tuning contrasts exclude zero only in draw 3;
- strict full-FT R1 fails in both new draws.

### Defensible replacement

> The coverage contrast was positive across the new draws; it excluded zero consistently for scratch and in draw 3 for full fine-tuning, while the strict full-fine-tuning collapse criterion failed in both draws.

---

# 4. MovingAI experimental-unit and external-validity problem

## 4.1 The analysis resamples instances nested inside very few base maps

**Severity: high scientific issue**

`c8_movingai_analysis.py:85–106` treats the 25 evaluation instances in each group as the bootstrap/McNemar units. Those instances are nested within only:

- three street base maps;
- two dungeon base maps.

The dungeon evaluation specifically contains:

- 17 instances from `den520d`;
- 8 instances from `den312d`.

The current uncertainty estimates therefore quantify variation across generated start/goal/roadmap instances conditional on a tiny fixed set of external geometries. They do not quantify transfer uncertainty across externally authored maps.

This matters because the scientific claim concerns geometry provenance.

### Required revision

1. Call the current intervals and tests **instance-level**.
2. Report instance counts by underlying base map.
3. Do not describe $n=25$ as 25 independent external geometries.
4. Cluster by base map if inferential claims are retained, recognizing that two or three clusters are too few for strong inference.
5. Expand the number of external base maps if a broad external-geometry claim is desired.

The negative result remains useful. It should be framed as a descriptive boundary on these five contributing base maps and generated instances.

---

## 4.2 Development and evaluation reuse the same external base maps

Development and evaluation instances use disjoint seeds but come from the same small map set. Thus weighted-A* calibration/tuning and evaluation are not separated at the base-geometry level.

The result can support instance-level performance on fixed external maps. It cannot support held-out-base-map generalization.

---

## 4.3 The conversion changes more than provenance alone

The MovingAI maps are:

- majority-coarsened to 64×64;
- decomposed into axis-aligned rectangle obstacles;
- paired with the paper’s own patroller process;
- evaluated on the same PRM/time-expanded formulation.

The manuscript acknowledges much of this, which is good. Continue to call it transfer to **externally authored geometry within the same raster/scale/task regime**, not general domain transfer.

---

# 5. C14 framing remains stronger than the design and results

## 5.1 The preregistered primary hypothesis was rejected

**Severity: high framing issue**

The supplement correctly states:

> “Preregistered verdict: H-C14 rejected as stated.”

at `supplementary_v6.tex:354`.

The required full-FT × log-label-count interaction was null. Nevertheless, the positive coverage interpretation is foregrounded in:

- abstract: `submission_v6.tex:37`;
- contribution heading: `submission_v6.tex:56`;
- main analysis: `submission_v6.tex:145`;
- conclusion: `submission_v6.tex:171`.

The coverage contrast was a prespecified secondary readout, so it is not an undisclosed post-hoc analysis. But the paper should explicitly say:

> The preregistered label-count hypothesis was rejected; a prespecified coverage analysis instead found…

Without this sentence in the abstract/introduction, the reader can reasonably infer that C14 confirmed its motivating mechanism.

---

## 5.2 “Controls,” “governs,” and “ties” are too strong

The factorial manipulates concentrated versus distributed retained labels under the tested procedural streams. It supports a statement about that manipulation in the tested maze-dense cells.

It does not establish that world count generally controls adaptation stability because:

- there is one target family per domain;
- “coverage” is number of worlds, not measured geometric diversity;
- distributed collection changes data-generation cost;
- the original design uses one sampled world set per cell;
- C14-R adds only two independent draws;
- C14-R uses one optimization seed;
- all draws reuse the same fixed test cohorts;
- per-cell readouts are uncorrected for multiplicity;
- there is no noninferiority margin for LoRA;
- the primary preregistered ratio hypothesis failed.

### Defensible claim

> Distributing a fixed number of retained labels across more sampled worlds improved held-out success for full fine-tuning and scratch in the tested maze-dense cells.

That statement is strong and supported. “Distinct-world coverage controls adaptation stability” is not yet.

---

## 5.3 The independent-draw verdict is mixed under its own strict criteria

The C14-R output reports:

| Draw | R1: concentrated FT+scratch harm | R2: distributed FT rescue | R3: static concentrated harm | R4: no LoRA degradation |
|---|---|---|---|---|
| 2 | FAIL | PASS | FAIL | PASS |
| 3 | FAIL | PASS | PASS | FAIL |

The evidence supports consistent rescue of distributed arms and especially stable scratch contrasts. It does not support an unqualified statement that the preregistered collapse/protection pattern replicated.

### Better synthesis

> The distributed arm was consistently rescued, while the magnitude and statistical detectability of concentrated harm depended on the sampled worlds; one replicate also showed significant LoRA degradation.

---

## 5.4 The adaptation headline drops its endpoint

**Severity: high**

The abstract, contribution list, and conclusion say full fine-tuning broadly “overtakes” LoRA:

- `submission_v6.tex:37`;
- `submission_v6.tex:56`;
- `submission_v6.tex:171`.

But `submission_v6.tex:141` supports the narrower result:

- full fine-tuning overtakes on **matched-solved expansion effort** by $K=16$;
- success remains near parity.

### Required revision

Retain the endpoint every time:

> Full fine-tuning often overtakes LoRA in matched-solved expansion effort as supervision spans more worlds, while success is near parity.

---

## 5.5 The chronology record needs clarification

The C14 design says:

> “the paper’s submitted claims do not depend on this study and will not be edited in response to it before review.”

at:

`artifact_package/preregistrations/c14/2026-07-23-c14-label-density-factorial.md:43`.

Yet C14 is central to the v6 abstract, main Figure 2, contribution list, and conclusion.

The README mentions a post-completion supersession record, but the supplied snapshot does not provide a clearly linked, independently dated record that unambiguously resolves this promise.

### Required revision

Add a clear dated explanation of:

- which manuscript version the promise referred to;
- when and why it was superseded;
- which analyses were prespecified before execution;
- which interpretation was formed after seeing the results.

If the documents are not immutably externally timestamped, “pre-specified” or “frozen before execution” is safer than repeatedly saying “preregistered.”

---

# 6. Dynamic zero-shot result: supported core and necessary limits

## 6.1 What is supported

The independent empirical audit found the central C8 rows credible. The paper correctly distinguishes:

- three held-out suites;
- three trained families evaluated on disjoint maps;
- Crossing as the clearest held-out topology;
- dense maze as a parameter shift;
- large rooms as a parameter/scale shift.

The fixed blind U-Net substantially improves success over Euclidean A* at the binding budgets and lowers matched-solved expansion effort.

No evidence was found that the headline rows were fabricated, reversed by a basic experimental-unit mistake, or caused by obvious train/test cohort leakage.

---

## 6.2 Retraining robustness is not independent experimental replication

The two additional pipelines:

- use independent training seeds;
- resample training worlds;
- reuse the same 50-map confirmation cohort;
- use substantially larger training collections than the canonical run.

The manuscript now discloses this well. Continue to call them **training-pipeline robustness checks**, not independent experimental replications.

---

## 6.3 The weighted-A* control is valuable but not fully effort-tuned

The control is selected by highest development success, ties to smaller $w_h$. Three selected weights hit the maximum tested value of 5:

- maze;
- dense maze;
- spiral.

This leaves open whether weights above 5, or a success-then-effort tie-break, would reduce weighted-A* expansions further. Since the paper’s residual claim against weighted A* is primarily about effort, calling the comparator simply “tuned weighted A*” is somewhat stronger than the tuning objective warrants.

### Recommended revision

- say “success-tuned weighted A*”;
- extend the weight grid when the selected value hits its boundary;
- report a success/effort/path-quality Pareto curve;
- or tune lexicographically by success, then effort, while retaining path quality.

---

## 6.4 SIPP correctly bounds planner superiority

The SIPP addition is a major strength:

- all 300 correctness gates pass;
- success is 0.98–1.00;
- path quality is optimal;
- wall time is comparable to or better than the learned pipeline;
- the manuscript explicitly states that the learned contribution is not superiority over a dedicated dynamic planner.

One remaining presentation issue is the use of the same numeric binding thresholds for interval-state expansions and $(v,t)$ expansions despite admitting that the units differ. That comparison should be treated as contextual only. Wall time and unbudgeted success are the meaningful cross-representation comparisons.

---

## 6.5 End-to-end latency does not favor the learned method

The supplement reports approximately:

- learned pipeline: 1.54 s per map;
- weighted A*: 0.39 s;
- anchor: 0.76 s;
- SIPP: roughly 1.0–1.7 s depending on suite.

Therefore the contribution is about search success and expansion structure, not runtime efficiency. Version 6 says this, and that qualification should remain prominent.

---

# 7. Statistical wording and endpoint discipline

## 7.1 Use “marginal interval excludes zero,” not “significant,” for uncorrected endpoints

The stated policy says:

- BH-corrected McNemar tests are primary for success;
- effort and path-quality intervals are marginal and uncorrected.

Nevertheless, `submission_v6.tex:95` and the main Table 1 caption call some path-quality premiums “significant.”

### Required revision

Either:

- say “the marginal 95% interval excludes zero,” or
- apply and report a declared multiplicity correction across path-quality comparisons.

The same discipline applies to C14 per-cell stars.

---

## 7.2 A null ratio interaction does not prove label count is irrelevant to success

The preregistered C14 regression models the matched-solved expansion ratio, while the promoted positive result concerns success.

Therefore the sentence that state count “did not explain the observed success degradation” is stronger than the preregistered model directly supports.

Either:

- provide a prespecified success-scale model; or
- present the success pattern descriptively without using a null ratio interaction as the mechanism test.

---

## 7.3 MovingAI multiplicity policy is underspecified

The MovingAI analysis reports unadjusted McNemar $p$ values for four natural success comparisons: two groups by two comparators. The general statistics section does not clearly declare the multiplicity family for this newly added experiment.

The dungeon learned-versus-weighted-A* value of approximately 0.016 can yield a different corrected conclusion depending on whether the family is:

- two group comparisons per comparator; or
- all four external-benchmark success comparisons.

Declare the family and report $q$ values before using inferential language.

---

# 8. Artifact reproducibility and provenance

## 8.1 What reproduced

After path repair in a disposable copy:

- SIPP output reproduced exactly;
- MovingAI output reproduced exactly;
- C14-R output reproduced exactly.

This is positive evidence for the new result summaries.

---

## 8.2 README commands are not turnkey

The README says Python 3.11, NumPy, and Matplotlib are sufficient, but several commands fail after installing those dependencies because scripts point to external paths such as:

- `hrm-cloud/continuous_prm/runs/...`;
- `docs/experiments/analysis/...`.

The README discloses this at `artifact_package/README.md:57–63`, but the package is still not self-contained for reviewer execution.

### Required revision

- use package-relative paths;
- add one clean reproduction command;
- execute it in a fresh environment;
- compare outputs to committed expected files;
- document dependency versions.

---

## 8.3 Packaged generators do not reproduce the exact submitted figures

The submitted and packaged/regenerated assets differ:

- `fig3_transfer.pdf` has different geometry;
- `fig4_c11.pdf` differs in geometry and extracted text;
- `fig_c14_factorial.pdf` differs from the packaged and regenerated version;
- `make_fig_dynamic.py` describes an older one-pipeline Figure 1, while the submitted Figure 1 contains three pipelines;
- the exact submitted one-panel `fig_c14_dynamic.pdf` is not reproduced by the packaged generator path.

The README claims coverage of all paper figures, which is not currently true for the exact submitted assets.

### Required revision

- designate one canonical generator per submitted figure;
- read all plotted values from packaged analysis outputs;
- regenerate the paper figures from the artifact;
- compare hashes between generated and submitted assets;
- remove stale scripts and stale copies;
- update provenance comments and README commands.

---

## 8.4 MovingAI map manifest is missing or inadequate

The supplement and README claim a MovingAI map manifest ships with the artifact. The package contains `raw/c8r_movingai/raw.csv`, including map names and seeds, but no separate map manifest with:

- source-file hashes;
- exact downloaded map files;
- conversion windows/parameters;
- accounting for the absent `brc202d` rows.

Add the actual source maps if licensing permits, or provide hashes, retrieval instructions, and a complete conversion manifest.

---

## 8.5 C14 result/provenance links are stale

The C14 result memo contains links to repository-layout paths such as `../design/...` that do not resolve within the submitted package. Repair every internal link after packaging.

---

# 9. Anonymity audit

## 9.1 Rendered PDFs are clean

The rendered main PDF, supplement PDF, and six used figure PDFs contained no detected username/project-path strings in their objects, streams, or metadata.

## 9.2 Auxiliary package files leak local identity/context

**Severity: critical if uploaded**

`artifact_package/raw/c13m/integrity.json` contains approximately 172 absolute paths exposing:

`C:\Users\hrish\Code Projects\HRMv2\...`

The LaTeX logs also expose local user and MiKTeX paths beginning near line 9:

- `submission_v6.log`;
- `supplementary_v6.log`.

### Required pre-submission procedure

1. Recursively scan every text-like file for:
   - `C:\Users\`;
   - `hrish`;
   - `Code Projects`;
   - repository roots;
   - workstation-specific paths.
2. Replace absolute paths with package-relative paths.
3. Omit logs unless required.
4. Regenerate integrity manifests after sanitization.
5. Unpack the final archive into a clean directory and scan it again.
6. Run the reproduction workflow from that clean directory.

Do not upload the current unsanitized integrity manifest.

---

# 10. Rendered-PDF and figure assessment

## 10.1 Main PDF strengths

- no main-paper overfull boxes;
- no visible clipping or gutter intrusion;
- correct letter page size;
- all fonts embedded;
- table captions below tables;
- figure captions below figures;
- no undefined references or citations reported;
- bibliography starts on page 8;
- rendered source/PDF order is generally coherent.

## 10.2 Main page-flow defects

- the third contribution bullet is split across pages 1–2;
- Figure 1 is extremely dense and functions partly as a results table;
- page 9 contains only a few references and substantial unused whitespace;
- multiple major result sections are compressed into long uninterrupted paragraphs.

## 10.3 Figure 1

### Strengths

- separates success, effort, and aware/blind variation;
- reports jointly solved sample sizes;
- exposes disagreement across three pipelines;
- avoids collapsing success, effort, and path quality into one score.

### Defects

- internal text is too small;
- panel (a)’s legend competes with lower-suite data;
- panel (c)’s legend overlaps the data region;
- panel (c) omits repeated suite labels/horizontal guides;
- row identity must be inferred from panel (a);
- multiple seed colors fail the venue contrast rule.

## 10.4 Figure 2

### Strengths

- concentration/distribution differ by line style and marker fill as well as color;
- individual adaptation seeds are shown;
- the log-$N$ axis is labeled.

### Defects

- too small at one-column width;
- no confidence intervals;
- connected means visually suggest a smooth dose response not established by the design;
- the independent-draw results are asserted in the caption but not visualized;
- the legend consumes much of the available area;
- the orange line fails the literal contrast requirement.

## 10.5 Supplementary budget curves

The curves usefully reduce dependence on one budget, but:

- sample sizes are not printed;
- uncertainty is absent;
- “oracle” needs a concise explicit definition in the caption;
- four methods across six small panels are difficult at the current font size;
- the binding-budget line can be confused with other dashed styling.

## 10.6 C11 figure

The right panel uses record-level means while the main inference is map-level. The values happen to be nearly identical, but the figure should display the primary inferential grain directly. Add uncertainty or map-level distributions rather than three unadorned bars.

---

# 11. Editorial and structural review

## 11.1 Main-paper density

Many TeX lines contain entire 140–288-word paragraphs:

- abstract: `submission_v6.tex:37`;
- first contribution: `:54`;
- formulation: `:69–73`;
- environment/evaluation setup: `:79–85`;
- weighted-A* control: `:95`;
- C14 synthesis: `:145`;
- limitations: `:165`.

These paragraphs often combine:

- protocol;
- result;
- interpretation;
- caveat;
- chronology;
- artifact pointer.

The result is technically careful but difficult to absorb.

## 11.2 Observable audit-like prose patterns

The main and supplement repeatedly use:

- “fixed”;
- “frozen”;
- “confirmation”;
- “gate”;
- “verdict”;
- “reported as-is”;
- “we disclose”;
- “audit trail.”

The supplement contains dozens of stage-code and gate mentions. This creates assurance saturation: the paper repeatedly tells the reader that the process was controlled instead of allowing compact design descriptions and evidence to demonstrate that control.

This is an observable register issue, not an authorship allegation.

## 11.3 Research-log chronology displaces scientific hierarchy

The supplement is organized partly as an experiment ledger—C7 through C14, gates, repairs, rungs, and execution-integrity events. That is useful provenance, but it is not the clearest scientific hierarchy for a reviewer.

A claim-oriented supplement would be easier to navigate:

1. dynamic transfer protocol and controls;
2. classical baselines and external boundary;
3. adaptation protocol and factorial;
4. interface/target controls;
5. negative hierarchy/composition studies;
6. statistics and robustness;
7. reproducibility/provenance.

Retain detailed stage chronology in the artifact’s experiment records.

## 11.4 Architecture claim is broader than the controlled comparisons

The abstract says MLPs, LSTMs, HRMs, and U-Nets can all support transfer and that no architecture is uniformly best. Those observations combine heterogeneous tasks, input representations, backbones, and training formulations.

The manuscript should describe this as a cross-experiment descriptive pattern, not an identified architecture comparison.

## 11.5 “Success-versus-budget curves remove dependence” is too strong

`submission_v6.tex:83` says curves up to $4\times$ binding remove dependence on a single threshold. They reduce or expose threshold sensitivity; they do not remove dependence on:

- the range endpoint;
- task horizon;
- candidate budgets;
- suite construction.

Use “mitigate” or “characterize” rather than “remove.”

---

# 12. Bibliography audit

There are no BibTeX warnings, but several entries are structurally incomplete or mistyped:

- `phillips2011sipp` is encoded as `@article` with a conference title in `journal`;
- `koenig2006rtaa` has the same problem;
- `silver2005cooperative` and `aine2016mha` should be checked as proceedings entries;
- several classic journal papers lack volume, issue, and page metadata;
- the generated Czechowski entry renders one author inconsistently as “Łukasz Kuciński” while neighboring names use surname/initial style;
- stale keys such as `aine2016mha` for a 2014 paper are confusing;
- many proceedings entries omit pages or stable identifiers.

This is not a scientific blocker, but it should be cleaned before submission.

---

# 13. Version-5 regression assessment

| Version-5 issue | Version-6 status |
|---|---|
| Main Table 1 used `\scriptsize` | **Fixed:** now `\small`, nominal 9 pt |
| Supplement tables used 7–8 pt text | **Largely fixed:** table text is nominal 9 pt under the AAAI class |
| “Single crossing interval” ambiguity | **Fixed** |
| “Statistically indistinguishable” without equivalence tests | **Fixed** |
| Model presented as prospectively fixed before target outcomes | **Fixed:** development selection is explicit |
| Retraining described as clean replication despite larger data | **Fixed:** confound disclosed |
| Weighted-A* weakened broad dominance | **Fixed/framed honestly** |
| No strong dynamic-planner baseline | **Fixed:** SIPP added |
| No external-geometry boundary test | **Fixed in principle:** negative MovingAI study added |
| No independent C14 world draws | **Improved:** two draws added, but strict verdicts are mixed |
| C14 coverage claim overstates identification | **Still present and now central** |
| Table/figure venue compliance | **Tables improved; figures still noncompliant** |
| Supplement overflow/readability | **Still problematic; new clear page-1 overflow** |
| Artifact self-containment | **Still unresolved** |
| Figure-generator divergence | **Still unresolved and directly demonstrable** |
| Sparse/inconsistent bibliography | **Still present** |

---

# 14. Recommended revision order

## Immediate blockers

1. Sanitize the anonymous artifact package.
2. Fix the SIPP “we do not evaluate” contradiction.
3. Correct the five-contributing-map versus six-listed-map MovingAI discrepancy.
4. Correct the false C14-R “all CIs excluding zero” statement.
5. Restore “matched-solved expansion effort” to every “full FT overtakes” headline.
6. Synchronize the inconsistent main/supplement path-cost values.

## Scientific framing

7. Reframe C14 as a rejected primary hypothesis followed by a positive prespecified secondary coverage result.
8. Replace “controls,” “governs,” and broad “replicates” language with cell- and endpoint-specific claims.
9. Recast MovingAI inference as instance-level within five contributing base maps.
10. Declare the MovingAI multiplicity family.
11. Call weighted A* success-tuned and address grid-boundary/tie-break limitations.

## Formatting and layout

12. Regenerate all figures with compliant font sizes, strokes, and contrast.
13. Remove the supplementary page-1 overflow.
14. Interleave supplementary tables with their discussion.
15. Reduce legend overlap and improve Figure 1 row readability.

## Artifact and editorial quality

16. Make every artifact path package-relative.
17. Reproduce the exact submitted figures from canonical packaged generators.
18. Add the missing external-map manifest/accounting.
19. Repair stale internal links.
20. Reduce chronology/provenance narration in the main paper.
21. Reorganize the supplement by claim rather than stage ledger.
22. Clean bibliography metadata and entry types.

---

# 15. Bottom line

The empirical honesty of version 6 is substantially better than version 5. Independent audit found no evidence that the central C8, SIPP, MovingAI, weighted-A*, or C14-R rows were fabricated or reversed by a simple analysis error. The new analyses can be reproduced once path constants are corrected.

The remaining problems are nevertheless material:

- hard AAAI figure and margin violations;
- a stale SIPP contradiction;
- incorrect MovingAI map accounting;
- a mismatch in Table 1 values;
- instance-level inference presented too close to map-level geometry generalization;
- C14’s rejected primary hypothesis being overshadowed by a stronger secondary-mechanism headline;
- a false C14-R interval statement;
- stale figure provenance and non-turnkey artifact commands;
- auxiliary-file anonymity leakage.

The defensible paper is not “learned heuristics dominate dynamic planning.” It is:

> A fixed learned heuristic can make tightly budgeted time-expanded A* much more effective on related procedural deterministic dynamics, while tuned weighted A*, SIPP, latency measurements, and external geometry sharply delimit that benefit. Distribution of retained target labels across procedural worlds also appears to affect adaptation success in the tested maze-dense cells, but the primary state-count hypothesis was rejected and the broader mechanism remains provisional.

That is a credible, bounded, and more interesting conclusion. A focused major revision can make the paper substantially stronger.

---

# 16. Completion and provenance note

- No authoritative manuscript, supplement, figure, raw-data, or artifact file was changed during review.
- Temporary renders and repaired-path reproduction runs were created outside the project.
- Independent editorial reports were produced at:
  - `C:\Users\hrish\AppData\Local\Temp\hrmv2_v6_independent_audit\v6_independent_ledger.md`
  - `C:\Users\hrish\AppData\Local\Temp\hrmv2_v6_independent_audit\v5_to_v6_comparison.md`
- This Markdown document consolidates the primary deep-pass review and the later three-agent independent-audit addendum.
