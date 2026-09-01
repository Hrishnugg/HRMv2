# Prose & Structure Analysis of High-Impact Related Papers

**Purpose:** an evidence-based style guide for rewriting our submission, built by close-reading the field's best-received papers rather than from general writing advice. Compiled 2026-07-24 in response to the professor's critique (reproducibility specificity, "this is not how you write papers," per-hypothesis results, AI-register prose). **Advisory only — no paper edits made from this document yet.**

**Corpus (all PDFs in scratchpad `style_papers/`, close-read pages noted):**

| Paper | Venue | Standing | Pages read |
|---|---|---|---|
| Value Iteration Networks (Tamar et al., 1602.02867) | NeurIPS 2016, best paper | ~1.5k+ cites | 5–6 (experiments) |
| Path Planning using Neural A* Search (Yonetani et al., 2009.07476) | ICML 2021 | hundreds | 1–2, 5–6 |
| TransPath (Kirilenko et al., 2212.11730) | AAAI 2023 | 38 (verified S2) | 1–2 |
| Generalized Reactive Policies (Groshev et al., 1708.07280) | ICAPS 2018 | 133 (verified S2) | 1–2 |
| Optimize Heuristics to Rank (Chrestien et al., 2310.19463) | NeurIPS 2023 | recent | 1–2 |
| MPNet (1907.06013), Ichter sampling (1709.05448), SAIL (1707.03034) | T-RO / ICRA / CoRL | high | downloaded, not yet close-read |

The corpus deliberately spans the four venue registers that matter to us: AAAI (TransPath), ICAPS (Groshev — the fallback venue), NeurIPS (VIN, Chrestien), ICML (Neural A*).

---

## Part I — Abstract anatomy

All five abstracts follow one skeleton, with venue-specific weighting:

1. **Sentence 1 either names the problem in textbook terms or names the system.** Two observed openings: system-first (Neural A*: present the named method + its category in one sentence) and problem-first (TransPath: heuristic search on grids and where instance-independent heuristics fail; Groshev: a new approach to learning-for-planning stated as knowledge reuse across instances). Never an aphorism, never a question, never scene-setting.
2. **One gap sentence** with a *mechanistic* reason ("challenging due to the discrete nature of search," "instance-independent heuristics do not take obstacles into account") — not a rhetorical gap ("little attention has been paid…").
3. **What-we-do sentences are mechanically concrete.** They name the operation, not the aspiration: reformulate A* to be differentiable and couple with a convolutional encoder; learn the ratio between the instance-independent estimate and the perfect one; represent the policy as a DNN trained from an off-the-shelf planner's traces.
4. **Definitions appear inline in the abstract when a coined term is load-bearing.** TransPath defines its correction factor *inside the abstract* with an "i.e., the ratio between…" clause. Groshev italicizes and defines generalized reactive policy on first use. Rule: a coined term never survives an abstract undefined.
5. **The evidence sentence carries numbers or countable scope.** TransPath closes with an enumerated result list — roman numerals *i)*, *ii)* — including "up to a factor of 4x" and "less than 0.3% on average" *in the abstract*. Chrestien counts domains ("a diverse set" in the abstract, "eight problems, three grid and five PDDL" in the contributions). VIN-style papers put success rates in the intro. Our equivalent: the fixed-provider deltas and the 73–95% effort reduction belong in the abstract as numbers, not adjectives.
6. **Guarantee/scope sentence when applicable** (TransPath: the second heuristic keeps bounded-suboptimality via Focal search). We have the exact analogue (the w=1.10 certified control) and should use this slot the same way.
7. **Optional final capability sentence** (Neural A*: also works directly on raw images) — one bonus claim maximum.

**Register notes.** No first-person plural of emotion (no "we believe" in abstracts except Groshev's single low-key instance in the intro), no "novel" more than once, no hedging stack ("may potentially help"). Verbs are do-verbs: reformulate, learn, evaluate, show, outperform, reduce.

**Calibration datum:** TransPath's English is visibly non-native ("which costs exceed the costs of the optimal solutions") and it was still accepted at AAAI. Specificity, structure, and numbers dominate prose polish. The professor's bar is reproducibility, not literary style — the corpus confirms that ordering.

---

## Part II — Introduction architecture

The five intros share a paragraph-role sequence. Ours should map onto it one-to-one:

- **P1 — Textbook framing with application citations.** Define the problem in one or two sentences a non-specialist can parse, then ground it: applications (autonomous vehicles, arm manipulation, game AI — each with a citation) or complexity results (Groshev cites PSPACE-completeness). ICAPS flavor (Groshev) opens with an everyday robot scenario (arranging a dinner table) before formalizing — the venue tolerates, even rewards, one sentence of concreteness before formalism.
- **P2 — The specific dependence/gap.** Why the classical solution is limited, stated mechanically (search performance is heavily dependent on the input heuristic; instance-independent heuristics ignore obstacles). Define key terms inline with italics on first use (cost-to-go heuristic).
- **P3 — Prior-work lineage in miniature.** Two to four sentences naming the closest systems with one-clause summaries each (Takahashi learned perfect cost-to-go supervised; Yonetani made matrix A* end-to-end). This is not the related-work section; it is the minimal genealogy needed to position the contribution.
- **P4 — "In this work/study, we…" positioning paragraph.** Every paper has the literal marker phrase. It states the delta from P3's lineage in one sentence, often with an explicit list: TransPath announces "the distinguishable features of our work are as follows" and then lists them. Neural A* italicizes its thesis clause (reformulating canonical A* to be differentiable) — *typographic emphasis reserved for the single load-bearing claim*, used once.
- **P5 — Mechanism walkthrough with numbered stages that mirror the pipeline figure.** Neural A*'s intro walks training as (1) encode, (2) search, (3) backpropagate — and its Figure 2 labels the same three numbered stages. The figure and the prose share numbering. This is the single most stealable organizational trick in the corpus.
- **P6 — Intuition paragraph.** One paragraph translating the mechanism into a plain-language why-it-works (the encoder learns visual cues like dead-end shapes; attention lets the model reason "there is a passage between two regions"). A short quoted-intuition phrase in scare quotes appears in two of five papers — human, memorable, cheap.
- **P7 — Evaluation preview with dataset provenance.** One paragraph: what we evaluate on (named datasets with citations — synthetic + real), what we compare against, and the one-line outcome. Neural A* cites Bhardwaj's MP dataset and Sturtevant's city maps here — *in the intro*.
- **Contributions block.** Chrestien: a literal numbered list of five, each one sentence, the last one carrying the empirical count ("eight problems… *always* better" — italics on the strongest word, used exactly once). Groshev instead numbers *architectural findings* — the experiments-revealed-that-(1)-(2)-(3) pattern — turning results into design lessons. Either form beats prose-buried contributions.
- **Scope-limiting paragraph (Chrestien's device).** Immediately *before* the contributions: an explicit paragraph saying what the work does **not** claim or address, phrased as emphasis, not apology. This is the professional version of our claim-boundary discipline and should be adopted verbatim as a device: one paragraph, starts "We emphasize that this work neither…," placed before the contribution list.

---

## Part III — Related work conventions

- **Open by naming the threads:** TransPath begins by stating the two relevant lines of research as an enumerated sentence, then gives each a bold run-in subsection. The reader knows the taxonomy before any citation appears.
- **Every thread ends with a differencing sentence.** TransPath: a literal "The main difference between the mentioned approaches and our work is…" sentence per thread. Groshev quantifies the difference where possible (three orders of magnitude fewer training instances than the concurrent DNN-planning work). Our related work mostly lacks these closers — every paragraph should earn its keep with one.
- **Chronological micro-narratives inside threads** (first X was proposed, then Y extended it, more recently Z) — two to four citations per sentence cluster, each with a one-clause characterization. Never bare citation dumps without characterization, and never characterization without citations.
- Groshev's related work is long and generous (a full column+) — ICAPS culture rewards scholarly coverage of the planning-and-learning interface. For the ICAPS fallback, our related work should *grow*, not shrink.

---

## Part IV — Experiments/methodology sections (the professor's core complaint)

This is where the corpus is most prescriptive. The conventions, in the order sections use them:

**1. Open with the research questions.** VIN's experiments section opens with a numbered list of the two questions the experiments answer, before any setup. This is exactly the "per-hypothesis" organization the professor demanded, in the field's most-cited exemplar. Our stages map naturally: each C-stage's frozen question becomes a numbered Q.

**2. Datasets/domains as named, cited, counted entities.** Neural A* §4.1 is the model:
- Each dataset gets a bold/bulleted name, a provenance citation, and *exact* counts: 800/100/100 train/val/test maps per environment group; 3,200/400/400 after tiling; 20 of 30 city maps for training and the remaining 10 for test.
- Split hygiene is stated as a sentence, not implied ("we ensured that no maps were shared between training/validation and test splits" — the pattern, paraphrased).
- Instance generation is specified to the last die roll: goal drawn from one of four corner regions; 1/6/15 start locations for train/val/test; ground truth via Dijkstra.
- Resizing/adaptation of third-party data is disclosed with its reason (resized to 32×32 "to complete the whole experiment in a reasonable time").
- **Our gap:** our suite names (dense maze, spiral, bugtrap) appear without definitions or generator citation. Following this convention: one bulleted paragraph per family — construction rule, obstacle counts/radii ranges, side length — plus a pointer to the released generator. The §8 spec extraction in FEEDBACK_2026-07-24_ACTION_PLAN.md has every number needed.

**3. Architecture/implementation paragraph discloses every empirical constant.** Neural A* names the encoder backbone (U-Net on VGG-16, with the *implementation source cited*), the output activation, the heuristic used inside search (Chebyshev), the tie-breaking epsilon (Euclidean × 0.001), and the temperature rule (square root of map width) — each with its value and, where arbitrary, the word "empirically." VIN describes baselines with layer counts in prose (a 5-conv-layer CNN inspired by DQN; a 3-layer FCN whose first filter spans the image). **Rule: any number a reader would need to re-run the experiment appears in the text or a cited appendix table — none survive as "approximately."** (Directly answers the professor's 192-nodes point: state 192 exactly, k=7, and the sampling rule.)

**4. Baselines are constructed for causal isolation and framed that way.** Neural A* builds BB-A* by changing exactly one thing (differentiable vs. black-box search) while "keeping other setups… the same," and then *uses that in the results prose*: because the only difference is X, the comparison isolates X. Neural BF is defined by altering one equation term. This is our twin/matched-control discipline — the corpus shows how to narrate it: each baseline gets one sentence of mechanism, one clause of what is held fixed, and the results section cashes in the isolation explicitly.

**5. Metrics get bold names, formulas, and aggregation rules.** Neural A* defines Opt, Exp (with the max(100·(E*−E)/E*, 0) formula and its averaging scope), and Hmean as named bullets, then states model selection (best validation Hmean) and inference ("bootstrap mean and 95% confidence bounds per metric"). Our matched-solved ratio needs this treatment: bold name, formula, the joint-success conditioning stated as a property (it can flatter arms that solve easier maps — we already say this; keep it, it is exactly the genre's honesty register).

**6. Budgets and other swept parameters live here, not in the setting.** The professor's point is confirmed by the corpus: parameters that vary (budget, w) are introduced with the evaluation protocol, and results are presented as a function of them (our budget-curves figure is precisely the genre-standard artifact — VIN's Table 1 sweeps domain size the same way).

**7. Appendix discipline.** Neural A* states in one sentence that detailed setups are in Appendix A — *and the main text still carries the load-bearing numbers*. The appendix holds completeness (full per-cell tables, extra domains, hyperparameter grids), not the primary specification.

---

## Part V — Results prose conventions

- **Verdict-first paragraphs.** The topic sentence states the outcome (overall, X outperformed the baselines on both metrics); nuance follows (SAIL is sometimes more efficient, but at low optimality); mechanism-level explanation closes (the differentiable module exposes per-step information a black box cannot use). Never suspense, never "interestingly" as a substitute for analysis.
- **Honest negatives inside positive paragraphs.** Every exemplar concedes cells it loses (VIN: reactive nets have comparable prediction loss — the failure is specifically in task success, which sharpens the claim; Neural A*: classical planners are comparable on some datasets). Concessions are always paired with the reason and never buried — the professor's "only positives in the main paper" should be implemented as *lead* with positives, not as *hide* negatives; the corpus keeps sharpening-negatives adjacent to the claims they sharpen and moves *orphan* negatives (whole failed directions) to appendices.
- **Numbers in captions.** VIN's Table 1 caption states the takeaway (VIN significantly outperforms; the gap grows with problem size). Figure captions in Neural A* explain the color semantics (expanded nodes in green, path in red) so figures are self-contained. Every one of our captions should close with its takeaway sentence — most already do; audit the rest.
- **Per-question closure.** Where a section opened with numbered questions, each results subsection answers one by number. VIN's grid-world subsection ends by answering generalization; the Mars subsection extends it to real input. Nothing is left as data-without-verdict.
- **Mechanism paragraphs are allowed one conjecture, marked as such** (VIN: "we conjecture that…" appears in the domain setup; the results then test it). Conjecture verbs are used before evidence, claim verbs after — tense and verb choice track evidential status.

---

## Part VI — Figures, tables, naming, typography

- **Figure 1 is a what-we-do picture on page 1** in Neural A* and TransPath: input vs. classical-search vs. ours, with the visual convention (green = expansions) legible in seconds. We have no equivalent qualitative figure — our Figure 1 is quantitative. Worth considering a compact maps-with-expansions panel; a reviewer's first 10 seconds currently meet dumbbell plots instead of a picture of the task. (Advisory; no change made.)
- **Pipeline figures number their stages; the prose reuses the numbers** (Neural A* Fig. 2). If we add a provider→integration→search schematic, number it and mirror the numbering in the methods walkthrough.
- **Algorithm boxes with explicit Input/Output lines** (Neural A* Alg. 2) — worth adopting for the space-time A* + provider interface in the appendix.
- **Named systems and named devices.** Every exemplar names its artifact (VIN, Neural A*, TransPath, GRP, SAIL) and some name techniques (Groshev's leapfrogging, in quotes at first use). Our paper has no name for the fixed blind provider or the local-Bellman method — C13-M-style stage codes are internal jargon, exactly what the professor bounced off. A memorable name for the bounded-observation method would carry real weight. (Decision item — naming is the authors' call.)
- **Typographic emphasis budget:** italics for first-use definitions and for at most one or two load-bearing claim words per paper (Chrestien's *always*; Neural A*'s italicized thesis). Bold only for run-in labels, table winners, and metric names. Our draft is close to this already.

---

## Part VII — Venue register differences (for the AAAI-now / ICAPS-fallback decision)

- **AAAI (TransPath):** longest abstract of the corpus, numbers-in-abstract, page-1 qualitative figure, compact related work with differencing sentences, guarantees emphasized (Focal bound). Tolerant of imperfect prose when structure and specificity are right.
- **ICAPS (Groshev):** planner-literate framing (PSPACE, PDDL lineage, curated-knowledge history), generous related work, findings-as-design-lessons, everyday-robot motivation permitted. The audience knows weighted A*, Focal search, and SIPP cold — our planner-alignment material will land *better* there, and the "harness" framing can be translated into the planning community's own vocabulary (heuristic computation vs. search control).
- **NeurIPS (VIN, Chrestien):** research-questions-first experiments, theory-forward abstracts (Chrestien) or capability-forward (VIN), formal preliminaries sections with complete notation before any method text.
- **ICML (Neural A*):** the most polished specification discipline of the corpus; run-in paragraph labels everywhere (our draft already uses this device).

---

## Part VIII — Anti-pattern purge list (the "AI-written" register)

Concrete tells to sweep from our draft, each with the corpus-backed replacement:

1. **Aphoristic openers and slogans** ("two coupled programs share one question") → textbook problem statement with citations (every exemplar's P1).
2. **Coined framings used before definition** ("harness," "proving ground," stage codes C7/C13-M in main text) → either define-inline-with-italics at first use (the corpus does this for every coined term) or translate to community vocabulary (training objective, planner integration, evaluation protocol).
3. **Em-dash chains and semicolon stacks** carrying three claims per sentence → one claim per sentence in methods; the corpus's methods sentences are short, declarative, and singular. (Longer sentences appear in intros, but always with one subject.)
4. **Abstract nouns doing verb work** ("supervision density governs adaptation") → mechanism verbs with agents ("with fewer than ~N labeled states, full fine-tuning destroys the transferred prior; LoRA preserves it").
5. **Unanchored quantities** ("approximately 192 nodes," "several families held out") → exact values and enumerated lists (the corpus never approximates a quantity the authors control).
6. **Symmetric rule-of-three constructions** repeated across paragraphs → varied paragraph rhythm; the corpus uses lists when listing and prose when narrating, never decorative triads.
7. **Hedged verdicts** ("appears to suggest") → calibrated claim verbs: show/demonstrate for confirmed, conjecture for pre-evidence, "we report a failure to observe" for our negatives (this phrase of ours is actually genre-correct — keep it).
8. **Results narrated as process** ("we then ran… we then analyzed…") → verdict-first paragraphs (Part V).

---

## Part IX — Bonus discoveries relevant to the action plan (not style)

1. **Neural A* evaluates on exactly the external benchmarks the professor asked us to find:** Bhardwaj et al.'s Motion Planning dataset (eight obstacle-type groups) and Sturtevant's city/street maps (MovingAI). This is both the precedent (top venues treat these as the standard external validation) and the shopping list for our real-world credibility experiment — porting our frozen provider to MovingAI city maps mirrors a celebrated ICML paper's protocol, which itself is a defensible sentence in our rebuttal of "entirely synthetic."
2. **VIN's Mars Rover subsection** is the template for a compact "real input" section: one paragraph of data provenance (image patch sizes, elevation threshold, the detail that elevation is *not* an input), one paragraph of what transfers, done in half a page.
3. **TransPath extends the dataset of the closest prior work and says so** — dataset lineage as a legitimacy device; our fresh-cohort/seed-replication protocol can be narrated the same way.

---

## Part X — Priority translation to our rewrite (checklist, no edits made yet)

1. Abstract: add the headline numbers (success deltas, 73–95%, three-seed replication count) in TransPath's *i)/ii)* enumerated-results form.
2. Intro: verify the P1–P7 role sequence; add the scope-limiting paragraph before contributions (Chrestien device); keep the three contribution bullets but make each end with its countable evidence.
3. Methods: rename to Experimental Setting / Experimental Evaluation; suite-definition block with exact generator parameters (spec table already extracted); every constant disclosed; budgets moved to evaluation and introduced beside the budget-curves figure.
4. Experiments: open with numbered research questions mapped from the stage hypotheses; per-question verdicts in results.
5. Related work: add a differencing closer sentence to every thread.
6. Prose: run the Part VIII purge; captions end with takeaways; one italicized thesis clause maximum.
7. Consider (decision items): page-1 qualitative figure; naming the method; MovingAI port as the external-validation section.
