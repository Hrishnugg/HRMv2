# Program Audit — Hierarchy, Substrates, and Where To Point Next

**Date:** 2026-07-05
**Scope:** the *entire* experimental program — discrete-grid learned A\* (2025-11 → 2026-06: world-model benchmarks, clean transfer, CondLoRA, multitask/residual TaskLoRA, learned focal search) **and** continuous-PRM (C1–C10, C9b).
**Companions:** [`EXPERIMENT_RESULTS_COMPENDIUM.md`](EXPERIMENT_RESULTS_COMPENDIUM.md) (discrete record) · [`EXPERIMENT_RESULTS_FOCAL_REDESIGN.md`](EXPERIMENT_RESULTS_FOCAL_REDESIGN.md) · [`continuous_prm/CONTINUOUS_PRM_STORY.md`](continuous_prm/CONTINUOUS_PRM_STORY.md) (continuous record) · [`continuous_prm/CONTINUOUS_PRM_STRATEGY.md`](continuous_prm/CONTINUOUS_PRM_STRATEGY.md) (post-C9b step-back this audit refines).
**Method:** re-read both experimental records end-to-end; read the actual model implementations (`continuous_prm_common.py:1042-1223`, `residual_tasklora_v2.py:942-1298`) and the exact inputs they receive; derive substrate requirements from the architectures' real inductive biases instead of picking a direction by feel.

---

## 1. Why this audit

The program began with a thesis about **hierarchical models** (HRM, ON-LSTM) and has spent ~8 months testing it across two substrates. The continuous strategy memo diagnosed *substrate saturation* and recommended "a new substrate," but pointing at one without first asking **what our architectures would actually need from a substrate** risks a third round of nulls. And the program has a documented history of a specific failure class — the model was fine, the *harness* was wrong — which obliges us to check whether the hierarchy negative is a third instance of it before accepting it as a finding about the models:

1. **C5 → C6 (representation).** The per-node scalar residual collapsed HRM to a constant; the value-field representation rescued it to best-in-class (maze 0.975 vs oracle 0.984).
2. **Additive → focal (integration).** The discrete learned heuristic was *net-harmful* as an additive magnitude (α-tuning pinned α to the floor on all 10 models) while being a near-perfect ranker (ρ≈0.99 with true cost-to-go). Re-integrated as a focal-band ranker — same weights — it became a regression-free ~15% expansion win on **both** HRM and ON-LSTM.

Twice, an apparent model failure was a harness failure. This audit asks: is the six-phase "no hierarchical edge" result the third one?

**Short answer: substantially yes — but at a deeper layer than the previous two.** The evidence below says the *task formulation itself* (per-node scalar regression from pre-digested local features) is one an MLP can saturate, so *every* sequence architecture reduces to an expensive MLP on it. The models were never handed a problem in which their inductive biases could express themselves. That is a misapplication in the same family as C5's representation error and the additive-integration error — one layer further out.

---

## 2. The two programs at a glance

**Discrete grid (space-time A\*, Manhattan baseline, receding horizon):**

| Era | Task | Result |
|---|---|---|
| 2025-11 → 2026-02 | **World-model forecasting** (predict obstacle trajectories; planner consumes rollouts) | Real architecture separations, *task-dependent sign*: scaled HRM (28.97M, 8×H100, 18M samples) 68% vs LSTM 66%; hrm_3m/10m 0.71 vs lstm 0.67–0.69; but Preset M+ (harder mix) **ON-LSTM 0.388 > HRM 0.265** |
| 2026-02 → 2026-04 | **Learned heuristic regression** (residual cost-to-go, additive `f = g + h + α·δ`) | Near-null to negative: clean transfer *all* learned arms below Manhattan (best: fullft_hrm −0.59pp); multitask `avgbase__hrm` **+2.11pp** (the lone completed positive); specialists ≤ base |
| 2026-05 → 2026-06 | **Residual TaskLoRA v2 + diagnosis + focal redesign** | avgbase +1.0pp, expert −1.0pp vs avgbase; diagnosis: ρ≈0.99 ranker / flat magnitude (pred/true 0.73→0.19 with map size), α suppressed; **focal**: ~15% fewer expansions at matched-or-better success on both backbones; experts rank *identically* to base (bounded residual can't reorder) |

**Continuous PRM (A\* on roadmaps, Euclidean baseline, calibrated binding budgets):**

| Phase | Question | Result |
|---|---|---|
| C6 | representation fix | field rescues HRM; all models ≫ euclid; multi-suite closes OOD |
| C7 | integration matrix | additive cuts 15–48%; **additive > focal vs a loose baseline** (inverse of discrete); HRM ≈ ON-LSTM ≈ U-Net |
| C8 | dynamics + temporal spotlight | learned ≫ euclid-time (65–95%); **aware ≈/< blind** (7 sig blind-wins vs 1; MAE aware −0.25 worse); U-Net strongest |
| C9/C9h | few-shot transfer | transfer ≫ scratch (q=0.000); LoRA robust-but-plateaued vs full-FT crossover; bound irrelevant; no architecture edge |
| C10 | zero-label interpolation | clean null vs zero-shot (can't pass the LoRA plateau); RBF selectivity ~99% works |
| C9b | transfer under dynamics | C8 negative **robust to adaptation** (0/9 aware-wins @ full-FT K16); C9 crossover out-of-regime (one dynamic world ≈ 25k+ supervised (node,t) targets) |

---

## 3. Cross-space shared patterns

Seven patterns hold across **both** substrates. Receipts cited from both records.

**P1 — Learned wins are efficiency margins, not success-regime changes.** Discrete: the only safe operating point is focal `w=1.0` (learned *tie-breaking*): ~15% fewer expansions; `w=1.05` already regresses success on some suites. Continuous: 15–95% expansion cuts at matched-or-better success — but at budgets *calibrated to make Euclid struggle*. Neither program ever turned an unsolvable regime into a solved one on merit (discrete family-B stayed ≈0.01 for *every* method, learned or not).
*Corollary worth noting:* the continuous program engineered its fair fight (binding-budget calibration); the discrete program didn't (generous budgets, strong Manhattan) — which is why discrete results look weaker for what may be the same underlying signal quality.

**P2 — Every "model failure" so far has actually been a harness failure.** C5 scalar collapse → representation (C6). Discrete net-harmful heuristic → integration (focal). Each rescue came from questioning a layer previously held fixed. The layer never yet questioned: the **task formulation** (per-node scalar regression from local features). §5 argues this is the third instance.

**P3 — Pooled base ≥ specialists, everywhere.** Discrete: `avgbase__hrm` is the only completed positive; TaskLoRA experts mostly below matched baselines; under focal, expert exp-ratio **identical** to base (0.85=0.85, 0.82=0.82, 0.78=0.78). Continuous: LoRA plateaus at zero-shot (C9, matched-compute C9h); C10 interpolation of experts ≈ zero-shot ≈ uniform. Specialization-on-top has never beaten the pooled prior, in either space.

**P4 — Bounded corrections are safe but impotent.** Discrete mechanism (proven): the tanh-bounded residual is too small to change node *ordering*, so experts collapse onto the base. Continuous mechanism (proven): the clamp almost never binds (bounded ≈ unbounded, Δ=0.000±0.008) and low-rank capacity, not the bound, causes the plateau. Same design philosophy — "safe correction around a frozen base" — same result: safety purchased at the price of impotence.

**P5 — Architecture separations existed *only* on forecasting tasks, and vanished the day the task became cost-to-go regression.** This is the most important cross-space pattern for the hierarchy question:
- Obstacle-trajectory *forecasting* (world-model era): architectures separated **strongly** — scaled HRM beat LSTM (68/66); on Preset M+ ON-LSTM beat HRM by +0.12 mean success. The *sign* flipped with the task mix, but the tasks could **distinguish** architectures.
- Every *heuristic-regression* experiment since — discrete clean-transfer/TaskLoRA family and all of C6–C10/C9b — shows HRM ≈ ON-LSTM (≈ U-Net where present) with deltas of ±0.01–0.05 and no stable sign.
The distinguishing power didn't fade gradually; it disappeared exactly at the task boundary. Forecasting requires modeling *dynamics over time* (a sequence-model problem). Per-node residual regression, as we posed it, does not.

**P6 — Transfer helps exactly where supervision is scarce, and dissolves where it's dense.** C9's crossover lives at K≤4 worlds; C9b showed one *dynamic* world already carries ~25k (node,t) labels and the crossover evaporates; C10 showed composition can't add anything when the base already covers the region. Discrete rhyme: pooled training on all stages (avgbase) beat every attempt to specialize from less data.

**P7 — On every substrate we've built, the cheap signal saturates the achievable gap.** Discrete: learned signal only trustworthy as a tie-breaker. Continuous: present frame ≈ future window (C8, C9b); low-rank ≈ full model at the binding budget for most suites. Both programs independently arrived at "the learned signal is a margin, not a regime change" — because both substrates admit a mostly-local, mostly-smooth cost-to-go.

---

## 4. Personal audit of the direction

**What the program does well** (and should keep): matched paired evaluation with real statistics (McNemar+BH, bootstrap CIs, binding budgets); pre-registered gates; honest negatives written up as first-class results; frozen-reuse discipline between phases; TDD harnesses with adversarial review; and — twice — the willingness to diagnose *mechanism* (ρ-probe, MAE ablation) rather than stop at "it didn't work."

**Where the process failed us — seven specific self-criticisms:**

1. **Formulation inertia.** The core task — *regress a per-node scalar residual from hand-built features, integrate additively into A\** — survived unmodified from the first discrete transfer run through C9b. We varied representation (scalar/field), integration (additive/focal), adaptation (LoRA/FT/interpolation), substrate physics (static/dynamic), and *never* the formulation. Both historic rescues came from questioning a frozen layer; the frozen-est layer got no scrutiny.
2. **We never ran the degenerate control.** No experiment ever included a plain-MLP arm. The closest thing — C8/C9b's "time-blind" models with sequence length 1, which *are* MLPs over a single token — was built as a physics ablation, not an architecture control. It tied or beat everything. The most informative architecture result in the program was produced *by accident*.
3. **Name–mechanism gap.** "Hierarchical" was carried by class names, not verified mechanisms (§5): the HRM variant we test lacks the HRM paper's core machinery, its attention is degenerate (length-1 sequences), and ON-LSTM's structure-discovery gates were fed inputs with no structure to discover. We compared *labels*, and the labels tied.
4. **Scale confound, unexamined.** HRM's one historical win required 28.97M params, 18M samples, 8×H100. The C-series backbones are hidden-dim 128–256 (≈1–3M params) on a local GPU. Six phases of "no hierarchical edge" were run at 1/10–1/100 of the only scale at which hierarchy ever showed value here. (This does not predict scale would fix it — P5 suggests the task, not the scale, is the binding constraint — but an audit must flag it, and one scaled control would settle it.)
5. **Metric monoculture.** Everything optimizes expansions-at-binding-budget. Solution *quality* (the inadmissible additive arms run 10–18% over optimal makespan — measured once in C8, never studied), anytime profiles, and replan stability were never first-class. There may be headroom on axes we don't measure.
6. **Greedy search of design space.** Each phase asked "does X beat Y on the current substrate?" — a local move. We never asked the inverse question — "what would have to be true of the *problem* for X to matter?" — until after four consecutive near-nulls. This audit is the corrective, late.
7. **Headroom was diagnosed post-hoc, not gated pre-hoc.** The saturation diagnosis (strategy memo) came after C8–C10. A phase should never start without a measured oracle-vs-cheap-baseline gap large enough for the hypothesis to show up in. That becomes a standing gate (§7).

---

## 5. The architectures under the microscope

This section is the requested second look at HRM and ON-LSTM: what they actually are (as designed and as implemented here), what we actually feed them, and the verdict on misintegration.

### 5.1 ON-LSTM — as designed vs as used

**Designed** (Shen et al., *Ordered Neurons*): an LSTM whose *master forget/input gates* pass through `cumax` (cumulative softmax), imposing a total order on neurons: high-order neurons update rarely (slow, global state), low-order churn every step (fast, local detail). The split point between them is input-dependent. The bias exists to discover **latent nested structure** — constituency trees in language; more generally, sequences composed of segments-within-segments.

**As implemented here** (`continuous_prm_common.py:1086-1143`, identical maths in the discrete file): faithful cumax cell, chunk 8, hidden 256/480 → 32/60 orderable levels. The implementation is fine. The **inputs** are the problem:

| Pipeline | What the ON-LSTM actually scans | Nested structure available |
|---|---|---|
| Continuous static (C6–C10) | **24 tokens** = 1 state/goal + 6 nearest-obstacles + 16 rays + 1 descriptor (`FeatureConfig.seq_len = 1+6+16+1`), in fixed arbitrary order, token-type flags in dims 0–3 | none — it's a serialized *bag of features*; scan order is a file-format choice |
| Continuous dynamic (C8/C9b) aware | 9 rollout frames of nearest-patroller features | one timescale, 9 steps |
| Continuous dynamic blind | **1 token** | none — the cell fires once from zero state: an MLP |
| Discrete transfer family | 20 CNN-encoded global occupancy frames | in *static* suites the walls/goal channels are constant; only the agent-position channel moves — near-constant sequence |

A mechanism that allocates neurons by *update frequency* has nothing to allocate when the sequence is an unordered feature bag, one token, or twenty nearly-identical frames. Its one genuine showing — Preset M+ *forecasting*, where the input was 20 steps of real multi-object motion — is exactly where it beat HRM (0.388 vs 0.265).

### 5.2 HRM — as designed vs as implemented

**Designed** (Sapient's Hierarchical Reasoning Model): a slow **H module** and fast **L module** that implement *iterative refinement on a fixed input* — L runs T micro-steps to local convergence, H integrates and re-contextualizes L, repeated N cycles (effective depth N×T with O(1) memory via a 1-step gradient), trained with **deep supervision** per segment and **adaptive computation (ACT)** to spend more cycles on harder instances. Its published wins (Sudoku-Extreme, 30×30 Maze-Hard, ARC) are cases where the model itself performs an *algorithm-like latent computation* over a spatial input — the model is the solver, and depth-of-compute is the resource.

**As implemented here** (`DeepSapientHRMBackbone`, both files): a **two-timescale streaming RNN**. It scans input tokens once; H updates every `k_step=2` tokens (with the 1-step `detach`), L every token; the output is the last L state. What was kept: two timescales, the detach trick. What was dropped: **iterate-to-convergence on a fixed input, deep supervision, ACT — i.e., every mechanism the HRM paper identifies as the source of its reasoning power.** And one outright degeneracy: `GatedRecurrentBlock` applies `nn.MultiheadAttention` to a **length-1 sequence** (`h_norm.unsqueeze(1)`; `continuous_prm_common.py:1079`) — softmax over a single key is identically 1, so the "4-head attention" is a linear map in disguise. There is *no cross-token attention anywhere in the model*; all interaction across the 24 tokens flows through the recurrent scan.

So the honest description of what six phases compared: **a cumax-gated RNN vs a two-timescale gated RNN vs (sometimes) a conv net, all compressing a small feature bag into one vector for a 3-layer MLP head.** On a per-node smooth-regression task, these are three parametrizations of the same function class. The observed three-way tie is not a mystery; it is the *expected outcome of the formulation*.

### 5.3 The task, as formulated, is MLP-complete — the evidence

Four independent lines:

1. **The accidental MLP control wins.** C8 heavy: time-blind (seq-len-1 ⇒ MLP) beats aware in 7/8 significant cells; C9b: 0/9 aware-wins even after full fine-tuning at K=16. On this task, removing the sequence entirely costs nothing.
2. **The only model with a global view is the persistent (mild) winner.** The U-Net sees the *entire map* and computes a field in one pass; HRM/ON-LSTM see 6 obstacles + 16 rays + 3 non-local scalars (LOS bit, corridor score, free-fraction). The U-Net posting the best ratios under dynamics (C8: maze 0.064, spiral 0.076) is consistent with *information* (global receptive field), not *architecture*, being the differentiator.
3. **Non-locality is where all scalar models are weakest.** Bugtrap — the one family whose cost-to-go is genuinely non-local (you must back *out* of a trap) — has the tightest binding budget (24), the weakest zero-shot (0.696 @ 0.83), and the only LoRA that *degrades* with K (0.950 @ 0.48 by K=32). Locality starvation is visible exactly where locality fails.
4. **P5.** Architectures separated on forecasting tasks (both directions!) and never on regression tasks. The distinguisher is the task's demand for temporal/structured computation, not the model zoo.

### 5.4 Verdict on the misapplication question

**Yes — there is a misintegration we hadn't named, and it is one layer deeper than the two we fixed.** The program's ladder of discovered errors:

| Layer | Error | Fixed by | Cost of not seeing it |
|---|---|---|---|
| Representation | per-node scalar collapsed HRM (C5) | value field (C6) | 1 phase |
| Integration | ranker used as additive magnitude, α-suppressed | focal search | ~3 experiment families |
| **Formulation** | **hierarchical/sequence models pointed at a local, smooth, single-shot regression with pre-digested inputs** | **not yet — this audit names it** | 6 phases of architecture ties |

The correct conclusion from six phases is therefore *not* "hierarchy doesn't help planning." It is: **"on a formulation an MLP can saturate, nothing beats an MLP, and our formulation is one an MLP can saturate."** The hierarchy hypothesis has, strictly speaking, *not yet been tested* in this program — outside the early forecasting era, where architecture did matter and the sign flipped with task mix.

### 5.5 Where these architectures could actually shine (in this planning context)

Mapping verified bias → binding conditions:

- **HRM-as-designed (iterative refinement, two timescales, adaptive depth):** shines when the *output requires multi-step latent computation* — value/cost-to-go propagation the model must perform itself (its own maze benchmark is literally this), multi-stage constraint chaining, plan sketching. Binding conditions: feed it the *problem* (map/graph/mission), let it **iterate** on that fixed input (k refinement cycles, weight-tied), supervise per-iteration (deep supervision), and measure quality-vs-k. None of these conditions has ever been present in our pipelines.
- **ON-LSTM (ordered neurons, nested-segment discovery):** shines when inputs have *latent nested structure* — mission traces (mission → legs → maneuvers), multi-timescale environments (slow doors/gates over fast patrollers), episodic histories with boundaries. Binding condition: the sequence must be real and structured, not a serialized feature bag.
- **Both (two-timescale separation):** shines when a *slow* state (mission stage, region, regime) genuinely modulates *fast* decisions (local motion) — i.e., when the problem itself is hierarchical.

---

## 6. Substrate requirements — derived, not pointed at

From §3–§5, a substrate that can actually adjudicate the hierarchy thesis must satisfy:

- **R1 — Measured headroom, gated up front.** Oracle heuristic must beat the best *cheap admissible* baseline by a wide, pre-measured margin (target: ≥40–50% expansion reduction at the binding budget, and/or a large success gap). No phase starts without this number (new standing gate **G0-H**). *Lesson of C8–C10.*
- **R2 — Non-local cost-to-go.** Local geometry must underdetermine h\*: correct early decisions must depend on distant structure (traps, doors, ordering constraints). *Lesson of §5.3 (2,3).*
- **R3 — Compositional / nested task structure.** An explicit stage/subgoal hierarchy, so "slow state modulating fast decisions" is a property of the *problem*, giving two-timescale and ordered-neuron biases a real referent. *Lesson of §5.5.*
- **R4 — (optional second axis) Multi-timescale dynamics.** Slow topology change (doors/gates on long periods) *composed with* fast obstacles — constructed so a present-frame model provably cannot suffice (make "blind" fail *by design*, then see who can exploit time). *Lesson of C8/C9b.*
- **R5 — Structure-exposing I/O and compute.** The model must receive the map/graph/mission (not 24 pre-chewed tokens); iterative or message-passing computation must be allowed; and **every comparison must include an explicit MLP control arm** from day one. *Lesson of §4 (2,3).*
- **R6 — Harness continuity.** Keep the PRM + matched-A\*-expansion methodology, oracle labels, calibrated budgets, gates, and stats machinery — it is validated and cheap to reuse. *Keep what works.*

### Candidates evaluated against R1–R6

| Candidate | R1 headroom | R2 non-local | R3 compositional | R4 timescales | R5 I/O fit | R6 reuse | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| **S1. High-DOF arm PRM** (3–7 DOF joint space) | ✓✓ (Euclid poor in C-space) | ✓✓ | ✗ (hard ≠ hierarchical) | – | hard (C-space encoding unsolved for our models) | low-med | **high** | Real, but tests *difficulty*, not *hierarchy*; keep as follow-up |
| **S2. Compositional missions on existing hard maps** (ordered waypoints, keys→doors; state = PRM × mission-automaton) | ✓✓ *by construction* (euclid-to-goal is wrong in principle; leg-sum bound leaves a provable gap) | ✓✓ (doors/keys couple distant regions) | ✓✓ (explicit mission hierarchy) | optional add-on | ✓ (mission trace is a *real* sequence; map for field/GNN arms) | ✓✓ (C8 already built PRM×time product machinery; oracle = backward Dijkstra on PRM×automaton) | **medium** | **Primary — recommended (C11)** |
| S3. Multi-timescale dynamics extension of C8 | ~ | ~ | ✗ | ✓✓ | ✓ | ✓✓ | low-med | Not standalone (risks repeating the C8 null); fold into C11 later as the R4 axis |
| S4. Bigger/3-D versions of current maps | ✗ (same structure, more pixels) | ✗ | ✗ | – | – | ✓ | med | **Reject** — scale without structure reproduces saturation |
| S5. Iterative latent solver (recurrent weight-tied field refiner, deep supervision — HRM *as designed*, in field form) | n/a | n/a | n/a | n/a | ✓✓ | ✓✓ | low | Not a substrate — the **missing model arm**; fold into C11 |

---

## 7. Recommendation — C11: compositional-mission PRM, with the controls we never ran

**The substrate (S2).** Missions on the existing C5/C7 hard maps: visit K subgoals under precedence constraints, with keys that open doors (doors = removable obstacles gating corridors). Search state = `(node, stage)` on the product graph PRM × mission-automaton — mechanically the *same* product construction as C8's PRM × time, so the space-time A\*, oracle (backward Dijkstra over the product), calibration, eval, and stats layers port directly. Baselines: `euclid-to-next-subgoal` (weak admissible) and `remaining-leg-sum` lower bound (strong admissible — the one to beat). Targets: held-out mission *structures* (new orderings/key-door graphs) and held-out map families, so the C9-style transfer questions carry over.

**Why this substrate and not another:** it is the only candidate that (a) creates headroom *by construction* — no single-goal geometric heuristic can encode "you must detour for the key first," so the oracle-vs-baseline gap is structural, not tuned; (b) makes the problem itself hierarchical (mission over motion), giving both architectures' biases a referent (R3); (c) reuses ~everything (R6), keeping cost at one phase, not a re-platform.

**The arms — designed to answer the architecture question this time:**
1. **MLP control** (explicit, first-class — the arm we never ran);
2. U-Net field per stage (current champion);
3. **GNN over the PRM product graph** (message passing = the natural propagation prior);
4. HRM / ON-LSTM fed the **mission trace + geometry** (a real nested sequence at last);
5. **Iterative field refiner** (S5): weight-tied recurrent U-Net/GNN, k ∈ {1,2,4,8} refinement cycles, deep supervision per cycle — the first faithful test of the HRM *mechanism* (iterative refinement) rather than the HRM *name*;
6. (cheap addendum, once, at final config) one ~10–20M-param variant of the best and worst arm — kills or confirms the §4.4 scale confound.

**Pre-registered spotlights (gates):**
- **G0-H (headroom):** measure oracle vs leg-sum baseline on candidate mission configs *before* building the phase; require ≥40–50% expansion gap at binding budget. If no config clears it, the substrate is rejected *before* we spend a phase on it. (~1 day: product-graph oracle + existing calibrate machinery.)
- **G1 (dose-response in structure):** architecture gap as a function of mission depth `n_stages ∈ {1, 2, 4, 8}`. At `n_stages=1` the task degenerates to C7 and **must reproduce the three-way tie** — a built-in control validating continuity. The hierarchy thesis predicts a *widening* gap (GNN/iterative/hierarchical over MLP) as stages deepen.
- **G2 (depth-of-compute):** iterative refiner quality vs k. If iteration buys accuracy on deep missions (and the MLP/one-shot arms plateau), the HRM mechanism finally registers; if quality is flat in k, iterative latent computation is genuinely unnecessary even here.
- **G3 (honest closure):** if, with measured headroom (G0-H passed), real compositional structure, structure-exposing I/O, and an explicit MLP control, **everything still ties** — then "learned planning heuristics are architecture-agnostic" graduates from a suspicion to a strong, general, publishable claim, and the program pivots cleanly to the transfer+integration paper (strategy memo §5, thrust #5) with the architecture question *closed*, not abandoned.

**Decision rule after C11:** dose-response positive → the hierarchy thesis lives; invest next in S1 (high-DOF arm) to test generality. Flat → close the architecture chapter with confidence; the program's durable contributions are transfer + integration + the negatives, and we write that paper.

### Immediate next actions
1. **Nothing further on the flat 2-D formulation** — no more method variations there (reaffirms the strategy memo, now with mechanism).
2. **G0-H headroom pre-check script** (product-graph oracle vs leg-sum on 2–3 mission configs on existing maps) — the go/no-go for C11, ~a day of work, zero new substrate risk.
3. If G0-H clears: **brainstorm → spec C11** with the six arms and four gates above.
4. Keep S1 (arm) parked as the post-C11 follow-up; fold S3 (multi-timescale dynamics) in only if C11's static version shows an architecture signal worth stressing along a second axis.

---

## 8. One-paragraph summary

Across eight months and two substrates, the program has proven that learned heuristics beat their planners' baselines as *efficiency margins* and that *transfer from a pooled prior* is real and robust — and it has watched architecture differences matter only on forecasting tasks, never on heuristic regression. Reading the actual code explains why: both "hierarchical" models are used as small feature-bag compressors feeding an MLP head, on a task (local, smooth, single-shot scalar regression) that an MLP saturates — with the accidental proof that seq-len-1 "blind" models tie everything. This is the third instance of the program's recurring error class (representation → integration → **formulation**), so the hierarchy hypothesis is not refuted; it is untested. The corrective is not "a new substrate" in the abstract but a substrate derived from the architectures' verified biases: compositional missions on the existing maps (PRM × mission-automaton), with measured-headroom gating, a real mission sequence for the sequence models, an iterative-refinement arm that finally tests the HRM mechanism, an explicit MLP control, and a stage-depth dose-response that will either revive the hierarchy thesis or close it with confidence — either outcome ending the era of accidental nulls.
