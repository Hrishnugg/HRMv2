# C12 Persistent Hierarchical Planning — Results

**Status (2026-07-14): COMPLETE.** C12-A's frozen one-seed pilot is complete and development-only with a `strong_negative` closure. C12-B completed the full G0 → smoke → pilot/runtime → three-seed TEST sequence. C12-B is also negative for the registered hierarchy-depth claim: G1-B fails the K-dose-response requirement, although refinement improves monotonically within every cell and C/K=8 passes the matched-control G2-B comparison.

**Primary sources:** [design](../design/2026-07-10-c12-persistent-hierarchical-planning-design.md), [implementation plan](../plans/2026-07-10-c12-persistent-hierarchical-planning.md), [C12-A pilot analysis](../../../../../hrm-cloud/continuous_prm/runs/c12_persistent_pilot_v6_final/results/C12A_ANALYSIS.md), [C12-B probe summary](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/probe/c12b_probe_summary.json), [C12-B computed summary](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/c12b_summary.json), [significance table](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/c12b_significance.csv), and [integrity audit](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/integrity.json).

## Executive verdict

C12 establishes two useful boundaries. First, hidden slow/fast dynamics really do create decision-relevant memory headroom (C12-A G0-A passes), but the tested learned temporal hierarchies do not convert it into a reliable forecast, planning, or persistent-carry advantage. Second, repeated graph propagation really does improve bounded-search behavior within a checkpoint (C12-B cycle curves are monotone), but the gain is not larger at K=8 than K=2. The strict hierarchy/depth hypothesis therefore remains unsupported.

C12-B also contains a localized architectural result that should be retained, not promoted to the headline: on C/K=8, the 101,505-parameter tied refiner at cycle 8 beats both the equally sized one-step shallow control and the 681,217-parameter eight-step untied control after BH correction, with no completion regression. This does not satisfy G1-B because the cycle gain lacks the preregistered K-dose response, and it does not replicate against untied on A/K=8 (untied is slightly but significantly better there).

## C12-A — persistent hidden-regime dynamics

### Methodology and reason for the formulation

C12-A was motivated by a limitation of the earlier static continuous-space ladder: if the current observation is sufficient, persistent memory has little reason to win. It therefore constructs paired PRM decisions whose present observation is aliased while the hidden direction, gate phase, or route mode changes the correct future action. A present-sufficient stratum is the negative control. Static-map, roadmap, goal, and latent-regime seeds are separated; paired variants share the visible decision state but differ in hidden dynamics and oracle action. Providers plan with the same 32-step space-time A* and are scored against a separate true simulator.

The full G0-A probe used 100 counterfactual pairs (200 episodes) in each of four strata, totaling 3200 provider rows. World-pair clustered intervals were used. The privileged `true_mode` provider is an authorization diagnostic, not a learned candidate.

### G0-A authorization

| Condition | Result | Gate |
|---|---:|---|
| Constructed alias rate | 71.3% | Pass |
| History completion gain | +0.467 | Pass |
| Collision-adjusted regret reduction | 65.1% (95% CI 63.0%–66.9%) | Pass |
| Oracle completion | 97.5% | Pass |
| Oracle ceiling gap | 75.1% | Pass |
| Present-sufficient history headroom | 0.000 | Pass |

### Learned pilot outcome

The frozen pilot selected `onlstm` as the hierarchy and `lstm` as the flat comparator using VALIDATION only. G1-A forecast: **FAIL**; G2-A planning: **FAIL**; G3-A carry: **FAIL**. G4-A closes as **`strong_negative`**: Matched temporal hierarchy adds no value even when history is necessary.

This is development-only (`official_final=false`, one model seed). It is sufficient to close the approved pilot sequence, not to support a final multi-seed temporal-hierarchy claim.

## C12-B — tied iterative product-graph refinement

### Hypothesis and methodology

C12-B asks whether C11's global product-graph model was limited by a fixed shallow computation, and whether a shared recurrent update can progressively propagate value information across long mission graphs. It reuses C11 world generation, exact product-oracle labels, C11 node/edge tensors, leg-sum initialization, and matched product A*. C11's forward motion edges are consumed in the reverse value direction (destination → source), so one recurrent cycle moves downstream cost-to-go information one product-graph hop toward predecessors.

The tied model applies one shared graph block eight times and exposes cycles 1/2/4/8. Deep-supervision weights are 0.1/0.2/0.3/0.4. Controls isolate the relevant alternatives:

| Arm | Parameters | Edge applications | Purpose |
|---|---:|---:|---|
| `c11_gnn8` | 681,089 | 8 | Exact C11 untied forward-message architecture, retrained |
| `shallow_param_match` | 101,505 | 1 | Same parameter count as tied, one propagation step |
| `untied_compute_match` | 681,217 | 8 | Same reverse edge-compute depth, distinct weights per step |
| `tied_refiner` | 101,505 | 8 | Shared recurrent block; primary method |

All arms use the same residual-over-leg-sum target, smooth-L1 loss, AdamW (2e-4 learning rate, 1e-4 weight decay), gradient clip 1.0, 40 epochs, common node-budget graph batching, and validation-only checkpoint selection. Each authorized cell has 40 TRAIN, 10 VALIDATION, and the existing 25 C11 TEST worlds. Three model seeds are averaged inside each TEST world; the world is the independent unit. The analysis uses 10,000 world bootstraps, 20,000 sign flips, completion bootstraps, and BH correction over the four deep-cell G2 comparisons.

### G0-B: K=16 feasibility and authorization

K=16 was evaluated before learning under a frozen 300-attempt envelope. Both cells clearly contained the intended headroom/depth regime when a valid mission was found, but construction was too rare to meet the required 20 valid worlds. K=16 was therefore dropped before training; K=2/8 remained authorized from the existing C11 substrate.

| Cell | Valid / attempts | Oracle/leg-sum ratio | Median final-transition hops | Max graph bytes | G0-B |
|---|---:|---:|---:|---:|---|
| A/K16 | 2 / 300 | 0.0690 | 188 | 832,048 | Fail: insufficient valid worlds |
| C/K16 | 1 / 300 | 0.0618 | 166 | 830,200 | Fail: insufficient valid worlds |

Frozen probe hashes: raw `5c986102843ed7db763e98c67b63f7db9a8975f64f41baad95c8b775dadd1b5e`, seed ledger `28bcf0f1f7b35a3c22315acb244f8eae529ba5a2adcedfcf96b4a6d3fbe5ab57`.

### Full TEST cycle curves

The primary planning metric is expansion burden = expansions / binding budget; lower is better. Completion is shown alongside it. Values average three model seeds within each of 25 TEST worlds.

| Cell | Cycle 1 burden / success | Cycle 2 | Cycle 4 | Cycle 8 | C1−C8 improvement (95% CI) |
|---|---:|---:|---:|---:|---:|
| A/K2 | 0.8591 / 32% | 0.8463 / 35% | 0.8339 / 43% | 0.8154 / 52% | 0.0437 (0.0258, 0.0625) |
| A/K8 | 0.6237 / 100% | 0.6082 / 100% | 0.5927 / 100% | 0.5762 / 100% | 0.0475 (0.0324, 0.0641) |
| C/K2 | 0.8675 / 48% | 0.8351 / 52% | 0.7925 / 59% | 0.7772 / 61% | 0.0903 (0.0541, 0.1285) |
| C/K8 | 0.5765 / 100% | 0.5585 / 100% | 0.5394 / 100% | 0.5167 / 100% | 0.0598 (0.0421, 0.0778) |

Every within-cell adjacent cycle comparison has a bootstrap interval excluding worsening, and cycle 1 vs 8 is separated in all four cells. That is a genuine progressive-compute signal. It is not the registered hierarchy-depth signal because the gain does not grow from K=2 to K=8:

| Config | K8 improvement − K2 improvement | 95% CI | Dose-response |
|---|---:|---:|---|
| A | 0.0038 | (-0.0199, 0.0281) | Fail |
| C | -0.0305 | (-0.0726, 0.0096) | Fail |

### G2-B matched controls at K=8

Positive values mean the control uses more normalized expansions than tied cycle 8.

| Config | Control | Control−tied burden (95% CI) | BH q | Completion regression | Verdict |
|---|---|---:|---:|---|---|
| A | `shallow_param_match_cycle1` | 0.0232 (0.0148, 0.0324) | 0.000067 | None | Tied better |
| A | `untied_compute_match_cycle8` | -0.0031 (-0.0055, -0.0009) | 0.016799 | None | Tied worse |
| C | `shallow_param_match_cycle1` | 0.0177 (0.0109, 0.0252) | 0.000067 | None | Tied better |
| C | `untied_compute_match_cycle8` | 0.0076 (0.0044, 0.0110) | 0.000067 | None | Tied better |

G2-B passes in C/K=8 because tied beats both controls with no completion loss. A/K=8 does not pass: tied beats shallow, but the untied compute match is better by 0.0031 burden (about five expansions at budget 1,600). This config dependence prevents a general tying advantage claim.

### Value quality and planner caveat

The cycle improvement is stronger in bounded-search behavior than in oracle-value fidelity. This matters because the learned heuristics are intentionally not constrained to admissibility.

| Cell | Tied MAE C1 → C8 | Bellman residual C1 → C8 | Mean solved cost ratio at C8 |
|---|---:|---:|---:|
| A/K2 | 0.5290 → 0.5007 | 0.0404 → 0.0770 | 1.0220 |
| A/K8 | 0.9359 → 0.9192 | 0.0551 → 0.0589 | 1.0142 |
| C/K2 | 0.5579 → 0.5306 | 0.0464 → 0.0842 | 1.0209 |
| C/K8 | 0.8027 → 0.8052 | 0.0503 → 0.0552 | 1.0152 |

MAE improves in three cells but slightly worsens in C/K=8; Bellman residual worsens from cycle 1 to 8 in every cell. Cycle-8 solved paths average roughly 1.4–2.2% above oracle cost. The expansion gains should therefore be described as bounded-search efficiency under an inadmissible learned heuristic, not as uniformly better value iteration or optimal planning.

### Reference headroom on the exact TEST worlds

| Cell | Leg-sum success / burden | Oracle success / burden |
|---|---:|---:|
| A/K2 | 24% / 0.9444 | 100% / 0.1912 |
| A/K8 | 80% / 0.8436 | 100% / 0.0672 |
| C/K2 | 20% / 0.9652 | 100% / 0.1952 |
| C/K8 | 92% / 0.8645 | 100% / 0.0687 |

Large oracle headroom remains in every cell. C12-B therefore does not fail because the planning problem is saturated; it fails the specific claim that tied recurrent computation produces a larger benefit as mission depth grows.

### Gate resolution

- **G0-B:** K16 not authorized; A/C at K2/8 authorized.
- **G1-B:** **FAIL**. Monotone within-cell gains and deep C1-vs-C8 separation are present, but the required K-dose response is absent.
- **G2-B:** **PASS**, localized to C/K8.
- **G3-B registered closure:** `no_progressive_refinement`. Read literally as a failed registered progressive-refinement gate, not as "no cycle ever helps"; all four cycle curves improve, but not preferentially on the deeper task.

## Reproducibility and integrity

The full C12-B run completed in 14.95 minutes with 48 checkpoint records and no failures. Integrity expected and observed 48 checkpoints plus 3200 state rows and 3200 evaluation rows; every count, uniqueness check, finite-metric check, and required artifact check passed.

Accepted TRAIN/VALIDATION/TEST seeds are 40/10/25 per cell with zero overlap. Reanalysis from the two raw CSVs passed: 3200 state rows and 3200 evaluation rows; the significance CSV is byte-identical and the summary is semantically identical within 1e-12 float tolerance.

| Artifact | SHA-256 |
|---|---|
| `c12b_state_metrics.csv` | `ab7f4d6e1d66842c27a13a96a88080b0f6f3147f55f454fee0ba019c94c19c8f` |
| `c12b_eval_raw.csv` | `5502f63f185c135cb0372ae64eedf48e521ff6048ca934b1af71a71817bed75b` |
| `c12b_summary.json` | `cf8655ebd5b080a13ab21c835b60ac66781df072d5ab11f620e310c5b2a351a3` |
| `c12b_significance.csv` | `fc6ad13653abf70a5b641c3168ffcd64147dde139f06f4a4e0bc43c070f105fd` |

Regression verification: 217 C8/C11/C12 tests passed. The first two attempts were invalid environment runs because pytest could not create its default or `C:\tmp` fixture root; the clean run used a repo-local writable `--basetemp` and had zero failures/errors.

Environment: Python 3.13.7, PyTorch 2.9.0+cu130, `NVIDIA GeForce RTX 5090`. Source base: git `95be75a` on branch `c11-mission` with C12-B changes in the working tree at run time.

## Deviations and limitations

- K=16 was not silently shrunk after training; it failed G0-B construction feasibility and was removed before any model fit, exactly as preregistered.
- The execution CLI uses explicit `probe`, `smoke`, `pilot`, `full`, and `analyze` modes instead of a separate mode-plus-scale pair. Training/evaluation remain one resumable full transaction so partial TEST artifacts cannot be mistaken for a completed verdict.
- The C12-B implementation is split into a model/probe module, an execution pipeline module, and an independent reanalysis module for reviewability.
- C12-A remains a one-seed development pilot (`official_final=false`); C12-B is the completed three-seed full grid.
- The registered G3-B wording is coarser than the observed pattern: G1-B fails because the dose response is absent, even though cycles improve within every cell. Both facts are reported.
- Learned heuristic paths are usually slightly suboptimal. Any paper-facing efficiency claim must include path-cost ratio alongside expansion counts.

## Program implication

C12 closes the two most plausible remaining substrate objections to C11 without producing a general hierarchy win. Memory-relevant dynamics exist, but the tested temporal hierarchies do not exploit them robustly. Recurrent graph computation helps bounded search, and tying can help in one gated deep configuration, but the benefit is not depth-selective. The next experiment should not be another backbone swap. A defensible follow-up would change the planner/learning objective itself—e.g., cost-aware focal ordering, admissibility-aware correction, learned macro-actions, or explicit multilevel search—and preregister path quality as a co-primary outcome.
