# Continuous PRM Heuristic Learning: Gradated Experiment Ladder C1-C4

**Status:** implemented as a staged, repo-oriented experiment suite.

**Primary location in repo layout:** `hrm-cloud/continuous_prm/`

**Files:**

| File | Role |
|---|---|
| `continuous_prm_common.py` | Shared continuous-world generator, PRM builder, Dijkstra labels, A* evaluator, HRM/ONLSTM models, LoRA parametrization, training/eval helpers. |
| `continuous_prm_stage_runner.py` | Unified CLI runner for `--stage c1/c2/c3/c4/full`. This is the main orchestration entry point. |
| `continuous_prm_c1_baseline.py` | Thin C1 wrapper with default `--stage c1`. |
| `continuous_prm_c2_avgbase.py` | Thin C2 wrapper with default `--stage c2`. |
| `continuous_prm_c3_residual_tasklora_experts.py` | Thin C3 wrapper with default `--stage c3`. |
| `continuous_prm_c4_rbf_mixture.py` | Thin C4 wrapper with default `--stage c4`. |

The implementation is intentionally self-contained, following the same practical style as the current grid experiment scripts: a single mounted experiment directory can run without importing the full historical grid pipeline. The continuous domain is different enough that tight reuse of grid-specific data generation would create more coupling than scientific value. Instead, the coupling is through naming conventions, CLI structure, checkpoint/result layout, HRM/ONLSTM comparison, bounded residual LoRA framing, and A* metrics.

---

## 1. Scientific motivation

The discrete-grid experiments showed that the average-base + bounded residual task-LoRA framing is promising after fixing the earlier unstable expert design. The next question is whether the same principle holds in a continuous geometric planning setting:

\[
\text{continuous world} \rightarrow \text{sampled PRM graph} \rightarrow \text{Dijkstra labels} \rightarrow \text{learned A* heuristic}
\]

This is not yet a fully kinodynamic continuous planner. It is a controlled bridge: state coordinates, obstacles, starts, and goals are continuous, while search is evaluated on a sampled roadmap graph so we can still measure A* expansions, success, and cost ratio.

The core hypothesis is:

> A pooled average heuristic can learn generic detour structure beyond Euclidean distance, while bounded residual LoRA experts can specialize to continuous task regimes such as open maps, random clutter, narrow passages, and larger-scale clutter.

---

## 2. Continuous planning formulation

### 2.1 World

Each world is a 2D square continuous domain. A point robot moves from a continuous start coordinate to a continuous goal coordinate while avoiding obstacles. Training anchors use circular obstacles first; evaluation also includes rectangular-obstacle shift.

Default training anchors:

| Anchor | Description | Intended pressure |
|---|---|---|
| `C_open` | Sparse circular obstacles | Euclidean should already be strong; tests whether model avoids hurting easy cases. |
| `C_clutter` | Medium random clutter | Tests generic detour prediction. |
| `C_narrow` | Barrier/gap structure plus clutter | Tests passage reasoning. |
| `C_large_clutter` | Larger side length and clutter | Tests scale/generalization pressure. |

Default eval suites:

| Suite | Type |
|---|---|
| `C_open` | ID anchor |
| `C_clutter` | ID anchor |
| `C_narrow` | ID anchor |
| `C_large_clutter` | ID anchor |
| `C_extra_dense` | OOD density |
| `C_tiny_passage` | OOD passage width |
| `C_large_open` | OOD scale/open |
| `C_large_narrow` | OOD scale + passage |
| `C_rectangles` | OOD obstacle shape |

### 2.2 PRM graph

For each generated world:

1. Sample `N` free continuous points.
2. Add start and goal.
3. Connect each point to `k` nearest neighbors.
4. Keep only collision-free straight-line segments.
5. Edge cost is Euclidean segment length.
6. Run Dijkstra from goal to get true roadmap cost-to-go.
7. Run A* from start using different heuristics.

This gives exact labels on the sampled graph while preserving continuous geometry.

### 2.3 Learning target

The base heuristic is Euclidean distance:

\[
h_0(x, g) = \lVert x - g \rVert_2
\]

Dijkstra gives the graph cost-to-go:

\[
d^*(x, g; W)
\]

The normalized training target is the detour residual:

\[
y(x, g, W) = \frac{d^*(x, g; W) - h_0(x, g)}{\text{side\_len}}
\]

The final deployed learned heuristic is:

\[
h(x, g, W) = h_0(x, g) + \text{side\_len} \cdot \widehat{y}(x, g, W)
\]

Predicted residuals are constrained to stay nonnegative and bounded:

\[
0 \leq \widehat{y} \leq B_{\max}
\]

where `B_max` defaults to `--max-norm-residual 4.0`.

---

## 3. Feature representation

The continuous model cannot rely only on local coordinates. It needs explicit geometry features. Each candidate PRM node is encoded as a short sequence of fixed-width tokens.

Feature groups:

1. **State-goal geometry**
   - normalized node coordinate
   - normalized goal coordinate
   - relative vector to goal
   - Euclidean distance to goal
   - line-of-sight-to-goal flag
   - corridor blockage/obstacle interaction indicators

2. **Nearest obstacle tokens**
   - relative obstacle centers
   - obstacle radius or rectangle half-width/half-height summary
   - clearance-like features

3. **Raycast tokens**
   - fixed set of radial distance probes from the current point
   - helps represent local visibility and obstacle density

4. **Task/world descriptor token**
   - side length
   - obstacle count
   - mean obstacle radius/size
   - obstacle density/free-space proxy
   - narrow-passage indicator
   - OOD flag for evaluation metadata

The model therefore sees both local geometry and global task context.

---

## 4. Model setup

Both backbones are included because the professor was specifically interested in ONLSTM performance, even though HRM was the stronger primary backbone in the previous discrete experiment.

Default backbones:

| Backbone | Role | LoRA rank |
|---|---|---:|
| HRM-style recurrent block | Primary decision backbone | 8 |
| ONLSTM-style sequence encoder | Secondary/professor-interest comparison | 24 |

The default hidden sizes are intentionally moderate because PRM construction and collision checks dominate local runtime. They can be increased later.

---

## 5. Experiment ladder

## C1: continuous PRM sanity baseline

**File:** `continuous_prm_c1_baseline.py`

**Runner equivalent:** `python continuous_prm_stage_runner.py --stage c1 ...`

C1 validates the continuous benchmark before any learning.

Methods:

| Method | Meaning |
|---|---|
| `euclidean` | A* with straight-line distance to goal. |
| `dijkstra` / zero heuristic equivalent | A* with zero heuristic; used as a control for graph search. |

The staged implementation logs Euclidean as the primary baseline. The zero-heuristic control is retained in the earlier standalone C1 implementation and can be re-added to the staged runner if desired; for C1 acceptance, Euclidean vs search budget is the main diagnostic.

C1 questions:

1. Are roadmaps connected often enough?
2. Does Euclidean A* solve easy/open cases with low expansions?
3. Are hard suites hard because of geometry rather than implementation bugs?
4. Are path-cost ratios sensible when a solution is found?

Outputs:

```text
results/continuous_prm_c1_raw.csv
results/continuous_prm_c1_summary.csv
results/continuous_prm_c1_summary.json
figures/c1/success__<suite>.png
```

Smoke:

```bash
python continuous_prm_c1_baseline.py \
  --smoke-test \
  --out-dir runs/smoke_c1 \
  --cpu
```

Serious run:

```bash
python continuous_prm_c1_baseline.py \
  --out-dir runs/continuous_prm_c1_baseline \
  --eval-worlds 80 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --budgets 100,200,500,1000
```

Acceptance criteria:

- Roadmap generation does not silently fail.
- At least the open/clutter suites have healthy valid-world counts.
- Euclidean success curves are monotone or near-monotone with budget.
- Cost ratios are finite for solved worlds.

---

## C2: pooled continuous average-base heuristic

**File:** `continuous_prm_c2_avgbase.py`

**Runner equivalent:** `python continuous_prm_stage_runner.py --stage c2 ...`

C2 trains one pooled average-base residual model over the balanced union of the four continuous anchor tasks.

Methods:

| Method | Meaning |
|---|---|
| `euclidean` | Base geometric heuristic. |
| `avgbase__hrm` | HRM average residual model. |
| `avgbase__onlstm` | ONLSTM average residual model. |

C2 questions:

1. Does a learned average detour residual improve over Euclidean?
2. Is HRM still the stronger primary backbone in continuous space?
3. Does ONLSTM recover or remain weaker in this geometry setting?
4. Does the model help in clutter/narrow-passage suites without hurting open maps?

Outputs:

```text
datasets/<task>_train.npz
datasets/<task>_train.json
checkpoints/avgbase__hrm.pt
checkpoints/avgbase__onlstm.pt
logs/train_avgbase__<backbone>.json
results/continuous_prm_c2_raw.csv
results/continuous_prm_c2_summary.csv
results/continuous_prm_c2_summary.json
figures/c2/success__<suite>.png
```

Smoke:

```bash
python continuous_prm_c2_avgbase.py \
  --smoke-test \
  --out-dir runs/smoke_c2 \
  --cpu
```

Serious run:

```bash
python continuous_prm_c2_avgbase.py \
  --out-dir runs/continuous_prm_c2_avgbase \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --backbones hrm,onlstm \
  --budgets 100,200,500,1000
```

Acceptance criteria:

- No nonfinite predictions or losses.
- HRM/ONLSTM checkpoints are saved and reloadable.
- Average-base curves appear in every eval suite.
- Learned heuristics do not catastrophically reduce success in open maps.

---

## C3: bounded residual task-LoRA experts

**File:** `continuous_prm_c3_residual_tasklora_experts.py`

**Runner equivalent:** `python continuous_prm_stage_runner.py --stage c3 ...`

C3 is the direct continuous analogue of the successful corrected discrete residual LoRA setup.

Procedure:

1. Train pooled average-base over all anchor tasks.
2. Freeze the average-base.
3. For each anchor task, train one bounded residual LoRA expert.
4. Evaluate:
   - Euclidean
   - avgbase
   - every individual task expert
   - oracle expert per suite/budget/world diagnostic

Expert equation:

\[
\text{base} = f_{\theta}(x)
\]

\[
\text{raw expert} = f_{\theta + A_t}(x)
\]

\[
\Delta_t(x) = B_t \tanh\left(\frac{f_{\theta + A_t}(x)-f_{\theta}(x)}{B_t}\right)
\]

\[
\widehat{y}_t(x) = \text{clip}(f_{\theta}(x) + \Delta_t(x), 0, B_{\max})
\]

`B_t` is calibrated from frozen-base residual errors on the corresponding anchor dataset.

C3 questions:

1. Do matched continuous experts beat or match avgbase on their own anchors?
2. Does the oracle expert show useful diversity among experts?
3. Are experts bounded and stable off-task?
4. Do narrow/clutter experts encode different residual functions than open-map experts?

Outputs:

```text
checkpoints/expert__<backbone>__<task>__alpha=<a>.pt
logs/train_expert__<backbone>__<task>__alpha=<a>.json
results/continuous_prm_eval_raw.csv
results/continuous_prm_eval_summary.csv
results/continuous_prm_eval_summary.json
figures/success__<suite>.png
```

Note: C3 currently uses the shared `evaluate_all` helper from `continuous_prm_common.py`, so its result filenames are intentionally generic. C1/C2/C4 use stage-specific filenames in the staged runner.

Smoke:

```bash
python continuous_prm_c3_residual_tasklora_experts.py \
  --smoke-test \
  --out-dir runs/smoke_c3 \
  --cpu
```

Serious run:

```bash
python continuous_prm_c3_residual_tasklora_experts.py \
  --out-dir runs/continuous_prm_c3_residual_tasklora \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --expert-epochs 8 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --budgets 100,200,500,1000
```

Larger alpha sweep:

```bash
python continuous_prm_c3_residual_tasklora_experts.py \
  --out-dir runs/continuous_prm_c3_alpha_sweep \
  --train-worlds 300 \
  --nodes-per-world 220 \
  --eval-worlds 80 \
  --roadmap-nodes 384 \
  --roadmap-k 18 \
  --base-epochs 20 \
  --expert-epochs 12 \
  --backbones hrm,onlstm \
  --lora-alphas 0.5,1.0,1.5,2.0 \
  --budgets 100,200,500,1000,2000
```

Acceptance criteria:

- Expert corrections remain bounded.
- No NaN/Inf losses, predictions, or heuristic arrays.
- Matched experts are not systematically worse than avgbase.
- Oracle expert provides a meaningful upper-bound gap over avgbase.
- Dynamic off-task explosion from the earlier discrete task-LoRA run does not reappear.

---

## C4: descriptor-based nearest/RBF residual expert mixture

**File:** `continuous_prm_c4_rbf_mixture.py`

**Runner equivalent:** `python continuous_prm_stage_runner.py --stage c4 ...`

C4 is the first task-conditioned specialization experiment. It deliberately avoids a learned router at this stage. Instead, it uses task/world descriptors to choose or mix residual experts.

C4 methods:

| Method | Meaning |
|---|---|
| `euclidean` | Straight-line heuristic. |
| `avgbase` | Pooled average residual model. |
| `nearest_tasklora` | Use the expert whose anchor descriptor is nearest to the evaluation world descriptor. |
| `rbf_mix_tasklora` | Prediction-space weighted mixture of expert corrections using RBF weights. |
| `tasklora` | Optional individual expert rows if `--include-expert-matrix` is set. |
| `oracle_tasklora` | Optional diagnostic upper bound if `--include-expert-matrix` is set. |

RBF mixture:

\[
w_i(z) = \frac{\exp(-\frac{1}{2}\lVert (z-z_i)/s \rVert^2 / \sigma^2)}{\sum_j \exp(-\frac{1}{2}\lVert (z-z_j)/s \rVert^2 / \sigma^2)}
\]

where:

- `z` is the current world/task descriptor,
- `z_i` is the estimated descriptor centroid for anchor expert `i`,
- `s` is the empirical descriptor scale across anchors,
- `sigma` is controlled by `--rbf-sigma`.

The mixture is performed in prediction/correction space, not LoRA parameter space:

\[
\Delta_{mix}(x) = \sum_i w_i(z) \Delta_i(x)
\]

\[
\widehat{y}_{mix}(x) = \text{clip}(f_{base}(x) + \Delta_{mix}(x), 0, B_{\max})
\]

This is intentionally conservative and debuggable. Weight-space LoRA interpolation can be explored later if prediction-space mixing is useful.

C4 questions:

1. Does descriptor-based nearest expert improve over avgbase?
2. Does smooth RBF mixing improve over nearest expert?
3. Are interpolation suites helped more than OOD suites?
4. Do descriptor weights behave sensibly for open, clutter, narrow, large, and rectangle-shift worlds?

Outputs:

```text
results/c4_anchor_descriptor_refs.json
results/continuous_prm_c4_raw.csv
results/continuous_prm_c4_summary.csv
results/continuous_prm_c4_summary.json
figures/c4/success__<suite>.png
```
Smoke:

```bash
python continuous_prm_c4_rbf_mixture.py \
  --smoke-test \
  --out-dir runs/smoke_c4 \
  --cpu
```

C4 eval from existing C3 assets:

```bash
python continuous_prm_c4_rbf_mixture.py \
  --mode eval \
  --out-dir runs/continuous_prm_c3_residual_tasklora \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --rbf-sigma 1.0 \
  --budgets 100,200,500,1000
```

C4 full train/eval:

```bash
python continuous_prm_c4_rbf_mixture.py \
  --out-dir runs/continuous_prm_c4_rbf_mixture \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --expert-epochs 8 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --rbf-sigma 1.0 \
  --budgets 100,200,500,1000
```

C4 with individual expert matrix and oracle diagnostic:

```bash
python continuous_prm_c4_rbf_mixture.py \
  --mode eval \
  --out-dir runs/continuous_prm_c3_residual_tasklora \
  --include-expert-matrix \
  --rbf-sigma 1.0
```

Acceptance criteria:

- `nearest_tasklora` and `rbf_mix_tasklora` both appear in summary outputs.
- Mixture weights are logged in raw rows.
- RBF mixture does not produce nonfinite heuristics.
- RBF or nearest improves at least some clutter/narrow/scale slices without damaging open maps.

---

## 6. Full ladder run

The stage runner can execute the whole ladder in one output directory:

```bash
python continuous_prm_stage_runner.py \
  --stage full \
  --out-dir runs/continuous_prm_full_ladder \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --expert-epochs 8 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --rbf-sigma 1.0 \
  --budgets 100,200,500,1000
```

This shares datasets and checkpoints across C2-C4.

---

## 7. Metrics and diagnostics

Primary planning metrics:

| Metric | Meaning |
|---|---|
| `success_rate` | Fraction of eval worlds solved within budget. |
| `mean_cost_ratio` | Found path cost divided by Dijkstra/oracle graph path cost. |
| `mean_expansions` | Average expansions for method/budget/suite. |
| `mean_closed` | Number of closed nodes. |
| `heuristic_max/min` | Sanity range of heuristic values. |
| `nonfinite_heuristic` | Count/flag for NaN/Inf heuristic values. |

Learning/residual diagnostics:

| Metric | Meaning |
|---|---|
| `delta_mean` | Mean predicted normalized detour residual. |
| `correction_abs_mean` | Mean absolute expert correction around avgbase. |
| `mixture_weights` | JSON-encoded C4 selector weights. |
| `rbf_sigma` | Selector bandwidth used in C4. |
| `rbf_topk` | Top-k truncation used in C4. |

Roadmap diagnostics:

| Metric | Meaning |
|---|---|
| `roadmap_nodes` | Number of PRM nodes including start/goal. |
| `roadmap_edges` | Undirected edge count. |
| `obstacle_count` | Obstacle count in eval world. |
| `side_len` | Continuous map scale. |
| `oracle_cost` | Dijkstra cost from start to goal on PRM. |

---

## 8. Implementation reasoning

### Why PRM instead of a pure continuous planner?

The previous work measured A* heuristic quality. A* requires a graph or graph-like successor structure. PRM lets us keep continuous geometry while retaining exact graph labels and comparable search metrics.

### Why normalize residuals by side length?

The eval suite includes larger worlds. Without normalization, the model would need to learn scale-dependent target magnitudes. Normalization makes residuals more comparable across side lengths and lets C4 mix anchor experts more safely.

### Why bounded residuals?

The earlier discrete task-LoRA attempt produced off-task explosions. Bounded corrections preserve the corrected residual-LoRA interpretation:

\[
\text{avgbase} + \text{bounded task correction}
\]

This protects A* from pathological heuristic magnitudes and makes diagnostics interpretable.

### Why prediction-space mixture in C4?

Parameter-space LoRA interpolation is more elegant but harder to debug. Prediction-space mixing is explicit:

1. Load each expert.
2. Compute each expert correction.
3. Weight corrections by descriptor similarity.
4. Clip final residual.

If this works, weight-space interpolation or a learned router becomes justified.

### Why keep ONLSTM?

HRM remains the primary decision backbone based on previous results, but ONLSTM is retained because the professor is interested in its behavior. Continuous geometry may also stress sequence encoders differently from discrete grid metadata, so it is worth measuring rather than assuming it will fail.

---

## 9. Known limitations

1. **The planner is continuous-coordinate but graph-discretized.** It is a PRM bridge, not a full analytic continuous planner.
2. **Roadmap label quality depends on PRM density.** Sparse roadmaps can create noisy detour labels.
3. **Obstacle encoding is handcrafted.** A richer set/attention encoder over obstacles may outperform nearest-obstacle + ray features.
4. **Dynamic obstacles are not included yet.** That should be a later C5/C6 direction.
5. **C4 selector is non-learned.** That is deliberate for interpretability and debugging.

---

## 10. Future experiments after C4

### C5: learned descriptor router

Train a small MLP to map task/world descriptor to expert weights. Compare against nearest and RBF.

### C6: richer obstacle encoder

Replace handcrafted nearest-obstacle/raycast features with a set transformer or cross-attention encoder over obstacle primitives.

### C7: roadmap density robustness

Train at one PRM density and evaluate at several other densities to see whether the heuristic learns geometry or graph artifacts.

### C8: continuous dynamic obstacles

Add time as a state coordinate and moving circular obstacles. The graph becomes a time-expanded PRM or kinodynamic roadmap.

### C9: parameter-space LoRA interpolation

Interpolate LoRA adapter weights directly as a function of task descriptor and compare against C4 prediction-space residual mixing.

### C10: learned sampling + learned heuristic

Use the heuristic model to bias PRM node sampling or edge expansion order, not only A* node priority.

---

## 11. Smoke-test status

The staged files were syntax checked. CPU smoke tests were run successfully for:

- C1 baseline
- C2 avgbase, using the same tiny two-anchor smoke configuration as the standalone C2 test
- C3 residual task-LoRA experts
- C4 nearest/RBF selector

One practical note: avoid shipping `__pycache__/` files in bundles. A stale cache created during earlier local patching caused an import issue in the sandbox. The clean bundle excludes caches.

---

## 12. Suggested first serious sequence

Run these in order:

```bash
# C1: benchmark validation
python continuous_prm_c1_baseline.py \
  --out-dir runs/continuous_prm_c1_baseline \
  --eval-worlds 80 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --budgets 100,200,500,1000

# C2: average-base check
python continuous_prm_c2_avgbase.py \
  --out-dir runs/continuous_prm_c2_avgbase \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --backbones hrm,onlstm \
  --budgets 100,200,500,1000

# C3: residual experts
python continuous_prm_c3_residual_tasklora_experts.py \
  --out-dir runs/continuous_prm_c3_residual_tasklora \
  --train-worlds 120 \
  --nodes-per-world 160 \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --base-epochs 10 \
  --expert-epochs 8 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --budgets 100,200,500,1000

# C4: descriptor-conditioned nearest/RBF mixture using the C3 output directory
python continuous_prm_c4_rbf_mixture.py \
  --mode eval \
  --out-dir runs/continuous_prm_c3_residual_tasklora \
  --eval-worlds 40 \
  --roadmap-nodes 256 \
  --roadmap-k 14 \
  --backbones hrm,onlstm \
  --lora-alphas 1.0 \
  --rbf-sigma 1.0 \
  --budgets 100,200,500,1000
```
