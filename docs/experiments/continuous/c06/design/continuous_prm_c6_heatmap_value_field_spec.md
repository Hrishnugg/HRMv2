# C6 Heatmap / Value-Field Continuous Heuristic Investigation

## Why this is the right next continuous experiment

C5 established that the hard continuous PRM maps are useful: Euclidean A* lands
in the desired 50-70% success band at budget 144 on `C_hard_maze` and
`C_hard_maze_dense`, while ONLSTM produces large, statistically significant
success and expansion improvements.

It also exposed a failure mode in the current HRM formulation. The C5 model
learns a per-roadmap-node residual:

```text
h(node) = euclidean(node, goal) + learned_nonnegative_residual(node)
```

For HRM, that residual collapsed to a constant cap. A constant additive term
does not change A* ordering, so HRM exactly matches Euclidean even when the map
is hard enough for ONLSTM to help. That points to an objective/representation
problem, not another map-difficulty or residual-cap tuning problem.

The professor's heatmap idea changes the learned object from a scalar residual
at each independently featurized node to a goal-conditioned spatial value field:

```text
H_theta(x, y | map, goal) ~= cost-to-go or search priority
```

This is a better match for continuous planning because bottlenecks, gate order,
dead ends, and topology are field-level structure.

## Literature anchors

Primary references that support this direction:

- Takahashi et al., "Learning Heuristic Functions for Mobile Robot Path
  Planning Using Deep Neural Networks", ICAPS 2019.
  - Learns a heuristic function over the whole environment from an occupancy map
    and goal tensor.
  - Uses a U-Net-like image-to-image architecture.
  - Explicitly shows that MSE/MAE to optimal heuristic value can be misaligned
    with A* search cost, motivating search-aware losses.
- Yonetani et al., "Path Planning using Neural A* Search", ICML 2021.
  - Encodes the planning instance into a guidance map, then uses differentiable
    A* to learn search-efficient maps from expert paths.
  - This is directly adjacent to the heatmap framing.
- Tamar et al., "Value Iteration Networks", NeurIPS 2016.
  - Embeds a differentiable value-iteration-style planning computation in a
    neural model and evaluates on path-planning domains.
  - Useful as a conceptual baseline for iterative value-field refinement.
- Veerapaneni et al., "Learning Local Heuristics for Search-Based Navigation
  Planning", ICAPS 2023.
  - Shows that learning local heuristics can reduce node expansions
    substantially while preserving bounded-suboptimal search framing.
  - Important caution: a smaller/local objective may generalize better than a
    single global cost-to-go regressor.
- Bhardwaj et al., "Learning Heuristic Search via Imitation", CoRL 2017.
  - Trains heuristic policies to minimize search effort by imitating
    clairvoyant oracle expansion decisions.
  - Supports adding ranking/imitation losses instead of relying only on value
    regression.
- Qureshi et al., "Motion Planning Networks", T-RO 2020.
  - Uses learned motion-planning priors together with classical sampling-based
    planning to retain guarantees.
  - Useful as a neighboring baseline family, though it is less directly aligned
    with A* heatmaps.
- Valero-Gomez et al., "Fast Marching Methods in Path Planning".
  - Potential-field / fast-marching planners provide a non-neural value-field
    baseline and warn about local-minimum issues in naive potential fields.

## Proposed experiment

Add a new script rather than mutating C5:

```text
hrm-cloud/continuous_prm/continuous_prm_c6_heatmap_value_field.py
```

and a new Modal entrypoint:

```text
continuous_prm_modal.py::run_c6_heatmap
```

C6 should reuse the C5 hard-map generators and PRM evaluation setup, but replace
the learned target and model interface.

## Initial oracle feasibility probe

Before implementing learned models, a quick local oracle-only probe checked
whether a 64x64 rasterized cost-to-go field has leverage when interpolated back
onto C5 PRM nodes.

Probe setup:

- no model training;
- C5 hard world generator installed at runtime;
- 12 valid worlds per suite;
- PRM: 192 nodes, k=7;
- grid oracle: 8-neighbor Dijkstra over free raster cells;
- heuristic: `max(euclidean, interpolated_grid_distance)`;
- budgets: 128, 136, 144, 152.

Results:

| Suite | Budget | Euclidean | Grid Oracle | Delta | Expansion Delta | Gains | Losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C_hard_maze` | 128 | 0.167 | 0.917 | +0.750 | -28.8 | 9 | 0 |
| `C_hard_maze` | 136 | 0.333 | 0.917 | +0.583 | -34.0 | 7 | 0 |
| `C_hard_maze` | 144 | 0.667 | 1.000 | +0.333 | -38.1 | 4 | 0 |
| `C_hard_maze` | 152 | 0.917 | 1.000 | +0.083 | -40.2 | 1 | 0 |
| `C_hard_maze_dense` | 128 | 0.000 | 0.833 | +0.833 | -16.9 | 10 | 0 |
| `C_hard_maze_dense` | 136 | 0.000 | 0.917 | +0.917 | -23.8 | 11 | 0 |
| `C_hard_maze_dense` | 144 | 0.250 | 1.000 | +0.750 | -30.8 | 9 | 0 |
| `C_hard_maze_dense` | 152 | 0.417 | 1.000 | +0.583 | -36.6 | 7 | 0 |
| `C_hard_rooms` | 128 | 0.000 | 0.917 | +0.917 | -21.1 | 11 | 0 |
| `C_hard_rooms` | 136 | 0.167 | 1.000 | +0.833 | -27.8 | 10 | 0 |
| `C_hard_rooms` | 144 | 0.583 | 1.000 | +0.417 | -32.4 | 5 | 0 |
| `C_hard_rooms` | 152 | 0.833 | 1.000 | +0.167 | -34.8 | 2 | 0 |

Interpretation:

- The heatmap/value-field representation passes the first feasibility gate.
- The raster oracle is not merely changing path quality; it consistently reduces
  expansions and wins paired success cases in hard budget bands.
- This is a small probe and should not be treated as a research claim, but it is
  enough to justify implementing C6.

## Dataset design

For each generated C5-style world:

1. Rasterize the continuous world to a fixed grid, initially 64x64.
2. Store input channels:
   - obstacle occupancy
   - signed or clipped clearance
   - goal Gaussian / goal one-hot
   - normalized x coordinate
   - normalized y coordinate
   - optional start Gaussian for start-conditioned variants
3. Compute oracle cost-to-go over free grid cells using 8-neighbor Dijkstra.
   - This is the cheap first oracle.
   - Later we can replace or augment it with fast marching for a smoother
     continuous-field oracle.
4. Keep the PRM roadmap for evaluation.
   - During evaluation, bilinearly interpolate the predicted heatmap at PRM node
     coordinates.
   - A* still runs on the same PRM graph as C5, so success/expansion metrics are
     comparable.

Recommended initial targets:

```text
D(x, y) = grid oracle cost-to-go
E(x, y) = Euclidean distance to goal
R(x, y) = max(0, D(x, y) - E(x, y))
```

Train the model to predict either `D` directly or `R`, but evaluate with:

```text
h(node) = max(E(node), predicted_D(node))
```

or:

```text
h(node) = E(node) + relu(predicted_R(node))
```

The residual version is safer for comparison to C5; the direct value version is
closer to the professor's heatmap framing.

## Model families

Run these in order.

### 1. Non-neural baselines

- Euclidean PRM A*.
- Oracle grid-distance heatmap interpolated at PRM nodes.
- Optional fast-marching heatmap if grid Dijkstra is too stair-stepped.

The oracle heatmap gives an upper bound for whether field guidance can help on
the hard maps before spending GPU time.

### 2. U-Net heatmap baseline

A small U-Net should be the first learned model. It is the cleanest baseline
because the ICAPS 2019 learned-heuristic paper used a U-Net-like map-to-map
architecture.

This answers: "Does the heatmap framing itself work?"

### 3. ONLSTM heatmap baseline

Flatten map rows or patches into a deterministic sequence and predict a field
through a deconvolution head. This is not necessarily the best architecture, but
it gives continuity with the C5 ONLSTM win.

This answers: "Does ONLSTM still dominate when the target is a field rather
than independent node residuals?"

### 4. HRM field refiner

Use a CNN stem to produce grid or patch tokens, then let HRM recurrently refine
latent map state before a decoder outputs the heatmap.

Initial practical version:

```text
input grid -> CNN stem -> 8x8 or 16x16 patch tokens -> HRM core -> decoder -> heatmap
```

This gives HRM a real hierarchical spatial refinement job. The key test is
whether it learns nonconstant, topology-aware corrections around gates and
rooms.

## Losses

Do not rely on plain MSE alone. The literature and our C5 failure both suggest
that value accuracy is not the same as useful search ordering.

Use a combined objective:

```text
L = L_value + lambda_rank * L_rank + lambda_path * L_path + lambda_consistency * L_consistency
```

Recommended components:

- `L_value`: Smooth L1 on free-cell `D` or `R`, with obstacle cells masked out.
- `L_rank`: pairwise ranking loss on sampled free-cell pairs:
  - if oracle says `D(a) < D(b)`, predicted priority should preserve that order.
  - oversample pairs near walls/gates and along Euclidean-failure regions.
- `L_path`: weighted BCE or focal loss on a soft shortest-path corridor mask.
  - Helps the model distinguish "good corridor" from "nearby dead space".
- `L_consistency`: soft edge consistency on neighboring free cells:
  - penalize large violations of `h(u) <= cost(u, v) + h(v)`.
  - Keep as a small regularizer; strict admissibility is not the goal.

Search-aware variant for phase 2:

- Run Euclidean A* and oracle-guided A* on training worlds.
- Sample "oracle before baseline" frontier pairs.
- Train the heatmap to prefer oracle-expanded nodes over misleading baseline
  nodes.

This is closer to SAIL / Neural A* and should be used if value/ranking loss is
not enough.

## Evaluation

Use the exact C5 hard suites and budget band first:

```text
train: C_hard_maze,C_hard_rooms
eval:  C_hard_maze,C_hard_maze_dense,C_hard_rooms
budgets: 128,136,144,152,168
roadmap: 192 nodes, k=7
```

For each world and method:

1. Generate PRM.
2. Produce a heatmap once per world/goal.
3. Interpolate heatmap values at all PRM nodes.
4. Run A* under each expansion budget.
5. Record:
   - success
   - expansions
   - path cost ratio
   - heuristic min/max/mean/std
   - Spearman correlation between predicted and oracle PRM cost-to-go
   - nonconstant-field diagnostics

Claim filter should match C5:

- Euclidean success in 0.50-0.70 target band.
- Learned method improves absolute success by at least +0.10.
- Paired McNemar p-value with BH-corrected q <= 0.05.
- Prefer expansion reduction among solved/all episodes.

## Smoke run

Local CPU/GPU smoke target:

```bash
python continuous_prm_c6_heatmap_value_field.py \
  --smoke-test \
  --out-dir runs/smoke_c6_heatmap \
  --grid-size 48 \
  --train-worlds 8 \
  --eval-worlds 4 \
  --epochs 2 \
  --models unet \
  --budgets 128,144 \
  --cpu
```

Smoke success criteria:

- Dataset collection completes.
- Oracle grid-distance heatmap improves or at least meaningfully changes A*
  expansions versus Euclidean.
- Learned U-Net produces finite, nonconstant heatmaps.
- Evaluation writes raw and summary CSVs.

Do not interpret smoke success rates as research evidence.

## Modal full run

First full C6 run:

```bash
python -m modal run continuous_prm_modal.py::run_c6_heatmap \
  --run-name continuous_prm_c6_heatmap_r1 \
  --train-worlds 160 \
  --eval-worlds 80 \
  --grid-size 64 \
  --roadmap-nodes 192 \
  --roadmap-k 7 \
  --epochs 16 \
  --models unet,onlstm,hrm \
  --train-tasks C_hard_maze,C_hard_rooms \
  --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms \
  --budgets 128,136,144,152,168 \
  --dataset-shard-worlds 10 \
  --eval-shard-worlds 1
```

Parallelization should follow the C5 Modal pattern:

- dataset shards by task and world range;
- model training by model family;
- eval shards by suite and world;
- merge raw CSVs and run significance analysis after all shards complete.

## Decision gates

Gate 1: oracle heatmap.

- If oracle grid-distance heatmap does not beat Euclidean on C5 hard PRMs, the
  raster oracle is misaligned with PRM evaluation. Fix raster resolution,
  interpolation, or use PRM-derived value supervision before training models.

Gate 2: U-Net.

- If U-Net does not learn a useful nonconstant heatmap, the problem is the
  target/loss/data pipeline, not HRM.

Gate 3: HRM.

- Only compare HRM seriously after the U-Net baseline works.
- HRM must show nonconstant field diagnostics and positive paired wins.
- If HRM still collapses, ablate:
  - HRM recurrent depth
  - high-level update cadence
  - decoder capacity
  - patch resolution
  - value-only vs ranking-augmented loss

## Expected research value

This experiment is stronger than C5 for the continuous story because it matches
the advisor's proposed framing:

- the learned object is a spatial heuristic field;
- the output is visualizable and diagnosable;
- the objective can be made search-aware;
- HRM has a meaningful hierarchical refinement role;
- the same C5 hard-map significance protocol remains available.

The main risk is that a heatmap becomes just a smoothed distance transform. The
ranking/path losses and gate-heavy C5 maps are specifically intended to prevent
that.
