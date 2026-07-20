# ON-LSTM vs HRM — Preset M+ (v2): Multi-step + Map Patch + Metrics + Ablations

This experiment is the *next iteration* of the **Preset M** ON‑LSTM vs HRM comparison. It keeps the same high-level training pipeline (synthetic data generation → supervised dynamics prediction → A* augmentation for planning), but implements four upgrades:

1. **Richer evaluation metrics** (beyond success rate)
2. **Multi-step rollout training** (+ scheduled sampling)
3. **Local map context** as an additional model input
4. **Ablation suites** (fixed vs random maps, gates on/off, horizon sweep)

The goal is to create a tougher, more diagnostic benchmark that (a) makes prediction errors matter more for planning and (b) exposes differences between architectural inductive biases.

---

## 1) Task definition

We evaluate **dynamic planning** on a grid world with:

- Static walls (maze-like rooms + corridors)
- A start cell `(0,0)` and a goal cell `(N-1,N-1)`
- Multiple **dynamic obstacles** moving according to rule-based behaviors

At each agent step, we:

1. Observe the last `H` obstacle positions (history window).
2. Use a learned predictor (ON‑LSTM or HRM) to roll out future obstacle positions for planning horizon `T`.
   - Rollout is computed *statefully* from the history window (autoregressive generation), matching the multi-step training objective.
3. Run **space-time A\*** in `(row, col, time)` to select the next agent move while avoiding predicted obstacle collisions.
4. Execute that move, advance obstacle physics by one step, and repeat.

---

## 2) What changed vs v1 (Preset M)

### 2.1 Local map patch input (recommendation #3)
In v1, the predictor received only `(x,y)` history. In v2, each training sample includes an **11×11 binary occupancy patch** centered on the obstacle’s *last observed* position.

- Patch values: `1 = wall`, `0 = free`
- Patch is encoded with a tiny CNN → **patch embedding**
- The embedding is concatenated to `(x,y)` at every timestep before entering the recurrent model.

**Motivation:** in maze-like environments, obstacle movement is highly constrained by *local topology*. This context should reduce ambiguity at intersections and corridor endpoints.

---

### 2.2 Multi-step rollout loss + scheduled sampling (recommendation #2)
In v1, the predictor was trained with a **one-step MSE** target.

In v2, the predictor is trained with a **k-step autoregressive rollout loss**:

- For each sample, we supervise the next `k` obstacle deltas.
- During training, we roll forward `k` steps and accumulate weighted MSE.
- **Implementation detail (performance + consistency):** rollouts are computed *statefully* (warm-start on the `H` history, then step forward autoregressively) so rollout cost scales as **O(H+k)** instead of **O(H·k)**.
- We apply **scheduled sampling**: early epochs use more teacher forcing; later epochs rely more on the model’s own predictions.

Default settings:
- `k = 5`
- Loss weights decay geometrically (e.g., `0.9^t`)

**Motivation:** one-step accuracy often looks great, but planning performance depends on *error compounding* over horizon. Multi-step losses better align training with downstream usage.

---

### 2.3 Expanded evaluation metrics (recommendation #1)
Success rate alone hides important failure modes. v2 reports:

- **Success rate**
- **Failure breakdown**
  - `collision` (dynamic obstacle collision)
  - `static_collision` (wall collision)
  - `timeout` (max steps reached)
- **A* planning cost proxy**
  - total/avg A* node expansions per episode
- **Prediction metrics**
  - **one-step prediction MSE** (overall + per obstacle class)
  - **k-step rollout MSE** (overall + per obstacle class, optional)

Obstacle classes:
- `gate`
- `patroller`
- `drifter`

---

### 2.4 Evaluation ablation suites (recommendation #4)
We evaluate across multiple suites:

1. **random_h20**  
   Random maps, horizon 20, gates on (main benchmark)

2. **random_h10**  
   Same as above, but horizon 10 (planner horizon sweep)

3. **fixedmap_h20**  
   A single fixed static map reused for all episodes (dynamics still vary)

4. **gatesoff_random_h20**  
   Gates removed; obstacle count kept constant by reallocating to drifters

---

## 3) Environment details (Preset M+)

- Grid size: `32×32`
- Rooms + corridors map generator
- Dynamic obstacles (default):
  - `2` gates (periodic open/close behavior, optional alcoves)
  - `4` patrollers (looping along precomputed BFS routes through room centers)
  - `6` drifters (heading-based motion with mode-switching: left/right/random)

Episodes:
- Physics steps per data episode: `70`
- Max agent steps at evaluation: `128`

---

## 4) Dataset generation

For each data episode:

1. Reset environment with random seed
2. Run obstacle physics for 70 steps
3. After warmup, for each timestep we create training samples for **each obstacle**:

A single sample contains:
- `pos_hist`: last `H=20` positions (normalized by grid size)
- `patch`: `11×11` local occupancy patch at the last observed position (uint8)
- `true_deltas`: the next `k=5` deltas (normalized)

This yields tens of millions of samples at the default scale (same order as v1).

---

## 5) Models

We keep the same parameter tiers as before, but now each model is wrapped with a patch encoder.

### ON‑LSTM tiers
- `onlstm_300k`
- `onlstm_1m`
- `onlstm_3m`
- `onlstm_10m`

### HRM tiers
- `hrm_302k`
- `hrm_3m`
- `hrm_10m` (4×GPU DDP)

**Note:** Adding the patch encoder introduces a small parameter increase, but the tier ordering and magnitude remain comparable.

---

## 6) Training setup

- Optimizer: AdamW
- LR schedules: OneCycleLR
- Epochs: same as v1 (30 for smaller ON‑LSTM tiers, 40 for larger models/HRM)
- Checkpointing: every 5 epochs (resumable)

---

## 7) Running the experiment

### Script
`hrm-cloud/onlstm_hrm_comparison_presetm_v2.py`

Run:
```bash
modal run hrm-cloud/onlstm_hrm_comparison_presetm_v2.py
```

Artifacts stored under:
- `/data/onlstm_comparison_presetm_v2/merged.pt`
- `/data/onlstm_comparison_presetm_v2/models/*.pt`
- `/data/onlstm_comparison_presetm_v2/results.json`

---

## 8) Results template

After completion, summarize for each suite:

| Suite | Model | Params | Success Rate | One-step MSE | Rollout MSE (k) | Avg A* Expansions |
|------:|-------|-------:|-------------:|-------------:|----------------:|------------------:|
| random_h20 | ... | ... | ... | ... | ... | ... |

Also compare:
- HRM vs ON‑LSTM **within the same parameter tier**
- Which obstacle classes dominate prediction error
- How horizon and gates change relative performance

---

## 9) Notes / expected outcomes

- If ON‑LSTM’s inductive bias helps with *hierarchical / structured motion* (e.g., gates/patrol loops), it should reduce **rollout drift** more than HRM at equal parameter count.
- HRM may show strengths in *short-range stability* (low one-step error) but may or may not maintain long-horizon consistency depending on gating dynamics.
- The ablations help diagnose whether gains come from:
  - better modeling of maze topology (fixed map vs random)
  - better handling of discontinuities (gates on/off)
  - robustness to compounding error (horizon sweep)

