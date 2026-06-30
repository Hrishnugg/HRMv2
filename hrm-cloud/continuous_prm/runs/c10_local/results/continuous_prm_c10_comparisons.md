# C10 Interpolation — Pre-registered Comparisons

Matched A* expansion-ratio vs euclid (median, 95% CI) at the per-target binding budget. Lower = fewer expansions.
Key contrasts: rbf_wmerge vs zero_shot (interpolation helps?), vs nearest/uniform (descriptor-weighting value), vs rbf_pmix (weight-space vs prediction-space).
Binding budgets: C10_maze_tgt=150, C10_rooms_t25=150, C10_rooms_t35=150.

## C10_maze_tgt / hrm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.529 [0.436,0.628] (succ 1.00, n24)|
|nearest|0.513 [0.462,0.597] (succ 1.00, n24)|
|uniform_wmerge|0.500 [0.438,0.600] (succ 1.00, n24)|
|rbf_wmerge|0.508 [0.475,0.617] (succ 0.97, n24)|
|rbf_pmix|0.519 [0.475,0.605] (succ 0.97, n24)|
- rbf_wmerge vs zero_shot: Δ=-0.022 (rbf_wmerge better)
- rbf_wmerge vs nearest: Δ=-0.005 (rbf_wmerge better)
- rbf_wmerge vs uniform_wmerge: Δ=+0.008 (uniform_wmerge better)
- rbf_wmerge vs rbf_pmix: Δ=-0.012 (rbf_wmerge better)

## C10_maze_tgt / onlstm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.455 [0.407,0.526] (succ 1.00, n24)|
|nearest|0.442 [0.378,0.504] (succ 0.97, n24)|
|uniform_wmerge|0.451 [0.402,0.585] (succ 0.97, n24)|
|rbf_wmerge|0.443 [0.390,0.556] (succ 0.97, n24)|
|rbf_pmix|0.445 [0.385,0.487] (succ 0.97, n24)|
- rbf_wmerge vs zero_shot: Δ=-0.012 (rbf_wmerge better)
- rbf_wmerge vs nearest: Δ=+0.000 (nearest better)
- rbf_wmerge vs uniform_wmerge: Δ=-0.008 (rbf_wmerge better)
- rbf_wmerge vs rbf_pmix: Δ=-0.002 (rbf_wmerge better)

## C10_rooms_t25 / hrm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.790 [0.655,0.863] (succ 1.00, n17)|
|nearest|0.766 [0.676,0.829] (succ 1.00, n17)|
|uniform_wmerge|0.781 [0.714,0.851] (succ 0.97, n17)|
|rbf_wmerge|0.766 [0.669,0.824] (succ 1.00, n17)|
|rbf_pmix|0.766 [0.669,0.831] (succ 1.00, n17)|
- rbf_wmerge vs zero_shot: Δ=-0.024 (rbf_wmerge better)
- rbf_wmerge vs nearest: Δ=-0.000 (rbf_wmerge better)
- rbf_wmerge vs uniform_wmerge: Δ=-0.015 (rbf_wmerge better)
- rbf_wmerge vs rbf_pmix: Δ=-0.000 (rbf_wmerge better)

## C10_rooms_t25 / onlstm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.610 [0.519,0.759] (succ 1.00, n17)|
|nearest|0.727 [0.685,0.782] (succ 0.97, n17)|
|uniform_wmerge|0.638 [0.574,0.699] (succ 0.97, n17)|
|rbf_wmerge|0.692 [0.660,0.789] (succ 0.97, n17)|
|rbf_pmix|0.692 [0.660,0.789] (succ 0.97, n17)|
- rbf_wmerge vs zero_shot: Δ=+0.082 (zero_shot better)
- rbf_wmerge vs nearest: Δ=-0.035 (rbf_wmerge better)
- rbf_wmerge vs uniform_wmerge: Δ=+0.054 (uniform_wmerge better)
- rbf_wmerge vs rbf_pmix: Δ=+0.000 (tie)

## C10_rooms_t35 / hrm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.748 [0.671,0.849] (succ 0.93, n18)|
|nearest|0.746 [0.655,0.833] (succ 0.97, n19)|
|uniform_wmerge|0.761 [0.678,0.872] (succ 0.97, n19)|
|rbf_wmerge|0.761 [0.669,0.833] (succ 0.97, n19)|
|rbf_pmix|0.768 [0.662,0.833] (succ 0.97, n19)|
- rbf_wmerge vs zero_shot: Δ=+0.012 (zero_shot better)
- rbf_wmerge vs nearest: Δ=+0.014 (nearest better)
- rbf_wmerge vs uniform_wmerge: Δ=+0.000 (tie)
- rbf_wmerge vs rbf_pmix: Δ=-0.007 (rbf_wmerge better)

## C10_rooms_t35 / onlstm
|arm|exp-ratio (median [CI]) |
|---|---|
|zero_shot|0.600 [0.537,0.694] (succ 1.00, n19)|
|nearest|0.808 [0.676,0.873] (succ 1.00, n19)|
|uniform_wmerge|0.610 [0.549,0.690] (succ 1.00, n19)|
|rbf_wmerge|0.716 [0.615,0.831] (succ 1.00, n19)|
|rbf_pmix|0.709 [0.611,0.831] (succ 1.00, n19)|
- rbf_wmerge vs zero_shot: Δ=+0.116 (zero_shot better)
- rbf_wmerge vs nearest: Δ=-0.092 (rbf_wmerge better)
- rbf_wmerge vs uniform_wmerge: Δ=+0.107 (uniform_wmerge better)
- rbf_wmerge vs rbf_pmix: Δ=+0.007 (rbf_pmix better)
