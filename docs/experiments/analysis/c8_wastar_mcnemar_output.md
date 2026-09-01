# WA* vs blind U-Net: exact McNemar + effort-ratio CIs (confirmation cohort, binding budgets)

| Suite | Discordant (learned-only / WA*-only) | exact p | BH q | Blind/WA* median ratio [CI] (n) | WA*/anchor median ratio [CI] (n) | Subopt WA* [CI] | Subopt learned [CI] |
|---|---|---|---|---|---|---|---|
| Crossing | 0/4 | 1.25e-01 | 3.75e-01 | 0.825 [0.531,1.167] (46) | 0.139 [0.115,0.171] (6) | 1.008 [1.003,1.013] | 1.164 [1.124,1.205] |
| Maze | 1/1 | 1.00e+00 | 1.00e+00 | 0.601 [0.440,0.835] (47) | 0.120 [0.058,0.243] (6) | 1.017 [1.010,1.026] | 1.012 [1.006,1.020] |
| Dense maze | 1/1 | 1.00e+00 | 1.00e+00 | 0.982 [0.823,1.089] (34) | 0.106 [0.084,0.228] (3) | 1.006 [1.003,1.010] | 1.008 [1.003,1.014] |
| Rooms | 1/0 | 1.00e+00 | 1.00e+00 | 0.554 [0.439,0.798] (49) | 0.158 [0.129,0.198] (21) | 1.015 [1.007,1.025] | 1.019 [1.010,1.029] |
| Large rooms | 1/0 | 1.00e+00 | 1.00e+00 | 0.728 [0.536,0.966] (49) | 0.335 [0.300,0.358] (41) | 1.003 [1.001,1.006] | 1.082 [1.060,1.106] |
| Spiral | 15/0 | 6.10e-05 | 3.66e-04 | 0.217 [0.183,0.312] (35) | 0.567 [0.346,0.897] (8) | 1.013 [1.006,1.021] | 1.011 [1.007,1.017] |

Exact two-sided binomial McNemar on discordant maps; BH across the six suites. Ratio CIs: 10k map-resampled bootstraps of the median on jointly solved maps (seed 20260724). Suboptimality: mean arrival / optimal arrival on solved maps with known optimum, bootstrap CI of the mean.

## Matched (jointly solved) path quality

| Suite | joint n | Subopt learned (joint) | Subopt WA* (joint) | Paired diff learned-WA* [CI] |
|---|---|---|---|---|
| Crossing | 46 | 1.164 | 1.008 | +0.156 [+0.116,+0.200] |
| Maze | 47 | 1.013 | 1.017 | -0.005 [-0.016,+0.006] |
| Dense maze | 34 | 1.008 | 1.006 | +0.002 [-0.005,+0.009] |
| Rooms | 49 | 1.019 | 1.015 | +0.004 [-0.008,+0.017] |
| Large rooms | 49 | 1.083 | 1.003 | +0.080 [+0.058,+0.105] |
| Spiral | 35 | 1.014 | 1.013 | +0.001 [-0.007,+0.010] |
