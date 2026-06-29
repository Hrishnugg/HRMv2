# C8 Dynamics Comparison — Pre-registered Comparisons

Binding budget per suite (lowest calibrated-band budget where euclid success >= 0.05, so a degenerate 0%-success edge is skipped; if no band budget qualifies, the highest band budget; else the first budget seen): C_dyn_crossing=150, C_dyn_maze=1800, C_dyn_maze_dense=2500, C_dyn_rooms=1300, C_dyn_rooms_large=600, C_dyn_spiral=2500

_Multiplicity: BH correction is applied ONLY to the success/McNemar grid over learned arms (see `continuous_prm_c8_significance.md`). The p-values in THESE six pre-registered comparisons are UNcorrected; treat the bootstrap 95% CIs as the primary inference._

_Small-n: a p shown as `n/a (n<6)` had too few discordant/matched pairs to trust (counts / median / CI are still reported)._

## 1. Time-aware learned vs euclid-time (expansions + success), per suite

Each time-aware learned arm (field_<bb>/astar, scalar_<bb>/astar) vs `euclid/astar`.

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.349 [0.217, 0.458]|0.031|
|C_dyn_crossing|150|field_unet/astar|20|0.300|1.000|0.700|<0.001|6|0.258 [0.132, 0.769]|0.062|
|C_dyn_crossing|150|scalar_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.308 [0.198, 0.662]|0.031|
|C_dyn_crossing|150|scalar_onlstm/astar|20|0.300|0.700|0.400|0.021|5|0.673 [0.292, 1.705]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.418 [0.201, 0.642]|0.031|
|C_dyn_maze|1800|field_unet/astar|20|0.300|1.000|0.700|<0.001|6|0.064 [0.046, 0.088]|0.031|
|C_dyn_maze|1800|scalar_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.266 [0.196, 0.576]|0.031|
|C_dyn_maze|1800|scalar_onlstm/astar|20|0.300|1.000|0.700|<0.001|6|0.387 [0.250, 0.789]|0.031|
|C_dyn_maze_dense|2500|field_hrm/astar|20|0.050|0.700|0.650|<0.001|1|0.337 [0.337, 0.337]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet/astar|20|0.050|0.750|0.700|<0.001|1|0.278 [0.278, 0.278]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm/astar|20|0.050|0.550|0.500|0.002|1|0.619 [0.619, 0.619]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm/astar|20|0.050|0.450|0.400|0.008|1|0.678 [0.678, 0.678]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.176 [0.099, 0.491]|0.031|
|C_dyn_rooms|1300|field_unet/astar|20|0.300|1.000|0.700|<0.001|6|0.096 [0.038, 0.191]|0.031|
|C_dyn_rooms|1300|scalar_hrm/astar|20|0.300|1.000|0.700|<0.001|6|0.200 [0.064, 0.429]|0.031|
|C_dyn_rooms|1300|scalar_onlstm/astar|20|0.300|1.000|0.700|<0.001|6|0.245 [0.142, 0.414]|0.031|
|C_dyn_rooms_large|600|field_hrm/astar|20|0.750|0.900|0.150|n/a (n<6)|15|0.760 [0.110, 0.832]|0.002|
|C_dyn_rooms_large|600|field_unet/astar|20|0.750|0.900|0.150|n/a (n<6)|14|0.380 [0.132, 0.591]|0.013|
|C_dyn_rooms_large|600|scalar_hrm/astar|20|0.750|0.950|0.200|0.219|14|0.326 [0.207, 0.532]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm/astar|20|0.750|0.650|-0.100|n/a (n<6)|12|0.682 [0.327, 0.899]|0.034|
|C_dyn_spiral|2500|field_hrm/astar|20|0.200|0.900|0.700|<0.001|4|0.046 [0.038, 0.073]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|20|0.200|0.900|0.700|<0.001|4|0.076 [0.055, 0.097]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|20|0.200|0.850|0.650|<0.001|4|0.363 [0.064, 0.614]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|20|0.200|0.850|0.650|<0.001|4|0.345 [0.180, 0.639]|n/a (n<6)|

## 2. Time-aware vs time-blind — THE SPOTLIGHT: does the future window help?

Matched expansion ratio of the time-AWARE arm vs its time-BLIND (W=0) twin (e.g. `scalar_hrm` vs `scalar_hrm_blind`, `field_unet` vs `field_unet_blind`), both astar.
Median ratio < 1 means the aware model expands fewer nodes than its blind twin (the future window helps). Success delta = aware - blind over shared worlds.

|Suite|Budget|Aware|Blind|n|Aware succ|Blind succ|Succ delta|n matched|Median ratio aware/blind (95% CI)|Wilcoxon p|
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|scalar_hrm_blind|20|1.000|1.000|0.000|20|1.313 [0.892, 1.710]|0.053|
|C_dyn_crossing|150|scalar_onlstm|scalar_onlstm_blind|20|0.700|1.000|-0.300|14|2.127 [1.257, 3.253]|<0.001|
|C_dyn_crossing|150|field_unet|field_unet_blind|20|1.000|0.950|0.050|19|1.000 [0.550, 1.294]|0.845|
|C_dyn_crossing|150|field_hrm|field_hrm_blind|20|1.000|0.950|0.050|19|1.000 [0.812, 1.267]|0.744|
|C_dyn_maze|1800|scalar_hrm|scalar_hrm_blind|20|1.000|1.000|0.000|20|0.823 [0.638, 0.974]|0.021|
|C_dyn_maze|1800|scalar_onlstm|scalar_onlstm_blind|20|1.000|1.000|0.000|20|1.243 [1.083, 2.048]|0.001|
|C_dyn_maze|1800|field_unet|field_unet_blind|20|1.000|1.000|0.000|20|0.880 [0.727, 1.163]|0.872|
|C_dyn_maze|1800|field_hrm|field_hrm_blind|20|1.000|1.000|0.000|20|2.210 [1.290, 3.291]|<0.001|
|C_dyn_maze_dense|2500|scalar_hrm|scalar_hrm_blind|20|0.550|0.650|-0.100|9|1.311 [0.886, 1.486]|0.074|
|C_dyn_maze_dense|2500|scalar_onlstm|scalar_onlstm_blind|20|0.450|0.600|-0.150|8|1.292 [1.151, 1.705]|0.008|
|C_dyn_maze_dense|2500|field_unet|field_unet_blind|20|0.750|0.750|0.000|15|1.624 [1.160, 2.064]|<0.001|
|C_dyn_maze_dense|2500|field_hrm|field_hrm_blind|20|0.700|0.650|0.050|12|0.902 [0.666, 1.222]|0.850|
|C_dyn_rooms|1300|scalar_hrm|scalar_hrm_blind|20|1.000|1.000|0.000|20|1.153 [0.674, 1.874]|0.261|
|C_dyn_rooms|1300|scalar_onlstm|scalar_onlstm_blind|20|1.000|1.000|0.000|20|1.508 [1.042, 1.837]|0.021|
|C_dyn_rooms|1300|field_unet|field_unet_blind|20|1.000|1.000|0.000|20|0.889 [0.616, 1.127]|0.452|
|C_dyn_rooms|1300|field_hrm|field_hrm_blind|20|1.000|1.000|0.000|20|1.270 [0.967, 2.084]|0.027|
|C_dyn_rooms_large|600|scalar_hrm|scalar_hrm_blind|20|0.950|0.750|0.200|15|0.762 [0.422, 1.880]|0.804|
|C_dyn_rooms_large|600|scalar_onlstm|scalar_onlstm_blind|20|0.650|0.900|-0.250|13|1.320 [0.857, 2.091]|0.146|
|C_dyn_rooms_large|600|field_unet|field_unet_blind|20|0.900|0.950|-0.050|17|1.101 [0.677, 1.800]|0.329|
|C_dyn_rooms_large|600|field_hrm|field_hrm_blind|20|0.900|0.850|0.050|17|0.776 [0.405, 1.259]|0.263|
|C_dyn_spiral|2500|scalar_hrm|scalar_hrm_blind|20|0.850|0.850|0.000|17|0.922 [0.779, 1.082]|0.306|
|C_dyn_spiral|2500|scalar_onlstm|scalar_onlstm_blind|20|0.850|0.850|0.000|17|0.910 [0.562, 1.140]|0.404|
|C_dyn_spiral|2500|field_unet|field_unet_blind|20|0.900|0.950|-0.050|18|1.061 [0.749, 1.224]|0.932|
|C_dyn_spiral|2500|field_hrm|field_hrm_blind|20|0.900|0.850|0.050|17|1.133 [0.735, 1.366]|0.306|

## 3. Additive (astar) vs focal — does C7's additive-wins hold under dynamics?

For each learned arm: its astar (additive) ratio vs euclid, and its best-w focal ratio vs euclid, side by side. Best w = lowest matched-median exp_ratio (post-hoc; the focal p is optimistic — treat the CI as primary).

|Suite|Budget|Arm|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|
|---|---:|---|---|---:|---|---:|
|C_dyn_crossing|150|field_hrm|0.349 [0.217, 0.458]|1.1|0.763 [0.673, 0.878]|0.031|
|C_dyn_crossing|150|field_hrm_blind|0.231 [0.163, 0.516]|1.1|0.802 [0.699, 0.881]|0.031|
|C_dyn_crossing|150|field_unet|0.258 [0.132, 0.769]|1.1|0.854 [0.482, 0.980]|0.031|
|C_dyn_crossing|150|field_unet_blind|0.191 [0.166, 0.878]|1.1|0.916 [0.645, 0.953]|0.031|
|C_dyn_crossing|150|scalar_hrm|0.308 [0.198, 0.662]|1.1|0.838 [0.628, 0.926]|0.031|
|C_dyn_crossing|150|scalar_hrm_blind|0.578 [0.172, 1.143]|1.1|0.771 [0.492, 0.965]|0.031|
|C_dyn_crossing|150|scalar_onlstm|0.673 [0.292, 1.705]|1.1|0.885 [0.643, 1.014]|0.094|
|C_dyn_crossing|150|scalar_onlstm_blind|0.644 [0.167, 1.048]|1.1|0.763 [0.495, 0.840]|0.031|
|C_dyn_maze|1800|field_hrm|0.418 [0.201, 0.642]|1.1|0.896 [0.788, 0.965]|0.031|
|C_dyn_maze|1800|field_hrm_blind|0.054 [0.035, 0.105]|1.1|0.833 [0.779, 0.933]|0.031|
|C_dyn_maze|1800|field_unet|0.064 [0.046, 0.088]|1.1|0.849 [0.749, 0.938]|0.031|
|C_dyn_maze|1800|field_unet_blind|0.093 [0.057, 0.137]|1.1|0.898 [0.803, 0.927]|0.031|
|C_dyn_maze|1800|scalar_hrm|0.266 [0.196, 0.576]|1.1|0.864 [0.779, 0.991]|0.031|
|C_dyn_maze|1800|scalar_hrm_blind|0.339 [0.217, 0.528]|1.1|0.882 [0.846, 0.977]|0.031|
|C_dyn_maze|1800|scalar_onlstm|0.387 [0.250, 0.789]|1.1|0.935 [0.782, 0.985]|0.031|
|C_dyn_maze|1800|scalar_onlstm_blind|0.329 [0.188, 0.425]|1.1|0.874 [0.804, 0.930]|0.031|
|C_dyn_maze_dense|2500|field_hrm|0.337 [0.337, 0.337]|1.1|0.991 [0.991, 0.991]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm_blind|0.261 [0.261, 0.261]|1.1|0.864 [0.864, 0.864]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet|0.278 [0.278, 0.278]|1.1|0.793 [0.793, 0.793]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet_blind|0.246 [0.246, 0.246]|1.1|0.947 [0.947, 0.947]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm|0.619 [0.619, 0.619]|1.1|0.793 [0.793, 0.793]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm_blind|0.472 [0.472, 0.472]|1.1|0.884 [0.884, 0.884]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm|0.678 [0.678, 0.678]|1.1|0.884 [0.884, 0.884]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm_blind|0.570 [0.570, 0.570]|1.1|0.884 [0.884, 0.884]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm|0.176 [0.099, 0.491]|1.1|0.833 [0.736, 0.891]|0.031|
|C_dyn_rooms|1300|field_hrm_blind|0.080 [0.034, 0.171]|1.1|0.833 [0.827, 0.953]|0.031|
|C_dyn_rooms|1300|field_unet|0.096 [0.038, 0.191]|1.1|0.896 [0.690, 0.927]|0.031|
|C_dyn_rooms|1300|field_unet_blind|0.099 [0.050, 0.214]|1.1|0.906 [0.810, 0.943]|0.031|
|C_dyn_rooms|1300|scalar_hrm|0.200 [0.064, 0.429]|1.1|0.919 [0.814, 0.957]|0.031|
|C_dyn_rooms|1300|scalar_hrm_blind|0.184 [0.107, 0.470]|1.1|0.918 [0.840, 0.962]|0.031|
|C_dyn_rooms|1300|scalar_onlstm|0.245 [0.142, 0.414]|1.1|0.907 [0.844, 0.966]|0.031|
|C_dyn_rooms|1300|scalar_onlstm_blind|0.188 [0.105, 0.297]|1.1|0.838 [0.744, 0.932]|0.031|
|C_dyn_rooms_large|600|field_hrm|0.760 [0.110, 0.832]|1.1|0.892 [0.843, 0.955]|<0.001|
|C_dyn_rooms_large|600|field_hrm_blind|0.581 [0.337, 0.782]|1.1|0.885 [0.872, 0.957]|<0.001|
|C_dyn_rooms_large|600|field_unet|0.380 [0.132, 0.591]|1.1|0.825 [0.739, 0.921]|<0.001|
|C_dyn_rooms_large|600|field_unet_blind|0.228 [0.199, 0.418]|1.1|0.862 [0.807, 0.896]|<0.001|
|C_dyn_rooms_large|600|scalar_hrm|0.326 [0.207, 0.532]|1.1|0.862 [0.737, 0.886]|<0.001|
|C_dyn_rooms_large|600|scalar_hrm_blind|0.560 [0.147, 0.766]|1.1|0.869 [0.813, 0.927]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm|0.682 [0.327, 0.899]|1.1|0.892 [0.819, 0.940]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm_blind|0.514 [0.269, 0.766]|1.1|0.943 [0.895, 0.982]|<0.001|
|C_dyn_spiral|2500|field_hrm|0.046 [0.038, 0.073]|1.1|0.894 [0.801, 0.993]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind|0.053 [0.050, 0.060]|1.1|0.971 [0.949, 0.989]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|0.076 [0.055, 0.097]|1.1|0.888 [0.818, 0.996]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind|0.087 [0.061, 0.287]|1.1|0.931 [0.802, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|0.363 [0.064, 0.614]|1.1|0.932 [0.803, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind|0.334 [0.068, 0.609]|1.1|0.912 [0.767, 0.973]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|0.345 [0.180, 0.639]|1.1|0.813 [0.795, 0.959]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind|0.353 [0.134, 0.552]|1.1|0.901 [0.803, 0.989]|n/a (n<6)|

## 4. Recurrent/hierarchical vs field U-Net — do temporal models win when timing matters?

exp_ratio vs euclid (astar) side by side: the recurrent/hierarchical arms (scalar_hrm, scalar_onlstm, field_hrm, field_onlstm) vs the convolutional `field_unet`.

|Suite|Budget|Arm|n matched|Median ratio vs euclid (95% CI)|Wilcoxon p|
|---|---:|---|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|6|0.308 [0.198, 0.662]|0.031|
|C_dyn_crossing|150|scalar_onlstm|5|0.673 [0.292, 1.705]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm|6|0.349 [0.217, 0.458]|0.031|
|C_dyn_crossing|150|field_unet|6|0.258 [0.132, 0.769]|0.062|
|C_dyn_maze|1800|scalar_hrm|6|0.266 [0.196, 0.576]|0.031|
|C_dyn_maze|1800|scalar_onlstm|6|0.387 [0.250, 0.789]|0.031|
|C_dyn_maze|1800|field_hrm|6|0.418 [0.201, 0.642]|0.031|
|C_dyn_maze|1800|field_unet|6|0.064 [0.046, 0.088]|0.031|
|C_dyn_maze_dense|2500|scalar_hrm|1|0.619 [0.619, 0.619]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm|1|0.678 [0.678, 0.678]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm|1|0.337 [0.337, 0.337]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet|1|0.278 [0.278, 0.278]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|6|0.200 [0.064, 0.429]|0.031|
|C_dyn_rooms|1300|scalar_onlstm|6|0.245 [0.142, 0.414]|0.031|
|C_dyn_rooms|1300|field_hrm|6|0.176 [0.099, 0.491]|0.031|
|C_dyn_rooms|1300|field_unet|6|0.096 [0.038, 0.191]|0.031|
|C_dyn_rooms_large|600|scalar_hrm|14|0.326 [0.207, 0.532]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm|12|0.682 [0.327, 0.899]|0.034|
|C_dyn_rooms_large|600|field_hrm|15|0.760 [0.110, 0.832]|0.002|
|C_dyn_rooms_large|600|field_unet|14|0.380 [0.132, 0.591]|0.013|
|C_dyn_spiral|2500|scalar_hrm|4|0.363 [0.064, 0.614]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|4|0.345 [0.180, 0.639]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm|4|0.046 [0.038, 0.073]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|4|0.076 [0.055, 0.097]|n/a (n<6)|

## 5. Learned vs oracle — gap-to-ceiling

Median over the triple-matched set (euclid, oracle, arm all solved) of
`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the
euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).

|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|
|---|---:|---|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|6|0.243|
|C_dyn_crossing|150|field_hrm_blind/astar|6|0.125|
|C_dyn_crossing|150|field_unet/astar|6|0.116|
|C_dyn_crossing|150|field_unet_blind/astar|6|0.058|
|C_dyn_crossing|150|scalar_hrm/astar|6|0.162|
|C_dyn_crossing|150|scalar_hrm_blind/astar|6|0.486|
|C_dyn_crossing|150|scalar_onlstm/astar|5|0.571|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|6|0.578|
|C_dyn_maze|1800|field_hrm/astar|6|0.350|
|C_dyn_maze|1800|field_hrm_blind/astar|6|0.022|
|C_dyn_maze|1800|field_unet/astar|6|0.026|
|C_dyn_maze|1800|field_unet_blind/astar|6|0.042|
|C_dyn_maze|1800|scalar_hrm/astar|6|0.196|
|C_dyn_maze|1800|scalar_hrm_blind/astar|6|0.325|
|C_dyn_maze|1800|scalar_onlstm/astar|6|0.377|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|6|0.250|
|C_dyn_maze_dense|2500|field_hrm/astar|1|0.325|
|C_dyn_maze_dense|2500|field_hrm_blind/astar|1|0.247|
|C_dyn_maze_dense|2500|field_unet/astar|1|0.265|
|C_dyn_maze_dense|2500|field_unet_blind/astar|1|0.232|
|C_dyn_maze_dense|2500|scalar_hrm/astar|1|0.612|
|C_dyn_maze_dense|2500|scalar_hrm_blind/astar|1|0.462|
|C_dyn_maze_dense|2500|scalar_onlstm/astar|1|0.672|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/astar|1|0.563|
|C_dyn_rooms|1300|field_hrm/astar|6|0.160|
|C_dyn_rooms|1300|field_hrm_blind/astar|6|0.056|
|C_dyn_rooms|1300|field_unet/astar|6|0.079|
|C_dyn_rooms|1300|field_unet_blind/astar|6|0.066|
|C_dyn_rooms|1300|scalar_hrm/astar|6|0.169|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|6|0.161|
|C_dyn_rooms|1300|scalar_onlstm/astar|6|0.217|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|6|0.161|
|C_dyn_rooms_large|600|field_hrm/astar|15|0.749|
|C_dyn_rooms_large|600|field_hrm_blind/astar|14|0.498|
|C_dyn_rooms_large|600|field_unet/astar|14|0.276|
|C_dyn_rooms_large|600|field_unet_blind/astar|15|0.151|
|C_dyn_rooms_large|600|scalar_hrm/astar|14|0.270|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|12|0.472|
|C_dyn_rooms_large|600|scalar_onlstm/astar|12|0.634|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|14|0.430|
|C_dyn_spiral|2500|field_hrm/astar|4|0.025|
|C_dyn_spiral|2500|field_hrm_blind/astar|4|0.034|
|C_dyn_spiral|2500|field_unet/astar|4|0.046|
|C_dyn_spiral|2500|field_unet_blind/astar|4|0.069|
|C_dyn_spiral|2500|scalar_hrm/astar|4|0.300|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|4|0.266|
|C_dyn_spiral|2500|scalar_onlstm/astar|4|0.270|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|4|0.284|

## 6. In-distribution vs held-out — best learned arm exp_ratio + success vs euclid

Best learned arm (lowest pooled matched-median exp_ratio vs euclid): **field_unet**/astar.  In-distribution (trained): C_dyn_maze, C_dyn_rooms, C_dyn_spiral.  Held-out (OOD): C_dyn_maze_dense, C_dyn_crossing, C_dyn_rooms_large.

|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|
|---|---|---:|---:|---|---:|
|in-dist|C_dyn_maze|1800|6|0.064 [0.046, 0.088]|0.700|
|in-dist|C_dyn_rooms|1300|6|0.096 [0.038, 0.191]|0.700|
|in-dist|C_dyn_spiral|2500|4|0.076 [0.055, 0.097]|0.700|
|held-out|C_dyn_maze_dense|2500|1|0.278 [0.278, 0.278]|0.700|
|held-out|C_dyn_crossing|150|6|0.258 [0.132, 0.769]|0.700|
|held-out|C_dyn_rooms_large|600|14|0.380 [0.132, 0.591]|0.150|

## Notes

- Each comparison uses the per-suite binding budget. Arms or suites absent from this run
  are skipped with a note rather than crashing.
- Comparison 2 (the spotlight) pairs each time-aware arm with its W=0 time-blind twin;
  median ratio < 1 means the future window reduces expansions.
- Comparison 3 picks the focal `w` with the lowest matched-median exp_ratio vs euclid;
  reporting that winner's p on the same data is optimistic — the CI is the primary inference.
- These six p-values are UNcorrected (BH applies only to the success/McNemar grid).
- The Wilcoxon p tests paired (ratio - 1) in ratio-space, matching the median ratio + CI.
- Bootstrap CIs are seeded (`np.random.default_rng(cfg.seed)`), so this analysis is reproducible.
