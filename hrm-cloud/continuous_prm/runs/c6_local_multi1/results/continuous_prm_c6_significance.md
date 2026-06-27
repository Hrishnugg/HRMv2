# C6 Heatmap Value-Field Significance Analysis

## Claim Candidates

|Suite|Budget|Euclid|Method|Delta|Gain|Loss|McNemar p|BH q|Exp Delta|
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
|C_hard_maze|144|0.625|hrm|0.350|14|0|0.000122|0.000209|-46.025|
|C_hard_maze|144|0.625|onlstm|0.325|14|1|0.000977|0.00153|-44.150|
|C_hard_maze|144|0.625|unet|0.350|14|0|0.000122|0.000209|-41.350|

## Target-Band Comparisons

|Suite|Budget|Euclid|Method|Method Success|Delta|Gain|Loss|BH q|Exp Delta|
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
|C_hard_maze|144|0.625|grid_oracle|0.950|0.325|13|0|0.0004|-45.325|
|C_hard_maze|144|0.625|hrm|0.975|0.350|14|0|0.000209|-46.025|
|C_hard_maze|144|0.625|onlstm|0.950|0.325|14|1|0.00153|-44.150|
|C_hard_maze|144|0.625|unet|0.975|0.350|14|0|0.000209|-41.350|

## Notes

- `grid_oracle` is an upper-bound diagnostic and is excluded from claim candidates.
- McNemar p-values are paired by suite, budget, and logical world index.
- BH q-values are corrected across all method-vs-Euclidean C6 comparisons.
