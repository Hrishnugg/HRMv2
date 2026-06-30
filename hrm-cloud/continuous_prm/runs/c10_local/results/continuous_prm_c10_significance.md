# C10 Interpolation — Significance

McNemar exact (arm found & euclid not = gain; euclid found & arm not = loss). BH across the table. n/a if <2 discordant.
At the per-target binding budget (C10_maze_tgt=150, C10_rooms_t25=150, C10_rooms_t35=150); pairs over worlds.

|target|backbone|arm|n|euclid_succ|arm_succ|gain|loss|McNemar p|BH q|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|C10_maze_tgt|hrm|zero_shot|30|0.800|1.000|6|0|0.0312|0.039|
|C10_maze_tgt|hrm|nearest|30|0.800|1.000|6|0|0.0312|0.039|
|C10_maze_tgt|hrm|uniform_wmerge|30|0.800|1.000|6|0|0.0312|0.039|
|C10_maze_tgt|hrm|rbf_wmerge|30|0.800|0.967|5|0|0.0625|0.062|
|C10_maze_tgt|hrm|rbf_pmix|30|0.800|0.967|5|0|0.0625|0.062|
|C10_maze_tgt|onlstm|zero_shot|30|0.800|1.000|6|0|0.0312|0.039|
|C10_maze_tgt|onlstm|nearest|30|0.800|0.967|5|0|0.0625|0.062|
|C10_maze_tgt|onlstm|uniform_wmerge|30|0.800|0.967|5|0|0.0625|0.062|
|C10_maze_tgt|onlstm|rbf_wmerge|30|0.800|0.967|5|0|0.0625|0.062|
|C10_maze_tgt|onlstm|rbf_pmix|30|0.800|0.967|5|0|0.0625|0.062|
|C10_rooms_t25|hrm|zero_shot|30|0.567|1.000|13|0|0.0002|0.001|
|C10_rooms_t25|hrm|nearest|30|0.567|1.000|13|0|0.0002|0.001|
|C10_rooms_t25|hrm|uniform_wmerge|30|0.567|0.967|12|0|0.0005|0.001|
|C10_rooms_t25|hrm|rbf_wmerge|30|0.567|1.000|13|0|0.0002|0.001|
|C10_rooms_t25|hrm|rbf_pmix|30|0.567|1.000|13|0|0.0002|0.001|
|C10_rooms_t25|onlstm|zero_shot|30|0.567|1.000|13|0|0.0002|0.001|
|C10_rooms_t25|onlstm|nearest|30|0.567|0.967|12|0|0.0005|0.001|
|C10_rooms_t25|onlstm|uniform_wmerge|30|0.567|0.967|12|0|0.0005|0.001|
|C10_rooms_t25|onlstm|rbf_wmerge|30|0.567|0.967|12|0|0.0005|0.001|
|C10_rooms_t25|onlstm|rbf_pmix|30|0.567|0.967|12|0|0.0005|0.001|
|C10_rooms_t35|hrm|zero_shot|30|0.633|0.933|10|1|0.0117|0.018|
|C10_rooms_t35|hrm|nearest|30|0.633|0.967|10|0|0.0020|0.003|
|C10_rooms_t35|hrm|uniform_wmerge|30|0.633|0.967|10|0|0.0020|0.003|
|C10_rooms_t35|hrm|rbf_wmerge|30|0.633|0.967|10|0|0.0020|0.003|
|C10_rooms_t35|hrm|rbf_pmix|30|0.633|0.967|10|0|0.0020|0.003|
|C10_rooms_t35|onlstm|zero_shot|30|0.633|1.000|11|0|0.0010|0.002|
|C10_rooms_t35|onlstm|nearest|30|0.633|1.000|11|0|0.0010|0.002|
|C10_rooms_t35|onlstm|uniform_wmerge|30|0.633|1.000|11|0|0.0010|0.002|
|C10_rooms_t35|onlstm|rbf_wmerge|30|0.633|1.000|11|0|0.0010|0.002|
|C10_rooms_t35|onlstm|rbf_pmix|30|0.633|1.000|11|0|0.0010|0.002|