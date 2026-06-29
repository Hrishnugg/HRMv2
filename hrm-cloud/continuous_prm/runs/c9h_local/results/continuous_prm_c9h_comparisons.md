# C9h Transfer Hardening — Pre-registered Comparisons

Adaptation curves: matched A* expansion-ratio vs euclid (median, 95% CI) per K.
lora_bounded vs lora_unbounded = clamp effect; lora_bounded vs full_ft = LoRA efficiency; lora_bounded vs scratch = transfer benefit. Lower = fewer expansions.
Binding budget per target: C_hard_bugtrap=24, C_hard_maze_dense=140, C_hard_rooms_large=56.

## C_hard_bugtrap / hrm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.696 [0.615,0.733] (succ 0.83, n15)|n/a|n/a|n/a|n/a|
|1|n/a|0.696 [0.625,0.700] (succ 0.83, n45)|0.696 [0.625,0.700] (succ 0.83, n45)|1.024 [1.000,1.288] (succ 0.22, n18)|1.000 [1.000,1.000] (succ 0.53, n48)|
|4|n/a|0.700 [0.667,0.733] (succ 0.83, n45)|0.700 [0.667,0.733] (succ 0.83, n45)|0.818 [0.714,0.955] (succ 0.52, n37)|1.098 [1.062,1.194] (succ 0.41, n36)|
|16|n/a|0.905 [0.810,0.952] (succ 0.59, n43)|0.913 [0.810,1.062] (succ 0.60, n43)|0.698 [0.620,0.753] (succ 0.73, n42)|1.603 [1.438,1.769] (succ 0.02, n2)|

## C_hard_bugtrap / onlstm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.650 [0.455,0.696] (succ 0.80, n15)|n/a|n/a|n/a|n/a|
|1|n/a|0.650 [0.533,0.682] (succ 0.80, n45)|0.650 [0.533,0.682] (succ 0.80, n45)|0.857 [0.750,0.913] (succ 0.73, n45)|1.000 [1.000,1.000] (succ 0.53, n48)|
|4|n/a|0.682 [0.533,0.696] (succ 0.73, n45)|0.682 [0.533,0.696] (succ 0.74, n45)|0.659 [0.636,0.762] (succ 0.73, n46)|1.000 [1.000,1.000] (succ 0.53, n48)|
|16|n/a|0.733 [0.652,0.773] (succ 0.72, n43)|0.733 [0.667,0.769] (succ 0.72, n43)|0.698 [0.633,0.750] (succ 0.80, n46)|1.257 [1.048,1.467] (succ 0.02, n2)|

## C_hard_bugtrap / unet
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.810 [0.562,0.933] (succ 0.63, n13)|n/a|n/a|n/a|n/a|
|1|n/a|0.731 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|0.886 [0.812,0.955] (succ 0.71, n46)|1.000 [0.952,1.045] (succ 0.42, n30)|
|4|n/a|0.753 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|0.923 [0.845,0.977] (succ 0.67, n44)|1.065 [0.958,1.106] (succ 0.30, n24)|
|16|n/a|0.753 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|0.977 [0.845,1.050] (succ 0.64, n42)|1.056 [1.000,1.100] (succ 0.49, n36)|

## C_hard_maze_dense / hrm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.650 [0.546,0.692] (succ 1.00, n10)|n/a|n/a|n/a|n/a|
|1|n/a|0.654 [0.549,0.684] (succ 1.00, n30)|0.658 [0.549,0.684] (succ 1.00, n30)|0.805 [0.722,0.896] (succ 0.71, n29)|1.000 [1.000,1.000] (succ 0.33, n30)|
|4|n/a|0.692 [0.549,0.727] (succ 0.97, n30)|0.694 [0.549,0.727] (succ 0.97, n30)|0.677 [0.615,0.745] (succ 0.97, n30)|1.000 [1.000,1.016] (succ 0.32, n27)|
|16|n/a|0.693 [0.649,0.723] (succ 0.98, n30)|0.686 [0.652,0.720] (succ 0.98, n30)|0.591 [0.551,0.637] (succ 0.99, n30)|0.984 [0.968,1.007] (succ 0.53, n29)|

## C_hard_maze_dense / onlstm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.873 [0.782,0.934] (succ 0.93, n10)|n/a|n/a|n/a|n/a|
|1|n/a|0.873 [0.789,0.898] (succ 0.92, n30)|0.873 [0.789,0.898] (succ 0.92, n30)|0.634 [0.598,0.718] (succ 0.92, n30)|1.000 [1.000,1.000] (succ 0.33, n30)|
|4|n/a|0.845 [0.752,0.861] (succ 0.97, n30)|0.845 [0.759,0.861] (succ 0.97, n30)|0.702 [0.615,0.787] (succ 0.91, n30)|1.000 [1.000,1.000] (succ 0.33, n30)|
|16|n/a|0.689 [0.636,0.716] (succ 1.00, n30)|0.682 [0.636,0.716] (succ 1.00, n30)|0.597 [0.532,0.642] (succ 0.98, n30)|0.993 [0.985,1.000] (succ 0.37, n30)|

## C_hard_maze_dense / unet
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.764 [0.725,0.867] (succ 1.00, n10)|n/a|n/a|n/a|n/a|
|1|n/a|0.768 [0.750,0.790] (succ 1.00, n30)|0.768 [0.750,0.790] (succ 1.00, n30)|0.803 [0.678,0.857] (succ 0.74, n30)|0.786 [0.743,0.814] (succ 0.42, n19)|
|4|n/a|0.768 [0.742,0.790] (succ 1.00, n30)|0.768 [0.742,0.790] (succ 1.00, n30)|0.683 [0.582,0.739] (succ 0.91, n30)|0.813 [0.760,0.869] (succ 0.71, n30)|
|16|n/a|0.768 [0.750,0.790] (succ 1.00, n30)|0.768 [0.750,0.790] (succ 1.00, n30)|0.695 [0.623,0.774] (succ 0.83, n30)|0.886 [0.829,0.970] (succ 0.58, n30)|

## C_hard_rooms_large / hrm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.771 [0.688,0.894] (succ 0.97, n11)|n/a|n/a|n/a|n/a|
|1|n/a|0.766 [0.725,0.889] (succ 0.97, n33)|0.766 [0.725,0.889] (succ 0.97, n33)|1.083 [1.056,1.104] (succ 0.41, n29)|1.028 [1.020,1.047] (succ 0.37, n33)|
|4|n/a|0.750 [0.725,0.889] (succ 0.93, n33)|0.750 [0.714,0.889] (succ 0.93, n33)|0.597 [0.496,0.650] (succ 0.87, n32)|0.680 [0.544,0.750] (succ 0.58, n30)|
|16|n/a|0.957 [0.781,1.000] (succ 0.72, n30)|0.957 [0.781,1.000] (succ 0.72, n30)|0.490 [0.465,0.543] (succ 0.91, n33)|0.679 [0.596,0.772] (succ 0.66, n32)|

## C_hard_rooms_large / onlstm
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.750 [0.549,1.083] (succ 1.00, n11)|n/a|n/a|n/a|n/a|
|1|n/a|0.750 [0.608,0.967] (succ 1.00, n33)|0.750 [0.608,0.967] (succ 1.00, n33)|0.722 [0.647,0.893] (succ 0.67, n33)|1.000 [1.000,1.000] (succ 0.37, n33)|
|4|n/a|0.806 [0.646,0.915] (succ 0.97, n33)|0.806 [0.646,0.915] (succ 0.97, n33)|0.574 [0.510,0.611] (succ 0.86, n33)|1.000 [1.000,1.000] (succ 0.37, n33)|
|16|n/a|0.609 [0.531,0.786] (succ 0.68, n30)|0.637 [0.510,0.771] (succ 0.68, n30)|0.500 [0.467,0.554] (succ 0.91, n33)|0.700 [0.627,0.792] (succ 0.66, n33)|

## C_hard_rooms_large / unet
|K|zero_shot|lora_bounded|lora_unbounded|full_ft|scratch|
|---:|---|---|---|---|---|
|0|0.982 [0.690,1.167] (succ 0.67, n10)|n/a|n/a|n/a|n/a|
|1|n/a|0.992 [0.804,1.093] (succ 0.67, n30)|0.992 [0.804,1.093] (succ 0.67, n30)|0.902 [0.843,0.979] (succ 0.68, n33)|0.625 [0.375,0.732] (succ 0.12, n3)|
|4|n/a|1.002 [0.804,1.093] (succ 0.67, n30)|1.002 [0.804,1.093] (succ 0.67, n30)|0.404 [0.372,0.469] (succ 0.97, n33)|0.514 [0.389,0.657] (succ 0.77, n33)|
|16|n/a|0.992 [0.804,1.093] (succ 0.67, n30)|0.992 [0.804,1.093] (succ 0.67, n30)|0.404 [0.361,0.535] (succ 0.97, n33)|0.660 [0.553,0.721] (succ 0.82, n33)|

## Bounded vs Unbounded

Comparison of lora_bounded vs lora_unbounded per (target, backbone, K).
A smaller exp_ratio for bounded means the clamp helps; larger means it hurts.

|target|backbone|K|lora_bounded|lora_unbounded|delta (bounded−unbounded)|
|---|---|---:|---|---|---|
|C_hard_bugtrap|hrm|1|0.696 [0.625,0.700] (succ 0.83, n45)|0.696 [0.625,0.700] (succ 0.83, n45)|0.000|
|C_hard_bugtrap|hrm|4|0.700 [0.667,0.733] (succ 0.83, n45)|0.700 [0.667,0.733] (succ 0.83, n45)|0.000|
|C_hard_bugtrap|hrm|16|0.905 [0.810,0.952] (succ 0.59, n43)|0.913 [0.810,1.062] (succ 0.60, n43)|-0.008|
|C_hard_bugtrap|onlstm|1|0.650 [0.533,0.682] (succ 0.80, n45)|0.650 [0.533,0.682] (succ 0.80, n45)|0.000|
|C_hard_bugtrap|onlstm|4|0.682 [0.533,0.696] (succ 0.73, n45)|0.682 [0.533,0.696] (succ 0.74, n45)|0.000|
|C_hard_bugtrap|onlstm|16|0.733 [0.652,0.773] (succ 0.72, n43)|0.733 [0.667,0.769] (succ 0.72, n43)|0.000|
|C_hard_bugtrap|unet|1|0.731 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|-0.022|
|C_hard_bugtrap|unet|4|0.753 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|0.000|
|C_hard_bugtrap|unet|16|0.753 [0.577,0.833] (succ 0.60, n36)|0.753 [0.577,0.833] (succ 0.60, n36)|0.000|
|C_hard_maze_dense|hrm|1|0.654 [0.549,0.684] (succ 1.00, n30)|0.658 [0.549,0.684] (succ 1.00, n30)|-0.004|
|C_hard_maze_dense|hrm|4|0.692 [0.549,0.727] (succ 0.97, n30)|0.694 [0.549,0.727] (succ 0.97, n30)|-0.002|
|C_hard_maze_dense|hrm|16|0.693 [0.649,0.723] (succ 0.98, n30)|0.686 [0.652,0.720] (succ 0.98, n30)|0.007|
|C_hard_maze_dense|onlstm|1|0.873 [0.789,0.898] (succ 0.92, n30)|0.873 [0.789,0.898] (succ 0.92, n30)|0.000|
|C_hard_maze_dense|onlstm|4|0.845 [0.752,0.861] (succ 0.97, n30)|0.845 [0.759,0.861] (succ 0.97, n30)|0.000|
|C_hard_maze_dense|onlstm|16|0.689 [0.636,0.716] (succ 1.00, n30)|0.682 [0.636,0.716] (succ 1.00, n30)|0.008|
|C_hard_maze_dense|unet|1|0.768 [0.750,0.790] (succ 1.00, n30)|0.768 [0.750,0.790] (succ 1.00, n30)|0.000|
|C_hard_maze_dense|unet|4|0.768 [0.742,0.790] (succ 1.00, n30)|0.768 [0.742,0.790] (succ 1.00, n30)|0.000|
|C_hard_maze_dense|unet|16|0.768 [0.750,0.790] (succ 1.00, n30)|0.768 [0.750,0.790] (succ 1.00, n30)|0.000|
|C_hard_rooms_large|hrm|1|0.766 [0.725,0.889] (succ 0.97, n33)|0.766 [0.725,0.889] (succ 0.97, n33)|0.000|
|C_hard_rooms_large|hrm|4|0.750 [0.725,0.889] (succ 0.93, n33)|0.750 [0.714,0.889] (succ 0.93, n33)|0.000|
|C_hard_rooms_large|hrm|16|0.957 [0.781,1.000] (succ 0.72, n30)|0.957 [0.781,1.000] (succ 0.72, n30)|0.000|
|C_hard_rooms_large|onlstm|1|0.750 [0.608,0.967] (succ 1.00, n33)|0.750 [0.608,0.967] (succ 1.00, n33)|0.000|
|C_hard_rooms_large|onlstm|4|0.806 [0.646,0.915] (succ 0.97, n33)|0.806 [0.646,0.915] (succ 0.97, n33)|0.000|
|C_hard_rooms_large|onlstm|16|0.609 [0.531,0.786] (succ 0.68, n30)|0.637 [0.510,0.771] (succ 0.68, n30)|-0.028|
|C_hard_rooms_large|unet|1|0.992 [0.804,1.093] (succ 0.67, n30)|0.992 [0.804,1.093] (succ 0.67, n30)|0.000|
|C_hard_rooms_large|unet|4|1.002 [0.804,1.093] (succ 0.67, n30)|1.002 [0.804,1.093] (succ 0.67, n30)|0.000|
|C_hard_rooms_large|unet|16|0.992 [0.804,1.093] (succ 0.67, n30)|0.992 [0.804,1.093] (succ 0.67, n30)|0.000|
