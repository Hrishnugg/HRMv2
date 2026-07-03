# C9b Dynamics Transfer — Aware-vs-Blind Success-Composite Probe

Per (target, backbone, method, K): succ_aware/succ_blind = mean `found` over ALL test
worlds at the binding budget (NOT just the shared-solved set); succ_delta = succ_aware - succ_blind.
Matched exp-ratio (vs euclid) is reported separately on the SHARED-SOLVED set only (worlds
both aware and blind solve) -- per C8's lesson, matched-ratio alone can miss/invert the read
when success differs, since it silently drops the worlds only one arm solves.

Pre-registered headline cell: full_ft @ K=16 with succ_delta > 0 (aware better) while the
time-coupling control (C_dyn_crossing) stays ~tied (|succ_delta| <= 0.05) is the
signature of 'the C8 spotlight negative flips under adaptation, but only on genuinely
time-coupled suites, not the control'.

|target|backbone|arm|K|succ_aware (n)|succ_blind (n)|succ_delta|n_shared|aware ratio [CI]|blind ratio [CI]|verdict|
|---|---|---|---:|---|---|---:|---:|---|---|---|
|C_dyn_maze_dense|scalar_hrm|zero_shot|0|0.500 (20)|0.600 (20)|-0.100|9|0.319 [0.319,0.319]|0.404 [0.404,0.404]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|scalar_hrm|lora|1|0.900 (20)|0.950 (20)|-0.050|18|0.285 [0.263,0.373]|0.275 [0.191,0.308]|-|
|C_dyn_maze_dense|scalar_hrm|lora|4|0.950 (20)|0.950 (20)|0.000|19|0.181 [0.170,0.369]|0.252 [0.161,0.295]|-|
|C_dyn_maze_dense|scalar_hrm|lora|16|1.000 (20)|0.950 (20)|0.050|19|0.126 [0.122,0.222]|0.184 [0.103,0.243]|-|
|C_dyn_maze_dense|scalar_hrm|full_ft|1|1.000 (20)|0.900 (20)|0.100|18|0.197 [0.169,0.690]|0.282 [0.091,0.558]|-|
|C_dyn_maze_dense|scalar_hrm|full_ft|4|0.950 (20)|0.950 (20)|0.000|19|0.197 [0.188,0.506]|0.193 [0.150,0.620]|-|
|C_dyn_maze_dense|scalar_hrm|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.100 [0.094,0.273]|0.220 [0.113,0.318]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_maze_dense|scalar_hrm|scratch|1|0.900 (20)|0.900 (20)|0.000|18|0.151 [0.132,0.171]|0.168 [0.158,0.178]|-|
|C_dyn_maze_dense|scalar_hrm|scratch|4|0.950 (20)|0.950 (20)|0.000|19|0.211 [0.132,0.236]|0.199 [0.121,0.229]|-|
|C_dyn_maze_dense|scalar_hrm|scratch|16|0.950 (20)|1.000 (20)|-0.050|19|0.161 [0.124,0.213]|0.187 [0.184,0.215]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|scalar_onlstm|zero_shot|0|0.400 (20)|0.600 (20)|-0.200|8|0.271 [0.271,0.271]|0.437 [0.437,0.437]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|scalar_onlstm|lora|1|0.900 (20)|0.900 (20)|0.000|18|0.348 [0.132,0.408]|0.285 [0.189,0.519]|-|
|C_dyn_maze_dense|scalar_onlstm|lora|4|0.950 (20)|0.950 (20)|0.000|19|0.237 [0.123,0.702]|0.224 [0.222,0.236]|-|
|C_dyn_maze_dense|scalar_onlstm|lora|16|1.000 (20)|0.950 (20)|0.050|19|0.247 [0.119,0.364]|0.218 [0.151,0.303]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|scalar_onlstm|full_ft|1|0.900 (20)|0.850 (20)|0.050|17|0.385 [0.101,0.457]|0.319 [0.226,0.412]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|scalar_onlstm|full_ft|4|1.000 (20)|1.000 (20)|0.000|20|0.178 [0.158,0.366]|0.250 [0.205,0.547]|-|
|C_dyn_maze_dense|scalar_onlstm|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.108 [0.092,0.451]|0.136 [0.109,0.290]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_maze_dense|scalar_onlstm|scratch|1|0.900 (20)|0.900 (20)|0.000|18|0.201 [0.105,0.297]|0.174 [0.148,0.200]|-|
|C_dyn_maze_dense|scalar_onlstm|scratch|4|1.000 (20)|0.900 (20)|0.100|18|0.163 [0.147,0.716]|0.234 [0.209,0.254]|-|
|C_dyn_maze_dense|scalar_onlstm|scratch|16|1.000 (20)|0.950 (20)|0.050|19|0.154 [0.100,0.297]|0.205 [0.142,0.258]|-|
|C_dyn_maze_dense|field_unet|zero_shot|0|0.600 (20)|0.800 (20)|-0.200|12|0.145 [0.145,0.145]|0.210 [0.210,0.210]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|field_unet|lora|1|0.950 (20)|0.900 (20)|0.050|18|0.165 [0.126,0.171]|0.120 [0.048,0.228]|DISAGREES-with-matched-ratio|
|C_dyn_maze_dense|field_unet|lora|4|0.900 (20)|0.950 (20)|-0.050|18|0.085 [0.059,0.089]|0.071 [0.067,0.149]|-|
|C_dyn_maze_dense|field_unet|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.059 [0.058,0.071]|0.075 [0.031,0.097]|-|
|C_dyn_maze_dense|field_unet|full_ft|1|0.900 (20)|0.900 (20)|0.000|18|0.112 [0.085,0.628]|0.117 [0.107,0.723]|-|
|C_dyn_maze_dense|field_unet|full_ft|4|0.950 (20)|0.950 (20)|0.000|19|0.167 [0.105,0.206]|0.069 [0.056,0.232]|-|
|C_dyn_maze_dense|field_unet|full_ft|16|0.950 (20)|1.000 (20)|-0.050|19|0.101 [0.094,0.204]|0.063 [0.056,0.099]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_maze_dense|field_unet|scratch|1|0.900 (20)|0.900 (20)|0.000|18|0.117 [0.086,0.147]|0.174 [0.153,0.195]|-|
|C_dyn_maze_dense|field_unet|scratch|4|0.900 (20)|0.950 (20)|-0.050|18|0.151 [0.137,0.320]|0.106 [0.095,0.302]|-|
|C_dyn_maze_dense|field_unet|scratch|16|1.000 (20)|1.000 (20)|0.000|20|0.106 [0.097,0.112]|0.136 [0.119,0.239]|-|
|C_dyn_crossing|scalar_hrm|zero_shot|0|1.000 (20)|0.950 (20)|0.050|19|0.315 [0.281,0.348]|0.121 [0.098,0.651]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|scalar_hrm|lora|1|0.950 (20)|0.950 (20)|0.000|18|0.322 [0.265,0.445]|0.255 [0.167,0.402]|CONTROL:tie|
|C_dyn_crossing|scalar_hrm|lora|4|1.000 (20)|0.950 (20)|0.050|19|0.136 [0.130,0.208]|0.121 [0.082,0.215]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|scalar_hrm|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.114 [0.089,0.221]|0.158 [0.103,0.228]|CONTROL:tie|
|C_dyn_crossing|scalar_hrm|full_ft|1|1.000 (20)|1.000 (20)|0.000|20|0.144 [0.121,0.322]|0.164 [0.121,0.289]|CONTROL:tie|
|C_dyn_crossing|scalar_hrm|full_ft|4|0.950 (20)|0.950 (20)|0.000|19|0.215 [0.098,0.275]|0.201 [0.121,0.255]|CONTROL:tie|
|C_dyn_crossing|scalar_hrm|full_ft|16|0.950 (20)|1.000 (20)|-0.050|19|0.189 [0.106,0.262]|0.130 [0.116,0.268]|HEADLINE(full_ft@K16); aware<=blind(succ); CONTROL:NOT-tie|
|C_dyn_crossing|scalar_hrm|scratch|1|0.550 (20)|0.550 (20)|0.000|11|0.196 [0.103,0.252]|0.207 [0.123,0.258]|CONTROL:tie|
|C_dyn_crossing|scalar_hrm|scratch|4|1.000 (20)|0.950 (20)|0.050|19|0.171 [0.121,0.282]|0.188 [0.123,0.219]|CONTROL:NOT-tie|
|C_dyn_crossing|scalar_hrm|scratch|16|1.000 (20)|1.000 (20)|0.000|20|0.123 [0.096,0.248]|0.159 [0.098,0.262]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|zero_shot|0|0.750 (20)|1.000 (20)|-0.250|15|0.394 [0.342,0.397]|0.390 [0.106,0.517]|CONTROL:NOT-tie|
|C_dyn_crossing|scalar_onlstm|lora|1|1.000 (20)|0.850 (20)|0.150|17|0.490 [0.267,0.671]|0.339 [0.322,0.470]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|scalar_onlstm|lora|4|0.950 (20)|0.950 (20)|0.000|19|0.195 [0.174,0.295]|0.302 [0.228,0.596]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.167 [0.089,0.235]|0.174 [0.144,0.273]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|full_ft|1|1.000 (20)|0.950 (20)|0.050|19|0.288 [0.178,0.349]|0.248 [0.192,0.356]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|scalar_onlstm|full_ft|4|0.950 (20)|0.950 (20)|0.000|19|0.221 [0.188,0.371]|0.201 [0.129,0.233]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|full_ft|16|0.950 (20)|1.000 (20)|-0.050|19|0.240 [0.151,0.362]|0.205 [0.129,0.235]|HEADLINE(full_ft@K16); aware<=blind(succ); CONTROL:NOT-tie|
|C_dyn_crossing|scalar_onlstm|scratch|1|0.600 (20)|0.550 (20)|0.050|11|0.190 [0.089,0.232]|0.207 [0.113,0.258]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|scratch|4|0.950 (20)|0.950 (20)|0.000|19|0.201 [0.114,0.248]|0.199 [0.144,0.268]|CONTROL:tie|
|C_dyn_crossing|scalar_onlstm|scratch|16|0.950 (20)|1.000 (20)|-0.050|19|0.248 [0.110,0.265]|0.152 [0.091,0.233]|CONTROL:NOT-tie|
|C_dyn_crossing|field_unet|zero_shot|0|1.000 (20)|1.000 (20)|0.000|20|0.192 [0.161,0.227]|0.159 [0.114,0.171]|CONTROL:tie|
|C_dyn_crossing|field_unet|lora|1|0.950 (20)|0.950 (20)|0.000|19|0.242 [0.197,0.570]|0.288 [0.167,0.384]|CONTROL:tie|
|C_dyn_crossing|field_unet|lora|4|0.950 (20)|0.950 (20)|0.000|19|0.242 [0.103,0.265]|0.205 [0.137,0.260]|CONTROL:tie|
|C_dyn_crossing|field_unet|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.114 [0.082,0.248]|0.185 [0.167,0.226]|CONTROL:tie|
|C_dyn_crossing|field_unet|full_ft|1|1.000 (20)|0.950 (20)|0.050|19|0.221 [0.144,0.336]|0.288 [0.195,0.606]|CONTROL:NOT-tie|
|C_dyn_crossing|field_unet|full_ft|4|0.950 (20)|1.000 (20)|-0.050|19|0.208 [0.114,0.267]|0.212 [0.123,0.356]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|field_unet|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.221 [0.123,0.303]|0.221 [0.152,0.423]|HEADLINE(full_ft@K16); aware<=blind(succ); CONTROL:tie|
|C_dyn_crossing|field_unet|scratch|1|0.600 (20)|0.500 (20)|0.100|10|0.366 [0.228,0.400]|0.262 [0.164,0.336]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|field_unet|scratch|4|1.000 (20)|0.950 (20)|0.050|19|0.261 [0.134,0.450]|0.208 [0.134,0.322]|CONTROL:NOT-tie; DISAGREES-with-matched-ratio|
|C_dyn_crossing|field_unet|scratch|16|0.950 (20)|0.950 (20)|0.000|19|0.121 [0.110,0.201]|0.227 [0.137,0.289]|CONTROL:tie|
|C_dyn_rooms_large|scalar_hrm|zero_shot|0|0.950 (20)|0.900 (20)|0.050|17|0.486 [0.352,0.822]|0.525 [0.139,0.889]|-|
|C_dyn_rooms_large|scalar_hrm|lora|1|1.000 (20)|1.000 (20)|0.000|20|0.366 [0.301,0.404]|0.302 [0.189,0.532]|-|
|C_dyn_rooms_large|scalar_hrm|lora|4|1.000 (20)|1.000 (20)|0.000|20|0.219 [0.148,0.256]|0.181 [0.145,0.221]|-|
|C_dyn_rooms_large|scalar_hrm|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.176 [0.131,0.185]|0.164 [0.147,0.226]|-|
|C_dyn_rooms_large|scalar_hrm|full_ft|1|1.000 (20)|1.000 (20)|0.000|20|0.221 [0.174,0.385]|0.252 [0.208,0.319]|-|
|C_dyn_rooms_large|scalar_hrm|full_ft|4|1.000 (20)|1.000 (20)|0.000|20|0.218 [0.196,0.277]|0.161 [0.126,0.230]|-|
|C_dyn_rooms_large|scalar_hrm|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.168 [0.141,0.219]|0.205 [0.185,0.239]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_rooms_large|scalar_hrm|scratch|1|0.950 (20)|0.950 (20)|0.000|19|0.191 [0.147,0.301]|0.190 [0.163,0.286]|-|
|C_dyn_rooms_large|scalar_hrm|scratch|4|1.000 (20)|1.000 (20)|0.000|20|0.135 [0.105,0.165]|0.132 [0.104,0.161]|-|
|C_dyn_rooms_large|scalar_hrm|scratch|16|1.000 (20)|1.000 (20)|0.000|20|0.130 [0.117,0.191]|0.147 [0.126,0.183]|-|
|C_dyn_rooms_large|scalar_onlstm|zero_shot|0|0.650 (20)|0.950 (20)|-0.300|12|0.856 [0.526,1.586]|0.794 [0.157,1.026]|-|
|C_dyn_rooms_large|scalar_onlstm|lora|1|1.000 (20)|1.000 (20)|0.000|20|0.428 [0.299,0.556]|0.473 [0.329,0.622]|-|
|C_dyn_rooms_large|scalar_onlstm|lora|4|1.000 (20)|1.000 (20)|0.000|20|0.165 [0.130,0.234]|0.226 [0.180,0.270]|-|
|C_dyn_rooms_large|scalar_onlstm|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.153 [0.129,0.220]|0.167 [0.139,0.197]|-|
|C_dyn_rooms_large|scalar_onlstm|full_ft|1|1.000 (20)|0.950 (20)|0.050|19|0.455 [0.320,0.623]|0.407 [0.225,0.541]|DISAGREES-with-matched-ratio|
|C_dyn_rooms_large|scalar_onlstm|full_ft|4|1.000 (20)|1.000 (20)|0.000|20|0.213 [0.153,0.293]|0.182 [0.147,0.227]|-|
|C_dyn_rooms_large|scalar_onlstm|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.185 [0.139,0.220]|0.211 [0.165,0.257]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_rooms_large|scalar_onlstm|scratch|1|0.950 (20)|0.950 (20)|0.000|19|0.155 [0.112,0.308]|0.189 [0.172,0.300]|-|
|C_dyn_rooms_large|scalar_onlstm|scratch|4|1.000 (20)|1.000 (20)|0.000|20|0.131 [0.097,0.218]|0.120 [0.089,0.205]|-|
|C_dyn_rooms_large|scalar_onlstm|scratch|16|1.000 (20)|1.000 (20)|0.000|20|0.137 [0.123,0.198]|0.156 [0.124,0.200]|-|
|C_dyn_rooms_large|field_unet|zero_shot|0|0.750 (20)|1.000 (20)|-0.250|15|0.418 [0.065,1.196]|0.261 [0.180,0.366]|-|
|C_dyn_rooms_large|field_unet|lora|1|1.000 (20)|1.000 (20)|0.000|20|0.196 [0.108,0.387]|0.274 [0.234,0.326]|-|
|C_dyn_rooms_large|field_unet|lora|4|1.000 (20)|1.000 (20)|0.000|20|0.113 [0.087,0.134]|0.160 [0.137,0.239]|-|
|C_dyn_rooms_large|field_unet|lora|16|1.000 (20)|1.000 (20)|0.000|20|0.166 [0.120,0.195]|0.155 [0.097,0.214]|-|
|C_dyn_rooms_large|field_unet|full_ft|1|1.000 (20)|0.950 (20)|0.050|19|0.277 [0.196,0.346]|0.270 [0.203,0.320]|DISAGREES-with-matched-ratio|
|C_dyn_rooms_large|field_unet|full_ft|4|1.000 (20)|1.000 (20)|0.000|20|0.124 [0.100,0.169]|0.194 [0.145,0.229]|-|
|C_dyn_rooms_large|field_unet|full_ft|16|1.000 (20)|1.000 (20)|0.000|20|0.129 [0.108,0.143]|0.136 [0.108,0.242]|HEADLINE(full_ft@K16); aware<=blind(succ)|
|C_dyn_rooms_large|field_unet|scratch|1|1.000 (20)|0.950 (20)|0.050|19|0.234 [0.169,0.390]|0.219 [0.188,0.350]|DISAGREES-with-matched-ratio|
|C_dyn_rooms_large|field_unet|scratch|4|1.000 (20)|1.000 (20)|0.000|20|0.130 [0.104,0.175]|0.146 [0.110,0.199]|-|
|C_dyn_rooms_large|field_unet|scratch|16|1.000 (20)|1.000 (20)|0.000|20|0.134 [0.102,0.188]|0.145 [0.103,0.205]|-|

## Summary

full_ft @ K=16: 0/9 (target, backbone) cell(s) show aware beating blind on the success composite (succ_delta > 0).
Control (C_dyn_crossing) did NOT stay a tie at full_ft@K16 (|succ_delta| > 0.05) -- re-examine whether this suite is truly time-decoupled.
17 cell(s) where the success composite and the shared-solved matched-ratio disagree on direction -- inspect these individually rather than trusting matched-ratio alone.