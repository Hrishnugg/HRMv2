# C9b Dynamics Transfer — Pre-registered Comparisons

Adaptation curves: matched A* expansion-ratio vs euclid (median, 95% CI) per K, split by awareness.
lora vs scratch = transfer helps; lora vs full_ft = sample-efficiency; vs K=0 (zero_shot) = adaptation helps. Lower = fewer expansions.
All results are at the per-target binding budget (C_dyn_crossing=150, C_dyn_maze_dense=2500, C_dyn_rooms_large=600); expansion-ratios are pooled over adapt-seeds (n = #worlds x #seeds).

## C_dyn_crossing / field_unet / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.192 [0.161,0.227] (succ 1.00, n3)|n/a|n/a|n/a|
|1|n/a|0.242 [0.197,0.570] (succ 0.92, n9)|0.221 [0.144,0.336] (succ 0.95, n9)|0.377 [0.275,0.403] (succ 0.52, n7)|
|4|n/a|0.242 [0.103,0.265] (succ 0.95, n9)|0.208 [0.114,0.267] (succ 0.95, n9)|0.261 [0.134,0.450] (succ 0.78, n8)|
|16|n/a|0.114 [0.082,0.248] (succ 0.97, n9)|0.221 [0.123,0.303] (succ 0.97, n9)|0.121 [0.110,0.201] (succ 0.95, n9)|
Crossover read: full_ft K1->K16: 0.221->0.221 (does not improve); lora K1 vs zero_shot: 0.242 vs 0.192 (diverges); lora vs scratch @K1: 0.242 vs 0.377 (transfer wins)

## C_dyn_crossing / field_unet / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.159 [0.114,0.171] (succ 1.00, n3)|n/a|n/a|n/a|
|1|n/a|0.288 [0.167,0.384] (succ 0.83, n9)|0.288 [0.195,0.606] (succ 0.93, n9)|0.262 [0.164,0.336] (succ 0.47, n6)|
|4|n/a|0.205 [0.137,0.260] (succ 0.95, n9)|0.212 [0.123,0.356] (succ 0.97, n9)|0.208 [0.134,0.322] (succ 0.78, n8)|
|16|n/a|0.185 [0.167,0.226] (succ 0.97, n9)|0.221 [0.152,0.423] (succ 0.97, n9)|0.227 [0.137,0.289] (succ 0.95, n9)|
Crossover read: full_ft K1->K16: 0.288->0.221 (improves); lora K1 vs zero_shot: 0.288 vs 0.159 (diverges); lora vs scratch @K1: 0.288 vs 0.262 (transfer does not win)

## C_dyn_crossing / scalar_hrm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.315 [0.281,0.348] (succ 1.00, n3)|n/a|n/a|n/a|
|1|n/a|0.322 [0.265,0.445] (succ 0.83, n9)|0.144 [0.121,0.322] (succ 0.90, n9)|0.196 [0.103,0.252] (succ 0.48, n6)|
|4|n/a|0.136 [0.130,0.208] (succ 0.97, n9)|0.215 [0.098,0.275] (succ 0.93, n9)|0.171 [0.121,0.282] (succ 0.92, n9)|
|16|n/a|0.114 [0.089,0.221] (succ 1.00, n9)|0.189 [0.106,0.262] (succ 0.95, n9)|0.123 [0.096,0.248] (succ 0.97, n9)|
Crossover read: full_ft K1->K16: 0.144->0.189 (does not improve); lora K1 vs zero_shot: 0.322 vs 0.315 (close); lora vs scratch @K1: 0.322 vs 0.196 (transfer does not win)

## C_dyn_crossing / scalar_hrm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.121 [0.098,0.651] (succ 0.95, n3)|n/a|n/a|n/a|
|1|n/a|0.255 [0.167,0.402] (succ 0.90, n9)|0.164 [0.121,0.289] (succ 0.97, n9)|0.207 [0.123,0.258] (succ 0.48, n6)|
|4|n/a|0.121 [0.082,0.215] (succ 0.95, n9)|0.201 [0.121,0.255] (succ 0.95, n9)|0.188 [0.123,0.219] (succ 0.82, n9)|
|16|n/a|0.158 [0.103,0.228] (succ 0.97, n9)|0.130 [0.116,0.268] (succ 0.97, n9)|0.159 [0.098,0.262] (succ 0.97, n9)|
Crossover read: full_ft K1->K16: 0.164->0.130 (improves); lora K1 vs zero_shot: 0.255 vs 0.121 (diverges); lora vs scratch @K1: 0.255 vs 0.207 (transfer does not win)

## C_dyn_crossing / scalar_onlstm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.394 [0.342,0.397] (succ 0.75, n3)|n/a|n/a|n/a|
|1|n/a|0.490 [0.267,0.671] (succ 0.93, n9)|0.288 [0.178,0.349] (succ 0.92, n9)|0.215 [0.103,0.242] (succ 0.50, n7)|
|4|n/a|0.195 [0.174,0.295] (succ 0.95, n9)|0.221 [0.188,0.371] (succ 0.88, n9)|0.201 [0.114,0.248] (succ 0.92, n9)|
|16|n/a|0.167 [0.089,0.235] (succ 0.97, n9)|0.240 [0.151,0.362] (succ 0.93, n9)|0.248 [0.110,0.265] (succ 0.95, n9)|
Crossover read: full_ft K1->K16: 0.288->0.240 (improves); lora K1 vs zero_shot: 0.490 vs 0.394 (diverges); lora vs scratch @K1: 0.490 vs 0.215 (transfer does not win)

## C_dyn_crossing / scalar_onlstm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.390 [0.106,0.517] (succ 1.00, n3)|n/a|n/a|n/a|
|1|n/a|0.339 [0.322,0.470] (succ 0.72, n8)|0.248 [0.192,0.356] (succ 0.83, n9)|0.207 [0.113,0.258] (succ 0.48, n6)|
|4|n/a|0.302 [0.228,0.596] (succ 0.75, n8)|0.201 [0.129,0.233] (succ 0.93, n9)|0.199 [0.144,0.268] (succ 0.88, n9)|
|16|n/a|0.174 [0.144,0.273] (succ 0.98, n9)|0.205 [0.129,0.235] (succ 0.98, n9)|0.152 [0.091,0.233] (succ 0.97, n9)|
Crossover read: full_ft K1->K16: 0.248->0.205 (improves); lora K1 vs zero_shot: 0.339 vs 0.390 (diverges); lora vs scratch @K1: 0.339 vs 0.207 (transfer does not win)

## C_dyn_maze_dense / field_unet / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.145 [0.145,0.145] (succ 0.60, n1)|n/a|n/a|n/a|
|1|n/a|0.165 [0.126,0.171] (succ 0.78, n3)|0.112 [0.085,0.628] (succ 0.70, n3)|0.117 [0.086,0.147] (succ 0.47, n2)|
|4|n/a|0.085 [0.059,0.089] (succ 0.88, n3)|0.167 [0.105,0.206] (succ 0.88, n3)|0.151 [0.137,0.320] (succ 0.78, n3)|
|16|n/a|0.059 [0.058,0.071] (succ 0.97, n3)|0.101 [0.094,0.204] (succ 0.95, n3)|0.106 [0.097,0.112] (succ 0.95, n3)|
Crossover read: full_ft K1->K16: 0.112->0.101 (improves); lora K1 vs zero_shot: 0.165 vs 0.145 (diverges); lora vs scratch @K1: 0.165 vs 0.117 (transfer does not win)

## C_dyn_maze_dense / field_unet / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.210 [0.210,0.210] (succ 0.80, n1)|n/a|n/a|n/a|
|1|n/a|0.120 [0.048,0.228] (succ 0.83, n3)|0.117 [0.107,0.723] (succ 0.62, n3)|0.174 [0.153,0.195] (succ 0.47, n2)|
|4|n/a|0.071 [0.067,0.149] (succ 0.90, n3)|0.069 [0.056,0.232] (succ 0.88, n3)|0.106 [0.095,0.302] (succ 0.88, n3)|
|16|n/a|0.075 [0.031,0.097] (succ 0.97, n3)|0.063 [0.056,0.099] (succ 0.97, n3)|0.136 [0.119,0.239] (succ 0.97, n3)|
Crossover read: full_ft K1->K16: 0.117->0.063 (improves); lora K1 vs zero_shot: 0.120 vs 0.210 (diverges); lora vs scratch @K1: 0.120 vs 0.174 (transfer wins)

## C_dyn_maze_dense / scalar_hrm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.319 [0.319,0.319] (succ 0.50, n1)|n/a|n/a|n/a|
|1|n/a|0.285 [0.263,0.373] (succ 0.72, n3)|0.197 [0.169,0.690] (succ 0.75, n3)|0.151 [0.132,0.171] (succ 0.47, n2)|
|4|n/a|0.181 [0.170,0.369] (succ 0.87, n3)|0.197 [0.188,0.506] (succ 0.88, n3)|0.211 [0.132,0.236] (succ 0.88, n3)|
|16|n/a|0.126 [0.122,0.222] (succ 0.95, n3)|0.100 [0.094,0.273] (succ 0.93, n3)|0.161 [0.124,0.213] (succ 0.93, n3)|
Crossover read: full_ft K1->K16: 0.197->0.100 (improves); lora K1 vs zero_shot: 0.285 vs 0.319 (diverges); lora vs scratch @K1: 0.285 vs 0.151 (transfer does not win)

## C_dyn_maze_dense / scalar_hrm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.404 [0.404,0.404] (succ 0.60, n1)|n/a|n/a|n/a|
|1|n/a|0.275 [0.191,0.308] (succ 0.78, n3)|0.282 [0.091,0.558] (succ 0.57, n3)|0.168 [0.158,0.178] (succ 0.47, n2)|
|4|n/a|0.252 [0.161,0.295] (succ 0.88, n3)|0.193 [0.150,0.620] (succ 0.93, n3)|0.199 [0.121,0.229] (succ 0.90, n3)|
|16|n/a|0.184 [0.103,0.243] (succ 0.95, n3)|0.220 [0.113,0.318] (succ 0.97, n3)|0.187 [0.184,0.215] (succ 0.92, n3)|
Crossover read: full_ft K1->K16: 0.282->0.220 (improves); lora K1 vs zero_shot: 0.275 vs 0.404 (diverges); lora vs scratch @K1: 0.275 vs 0.168 (transfer does not win)

## C_dyn_maze_dense / scalar_onlstm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.271 [0.271,0.271] (succ 0.40, n1)|n/a|n/a|n/a|
|1|n/a|0.348 [0.132,0.408] (succ 0.70, n3)|0.385 [0.101,0.457] (succ 0.67, n3)|0.201 [0.105,0.297] (succ 0.47, n2)|
|4|n/a|0.237 [0.123,0.702] (succ 0.90, n3)|0.178 [0.158,0.366] (succ 0.92, n3)|0.163 [0.147,0.716] (succ 0.93, n3)|
|16|n/a|0.247 [0.119,0.364] (succ 0.97, n3)|0.108 [0.092,0.451] (succ 0.93, n3)|0.154 [0.100,0.297] (succ 0.95, n3)|
Crossover read: full_ft K1->K16: 0.385->0.108 (improves); lora K1 vs zero_shot: 0.348 vs 0.271 (diverges); lora vs scratch @K1: 0.348 vs 0.201 (transfer does not win)

## C_dyn_maze_dense / scalar_onlstm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.437 [0.437,0.437] (succ 0.60, n1)|n/a|n/a|n/a|
|1|n/a|0.285 [0.189,0.519] (succ 0.83, n3)|0.319 [0.226,0.412] (succ 0.57, n2)|0.174 [0.148,0.200] (succ 0.47, n2)|
|4|n/a|0.224 [0.222,0.236] (succ 0.87, n3)|0.250 [0.205,0.547] (succ 0.93, n3)|0.234 [0.209,0.254] (succ 0.85, n3)|
|16|n/a|0.218 [0.151,0.303] (succ 0.95, n3)|0.136 [0.109,0.290] (succ 0.97, n3)|0.205 [0.142,0.258] (succ 0.95, n3)|
Crossover read: full_ft K1->K16: 0.319->0.136 (improves); lora K1 vs zero_shot: 0.285 vs 0.437 (diverges); lora vs scratch @K1: 0.285 vs 0.174 (transfer does not win)

## C_dyn_rooms_large / field_unet / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.418 [0.065,1.196] (succ 0.75, n10)|n/a|n/a|n/a|
|1|n/a|0.196 [0.108,0.387] (succ 0.97, n36)|0.277 [0.196,0.346] (succ 0.98, n36)|0.234 [0.169,0.390] (succ 0.53, n23)|
|4|n/a|0.113 [0.087,0.134] (succ 1.00, n36)|0.124 [0.100,0.169] (succ 1.00, n36)|0.130 [0.104,0.175] (succ 0.88, n31)|
|16|n/a|0.166 [0.120,0.195] (succ 1.00, n36)|0.129 [0.108,0.143] (succ 1.00, n36)|0.134 [0.102,0.188] (succ 1.00, n36)|
Crossover read: full_ft K1->K16: 0.277->0.129 (improves); lora K1 vs zero_shot: 0.196 vs 0.418 (diverges); lora vs scratch @K1: 0.196 vs 0.234 (transfer wins)

## C_dyn_rooms_large / field_unet / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.270 [0.209,0.405] (succ 1.00, n12)|n/a|n/a|n/a|
|1|n/a|0.274 [0.234,0.326] (succ 1.00, n36)|0.270 [0.203,0.320] (succ 0.95, n36)|0.219 [0.188,0.350] (succ 0.48, n21)|
|4|n/a|0.160 [0.137,0.239] (succ 1.00, n36)|0.194 [0.145,0.229] (succ 0.98, n36)|0.146 [0.110,0.199] (succ 0.93, n34)|
|16|n/a|0.155 [0.097,0.214] (succ 1.00, n36)|0.136 [0.108,0.242] (succ 0.98, n36)|0.145 [0.103,0.205] (succ 0.98, n36)|
Crossover read: full_ft K1->K16: 0.270->0.136 (improves); lora K1 vs zero_shot: 0.274 vs 0.270 (close); lora vs scratch @K1: 0.274 vs 0.219 (transfer does not win)

## C_dyn_rooms_large / scalar_hrm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.486 [0.352,0.822] (succ 0.95, n12)|n/a|n/a|n/a|
|1|n/a|0.366 [0.301,0.404] (succ 0.98, n36)|0.221 [0.174,0.385] (succ 0.93, n36)|0.191 [0.147,0.301] (succ 0.53, n24)|
|4|n/a|0.219 [0.148,0.256] (succ 0.97, n35)|0.218 [0.196,0.277] (succ 0.97, n36)|0.135 [0.105,0.165] (succ 0.92, n33)|
|16|n/a|0.176 [0.131,0.185] (succ 0.98, n36)|0.168 [0.141,0.219] (succ 0.98, n36)|0.130 [0.117,0.191] (succ 0.98, n36)|
Crossover read: full_ft K1->K16: 0.221->0.168 (improves); lora K1 vs zero_shot: 0.366 vs 0.486 (diverges); lora vs scratch @K1: 0.366 vs 0.191 (transfer does not win)

## C_dyn_rooms_large / scalar_hrm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.525 [0.139,0.889] (succ 0.90, n12)|n/a|n/a|n/a|
|1|n/a|0.302 [0.189,0.532] (succ 0.88, n35)|0.252 [0.208,0.319] (succ 0.97, n36)|0.190 [0.163,0.286] (succ 0.52, n23)|
|4|n/a|0.181 [0.145,0.221] (succ 1.00, n36)|0.161 [0.126,0.230] (succ 0.98, n36)|0.132 [0.104,0.161] (succ 0.97, n36)|
|16|n/a|0.164 [0.147,0.226] (succ 1.00, n36)|0.205 [0.185,0.239] (succ 0.98, n36)|0.147 [0.126,0.183] (succ 1.00, n36)|
Crossover read: full_ft K1->K16: 0.252->0.205 (improves); lora K1 vs zero_shot: 0.302 vs 0.525 (diverges); lora vs scratch @K1: 0.302 vs 0.190 (transfer does not win)

## C_dyn_rooms_large / scalar_onlstm / aware
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.856 [0.526,1.586] (succ 0.65, n9)|n/a|n/a|n/a|
|1|n/a|0.428 [0.299,0.556] (succ 0.93, n35)|0.455 [0.320,0.623] (succ 0.92, n36)|0.155 [0.112,0.308] (succ 0.55, n25)|
|4|n/a|0.165 [0.130,0.234] (succ 0.95, n35)|0.213 [0.153,0.293] (succ 0.98, n36)|0.131 [0.097,0.218] (succ 0.97, n36)|
|16|n/a|0.153 [0.129,0.220] (succ 0.98, n36)|0.185 [0.139,0.220] (succ 0.98, n36)|0.137 [0.123,0.198] (succ 0.98, n36)|
Crossover read: full_ft K1->K16: 0.455->0.185 (improves); lora K1 vs zero_shot: 0.428 vs 0.856 (diverges); lora vs scratch @K1: 0.428 vs 0.155 (transfer does not win)

## C_dyn_rooms_large / scalar_onlstm / blind
|K|zero_shot|lora|full_ft|scratch|
|---:|---|---|---|---|
|0|0.710 [0.264,0.900] (succ 0.95, n12)|n/a|n/a|n/a|
|1|n/a|0.473 [0.329,0.622] (succ 0.93, n36)|0.407 [0.225,0.541] (succ 0.88, n35)|0.189 [0.172,0.300] (succ 0.50, n22)|
|4|n/a|0.226 [0.180,0.270] (succ 0.98, n36)|0.182 [0.147,0.227] (succ 0.98, n36)|0.120 [0.089,0.205] (succ 0.97, n36)|
|16|n/a|0.167 [0.139,0.197] (succ 1.00, n36)|0.211 [0.165,0.257] (succ 1.00, n36)|0.156 [0.124,0.200] (succ 1.00, n36)|
Crossover read: full_ft K1->K16: 0.407->0.211 (improves); lora K1 vs zero_shot: 0.473 vs 0.710 (diverges); lora vs scratch @K1: 0.473 vs 0.189 (transfer does not win)
