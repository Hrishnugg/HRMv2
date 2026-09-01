# C8-S v2 analysis output

## N=192: R1 CI-excl 5/6 (q<.05 in 5/6) PASS; R2 all-ratios<1 True
- maze: succ 0.13->1.00 d=+0.87 [+0.73,+0.97] q=1.788e-07; ratio 0.062 [0.039,0.250] (n_joint 4)
- rooms: succ 0.40->1.00 d=+0.60 [+0.43,+0.77] q=1.526e-05; ratio 0.113 [0.088,0.200] (n_joint 12)
- spiral: succ 0.50->0.97 d=+0.47 [+0.30,+0.63] q=0.0001831; ratio 0.113 [0.072,0.152] (n_joint 15)
- maze_dense: succ 0.00->0.00 d=+0.00 [+0.00,+0.00] q=1; ratio nan [nan,nan] (n_joint 0)
- crossing: succ 0.40->1.00 d=+0.60 [+0.43,+0.77] q=1.526e-05; ratio 0.249 [0.177,0.422] (n_joint 12)
- rooms_large: succ 0.63->0.97 d=+0.33 [+0.17,+0.50] q=0.002344; ratio 0.220 [0.168,0.320] (n_joint 19)

## N=512: R1 CI-excl 5/6 (q<.05 in 5/6) PASS; R2 all-ratios<1 True
- maze: succ 0.17->1.00 d=+0.83 [+0.70,+0.97] q=3.576e-07; ratio 0.075 [0.046,0.088] (n_joint 5)
- rooms: succ 0.73->1.00 d=+0.27 [+0.13,+0.43] q=0.009375; ratio 0.151 [0.094,0.189] (n_joint 22)
- spiral: succ 0.37->1.00 d=+0.63 [+0.47,+0.80] q=1.144e-05; ratio 0.099 [0.077,0.161] (n_joint 11)
- maze_dense: succ 0.00->0.00 d=+0.00 [+0.00,+0.00] q=1; ratio nan [nan,nan] (n_joint 0)
- crossing: succ 0.40->1.00 d=+0.60 [+0.43,+0.77] q=1.526e-05; ratio 0.128 [0.065,0.181] (n_joint 12)
- rooms_large: succ 0.57->0.97 d=+0.40 [+0.23,+0.57] q=0.0007324; ratio 0.183 [0.159,0.326] (n_joint 17)

## N=1024: R1 CI-excl 4/6 (q<.05 in 4/6) FAIL; R2 all-ratios<1 True
- maze: succ 0.13->1.00 d=+0.87 [+0.73,+0.97] q=8.941e-08; ratio 0.072 [0.050,0.160] (n_joint 4)
- rooms: succ 0.33->1.00 d=+0.67 [+0.50,+0.83] q=3.815e-06; ratio 0.091 [0.060,0.130] (n_joint 10)
- spiral: succ 0.43->1.00 d=+0.57 [+0.40,+0.73] q=2.289e-05; ratio 0.120 [0.058,0.169] (n_joint 13)
- maze_dense: succ 0.00->0.00 d=+0.00 [+0.00,+0.00] q=1; ratio nan [nan,nan] (n_joint 0)
- crossing: succ 0.00->1.00 d=+1.00 [+1.00,+1.00] q=1.118e-08; ratio nan [nan,nan] (n_joint 0)
- rooms_large: succ 0.97->0.97 d=+0.00 [-0.10,+0.10] q=1; ratio 0.216 [0.162,0.366] (n_joint 28)

## N=2048: R1 CI-excl 3/6 (q<.05 in 3/6) FAIL; R2 all-ratios<1 True
- maze: succ 0.07->1.00 d=+0.93 [+0.83,+1.00] q=4.47e-08; ratio 0.267 [0.161,0.374] (n_joint 2)
- rooms: succ 0.57->1.00 d=+0.43 [+0.27,+0.60] q=0.0004883; ratio 0.204 [0.162,0.238] (n_joint 17)
- spiral: succ 0.00->0.00 d=+0.00 [+0.00,+0.00] q=1; ratio nan [nan,nan] (n_joint 0)
- maze_dense: succ 0.00->0.00 d=+0.00 [+0.00,+0.00] q=1; ratio nan [nan,nan] (n_joint 0)
- crossing: succ 0.20->1.00 d=+0.80 [+0.67,+0.93] q=3.576e-07; ratio 0.101 [0.042,0.200] (n_joint 6)
- rooms_large: succ 1.00->1.00 d=+0.00 [+0.00,+0.00] q=1; ratio 0.285 [0.207,0.400] (n_joint 30)

## R3 crossover (frozen rule; smallest N or none)

- maze: GPU None CPU None; slopes gpu 0.91 cpu 0.73 wa* 1.27 eu 1.17
    N=192: gpu-wa* dt +0.858s [+0.783,+0.928] q=0.0002285 succd +0.03 sub 1.027/1.021 x=False; t: eu 1.034 wa 0.372 cpu 2.088 gpu 1.230 sipp 1.702
    N=512: gpu-wa* dt +2.169s [+1.966,+2.365] q=0.0002285 succd +0.00 sub 1.023/1.036 x=False; t: eu 3.923 wa 1.008 cpu 4.315 gpu 3.176 sipp 5.962
    N=1024: gpu-wa* dt +3.815s [+3.476,+4.169] q=0.0002285 succd +0.00 sub 1.037/1.040 x=False; t: eu 8.079 wa 1.533 cpu 6.471 gpu 5.349 sipp 11.880
    N=2048: gpu-wa* dt +1.811s [+0.963,+2.682] q=0.0002285 succd +0.03 sub 1.008/1.008 x=False; t: eu 16.610 wa 9.211 cpu 12.100 gpu 11.021 sipp 17.353
- rooms: GPU None CPU None; slopes gpu 0.90 cpu 0.70 wa* 1.29 eu 1.26
    N=192: gpu-wa* dt +1.335s [+1.264,+1.404] q=0.0002285 succd +0.07 sub 1.019/1.033 x=False; t: eu 0.978 wa 0.439 cpu 3.014 gpu 1.774 sipp 2.271
    N=512: gpu-wa* dt +0.194s [+0.030,+0.360] q=0.02749 succd +0.10 sub 1.023/1.000 x=False; t: eu 2.323 wa 2.008 cpu 3.191 gpu 2.201 sipp 4.115
    N=1024: gpu-wa* dt +2.765s [+2.364,+3.159] q=0.0002285 succd +0.03 sub 1.018/1.004 x=False; t: eu 5.520 wa 2.722 cpu 6.681 gpu 5.488 sipp 11.492
    N=2048: gpu-wa* dt +3.604s [+2.838,+4.396] q=0.0002285 succd +0.00 sub 1.001/1.000 x=False; t: eu 20.095 wa 11.082 cpu 15.707 gpu 14.686 sipp 23.229
- spiral: GPU 512 CPU 1024; slopes gpu 0.81 cpu 0.59 wa* 0.18 eu 0.06
    N=192: gpu-wa* dt +0.036s [-0.125,+0.199] q=0.6597 succd +0.20 sub 1.009/1.006 x=False; t: eu 1.944 wa 1.406 cpu 2.702 gpu 1.442 sipp 1.730
    N=512: gpu-wa* dt -1.186s [-1.820,-0.565] q=0.0002285 succd +0.17 sub 1.010/1.007 x=True; t: eu 7.015 wa 4.904 cpu 4.863 gpu 3.718 sipp 5.897
    N=1024: gpu-wa* dt -5.644s [-6.883,-4.407] q=0.0002285 succd +0.17 sub 1.010/1.005 x=True; t: eu 15.564 wa 12.133 cpu 7.556 gpu 6.490 sipp 12.364
    N=2048: gpu-wa* dt +8.155s [+8.127,+8.185] q=0.0002285 succd +0.00 sub nan/nan x=False; t: eu 1.428 wa 1.409 cpu 10.726 gpu 9.564 sipp 25.010
- maze_dense: GPU None CPU None; slopes gpu 0.72 cpu 0.55 wa* 1.02 eu 1.02
    N=192: gpu-wa* dt +1.917s [+1.894,+1.943] q=0.0002285 succd +0.00 sub nan/nan x=False; t: eu 0.155 wa 0.155 cpu 3.472 gpu 2.072 sipp 3.653
    N=512: gpu-wa* dt +3.396s [+3.364,+3.435] q=0.0002285 succd +0.00 sub nan/nan x=False; t: eu 0.433 wa 0.431 cpu 5.304 gpu 3.827 sipp 9.551
    N=1024: gpu-wa* dt +5.528s [+5.501,+5.555] q=0.0002285 succd +0.00 sub nan/nan x=False; t: eu 0.871 wa 0.870 cpu 7.753 gpu 6.398 sipp 18.776
    N=2048: gpu-wa* dt +9.911s [+9.877,+9.945] q=0.0002285 succd +0.00 sub nan/nan x=False; t: eu 1.723 wa 1.708 cpu 13.037 gpu 11.619 sipp 36.821
- crossing: GPU None CPU None; slopes gpu 0.62 cpu 0.43 wa* 1.63 eu 1.35
    N=192: gpu-wa* dt +1.347s [+1.328,+1.370] q=0.0002285 succd +0.00 sub 1.149/1.009 x=False; t: eu 0.144 wa 0.025 cpu 2.507 gpu 1.371 sipp 2.653
    N=512: gpu-wa* dt +1.421s [+1.380,+1.462] q=0.0002285 succd +0.07 sub 1.108/1.000 x=False; t: eu 0.514 wa 0.319 cpu 2.734 gpu 1.740 sipp 5.140
    N=1024: gpu-wa* dt +2.807s [+2.768,+2.844] q=0.0002285 succd +0.00 sub 1.136/1.000 x=False; t: eu 0.608 wa 0.144 cpu 3.958 gpu 2.951 sipp 9.591
    N=2048: gpu-wa* dt +3.769s [+3.532,+4.054] q=0.0002285 succd +0.00 sub 1.072/1.000 x=False; t: eu 4.715 wa 2.176 cpu 6.994 gpu 5.945 sipp 19.184
- rooms_large: GPU None CPU None; slopes gpu 0.90 cpu 0.70 wa* 1.66 eu 1.35
    N=192: gpu-wa* dt +1.602s [+1.552,+1.649] q=0.0002285 succd +0.00 sub 1.057/1.003 x=False; t: eu 0.617 wa 0.264 cpu 3.156 gpu 1.866 sipp 3.735
    N=512: gpu-wa* dt +1.467s [+1.337,+1.603] q=0.0002285 succd +0.17 sub 1.079/1.000 x=False; t: eu 1.371 wa 1.060 cpu 3.783 gpu 2.527 sipp 7.339
    N=1024: gpu-wa* dt +2.887s [+2.339,+3.457] q=0.0002285 succd -0.03 sub 1.068/1.000 x=False; t: eu 4.099 wa 3.468 cpu 7.602 gpu 6.355 sipp 19.498
    N=2048: gpu-wa* dt +1.397s [-0.023,+3.037] q=0.05801 succd +0.00 sub 1.028/1.000 x=False; t: eu 14.997 wa 13.565 cpu 16.289 gpu 14.962 sipp 39.026

## Slope contrasts (paired world bootstrap; non-degenerate suites)

- maze: wastar-minus-learned_gpu +0.36 [+0.30,+0.43]; wastar-minus-learned_cpu +0.54 [+0.47,+0.62]; euclid-minus-learned_gpu +0.26 [+0.22,+0.30]
- rooms: wastar-minus-learned_gpu +0.38 [+0.32,+0.45]; wastar-minus-learned_cpu +0.58 [+0.52,+0.65]; euclid-minus-learned_gpu +0.35 [+0.32,+0.39]
- crossing: wastar-minus-learned_gpu +1.02 [+0.92,+1.12]; wastar-minus-learned_cpu +1.21 [+1.11,+1.31]; euclid-minus-learned_gpu +0.73 [+0.70,+0.77]
- rooms_large: wastar-minus-learned_gpu +0.76 [+0.67,+0.86]; wastar-minus-learned_cpu +0.96 [+0.86,+1.06]; euclid-minus-learned_gpu +0.45 [+0.40,+0.49]

## R4 SIPP (succ | mean t per suite)

- N=192: maze 1.00|1.70s; rooms 1.00|2.27s; spiral 0.97|1.73s; maze_dense 1.00|3.65s; crossing 1.00|2.65s; rooms_large 1.00|3.73s
- N=512: maze 1.00|5.96s; rooms 1.00|4.11s; spiral 1.00|5.90s; maze_dense 1.00|9.55s; crossing 1.00|5.14s; rooms_large 1.00|7.34s
- N=1024: maze 1.00|11.88s; rooms 1.00|11.49s; spiral 1.00|12.36s; maze_dense 1.00|18.78s; crossing 1.00|9.59s; rooms_large 1.00|19.50s
- N=2048: maze 1.00|17.35s; rooms 1.00|23.23s; spiral 1.00|25.01s; maze_dense 1.00|36.82s; crossing 1.00|19.18s; rooms_large 1.00|39.03s

## R5 GPU table-build mean (s): N=192 1.402, N=512 2.323, N=1024 4.420, N=2048 7.605

## Probe (median r | MAE | bias)

- N=192: maze 0.64|0.88|-0.72; rooms 0.49|0.89|-0.83; spiral 0.74|0.83|-0.69; maze_dense 0.79|1.05|-1.03; crossing 0.20|0.85|-0.15; rooms_large 0.54|1.32|-1.07
- N=512: maze 0.62|0.87|-0.81; rooms 0.50|0.90|-0.87; spiral 0.75|0.84|-0.81; maze_dense 0.77|0.90|-0.86; crossing 0.22|0.83|-0.27; rooms_large 0.56|1.39|-1.24
- N=1024: maze 0.58|0.88|-0.79; rooms 0.48|1.01|-1.01; spiral 0.74|0.83|-0.82; maze_dense 0.76|0.89|-0.87; crossing 0.23|0.84|-0.32; rooms_large 0.57|1.39|-1.21
- N=2048: maze 0.68|1.25|-1.25; rooms 0.59|1.37|-1.37; spiral 0.79|1.24|-1.24; maze_dense 0.79|1.35|-1.35; crossing 0.30|0.84|-0.68; rooms_large 0.58|1.64|-1.46

## CPU/GPU divergence: found mismatches 0, worlds with exp diff 100, max |d exp| 10