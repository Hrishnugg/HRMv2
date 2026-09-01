# C8-X v2 analysis output

## street (binding 400, tuned w_h=3, 12 held-out maps, N_target 65536)

- A1 zero-shot (held-out): learned 0.715 vs anchor 0.618 vs WA* 0.938; d_anchor +0.097 [+0.042,+0.160], d_WA* -0.222 [-0.278,-0.174]
- lora: a8 0.792 a1 0.745 zs 0.715; B1 +0.076 [+0.017,+0.132] p=0.0130 q=0.0347; B2 +0.046 [-0.012,+0.102] p=0.1202 q=0.2404
- full: a8 0.840 a1 0.803 zs 0.715; B1 +0.125 [+0.059,+0.191] p=0.0002 q=0.0016; B2 +0.037 [+0.012,+0.060] p=0.0032 q=0.0128
- scratch (descriptive): a8 0.868 a1 0.832; B1-style +0.153 [+0.097,+0.215]
- dose (held-out mean succ by M): lora M1=0.768 M2=0.780 M4=0.821 M8=0.792; full M1=0.803 M2=0.788 M4=0.800 M8=0.840; scratch M1=0.849 M2=0.838 M4=0.859 M8=0.868
- lora B3 pool-vs-heldout a8: 0.807 vs 0.792 (zs 0.667/0.715); B4 a8-vs-anchor +0.174 [+0.118,+0.233], a8-vs-WA* -0.146 [-0.194,-0.094]
- full B3 pool-vs-heldout a8: 0.828 vs 0.840 (zs 0.667/0.715); B4 a8-vs-anchor +0.222 [+0.167,+0.274], a8-vs-WA* -0.097 [-0.128,-0.063]
- lora B2 BALANCED (8 draws): a1bal 0.768; delta +0.023 [-0.027,+0.076] p=0.3688 q=0.7375
- full B2 BALANCED (8 draws): a1bal 0.803; delta +0.037 [+0.011,+0.063] p=0.0066 q=0.0264
- effort/path zeroshot_vs_anchor: ratio 0.449 [0.379,0.618]; subopt 1.059/1.000 (n_maps 12)
- effort/path zeroshot_vs_wastar: ratio 1.816 [1.303,2.853]; subopt 1.067/1.043 (n_maps 12)
- effort/path a8_lora_vs_anchor: ratio 0.450 [0.386,0.584]; subopt 1.056/1.000 (n_maps 12)
- effort/path a8_lora_vs_wastar: ratio 2.270 [1.556,3.054]; subopt 1.056/1.048 (n_maps 12)
- effort/path a8_full_vs_anchor: ratio 0.441 [0.398,0.553]; subopt 1.110/1.000 (n_maps 12)
- effort/path a8_full_vs_wastar: ratio 2.001 [1.782,2.556]; subopt 1.102/1.046 (n_maps 12)
- fidelity pool free 0.750 [0.742-0.761] comp 10.5; heldout free 0.755 [0.742-0.765] comp 5.7; failed: none

## dao (binding 400, tuned w_h=3, 9 held-out maps, N_target 65536)

- A1 zero-shot (held-out): learned 0.694 vs anchor 0.491 vs WA* 0.843; d_anchor +0.204 [+0.083,+0.324], d_WA* -0.148 [-0.278,-0.028]
- lora: a8 0.676 a1 0.684 zs 0.694; B1 -0.019 [-0.116,+0.088] p=0.6959 q=0.9279; B2 -0.008 [-0.042,+0.025] p=0.5999 q=0.9279
- full: a8 0.694 a1 0.691 zs 0.694; B1 -0.000 [-0.116,+0.116] p=0.9891 q=0.9891; B2 +0.003 [-0.032,+0.045] p=0.9055 q=0.9891
- scratch (descriptive): a8 0.699 a1 0.699; B1-style +0.005 [-0.116,+0.134]
- dose (held-out mean succ by M): lora M1=0.671 M2=0.654 M4=0.648 M8=0.676; full M1=0.702 M2=0.682 M4=0.713 M8=0.694; scratch M1=0.741 M2=0.731 M4=0.708 M8=0.699
- lora B3 pool-vs-heldout a8: 0.656 vs 0.676 (zs 0.564/0.694); B4 a8-vs-anchor +0.185 [+0.111,+0.259], a8-vs-WA* -0.167 [-0.245,-0.083]
- full B3 pool-vs-heldout a8: 0.664 vs 0.694 (zs 0.564/0.694); B4 a8-vs-anchor +0.204 [+0.116,+0.292], a8-vs-WA* -0.148 [-0.227,-0.074]
- lora B2 BALANCED (8 draws): a1bal 0.671; delta +0.005 [-0.022,+0.034] p=0.7497 q=0.8568
- full B2 BALANCED (8 draws): a1bal 0.702; delta -0.008 [-0.044,+0.027] p=0.6881 q=0.8568
- effort/path zeroshot_vs_anchor: ratio 0.580 [0.531,1.075]; subopt 1.084/1.000 (n_maps 9)
- effort/path zeroshot_vs_wastar: ratio 2.452 [1.371,6.473]; subopt 1.074/1.044 (n_maps 9)
- effort/path a8_lora_vs_anchor: ratio 0.521 [0.338,0.687]; subopt 1.027/1.000 (n_maps 9)
- effort/path a8_lora_vs_wastar: ratio 1.813 [1.690,3.244]; subopt 1.024/1.039 (n_maps 9)
- effort/path a8_full_vs_anchor: ratio 0.397 [0.281,0.463]; subopt 1.031/1.000 (n_maps 9)
- effort/path a8_full_vs_wastar: ratio 1.702 [1.328,2.086]; subopt 1.034/1.038 (n_maps 9)
- fidelity pool free 0.498 [0.128-0.916] comp 3.5; heldout free 0.616 [0.230-0.964] comp 1.8; failed: brc100d.map, brc300d.map, brc502d.map
