# C8-S Amendment 2 sensitivity output (post hoc, descriptive)

- maze_dense@192 (binding 3500, in_band, w*=5): anchor 0.27 learned 0.83 d=+0.57 [+0.40,+0.73] disc 17/0 p=1.526e-05; WA* 0.83 d=+0.00 [+0.00,+0.00] p=1
- maze_dense@512 (binding 9330, in_band, w*=5): anchor 0.23 learned 0.93 d=+0.70 [+0.53,+0.87] disc 21/0 p=9.537e-07; WA* 0.93 d=+0.00 [+0.00,+0.00] p=1
- maze_dense@1024 (binding 18670, in_band, w*=3): anchor 0.43 learned 0.93 d=+0.50 [+0.33,+0.67] disc 15/0 p=6.104e-05; WA* 0.93 d=+0.00 [+0.00,+0.00] p=1
- maze_dense@2048 (binding 74660, above_band, w*=1.1): anchor 0.93 learned 1.00 d=+0.07 [+0.00,+0.17] disc 2/0 p=0.5; WA* 0.93 d=+0.07 [+0.00,+0.17] p=0.5
- spiral@2048 (binding 74660, above_band, w*=1.1): anchor 1.00 learned 1.00 d=+0.00 [+0.00,+0.00] disc 0/0 p=1; WA* 1.00 d=+0.00 [+0.00,+0.00] p=1
- rooms_large@1024 (sens binding 3200 from recorded rates): anchor 0.43 learned 0.97 d=+0.53 [+0.37,+0.70] disc 16/0 p=3.052e-05; original-w WA* 0.73 d=+0.23 [+0.10,+0.40] (note: WA* weight tuned at the original binding, not re-tuned)
- rooms_large@2048 (sens binding 13870 from recorded rates): anchor 0.87 learned 0.97 d=+0.10 [-0.03,+0.23] disc 4/1 p=0.375; original-w WA* 0.93 d=+0.03 [-0.07,+0.13] (note: WA* weight tuned at the original binding, not re-tuned)