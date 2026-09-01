# MovingAI external benchmark (dynamic zero-shot)

## street (binding 600, tuned w_h=1.5, n=25)

- success: anchor 0.76 | WA* 0.76 | learned 0.68
- learned-anchor: -0.08 [-0.20,+0.00], discordant 0/2, exact p=5.00e-01
- learned-WA*: -0.08 [-0.20,+0.00], discordant 0/2, exact p=5.00e-01
- matched ratio learned/anchor: 0.481 [0.283,0.643] (n=17)
- matched ratio learned/WA*: 1.050 [0.812,1.218] (n=17)
- joint subopt learned 1.074 vs WA* 1.000, paired diff +0.074 [+0.030,+0.136] (n=17)

## dao (binding 900, tuned w_h=2, n=25)

- success: anchor 0.76 | WA* 0.96 | learned 0.68
- learned-anchor: -0.08 [-0.28,+0.12], discordant 2/4, exact p=6.88e-01
- learned-WA*: -0.28 [-0.48,-0.12], discordant 0/7, exact p=1.56e-02
- matched ratio learned/anchor: 0.796 [0.261,1.670] (n=15)
- matched ratio learned/WA*: 2.196 [1.254,6.360] (n=17)
- joint subopt learned 1.107 vs WA* 1.006, paired diff +0.101 [+0.021,+0.217] (n=17)

