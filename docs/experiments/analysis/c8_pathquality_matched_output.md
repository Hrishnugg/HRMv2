# Matched path quality: blind U-Net vs Euclid (fresh cohort, binding budgets)

| Suite | joint n | learned subopt | anchor subopt | paired diff [95% CI] |
|---|---:|---:|---:|---|
| Crossing | 6 | 1.216 | 1.000 | +0.216 [+0.107, +0.382] |
| Maze | 6 | 1.004 | 1.000 | +0.004 [+0.000, +0.013] |
| Dense maze | 3 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| Rooms | 21 | 1.031 | 1.000 | +0.031 [+0.014, +0.051] |
| Large rooms | 41 | 1.095 | 1.000 | +0.095 [+0.071, +0.121] |
| Spiral | 8 | 1.027 | 1.000 | +0.027 [+0.010, +0.046] |

Anchor arrival == optimal arrival on every anchor-solved map (all suites): True
