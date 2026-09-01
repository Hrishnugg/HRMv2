# C8 success-vs-budget curves (derived from one 4x-binding pass)

Run `c8r_budget_curves`; thresholding rule success(B) = [found and expansions <= B].

## Crossing (binding 150, high 600, n=50)
- euclid: success 0.12 @ binding -> 1.00 @ 4x binding
- field_unet_blind: success 0.92 @ binding -> 1.00 @ 4x binding
- field_unet: success 1.00 @ binding -> 1.00 @ 4x binding
- oracle: success 1.00 @ binding -> 1.00 @ 4x binding
- euclid catch-up to blind@binding (0.92): budget 374 (2.49x binding)

## Maze (binding 1800, high 7200, n=50)
- euclid: success 0.12 @ binding -> 0.98 @ 4x binding
- field_unet_blind: success 0.96 @ binding -> 0.98 @ 4x binding
- field_unet: success 0.98 @ binding -> 0.98 @ 4x binding
- oracle: success 0.98 @ binding -> 0.98 @ 4x binding
- euclid catch-up to blind@binding (0.96): budget 4131 (2.29x binding)

## Dense maze (binding 2500, high 10000, n=50)
- euclid: success 0.06 @ binding -> 0.94 @ 4x binding
- field_unet_blind: success 0.70 @ binding -> 0.94 @ 4x binding
- field_unet: success 0.58 @ binding -> 0.94 @ 4x binding
- oracle: success 0.86 @ binding -> 0.94 @ 4x binding
- euclid catch-up to blind@binding (0.70): budget 4880 (1.95x binding)

## Rooms (binding 1300, high 5200, n=50)
- euclid: success 0.42 @ binding -> 1.00 @ 4x binding
- field_unet_blind: success 1.00 @ binding -> 1.00 @ 4x binding
- field_unet: success 1.00 @ binding -> 1.00 @ 4x binding
- oracle: success 1.00 @ binding -> 1.00 @ 4x binding
- euclid catch-up to blind@binding (1.00): budget 2255 (1.73x binding)

## Large rooms (binding 600, high 2400, n=50)
- euclid: success 0.82 @ binding -> 1.00 @ 4x binding
- field_unet_blind: success 1.00 @ binding -> 1.00 @ 4x binding
- field_unet: success 0.80 @ binding -> 1.00 @ 4x binding
- oracle: success 1.00 @ binding -> 1.00 @ 4x binding
- euclid catch-up to blind@binding (1.00): budget 1162 (1.94x binding)

## Spiral (binding 2500, high 10000, n=50)
- euclid: success 0.16 @ binding -> 1.00 @ 4x binding
- field_unet_blind: success 1.00 @ binding -> 1.00 @ 4x binding
- field_unet: success 0.96 @ binding -> 1.00 @ 4x binding
- oracle: success 1.00 @ binding -> 1.00 @ 4x binding
- euclid catch-up to blind@binding (1.00): budget 6999 (2.80x binding)

## Binding-budget cross-check vs c8r_fresh_eval
PASS — thresholding at the binding budget reproduces every fresh-eval success value exactly (all suites, all providers).
