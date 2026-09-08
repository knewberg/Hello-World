# Sweep Head v1.2 — Regularized Calibration

This is the active score-shape layer for the NCAA Division I women's volleyball model. It sits **after** the locked 35/25/20/10/10 match-win model and never changes its match-win probability.

## Regularized update

For a fixed historical conditional-sweep head

`logit P(sweep | win) = a + b logit(P(win))`

the current-season bridge estimates only an intercept offset `delta`:

`a + delta + b logit(P(win))`

by maximizing current-season conditional-sweep log likelihood with the L2 penalty

`0.5 * lambda * delta^2`.

The operational penalty is **lambda = 5**.

## Leakage rule

For every target date, all forecasts use one state fit only on matches with `match_date < target_date`. Results from earlier first serves on the same date are not allowed to affect later matches that day.

## What updates online

- Favorite sweep intercept: yes.
- Underdog sweep intercept: only when clean upset-win observations exist.
- 3-1 versus 3-2 split: frozen.
- Probability slopes: frozen.
- Set-shape residual live weight: 0.
- Player residual live weight: 0.

## Current 2026 state

Using the 12 clean locked forecasts completed through 2026-09-07, the favorite sweep bridge is `-0.20340621161358413`. There were no upset wins, so the underdog sweep bridge remains zero.

The model preserves these identities exactly:

`P(A 3-0)+P(A 3-1)+P(A 3-2)=P(A wins)`

and all six exact-score probabilities sum to 1.

## Airtable audit integration

The project base now includes a `Set Calibration States` table. Each future `Set Simulations` record can link the exact state used before first serve. Historical simulations are not retroactively rewritten.
