# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanzzuan/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanzzuan/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — a program
(`python3 main.py`, Python 3.13, standard library only) that reads every
`*.in` case in the input directory and writes five quantiles of each
trophy's earn rate at the case's target to the output directory.

## Why rates move, and what this entry models

The score is a pinball loss on the logit scale, so a rate that halves costs
about as much as one that doubles. The rates themselves are ratios —
earners over players tracked — and the two sides move for different reasons:

- **Dilution.** Exophase refreshes a game's page every few hours, not
  continuously. On launch night the tracked base sat at 638 players for
  hours, then jumped to 3,868 at the first refresh — while the newcomers
  had earned almost nothing yet (the refreshed page's total trophies
  earned, 4,276, is barely more than the newcomers' one near-automatic
  trophy each). So at a refresh most rates fall roughly like 638/3868.
  A forecaster that says "rates stay" is wrong by ~1.6 logit units on
  almost every trophy.
- **Progression.** The players already tracked keep playing and earning,
  which pushes rates back up over a multi-day horizon — more for
  early-story trophies, less for late or grindy ones.

So this program models **counts, not rates**: players arrive in cohorts
(one per refresh, each with an age — how long its players had already been
playing when first counted), and each trophy has an earn curve over a
cohort's age, fitted from the case's history. The forecast is a scenario
distribution over future player growth and earn-curve uncertainty, reported
as weighted quantiles per trophy.

## The model, briefly

- **History as a staircase.** Consecutive identical reads are collapsed
  into distinct states; only refreshes change anything.
- **Earn curves.** Per trophy, a timescale `tau` (early/easy trophies are
  fast) under one of two families — Weibull (with a shape and an optional
  flat foot before any earning) or lognormal (a wide spread of player
  speeds). The shape, the first cohort's age, and the newcomers' age at
  first count are fitted globally on the logit scale; the best model of
  each family is kept and averaged by fit, because the two families fit a
  two-day history almost equally well but extrapolate differently.
- **Player growth.** Two curves fitted to the tracked-player counts — a
  power law in (t + c) with the offset fitted, and a saturating
  exponential — blended in log space and weighted by how well each fits.
  Their disagreement widens the spread. With a flat history (one distinct
  read, as on launch night) no growth can be fitted, so the program falls
  back to a refresh mixture: the page refreshes before the target with a
  probability that grows with the horizon and with how stale the page
  already is, and players then jump by a lognormal multiplier (median 6x,
  the size of the launch night's first refresh).
- **Quantiles.** Future player growth (5 log-points) crossed with
  earn-timescale uncertainty (5 multipliers), each scenario's rates from
  the cohort convolution, then weighted quantiles, widened a little more
  for Poisson count noise and a model floor.

## Local validation

Against a local pinball scorer built from the rules, under several
plausible refresh/dilution scenarios for the practice cases and synthetic
final cases with known ground truth (three player-growth families,
several earn-curve shapes, delays, lognormal truths):

- Practice (docs-like refresh, 638 → 3,868 players): **loss 0.23–0.34**,
  vs the house baseline's 1.17–1.48. Also beats it clearly under mild
  (2,000 players) and strong (8,000 players) refreshes.
- Synthetic finals (rich 2-day history, 4.5-day horizon): **loss
  0.08–0.24** in the standard worlds, vs the baseline's 0.32–1.25 —
  roughly 3–6x better. In deliberately hostile worlds (every trophy
  delayed, or a lognormal truth under a power-law player boom) it still
  beats the baseline, 0.15–0.81 vs 0.56–1.57.
- Shape, determinism, runtime: 55 rows per case, quantiles
  non-decreasing in 0–100, outputs ~2.8 KB (limit 65,536), two runs
  byte-identical, ~0.07 s per practice case and ~0.28 s per synthetic
  final case (limit 8 s).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
