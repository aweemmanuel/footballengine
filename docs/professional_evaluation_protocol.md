# Professional Evaluation Protocol

This project treats closing market odds as the strongest public baseline. The goal is not simply to predict winners; it is to identify calibrated probability differences large enough to survive vig, variance, and realistic staking.

## Fusion Layer Rules

1. Use rolling-origin validation only. A validation season may never influence feature engineering, model fitting, calibration, or fusion weights for that same season.
2. Evaluate model probabilities separately from directional accuracy.
3. Compare every fold against the closing-line benchmark.
4. Treat tipster consensus as a market-sentiment feature, not independent truth.
5. Only recommend value candidates when model probability exceeds closing implied probability by a defined edge threshold.

## Calibration Layer

Each fold fits a probability calibrator on the training seasons only. The current implementation searches a small grid of temperature-scaling values and market-blend weights, then applies the selected settings to the next unseen validation season. This prevents the fusion layer from learning weights on the same window it is judged against.

## Metrics

- Accuracy: directional hit rate for home/draw/away.
- Brier score: probability calibration and sharpness.
- Log loss: penalty for confident wrong probabilities.
- Market accuracy and market Brier: closing odds baseline.
- CLV-style edge: model probability minus market implied probability.
- Bankroll ROI: fractional Kelly simulation on value candidates.
- Max drawdown: worst peak-to-trough bankroll decline.
- Reliability bins: whether confidence levels match realized hit rates.
- Tipster-market correlation: double-counting risk indicator.

## Tipster Double-Counting Rule

If `tipster_*_consensus` has mean correlation above `0.80` with closing implied probabilities, it is marked `high_double_counting_risk`. In that case, it should be downweighted, residualized against market odds, or excluded from the fusion layer.

## Betting Discipline

The system should skip most matches. A match is actionable only when:

- data completeness is acceptable,
- model disagreement is not extreme,
- the edge clears the minimum threshold,
- the bet does not over-concentrate the day's portfolio,
- responsible gambling limits allow the stake.
