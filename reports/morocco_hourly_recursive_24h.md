# Morocco hourly recursive forecasting: Phase 3

Run `python src/evaluate_morocco_recursive.py` after Phase 2. It evaluates 72
non-overlapping 24-hour windows in the final 1,748-hour chronological holdout.
The final 20 hours cannot form a full 24-hour window and are excluded. Each
zone therefore has 1,728 forecasted hours and 72 residuals per exact forecast
horizon.

The evaluator refits each selected gradient-boosting model on only the first
6,988 hours before forecasting the holdout. Predictions feed into subsequent
hours within each 24-hour window. The comparison uses the previous-day actual
reading, which is available at each window origin. Actual weather is used at
each forecast horizon, so the backtest isolates recursive demand errors and
does not include weather forecast error.

## Overall holdout scores

| Zone | Recursive selected model MAE | Recursive selected model RMSE | Previous-day MAE | Previous-day RMSE |
|---|---:|---:|---:|---:|
| Zone 1 | 1,080.396 | 1,452.268 | 965.815 | 1,454.077 |
| Zone 2 | 1,144.921 | 1,482.527 | 1,162.432 | 1,755.195 |
| Zone 3 | 885.444 | 1,327.649 | 646.742 | 1,184.772 |

The selected model improves RMSE slightly for Zone 1 and improves both
metrics for Zone 2, while the previous-day baseline is better for Zone 1 MAE
and for both Zone 3 metrics. This is mixed performance, not evidence that the
model is reliable in current operation. One-step and recursive results are
shown separately in the dashboard.

Horizon-specific metrics and all predictions are in
`morocco_hourly_recursive_24h_metrics.csv` and
`morocco_hourly_recursive_24h_predictions.csv`. These residuals also provide
the app's empirical 90% error bands, calibrated separately by zone and lead
hour. Each band uses 72 same-horizon residuals from this single 2017 holdout;
it is not a guarantee and excludes weather forecast uncertainty.
