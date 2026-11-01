# Morocco hourly models: Phase 2

Run `python src/train_morocco_hourly.py` after preprocessing. The script trains
separate models for the three zones, compares them with previous-hour and
previous-day baselines, and saves a separate artifact at
`models/morocco_hourly_zone_forecasters.joblib`. It does not replace the
existing daily model.

## Evaluation setup

- Input: 8,736 hourly rows from 2017-01-01 00:00 through 2017-12-30 23:00.
- Chronological split: first 6,988 hours for training; final 1,748 hours for
  holdout.
- Model type is selected per zone on a validation block inside the first 6,988
  hours. The final holdout is not used for model selection.
- Each holdout prediction uses observed prior zone readings, so this is a
  rolling one-step evaluation, not a recursive 24-hour forecast backtest.
- Model features include weather readings, cyclic hour/weekday/day-of-year
  features, and zone-specific lag and rolling mean features.
- Scores are MAE and RMSE in the source target scale. The zone values are
  hourly means of source readings; the data dictionary does not establish
  whether they represent energy per interval or point load.

## Holdout comparison

| Zone | Model | MAE | RMSE |
|---|---|---:|---:|
| Zone 1 | Gradient boosting | 705.798 | 977.539 |
| Zone 1 | Previous-day baseline | 968.027 | 1,452.429 |
| Zone 1 | Previous-hour baseline | 1,785.825 | 2,361.852 |
| Zone 2 | Gradient boosting | 606.445 | 799.161 |
| Zone 2 | Previous-day baseline | 1,156.376 | 1,747.195 |
| Zone 2 | Previous-hour baseline | 1,490.462 | 1,990.341 |
| Zone 3 | Previous-day baseline | 644.463 | 1,179.403 |
| Zone 3 | Gradient boosting | 669.361 | 981.228 |
| Zone 3 | Previous-hour baseline | 907.198 | 1,334.350 |

The current saved artifact selects gradient boosting for all three zones based
on the earlier validation block. The final holdout table still compares every
candidate; for Zone 3, the previous-day baseline is better on that final
holdout, so the result indicates the chosen model does not generalize evenly.
Detailed metrics and holdout predictions are in
`morocco_hourly_model_comparison.csv` and
`morocco_hourly_holdout_predictions.csv`.

The final model artifact is refit on all available hourly data after model
selection. The recursive holdout evaluator separately refits on only the first
6,988 hours to avoid training on its holdout.
