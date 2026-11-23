# Preprocessing & Data Quality Report: 3-Zone Smart Grid Dataset

## Overview
- **Raw input file:** `df_3_zones_final.csv`
- **Total raw rows:** 393,152
- **Clean 5-minute rows:** 392,864 (first 288 rows reserved for 24-hour lag initialization)
- **Aggregated hourly rows:** 34,537
- **Coverage window:** 2021-01-03 00:30:00 to 2024-12-12 00:30:00 (~4 years: 2021 to 2024)

## Timestamps & Intervals
- **Modal interval:** 0 days 00:05:00 (predominantly 5-minute sampling, 99.7% of all intervals)
- **Non-modal interval count:** 1,150
- **Duplicate timestamps:** 0
- **Top interval frequencies:**
  * `0 days 00:05:00`: 392,001 occurrences
  * `0 days 00:10:00`: 272 occurrences
  * `0 days 00:15:00`: 267 occurrences
  * `0 days 00:20:00`: 93 occurrences
  * `0 days 01:20:00`: 52 occurrences

## Three-Zone Consistency vs Total Power Demand
- **Zone 1 mean demand:** 1538.48 MW
- **Zone 2 mean demand:** 1456.08 MW
- **Zone 3 mean demand:** 1088.39 MW
- **Zone Sum mean:** 4082.04 MW
- **Total Power Demand mean:** 3962.02 MW
- **Mean Absolute Difference:** 428.24 MW
- **Correlation:** 0.9471
> **Finding:** The total `Power demand` column is **not** strictly identical to the sum of the three zones. In real grid operations, total demand accounts for total state grid load/drawal, while the three zones represent major distribution areas (such as Delhi's DISCOM zones BRPL, BYPL, TPDDL), with residual unmetered / auxiliary / railway / institutional load. Therefore, both individual zone models and total grid load models are independently supported.

## Feature Engineering & Data Leakage Prevention
- **Strictly Non-Leaking Lags:**
  - `lag_1`: 5 minutes prior ($t - 1$)
  - `lag_12`: 1 hour prior ($t - 12$)
  - `lag_288`: 24 hours prior ($t - 288$)
  - `rolling_mean_1h`: Past 1-hour rolling mean calculated strictly as `.shift(1).rolling(12).mean()`
- **Contemporaneous Feature Exclusion:**
  - `moving_avg_3` was found to be computed with `rolling(3).mean()` including the current row ($t$). It is strictly excluded from all training and inference feature sets.
- **Cyclical Encoding:**
  - Hour, minute, day of week, and month are transformed into trigonometric sine/cosine pairs to capture daily and annual seasonality smoothly.

## Processed Outputs
- `data/processed/clean_3_zones_5min.csv`: Full high-resolution 5-minute clean dataset.
- `data/processed/hourly_powerconsumption.csv`: Hourly resampled dataset for macro-level analysis.
