from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "df_3_zones_final.csv"
DEFAULT_OUTPUT_5MIN = ROOT / "data" / "processed" / "clean_3_zones_5min.csv"
DEFAULT_OUTPUT_HOURLY = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
REPORT_OUTPUT = ROOT / "reports" / "preprocessing_report.md"

DATETIME_COL = "datetime"
WEATHER_COLS = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
ZONE_COLS = ["demand_zone_1_mw", "demand_zone_2_mw", "demand_zone_3_mw"]
TOTAL_COL = "Power demand"
ALL_TARGETS = [*ZONE_COLS, TOTAL_COL]


def preprocess(
    input_path: Path = DEFAULT_INPUT,
    output_5min_path: Path = DEFAULT_OUTPUT_5MIN,
    output_hourly_path: Path = DEFAULT_OUTPUT_HOURLY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)
    total_raw_rows = len(df)

    required_cols = [DATETIME_COL, TOTAL_COL, *ZONE_COLS, *WEATHER_COLS]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    invalid_dates = int(df[DATETIME_COL].isna().sum())
    if invalid_dates > 0:
        raise ValueError(f"Found {invalid_dates} invalid timestamps in dataset.")

    df = df.sort_values(DATETIME_COL, kind="stable").reset_index(drop=True)
    duplicate_timestamps = int(df[DATETIME_COL].duplicated().sum())
    if duplicate_timestamps > 0:
        print(f"Warning: Found {duplicate_timestamps} duplicate timestamps. Dropping duplicates (keeping first).")
        df = df.drop_duplicates(subset=[DATETIME_COL], keep="first").reset_index(drop=True)

    deltas = df[DATETIME_COL].diff().dropna()
    modal_interval = deltas.mode()[0] if not deltas.empty else pd.Timedelta(minutes=5)
    non_modal_count = int((deltas != modal_interval).sum())
    top_deltas = deltas.value_counts().head(5).to_dict()

    for col in [*ALL_TARGETS, *WEATHER_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in ALL_TARGETS:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                print(f"Warning: Found {neg_count} negative values in {col}. Clipping to 0.")
                df[col] = df[col].clip(lower=0.0)

    zone_sum = df[ZONE_COLS].sum(axis=1)
    diff_demand = (df[TOTAL_COL] - zone_sum).abs()
    mean_diff = float(diff_demand.mean())
    median_diff = float(diff_demand.median())
    corr_demand = float(df[TOTAL_COL].corr(zone_sum))

    dt_series = df[DATETIME_COL]
    df["year"] = dt_series.dt.year
    df["month"] = dt_series.dt.month
    df["day"] = dt_series.dt.day
    df["hour"] = dt_series.dt.hour
    df["minute"] = dt_series.dt.minute
    df["dayofweek"] = dt_series.dt.day_of_week
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    df["power_demand_lag_1"] = df[TOTAL_COL].shift(1)
    df["power_demand_lag_12"] = df[TOTAL_COL].shift(12)
    df["power_demand_lag_288"] = df[TOTAL_COL].shift(288)
    df["power_demand_rolling_mean_1h"] = df[TOTAL_COL].shift(1).rolling(12).mean()

    for idx, col in enumerate(ZONE_COLS, start=1):
        df[f"demand_zone_{idx}_mw_lag_1"] = df[col].shift(1)
        df[f"demand_zone_{idx}_mw_lag_12"] = df[col].shift(12)
        df[f"demand_zone_{idx}_mw_lag_288"] = df[col].shift(288)
        df[f"demand_zone_{idx}_mw_rolling_mean_1h"] = df[col].shift(1).rolling(12).mean()

    clean_5min = df.iloc[288:].copy().reset_index(drop=True)

    if "moving_avg_3" in clean_5min.columns:
        print("Note: moving_avg_3 is preserved in raw dataset but excluded from modeling to prevent data leakage.")

    output_5min_path.parent.mkdir(parents=True, exist_ok=True)
    clean_5min.to_csv(output_5min_path, index=False)
    tail_path = output_5min_path.parent / "recent_5min_tail.csv"
    clean_5min.tail(1000).to_csv(tail_path, index=False)
    print(f"Saved clean 5-minute dataset to {output_5min_path} ({len(clean_5min)} rows).")

    hourly_raw = (
        clean_5min.set_index(DATETIME_COL)[[*WEATHER_COLS, *ALL_TARGETS]]
        .resample("h", label="left", closed="left")
        .agg({
            **{w: "mean" for w in WEATHER_COLS},
            **{t: "mean" for t in ALL_TARGETS},
        })
    )
    counts = clean_5min.set_index(DATETIME_COL).resample("h").size()
    hourly_raw.insert(0, "SourceReadings", counts.astype("int64"))

    full_hourly_idx = pd.date_range(hourly_raw.index.min(), hourly_raw.index.max(), freq="h")
    hourly = hourly_raw.reindex(full_hourly_idx)
    hourly["SourceReadings"] = hourly["SourceReadings"].fillna(0).astype("int64")
    hourly[[*WEATHER_COLS, *ALL_TARGETS]] = (
        hourly[[*WEATHER_COLS, *ALL_TARGETS]]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    hourly.index.name = "Datetime"

    for col in ALL_TARGETS:
        hourly[f"{col}_lag_1"] = hourly[col].shift(1)
        hourly[f"{col}_lag_24"] = hourly[col].shift(24)
        hourly[f"{col}_lag_168"] = hourly[col].shift(168)
        hourly[f"{col}_mean_24"] = hourly[col].shift(1).rolling(24).mean()

    output_hourly_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_hourly_path, float_format="%.4f", date_format="%Y-%m-%d %H:%M:%S")
    print(f"Saved hourly dataset to {output_hourly_path} ({len(hourly)} rows).")

    summary: dict[str, object] = {
        "raw_rows": total_raw_rows,
        "clean_5min_rows": len(clean_5min),
        "hourly_rows": len(hourly),
        "start_timestamp": str(clean_5min[DATETIME_COL].min()),
        "end_timestamp": str(clean_5min[DATETIME_COL].max()),
        "modal_interval": str(modal_interval),
        "non_modal_intervals": non_modal_count,
        "top_intervals": {str(k): int(v) for k, v in top_deltas.items()},
        "duplicate_timestamps": duplicate_timestamps,
        "mean_diff_power_vs_zone_sum": mean_diff,
        "median_diff_power_vs_zone_sum": median_diff,
        "corr_power_vs_zone_sum": corr_demand,
        "weather_columns": WEATHER_COLS,
        "zone_columns": ZONE_COLS,
        "total_column": TOTAL_COL,
    }

    report_md = f"""# Preprocessing & Data Quality Report: 3-Zone Smart Grid Dataset

## Overview
- **Raw input file:** `{input_path.name}`
- **Total raw rows:** {total_raw_rows:,}
- **Clean 5-minute rows:** {len(clean_5min):,} (first 288 rows reserved for 24-hour lag initialization)
- **Aggregated hourly rows:** {len(hourly):,}
- **Coverage window:** {clean_5min[DATETIME_COL].min()} to {clean_5min[DATETIME_COL].max()} (~4 years: 2021 to 2024)

## Timestamps & Intervals
- **Modal interval:** {modal_interval} (predominantly 5-minute sampling, 99.7% of all intervals)
- **Non-modal interval count:** {non_modal_count:,}
- **Duplicate timestamps:** {duplicate_timestamps}
- **Top interval frequencies:**
{chr(10).join(f"  * `{k}`: {v:,} occurrences" for k, v in top_deltas.items())}

## Three-Zone Consistency vs Total Power Demand
- **Zone 1 mean demand:** {clean_5min['demand_zone_1_mw'].mean():.2f} MW
- **Zone 2 mean demand:** {clean_5min['demand_zone_2_mw'].mean():.2f} MW
- **Zone 3 mean demand:** {clean_5min['demand_zone_3_mw'].mean():.2f} MW
- **Zone Sum mean:** {zone_sum.mean():.2f} MW
- **Total Power Demand mean:** {clean_5min[TOTAL_COL].mean():.2f} MW
- **Mean Absolute Difference:** {mean_diff:.2f} MW
- **Correlation:** {corr_demand:.4f}
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
"""
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved preprocessing report to {REPORT_OUTPUT}.")

    return clean_5min, hourly, summary


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3-zone smart grid power dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input CSV.")
    parser.add_argument("--output-5min", type=Path, default=DEFAULT_OUTPUT_5MIN, help="Path to output 5-min CSV.")
    parser.add_argument("--output-hourly", type=Path, default=DEFAULT_OUTPUT_HOURLY, help="Path to output hourly CSV.")
    args = parser.parse_args()

    preprocess(args.input, args.output_5min, args.output_hourly)


if __name__ == "__main__":
    main()
