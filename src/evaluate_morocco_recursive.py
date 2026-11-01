"""Evaluate 24-hour recursive forecasts on chronological holdout windows.

At each 24-hour origin, only demand observations before the origin are used.
For model-based zones, each prediction feeds into later lags for that window.
Actual holdout weather is supplied at each forecast timestamp to isolate the
recursive demand component; this is an optimistic weather-input assumption,
not an end-to-end forecast using weather forecasts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_morocco_hourly import make_features, new_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
MODEL_PATH = ROOT / "models" / "morocco_hourly_zone_forecasters.joblib"
METRICS_PATH = ROOT / "reports" / "morocco_hourly_recursive_24h_metrics.csv"
PREDICTIONS_PATH = ROOT / "reports" / "morocco_hourly_recursive_24h_predictions.csv"
TIME_COLUMN = "Datetime"
TARGETS = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]
WEATHER = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
]


def make_one_hour_features(timestamp: pd.Timestamp, weather: pd.Series,
                           history: list[float], target: str) -> dict[str, float]:
    hour = timestamp.hour
    weekday = timestamp.day_of_week
    day_of_year = timestamp.day_of_year
    row = {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "year_day_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "year_day_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
        "is_weekend": int(weekday >= 5),
    }
    row.update({column: float(weather[column]) for column in WEATHER})
    for lag in (1, 2, 24, 168):
        row[f"{target}_lag_{lag}"] = float(history[-lag]) if len(history) >= lag else float("nan")
    row[f"{target}_mean_24"] = float(np.mean(history[-24:])) if len(history) >= 24 else float("nan")
    row[f"{target}_mean_168"] = float(np.mean(history[-168:])) if len(history) >= 168 else float("nan")
    return row


def main() -> None:
    for path in (DATA_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 3 input is missing: {path}")

    df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN]).sort_values(TIME_COLUMN).reset_index(drop=True)
    artifact = joblib.load(MODEL_PATH)
    required = {TIME_COLUMN, *WEATHER, *TARGETS}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Hourly data is missing columns: {sorted(absent)}")
    if df[TIME_COLUMN].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Hourly data has gaps; recursive evaluation requires continuous hours")

    split = int(len(df) * 0.8)
    evaluation_models = {}
    for target in TARGETS:
        selected = artifact["models"][target]
        if selected["kind"] == "hist_gradient_boosting":
            features = make_features(df, target)
            estimator = new_model()
            estimator.fit(features.iloc[:split], df[target].astype(float).iloc[:split])
            evaluation_models[target] = {
                "kind": "hist_gradient_boosting",
                "model": estimator,
                "features": list(features.columns),
            }
        else:
            evaluation_models[target] = selected

    # Use complete, non-overlapping 24-hour windows; restart each window with
    # its actual pre-origin history, as a daily operational forecast would.
    origins = list(range(split, len(df) - 23, 24))
    if not origins:
        raise ValueError("No complete 24-hour holdout windows are available")

    prediction_rows = []
    for origin in origins:
        for target in TARGETS:
            spec = evaluation_models[target]
            history = pd.to_numeric(df.loc[:origin - 1, target], errors="coerce").dropna().astype(float).tolist()
            if len(history) < 168:
                raise ValueError(f"Not enough historical values for {target} at {df.loc[origin, TIME_COLUMN]}")

            for step in range(1, 25):
                row_index = origin + step - 1
                row = df.iloc[row_index]
                timestamp = pd.Timestamp(row[TIME_COLUMN])
                baseline = history[-24] if step <= 24 else None
                if spec["kind"] == "seasonal_naive":
                    predicted = float(history[-int(spec["lag_hours"])])
                else:
                    feature_row = make_one_hour_features(timestamp, row, history, target)
                    X = pd.DataFrame([feature_row])[spec["features"]]
                    predicted = float(spec["model"].predict(X)[0])
                predicted = max(0.0, predicted)
                actual = float(row[target])
                prediction_rows.append({
                    "origin": df.loc[origin, TIME_COLUMN],
                    "target_time": timestamp,
                    "horizon_hour": step,
                    "zone": target,
                    "actual": actual,
                    "recursive_prediction": predicted,
                    "previous_day_baseline": float(baseline),
                    "model_kind": spec["kind"],
                    "weather_input": "observed_holdout_weather",
                })
                history.append(predicted)

    predictions = pd.DataFrame(prediction_rows)
    predictions["horizon_bucket"] = pd.cut(
        predictions["horizon_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["01-06", "07-12", "13-18", "19-24"],
    )

    metrics = []
    for (zone, bucket), group in predictions.groupby(["zone", "horizon_bucket"], observed=True):
        actual = group["actual"].to_numpy()
        for model_name, column in (
            ("recursive_selected_model", "recursive_prediction"),
            ("previous_day_baseline", "previous_day_baseline"),
        ):
            predicted = group[column].to_numpy()
            metrics.append({
                "zone": zone,
                "horizon_hours": str(bucket),
                "model": model_name,
                "records": len(group),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            })
    report = pd.DataFrame(metrics)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(METRICS_PATH, index=False)
    predictions.drop(columns="horizon_bucket").to_csv(PREDICTIONS_PATH, index=False, float_format="%.6f")
    print(f"Holdout: {len(df) - split:,} hours; complete 24-hour windows: {len(origins)}")
    print(f"Evaluated forecast steps: {len(predictions):,}")
    print("Weather input: actual holdout weather at each horizon (optimistic; isolates recursive demand behavior)")
    print("\nRecursive 24-hour performance by horizon bucket (lower is better):")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
