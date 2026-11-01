"""Train and chronologically evaluate separate hourly models for three zones.

Evaluation is one-step-ahead: at each held-out hour, prior observed zone
readings are available as lags. It is not a recursive 24-hour backtest.
Zone targets are hourly means of the documented source readings, not confirmed
hourly energy totals; see reports/morocco_preprocessing.md.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
MODEL_PATH = ROOT / "models" / "morocco_hourly_zone_forecasters.joblib"
METRICS_PATH = ROOT / "reports" / "morocco_hourly_model_comparison.csv"
PREDICTIONS_PATH = ROOT / "reports" / "morocco_hourly_holdout_predictions.csv"
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
LAGS = (1, 2, 24, 168)


def make_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    index = pd.DatetimeIndex(frame[TIME_COLUMN])
    features = pd.DataFrame(index=frame.index)
    features["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * index.day_of_week / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * index.day_of_week / 7)
    features["year_day_sin"] = np.sin(2 * np.pi * index.day_of_year / 365.25)
    features["year_day_cos"] = np.cos(2 * np.pi * index.day_of_year / 365.25)
    features["is_weekend"] = (index.day_of_week >= 5).astype(int)
    for column in WEATHER:
        features[column] = pd.to_numeric(frame[column], errors="coerce")

    values = pd.to_numeric(frame[target], errors="coerce")
    for lag in LAGS:
        features[f"{target}_lag_{lag}"] = values.shift(lag)
    features[f"{target}_mean_24"] = values.shift(1).rolling(24).mean()
    features[f"{target}_mean_168"] = values.shift(1).rolling(168).mean()
    return features


def new_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def score(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(actual, predicted)),
        float(np.sqrt(mean_squared_error(actual, predicted))),
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found at {DATA_PATH}. Run src/preprocess_morocco.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])
    required = {TIME_COLUMN, "SourceReadings", *WEATHER, *TARGETS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hourly dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    if df[TIME_COLUMN].duplicated().any() or not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("Hourly timestamps must be unique and chronological")
    if df[TIME_COLUMN].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Hourly dataset contains a gap; repair preprocessing before training")
    if df[TARGETS].isna().any().any():
        raise ValueError("Zone targets contain missing values")

    split = int(len(df) * 0.8)
    selection_split = int(split * 0.8)
    if split < 168 or split >= len(df):
        raise ValueError("Need at least 168 training rows and one holdout row")
    if selection_split < 168 or selection_split >= split:
        raise ValueError("Need enough data for an internal chronological model-selection split")

    metrics: list[dict[str, object]] = []
    predictions = pd.DataFrame({TIME_COLUMN: df.loc[split:, TIME_COLUMN].to_numpy()})
    final_models = {}

    for target in TARGETS:
        X = make_features(df, target)
        y = df[target].astype(float)
        actual = y.iloc[split:].to_numpy()
        holdout_model = new_model()
        holdout_model.fit(X.iloc[:split], y.iloc[:split])
        hgb_holdout_prediction = holdout_model.predict(X.iloc[split:])
        predictions[f"{target}_actual"] = actual

        mae, rmse = score(actual, hgb_holdout_prediction)
        metrics.append({
            "zone": target,
            "model": "hist_gradient_boosting",
            "features": X.shape[1],
            "train_hours": split,
            "test_hours": len(actual),
            "MAE": mae,
            "RMSE": rmse,
            "evaluation": "one_step_ahead_with_observed_test_lags",
        })

        selection_model = new_model()
        selection_model.fit(X.iloc[:selection_split], y.iloc[:selection_split])
        selection_actual = y.iloc[selection_split:split].to_numpy()
        selection_predictions = selection_model.predict(X.iloc[selection_split:split])
        selection_candidates = [
            ("hist_gradient_boosting", score(selection_actual, selection_predictions)[0], 0)
        ]
        baseline_predictions = {}
        for lag, name in ((1, "previous_hour_baseline"), (24, "previous_day_baseline")):
            baseline = y.shift(lag).iloc[split:].to_numpy()
            baseline_mae, baseline_rmse = score(actual, baseline)
            baseline_predictions[name] = baseline
            selection_baseline = y.shift(lag).iloc[selection_split:split].to_numpy()
            selection_mae, _ = score(selection_actual, selection_baseline)
            selection_candidates.append((name, selection_mae, lag))
            metrics.append({
                "zone": target,
                "model": name,
                "features": 0,
                "train_hours": 0,
                "test_hours": len(actual),
                "MAE": baseline_mae,
                "RMSE": baseline_rmse,
                "evaluation": "one_step_ahead_with_observed_test_lags",
            })

        # Choose model type using the earlier validation block; the final
        # holdout is reserved for evaluation and is never used for selection.
        selected_name, _, selected_lag = min(selection_candidates, key=lambda item: item[1])
        if selected_name == "hist_gradient_boosting":
            model_predicted = hgb_holdout_prediction
            final_model = new_model()
            final_model.fit(X, y)
            final_models[target] = {
                "kind": "hist_gradient_boosting",
                "model": final_model,
                "features": list(X.columns),
            }
            predictions[f"{target}_predicted"] = model_predicted
        else:
            model_predicted = baseline_predictions[selected_name]
            final_models[target] = {
                "kind": "seasonal_naive",
                "model": None,
                "lag_hours": selected_lag,
            }
        predictions[f"{target}_predicted"] = model_predicted

    report = pd.DataFrame(metrics).sort_values(["zone", "MAE"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": final_models,
        "targets": TARGETS,
        "weather_features": WEATHER,
        "frequency": "hourly",
        "target_semantics": "hourly mean of source readings; not verified hourly energy",
        "training_start": str(df[TIME_COLUMN].min()),
        "training_end": str(df[TIME_COLUMN].max()),
        "evaluation": "chronological 80/20 holdout; model type selected using earlier validation data",
    }, MODEL_PATH)
    report.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.6f")

    print(f"Data: {len(df):,} hourly rows, {df[TIME_COLUMN].min()} to {df[TIME_COLUMN].max()}")
    print(f"Chronological split: {split:,} train / {len(df) - split:,} holdout")
    print(f"Model selection used the earlier validation block from hour {selection_split:,} to {split:,}.")
    print("Targets: hourly mean readings for Zones 1–3")
    print("\nOne-step holdout comparison (lower MAE/RMSE is better):")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSelected by zone using earlier validation MAE:")
    for target, spec in final_models.items():
        print(f"{target}: {spec['kind']}" + (f" (lag {spec['lag_hours']}h)" if spec["kind"] == "seasonal_naive" else ""))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print("Evaluation uses observed prior-hour/day readings; it is not a recursive 24-hour forecast evaluation.")


if __name__ == "__main__":
    main()
