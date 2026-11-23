from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
MODEL_PATH = ROOT / "models" / "morocco_hourly_zone_forecasters.joblib"
METRICS_PATH = ROOT / "reports" / "morocco_hourly_model_comparison.csv"
PREDICTIONS_PATH = ROOT / "reports" / "morocco_hourly_holdout_predictions.csv"
TIME_COLUMN = "Datetime"
TARGETS = [
    "demand_zone_1_mw",
    "demand_zone_2_mw",
    "demand_zone_3_mw",
]
ALL_TARGETS = [*TARGETS, "Power demand"]
WEATHER = [
    "temp",
    "dwpt",
    "rhum",
    "wdir",
    "wspd",
    "pres",
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
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def score(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[valid], predicted[valid]
    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    r2 = float(r2_score(a, p))
    return mae, rmse, r2


def main() -> None:
    if not DATA_PATH.exists():
        from preprocess import preprocess
        preprocess()

    df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])
    required = {TIME_COLUMN, "SourceReadings", *WEATHER, *ALL_TARGETS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hourly dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    split = int(len(df) * 0.8)
    selection_split = int(split * 0.8)

    metrics: list[dict[str, object]] = []
    predictions = pd.DataFrame({TIME_COLUMN: df.loc[split:, TIME_COLUMN].to_numpy()})
    final_models = {}

    for target in ALL_TARGETS:
        X = make_features(df, target)
        y = df[target].astype(float)
        actual = y.iloc[split:].to_numpy()
        holdout_model = new_model()
        holdout_model.fit(X.iloc[:split], y.iloc[:split])
        hgb_holdout_prediction = holdout_model.predict(X.iloc[split:])
        predictions[f"{target}_actual"] = actual
        predictions[f"{target}_predicted"] = hgb_holdout_prediction

        mae, rmse, r2 = score(actual, hgb_holdout_prediction)
        metrics.append({
            "zone": target,
            "model": "hist_gradient_boosting",
            "features": X.shape[1],
            "train_hours": split,
            "test_hours": len(actual),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "evaluation": "one_step_ahead_with_observed_test_lags",
        })

        selection_model = new_model()
        selection_model.fit(X.iloc[:selection_split], y.iloc[:selection_split])
        selection_actual = y.iloc[selection_split:split].to_numpy()
        selection_predictions = selection_model.predict(X.iloc[selection_split:split])
        sel_mae, _, _ = score(selection_actual, selection_predictions)
        selection_candidates = [
            ("hist_gradient_boosting", sel_mae, 0)
        ]
        baseline_predictions = {}
        for lag, name in ((1, "previous_hour_baseline"), (24, "previous_day_baseline")):
            baseline = y.shift(lag).iloc[split:].to_numpy()
            baseline_mae, baseline_rmse, baseline_r2 = score(actual, baseline)
            baseline_predictions[name] = baseline
            selection_baseline = y.shift(lag).iloc[selection_split:split].to_numpy()
            sel_b_mae, _, _ = score(selection_actual, selection_baseline)
            selection_candidates.append((name, sel_b_mae, lag))
            metrics.append({
                "zone": target,
                "model": name,
                "features": 1,
                "train_hours": 0,
                "test_hours": len(actual),
                "MAE": round(baseline_mae, 2),
                "RMSE": round(baseline_rmse, 2),
                "R2": round(baseline_r2, 4),
                "evaluation": "persistence_baseline",
            })

        best_name, best_sel_mae, best_lag = min(selection_candidates, key=lambda c: c[1])
        if best_name == "hist_gradient_boosting":
            full_model = new_model()
            full_model.fit(X, y)
            final_models[target] = {
                "kind": "hist_gradient_boosting",
                "model": full_model,
                "features": list(X.columns),
            }
            predictions[f"{target}_best_selection"] = hgb_holdout_prediction
        else:
            final_models[target] = {
                "kind": "seasonal_naive",
                "lag_hours": best_lag,
                "model_name": best_name,
            }
            predictions[f"{target}_best_selection"] = baseline_predictions[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": final_models,
        "features": list(X.columns),
        "targets": ALL_TARGETS,
        "weather_columns": WEATHER,
        "lag_hours": list(LAGS),
    }, MODEL_PATH)
    print(f"Saved hourly models artifact to {MODEL_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    print(f"Saved model comparison to {METRICS_PATH}")

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.4f")
    print(f"Saved holdout predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
