
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "df_final.csv"
MODEL_PATH = ROOT / "models" / "load_forecaster.joblib"
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"

TARGET = "POWER_DEMAND"

WEATHER_COLUMNS = [
    "temp",
    "tempmax",
    "tempmin",
    "feelslike",
    "humidity",
    "dew",
    "precip",
    "precipprob",
    "windspeed",
    "windgust",
    "sealevelpressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
]

def main():
    df = pd.read_csv(DATA_PATH)

    required = {"DATE", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = (
        df.dropna(subset=["DATE", TARGET])
        .sort_values("DATE")
        .drop_duplicates(subset=["DATE"])
        .reset_index(drop=True)
    )

    if not df["DATE"].is_monotonic_increasing:
        raise ValueError("Dates must be in chronological order.")

    # Keep the original target intact. Zero-demand records are
    # excluded from fitting, pending data-quality verification.
    valid_target = df[TARGET] > 0
    demand_history = df[TARGET].where(valid_target).ffill()

    # Calendar features
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["day_of_year"] = df["DATE"].dt.dayofyear
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Past-demand features only; never shift backwards from the future.
    df["demand_lag_1"] = demand_history.shift(1)
    df["demand_lag_7"] = demand_history.shift(7)
    df["demand_mean_7"] = demand_history.shift(1).rolling(7).mean()

    weather = [c for c in WEATHER_COLUMNS if c in df.columns]
    calendar = [
        "year", "month", "day_of_year",
        "day_of_week", "is_weekend",
    ]
    lags = ["demand_lag_1", "demand_lag_7", "demand_mean_7"]

    # Compare weather-aware and weather-free models.
    feature_sets = {
        "calendar_and_lags": calendar + lags,
        "weather_calendar_and_lags": weather + calendar + lags,
    }

    # Hold out the latest 20% of dates. Do not shuffle time-series data.
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    results = []
    fitted_models = {}

    for name, features in feature_sets.items():
        train_ok = (
            train[TARGET].gt(0)
            & train[features].notna().any(axis=1)
        )
        test_ok = (
            test[TARGET].gt(0)
            & test[features].notna().any(axis=1)
        )

        X_train = train.loc[train_ok, features]
        y_train = train.loc[train_ok, TARGET]
        X_test = test.loc[test_ok, features]
        y_test = test.loc[test_ok, TARGET]

        # Drop features that are entirely missing in the training set.
        features = [
            c for c in features
            if X_train[c].notna().any()
        ]
        X_train = X_train[features]
        X_test = X_test[features]

        if len(X_train) < 30 or len(X_test) == 0:
            raise ValueError(f"Insufficient usable rows for {name}.")

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        results.append({
            "model": name,
            "features": len(features),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "MAE": mae,
            "RMSE": rmse,
        })

        fitted_models[name] = {
            "model": model,
            "features": features,
        }

    # Include a previous-day baseline on the same valid test dates.
    baseline_ok = test[TARGET].gt(0) & test["demand_lag_1"].notna()
    baseline_y = test.loc[baseline_ok, TARGET]
    baseline_pred = test.loc[baseline_ok, "demand_lag_1"]

    results.append({
        "model": "previous_day_baseline",
        "features": 1,
        "train_rows": 0,
        "test_rows": len(baseline_y),
        "MAE": mean_absolute_error(baseline_y, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(baseline_y, baseline_pred)),
    })

    report = pd.DataFrame(results).sort_values("MAE")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(REPORT_PATH, index=False)

    # Select the weather-aware model for the initial prototype.
    # Its performance must be checked against the baseline.
    selected = fitted_models["weather_calendar_and_lags"]

    joblib.dump({
        "model": selected["model"],
        "features": selected["features"],
        "target": TARGET,
        "weather_features": weather,
        "model_name": "HistGradientBoostingRegressor",
        "frequency": "daily",
    }, MODEL_PATH)

    print("\nDataset:", df.shape)
    print("Date range:", df["DATE"].min().date(), "to",
          df["DATE"].max().date())
    print("Zero-demand rows excluded from fitting/evaluation:",
          int((df[TARGET] <= 0).sum()))
    print("Weather features used:", weather)

    print("\nModel comparison (lower is better):")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved model:", MODEL_PATH)
    print("Saved comparison:", REPORT_PATH)
    print("\nEvaluation is rolling one-day-ahead, using observed prior demand.")
    print("This does not yet validate multi-day recursive forecasting.")

if __name__ == "__main__":
    main()
