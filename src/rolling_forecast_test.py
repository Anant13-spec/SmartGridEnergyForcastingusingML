
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "df_final.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "POWER_DEMAND"
HORIZON = 7
STEP = 7

WEATHER = [
    "temp", "tempmax", "tempmin", "feelslike", "humidity",
    "dew", "precip", "precipprob", "windspeed", "windgust",
    "sealevelpressure", "cloudcover", "visibility",
    "solarradiation", "solarenergy", "uvindex",
]

CALENDAR = [
    "year", "month", "day_of_year", "day_of_week", "is_weekend"
]
LAGS = ["demand_lag_1", "demand_lag_7", "demand_mean_7"]


def add_calendar(date):
    return {
        "year": date.year,
        "month": date.month,
        "day_of_year": date.day_of_year,
        "day_of_week": date.day_of_week,
        "is_weekend": int(date.day_of_week >= 5),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = (
        df.dropna(subset=["DATE", TARGET])
        .sort_values("DATE")
        .drop_duplicates("DATE")
        .reset_index(drop=True)
    )

    # Remove invalid targets from training and scoring.
    # Keep the original CSV untouched.
    demand = df[TARGET].where(df[TARGET] > 0)

    weather = [c for c in WEATHER if c in df.columns]
    features = weather + CALENDAR + LAGS

    # Feature construction uses only past demand.
    for key, values in add_calendar(pd.Timestamp("2024-01-01")).items():
        if key == "year":
            df["year"] = df["DATE"].dt.year
        elif key == "month":
            df["month"] = df["DATE"].dt.month
        elif key == "day_of_year":
            df["day_of_year"] = df["DATE"].dt.day_of_year
        elif key == "day_of_week":
            df["day_of_week"] = df["DATE"].dt.day_of_week
        elif key == "is_weekend":
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["demand_lag_1"] = demand.shift(1)
    df["demand_lag_7"] = demand.shift(7)
    df["demand_mean_7"] = demand.shift(1).rolling(7).mean()

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    train_ok = train[TARGET].gt(0) & train[features].notna().any(axis=1)
    X_train = train.loc[train_ok, features].copy()
    y_train = train.loc[train_ok, TARGET].copy()

    # Drop features that have no values in the training period.
    features = [c for c in features if X_train[c].notna().any()]
    X_train = X_train[features]

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

    # Each window is a new forecast issued using only demand
    # observed before that window. No actual demand inside a
    # window is used to update its recursive prediction history.
    all_predictions = []
    window_results = []

    start = 0
    while start < len(test):
        window = test.iloc[start:start + HORIZON]
        # Score the final partial block as well, keeping all chronological
        # holdout dates in the rolling evaluation.

        first_idx = window.index[0]
        history = demand.iloc[:first_idx].tolist()
        observed_before_window = demand.iloc[:first_idx].dropna()
        if observed_before_window.empty:
            raise ValueError("No positive demand available to initialize baseline.")
        baseline_value = float(observed_before_window.iloc[-1])

        predictions = []
        for _, row in window.iterrows():
            date = row["DATE"]
            x = add_calendar(date)

            x["demand_lag_1"] = history[-1] if history else np.nan
            x["demand_lag_7"] = (
                history[-7] if len(history) >= 7 else np.nan
            )
            x["demand_mean_7"] = (
                float(np.mean(history[-7:]))
                if len(history) >= 7 else np.nan
            )

            for col in weather:
                x[col] = row[col]

            X = pd.DataFrame([x])[features]
            pred = max(0.0, float(model.predict(X)[0]))
            predictions.append(pred)
            history.append(pred)

        actual = window[TARGET].to_numpy()
        predicted = np.array(predictions)
        valid = actual > 0

        if valid.any():
            mae = mean_absolute_error(actual[valid], predicted[valid])
            rmse = np.sqrt(mean_squared_error(actual[valid], predicted[valid]))

            window_results.append({
                "start_date": window["DATE"].iloc[0],
                "end_date": window["DATE"].iloc[-1],
                "days_scored": int(valid.sum()),
                "MAE": mae,
                "RMSE": rmse,
            })

        for date, y, p in zip(window["DATE"], actual, predicted):
            all_predictions.append({
                "DATE": date,
                "actual_demand": y,
                "predicted_demand": p,
                "previous_day_baseline": baseline_value,
                "absolute_error": abs(y - p) if y > 0 else np.nan,
            })

        start += STEP

    predictions_df = pd.DataFrame(all_predictions)
    windows_df = pd.DataFrame(window_results)
    scored = predictions_df[predictions_df["actual_demand"] > 0]

    if scored.empty:
        raise ValueError("No valid test demand values were available.")

    overall_mae = mean_absolute_error(
        scored["actual_demand"], scored["predicted_demand"]
    )
    overall_rmse = np.sqrt(mean_squared_error(
        scored["actual_demand"], scored["predicted_demand"]
    ))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(REPORT_DIR / "rolling_forecast_predictions.csv", index=False)
    windows_df.to_csv(REPORT_DIR / "rolling_forecast_metrics.csv", index=False)

    print("\n========== ROLLING SEVEN-DAY TEST ==========")
    print("Training ends:", train["DATE"].max().date())
    print("Test starts:", test["DATE"].min().date())
    print("Test ends:", test["DATE"].max().date())
    print("Forecast windows:", len(windows_df))
    print("\nWindow metrics:")
    print(windows_df.round(2).to_string(index=False))
    print(f"\nOverall MAE:  {overall_mae:.2f}")
    print(f"Overall RMSE: {overall_rmse:.2f}")
    print("\nSaved predictions: reports/rolling_forecast_predictions.csv")
    print("Saved metrics: reports/rolling_forecast_metrics.csv")
    print("\nWeather values are observed historical weather.")
    print("Demand predictions are recursive within each seven-day window.")


if __name__ == "__main__":
    main()
