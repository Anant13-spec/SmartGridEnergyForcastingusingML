
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
MODEL_PATH = ROOT / "models" / "forecast_test_model.joblib"
REPORT_PATH = ROOT / "reports" / "seven_day_forecast.csv"

TARGET = "POWER_DEMAND"

WEATHER_COLUMNS = [
    "temp", "tempmax", "tempmin", "feelslike",
    "humidity", "dew", "precip", "precipprob",
    "windspeed", "windgust", "sealevelpressure",
    "cloudcover", "visibility", "solarradiation",
    "solarenergy", "uvindex",
]

CALENDAR_COLUMNS = [
    "year", "month", "day_of_year",
    "day_of_week", "is_weekend",
]

LAG_COLUMNS = [
    "demand_lag_1", "demand_lag_7", "demand_mean_7",
]


def main():
    df = pd.read_csv(DATA_PATH)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = (
        df.dropna(subset=["DATE", TARGET])
        .sort_values("DATE")
        .drop_duplicates(subset=["DATE"])
        .reset_index(drop=True)
    )

    # Exclude zero-demand observations from fitting and evaluation.
    # Preserve the original CSV unchanged.
    valid_demand = df[TARGET].where(df[TARGET] > 0)
    demand_history = valid_demand.ffill()

    weather = [c for c in WEATHER_COLUMNS if c in df.columns]
    features = weather + CALENDAR_COLUMNS + LAG_COLUMNS

    # Build calendar features.
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["day_of_year"] = df["DATE"].dt.day_of_year
    df["day_of_week"] = df["DATE"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Training rows use observed past demand only.
    df["demand_lag_1"] = demand_history.shift(1)
    df["demand_lag_7"] = demand_history.shift(7)
    df["demand_mean_7"] = demand_history.shift(1).rolling(7).mean()

    # Chronological split: the last 20% is held out.
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    future = df.iloc[split:].copy()

    if len(future) < 7:
        raise ValueError("Not enough held-out rows for a seven-day test.")

    # Fit only on the historical training period.
    train_ok = (
        train[TARGET].gt(0)
        & train[features].notna().any(axis=1)
    )
    X_train = train.loc[train_ok, features].copy()
    y_train = train.loc[train_ok, TARGET].copy()

    # Remove features that are entirely missing in training.
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

    # Seed recursive predictions with demand observed before the cutoff.
    history = demand_history.iloc[:split].tolist()
    results = []

    # Forecast the first seven days after the cutoff.
    # Do not use actual demand from these seven days as input.
    for _, row in future.head(7).iterrows():
        date = row["DATE"]

        lag_1 = history[-1] if history else np.nan
        lag_7 = history[-7] if len(history) >= 7 else np.nan

        last_seven = history[-7:]
        mean_7 = (
            float(np.mean(last_seven))
            if len(last_seven) == 7
            else np.nan
        )

        x = {
            "year": date.year,
            "month": date.month,
            "day_of_year": date.day_of_year,
            "day_of_week": date.day_of_week,
            "is_weekend": int(date.day_of_week >= 5),
            "demand_lag_1": lag_1,
            "demand_lag_7": lag_7,
            "demand_mean_7": mean_7,
        }

        for col in weather:
            x[col] = row[col]

        X_future = pd.DataFrame([x])[features]
        prediction = float(model.predict(X_future)[0])
        prediction = max(0.0, prediction)

        actual = row[TARGET]

        results.append({
            "DATE": date,
            "actual_demand": actual,
            "predicted_demand": prediction,
            "absolute_error": (
                abs(actual - prediction) if actual > 0 else np.nan
            ),
        })

        # Critical: feed the prediction, not actual future demand,
        # into the history used for subsequent forecast days.
        history.append(prediction)

    result = pd.DataFrame(results)
    scored = result[result["actual_demand"] > 0].copy()

    if scored.empty:
        raise ValueError("No positive actual demand in the forecast window.")

    mae = mean_absolute_error(
        scored["actual_demand"], scored["predicted_demand"]
    )
    rmse = np.sqrt(mean_squared_error(
        scored["actual_demand"], scored["predicted_demand"]
    ))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_PATH, index=False)

    joblib.dump({
        "model": model,
        "features": features,
        "target": TARGET,
        "weather_features": weather,
        "frequency": "daily",
    }, MODEL_PATH)

    print("\n========== SEVEN-DAY FORECAST ==========")
    print("Training ends:", train["DATE"].max().date())
    print("Forecast starts:", result["DATE"].min().date())
    print("Forecast ends:", result["DATE"].max().date())
    print("\n", result.round(2).to_string(index=False))
    print(f"\nSeven-day MAE:  {mae:.2f}")
    print(f"Seven-day RMSE: {rmse:.2f}")
    print("\nSaved forecast:", REPORT_PATH)
    print("Saved test model:", MODEL_PATH)
    print("\nFuture actual demand was NOT used as a prediction input.")
    print("Historical weather observations were used for this backtest.")


if __name__ == "__main__":
    main()
