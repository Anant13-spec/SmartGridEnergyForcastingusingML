
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "df_final.csv"
MODEL_PATH = ROOT / "models" / "load_forecaster.joblib"

TARGET = "POWER_DEMAND"

WEATHER = [
    "temp", "tempmax", "tempmin", "feelslike",
    "humidity", "dew", "precip", "precipprob",
    "windspeed", "windgust", "sealevelpressure",
    "cloudcover", "visibility", "solarradiation",
    "solarenergy", "uvindex",
]

CALENDAR = [
    "year", "month", "day_of_year", "day_of_week", "is_weekend"
]

LAGS = ["demand_lag_1", "demand_lag_7", "demand_mean_7"]


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

    # Exclude zero/negative demand from fitting without modifying the CSV.
    demand = df[TARGET].where(df[TARGET] > 0)

    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["day_of_year"] = df["DATE"].dt.day_of_year
    df["day_of_week"] = df["DATE"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["demand_lag_1"] = demand.shift(1)
    df["demand_lag_7"] = demand.shift(7)
    df["demand_mean_7"] = demand.shift(1).rolling(7).mean()

    weather = [c for c in WEATHER if c in df.columns]
    features = weather + CALENDAR + LAGS

    # Use the first 80% of dates, matching the earlier experiment.
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()

    usable = [c for c in features if train[c].notna().any()]
    train_ok = (
        train[TARGET].gt(0)
        & train[usable].notna().any(axis=1)
    )

    X_train = train.loc[train_ok, usable]
    y_train = train.loc[train_ok, TARGET]

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

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": usable,
            "target": TARGET,
            "weather_features": weather,
            "frequency": "daily",
            "training_end": str(train["DATE"].max().date()),
            "model_name": "HistGradientBoostingRegressor",
        },
        MODEL_PATH,
    )

    print("Best candidate saved successfully.")
    print("Model path:", MODEL_PATH)
    print("Training ends:", train["DATE"].max().date())
    print("Training rows:", len(X_train))
    print("Features:", usable)


if __name__ == "__main__":
    main()
