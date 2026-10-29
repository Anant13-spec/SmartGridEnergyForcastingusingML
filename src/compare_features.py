
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "df_final.csv"
REPORT_PATH = ROOT / "reports" / "feature_experiment.csv"

TARGET = "POWER_DEMAND"
HORIZON = 7

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


def make_model(kind):
    if kind == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )

    return make_pipeline(SimpleImputer(strategy="median"), estimator)


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

    demand = df[TARGET].where(df[TARGET] > 0)
    weather = [c for c in WEATHER if c in df.columns]

    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["day_of_year"] = df["DATE"].dt.day_of_year
    df["day_of_week"] = df["DATE"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["demand_lag_1"] = demand.shift(1)
    df["demand_lag_7"] = demand.shift(7)
    df["demand_mean_7"] = demand.shift(1).rolling(7).mean()

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    feature_sets = {
        "weather_calendar": weather + CALENDAR,
        "weather_calendar_lags": weather + CALENDAR + LAGS,
    }

    all_results = []

    for feature_name, features in feature_sets.items():
        usable_features = [
            c for c in features
            if train[c].notna().any()
        ]

        train_ok = (
            train[TARGET].gt(0)
            & train[usable_features].notna().any(axis=1)
        )

        X_train = train.loc[train_ok, usable_features]
        y_train = train.loc[train_ok, TARGET]

        # Test successive seven-day windows, using recursive
        # predictions within each window.
        for model_name in ["HistGradientBoosting", "RandomForest"]:
            model = make_model(model_name)
            model.fit(X_train, y_train)

            start = 0
            while start + HORIZON <= len(test):
                window = test.iloc[start:start + HORIZON]
                first_index = window.index[0]
                history = demand.iloc[:first_index].tolist()
                predictions = []

                for _, row in window.iterrows():
                    date = row["DATE"]

                    x = {
                        "year": date.year,
                        "month": date.month,
                        "day_of_year": date.day_of_year,
                        "day_of_week": date.day_of_week,
                        "is_weekend": int(date.day_of_week >= 5),
                    }

                    for col in weather:
                        x[col] = row[col]

                    x["demand_lag_1"] = (
                        history[-1] if history else np.nan
                    )
                    x["demand_lag_7"] = (
                        history[-7] if len(history) >= 7 else np.nan
                    )
                    x["demand_mean_7"] = (
                        float(np.mean(history[-7:]))
                        if len(history) >= 7 else np.nan
                    )

                    X = pd.DataFrame([x])[usable_features]
                    prediction = max(
                        0.0, float(model.predict(X)[0])
                    )
                    predictions.append(prediction)

                    # Recursive forecast: use predictions, not
                    # actual demand inside the forecast window.
                    history.append(prediction)

                actual = window[TARGET].to_numpy()
                predicted = np.array(predictions)
                valid = actual > 0

                for date, y, p in zip(
                    window["DATE"], actual, predicted
                ):
                    all_results.append({
                        "model": model_name,
                        "features": feature_name,
                        "DATE": date,
                        "actual": y,
                        "predicted": p,
                    })

                start += HORIZON

    result = pd.DataFrame(all_results)
    result = result[result["actual"] > 0].copy()

    summary = (
        result.groupby(["model", "features"])
        .apply(
            lambda g: pd.Series({
                "records": len(g),
                "MAE": mean_absolute_error(
                    g["actual"], g["predicted"]
                ),
                "RMSE": np.sqrt(mean_squared_error(
                    g["actual"], g["predicted"]
                )),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("MAE")
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_PATH, index=False)
    result.to_csv(
        ROOT / "reports" / "feature_experiment_predictions.csv",
        index=False,
    )

    print("\n========== FEATURE EXPERIMENT ==========")
    print(summary.round(2).to_string(index=False))
    print("\nSaved metrics:", REPORT_PATH)
    print("Saved predictions: reports/feature_experiment_predictions.csv")
    print("All models use the same rolling test windows.")
    print("Weather values are historical observations.")


if __name__ == "__main__":
    main()
