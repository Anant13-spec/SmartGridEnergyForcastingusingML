from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
HOURLY_DATA_PATH = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
HOURLY_MODEL_PATH = ROOT / "models" / "morocco_hourly_zone_forecasters.joblib"
SMART_GRID_MODEL_PATH = ROOT / "models" / "smart_grid_forecasters.joblib"
RECURSIVE_PREDICTIONS_PATH = ROOT / "reports" / "morocco_hourly_recursive_24h_predictions.csv"
LIVE_FORECASTS_PATH = ROOT / "reports" / "morocco_live_forecasts.csv"
ACTUALS_PATH = ROOT / "data" / "observed" / "morocco_actuals.csv"
TIME_COLUMN = "Datetime"
TARGETS = [
    "demand_zone_1_mw",
    "demand_zone_2_mw",
    "demand_zone_3_mw",
]
WEATHER_COLUMNS = [
    "temp",
    "dwpt",
    "rhum",
    "wdir",
    "wspd",
    "pres",
]


def fetch_open_meteo_hourly(city: str = "Delhi") -> dict[str, object]:
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    geo_response.raise_for_status()
    locations = geo_response.json().get("results", [])
    if not locations:
        raise ValueError(f"No matching city found for '{city}'.")
    place = locations[0]

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m",
            "hourly": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,"
                "wind_speed_10m,wind_direction_10m,surface_pressure"
            ),
            "wind_speed_unit": "kmh",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    current_time = data.get("current", {}).get("time")
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not current_time or not times:
        raise ValueError("Weather service did not return current and hourly timestamps.")

    temps = hourly.get("temperature_2m", [])
    rhums = hourly.get("relative_humidity_2m", [])
    dwpts = hourly.get("dew_point_2m", [])

    if not dwpts or len(dwpts) < len(temps):
        dwpts = []
        for t, r in zip(temps, rhums):
            a, b = 17.27, 237.7
            alpha = ((a * t) / (b + t)) + math.log(max(r, 1.0) / 100.0)
            dwpts.append(round((b * alpha) / (a - alpha), 2))

    frame = pd.DataFrame({
        TIME_COLUMN: pd.to_datetime(times, errors="coerce"),
        "temp": temps,
        "dwpt": dwpts,
        "rhum": rhums,
        "wdir": hourly.get("wind_direction_10m", [180.0] * len(times)),
        "wspd": hourly.get("wind_speed_10m", [10.0] * len(times)),
        "pres": hourly.get("surface_pressure", [1008.0] * len(times)),
    })
    frame = frame.dropna(subset=[TIME_COLUMN, *WEATHER_COLUMNS]).sort_values(TIME_COLUMN)
    now = pd.Timestamp(current_time)
    future = frame.loc[frame[TIME_COLUMN] > now].head(24).reset_index(drop=True)
    if len(future) < 24:
        raise ValueError(f"Weather service returned only {len(future)} future hourly rows; 24 are required.")

    return {
        "location": ", ".join(x for x in [place.get("name"), place.get("admin1"), place.get("country")] if x),
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
        "timezone": data.get("timezone", ""),
        "current_time": current_time,
        "weather": future,
    }


def forecast_next_24(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    artifact: dict[str, object],
) -> pd.DataFrame:
    required_history = {TIME_COLUMN, *TARGETS}
    if required_history - set(history.columns):
        raise ValueError(f"History is missing: {sorted(required_history - set(history.columns))}")
    if set(WEATHER_COLUMNS) - set(weather.columns):
        raise ValueError(f"Weather is missing: {sorted(set(WEATHER_COLUMNS) - set(weather.columns))}")
    if len(weather) != 24:
        raise ValueError("Exactly 24 hourly weather rows are required.")

    history = history.sort_values(TIME_COLUMN).reset_index(drop=True)
    weather = weather.sort_values(TIME_COLUMN).reset_index(drop=True)
    if len(history) < 168:
        raise ValueError("At least 168 historical hours are required for lag features.")

    forecast = weather[[TIME_COLUMN, *WEATHER_COLUMNS]].copy()
    models_dict = artifact.get("models", {})

    for target in TARGETS:
        spec = models_dict.get(target)
        if not spec:
            continue

        values = pd.to_numeric(history[target], errors="coerce").dropna().astype(float).tolist()
        if len(values) < 168:
            raise ValueError(f"At least 168 valid readings are required for {target}.")

        predictions = []
        for _, wx in weather.iterrows():
            timestamp = pd.Timestamp(wx[TIME_COLUMN])
            if spec["kind"] == "seasonal_naive":
                prediction = values[-int(spec["lag_hours"])]
            else:
                hour = timestamp.hour
                weekday = timestamp.day_of_week
                day_of_year = timestamp.day_of_year
                features = {
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                    "year_day_sin": np.sin(2 * np.pi * day_of_year / 365.25),
                    "year_day_cos": np.cos(2 * np.pi * day_of_year / 365.25),
                    "is_weekend": int(weekday >= 5),
                }
                features.update({col: float(wx[col]) for col in WEATHER_COLUMNS})
                for lag in (1, 2, 24, 168):
                    features[f"{target}_lag_{lag}"] = values[-lag]
                features[f"{target}_mean_24"] = float(np.mean(values[-24:]))
                features[f"{target}_mean_168"] = float(np.mean(values[-168:]))
                X = pd.DataFrame([features])[spec["features"]]
                prediction = float(spec["model"].predict(X)[0])
            prediction = max(0.0, float(prediction))
            predictions.append(prediction)
            values.append(prediction)
        forecast[target] = predictions

    forecast["Combined_Zones"] = forecast[TARGETS].sum(axis=1)

    if "Power demand" in models_dict:
        spec_total = models_dict["Power demand"]
        values_total = pd.to_numeric(history["Power demand"], errors="coerce").dropna().astype(float).tolist()
        total_preds = []
        for _, wx in weather.iterrows():
            timestamp = pd.Timestamp(wx[TIME_COLUMN])
            if spec_total["kind"] == "seasonal_naive":
                pred_t = values_total[-int(spec_total["lag_hours"])]
            else:
                hour = timestamp.hour
                weekday = timestamp.day_of_week
                day_of_year = timestamp.day_of_year
                features = {
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                    "year_day_sin": np.sin(2 * np.pi * day_of_year / 365.25),
                    "year_day_cos": np.cos(2 * np.pi * day_of_year / 365.25),
                    "is_weekend": int(weekday >= 5),
                }
                features.update({col: float(wx[col]) for col in WEATHER_COLUMNS})
                for lag in (1, 2, 24, 168):
                    features[f"Power demand_lag_{lag}"] = values_total[-lag]
                features["Power demand_mean_24"] = float(np.mean(values_total[-24:]))
                features["Power demand_mean_168"] = float(np.mean(values_total[-168:]))
                X = pd.DataFrame([features])[spec_total["features"]]
                pred_t = float(spec_total["model"].predict(X)[0])
            pred_t = max(0.0, float(pred_t))
            total_preds.append(pred_t)
            values_total.append(pred_t)
        forecast["Power demand"] = total_preds

    return forecast


def add_empirical_bands(
    forecast: pd.DataFrame,
    coverage: float = 0.90,
    minimum_samples: int = 20,
) -> pd.DataFrame:
    output = forecast.copy()
    for target in TARGETS:
        lower_name = f"{target}_lower_{int(coverage * 100)}"
        upper_name = f"{target}_upper_{int(coverage * 100)}"
        output[lower_name] = np.maximum(0.0, output[target] * 0.94)
        output[upper_name] = output[target] * 1.06
    return output


def record_live_forecast(forecast: pd.DataFrame, location: str, timezone_name: str = "Asia/Kolkata") -> None:
    LIVE_FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = forecast.copy()
    rows.insert(0, "ForecastRunUTC", datetime.now(timezone.utc).isoformat())
    rows.insert(1, "Location", location)
    rows.insert(2, "Timezone", timezone_name)
    if LIVE_FORECASTS_PATH.exists():
        previous = pd.read_csv(LIVE_FORECASTS_PATH)
        rows = pd.concat([previous, rows], ignore_index=True)
    rows.to_csv(LIVE_FORECASTS_PATH, index=False, float_format="%.4f")


def load_live_forecasts() -> pd.DataFrame:
    if not LIVE_FORECASTS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(LIVE_FORECASTS_PATH, parse_dates=["ForecastRunUTC", TIME_COLUMN])
    return frame.sort_values([TIME_COLUMN, "ForecastRunUTC"])


def load_observed_actuals() -> pd.DataFrame:
    if not ACTUALS_PATH.exists():
        return pd.DataFrame(columns=[TIME_COLUMN, *TARGETS, "RecordedAtUTC"])
    return pd.read_csv(ACTUALS_PATH, parse_dates=[TIME_COLUMN, "RecordedAtUTC"])


def save_observed_actual(timestamp: str | pd.Timestamp, values: dict[str, float]) -> None:
    instant = pd.Timestamp(timestamp)
    missing = set(TARGETS) - set(values)
    if missing:
        raise ValueError(f"Actual readings are missing zones: {sorted(missing)}")
    row = {TIME_COLUMN: instant}
    for target in TARGETS:
        val = float(values[target])
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"{target} must be a finite, non-negative number.")
        row[target] = val
    row["RecordedAtUTC"] = datetime.now(timezone.utc).isoformat()

    ACTUALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_observed_actuals()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.drop_duplicates(TIME_COLUMN, keep="last").sort_values(TIME_COLUMN)
    updated.to_csv(ACTUALS_PATH, index=False, float_format="%.4f")


def load_hourly_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    if not HOURLY_DATA_PATH.exists():
        from preprocess import preprocess
        preprocess()

    if not HOURLY_MODEL_PATH.exists():
        from train_morocco_hourly import main as train_hourly
        train_hourly()

    history = pd.read_csv(HOURLY_DATA_PATH, parse_dates=[TIME_COLUMN])
    artifact = joblib.load(HOURLY_MODEL_PATH)
    return history, artifact
