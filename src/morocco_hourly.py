"""Open-Meteo inputs and recursive hourly forecasts for the Morocco model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
HOURLY_DATA_PATH = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
HOURLY_MODEL_PATH = ROOT / "models" / "morocco_hourly_zone_forecasters.joblib"
RECURSIVE_PREDICTIONS_PATH = ROOT / "reports" / "morocco_hourly_recursive_24h_predictions.csv"
LIVE_FORECASTS_PATH = ROOT / "reports" / "morocco_live_forecasts.csv"
ACTUALS_PATH = ROOT / "data" / "observed" / "morocco_actuals.csv"
TIME_COLUMN = "Datetime"
TARGETS = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]
WEATHER_COLUMNS = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
]


def fetch_open_meteo_hourly(city: str) -> dict[str, object]:
    """Resolve a city and return the next 24 hourly weather rows in its timezone."""
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    geo_response.raise_for_status()
    locations = geo_response.json().get("results", [])
    if not locations:
        raise ValueError(f"No matching city found for '{city}'. Try adding its country.")
    place = locations[0]

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m",
            "hourly": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                "shortwave_radiation,diffuse_radiation"
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

    frame = pd.DataFrame({
        TIME_COLUMN: pd.to_datetime(times, errors="coerce"),
        "Temperature": hourly.get("temperature_2m", []),
        "Humidity": hourly.get("relative_humidity_2m", []),
        "WindSpeed": hourly.get("wind_speed_10m", []),
        # The source calls these diffuse-flow fields and gives no units.
        # Use like-named solar-radiation proxies from Open-Meteo and disclose it.
        "GeneralDiffuseFlows": hourly.get("shortwave_radiation", []),
        "DiffuseFlows": hourly.get("diffuse_radiation", []),
    })
    frame = frame.dropna(subset=[TIME_COLUMN, *WEATHER_COLUMNS]).sort_values(TIME_COLUMN)
    now = pd.Timestamp(current_time)
    future = frame.loc[frame[TIME_COLUMN] > now].head(24).reset_index(drop=True)
    if len(future) < 24:
        raise ValueError(f"Weather service returned only {len(future)} future hourly rows; 24 are required.")

    return {
        "location": ", ".join(x for x in [place.get("name"), place.get("country")] if x),
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
        "timezone": data.get("timezone", ""),
        "current_time": current_time,
        "weather": future,
    }


def forecast_next_24(history: pd.DataFrame, weather: pd.DataFrame,
                     artifact: dict[str, object]) -> pd.DataFrame:
    """Generate zone forecasts recursively, using future weather rows supplied."""
    required_history = {TIME_COLUMN, *TARGETS}
    if required_history - set(history.columns):
        raise ValueError(f"History is missing: {sorted(required_history - set(history.columns))}")
    if set(WEATHER_COLUMNS) - set(weather.columns):
        raise ValueError(f"Weather is missing: {sorted(set(WEATHER_COLUMNS) - set(weather.columns))}")
    if len(weather) != 24:
        raise ValueError("Exactly 24 hourly weather rows are required.")
    if set(TARGETS) - set(artifact.get("models", {})):
        raise ValueError("Hourly model artifact does not contain all three zones.")

    history = history.sort_values(TIME_COLUMN).reset_index(drop=True)
    weather = weather.sort_values(TIME_COLUMN).reset_index(drop=True)
    if len(history) < 168:
        raise ValueError("At least 168 historical hours are required for lag features.")
    if weather[TIME_COLUMN].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Future weather timestamps must be one hour apart.")
    if pd.Timestamp(weather.loc[0, TIME_COLUMN]) <= pd.Timestamp(history.loc[len(history) - 1, TIME_COLUMN]):
        raise ValueError("Future weather must begin after the latest demand timestamp.")

    forecast = weather[[TIME_COLUMN, *WEATHER_COLUMNS]].copy()
    for target in TARGETS:
        spec = artifact["models"][target]
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
                features.update({column: float(wx[column]) for column in WEATHER_COLUMNS})
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
    return forecast


def add_empirical_bands(
    forecast: pd.DataFrame,
    coverage: float = 0.90,
    minimum_samples: int = 30,
) -> pd.DataFrame:
    """Add horizon-specific symmetric bands calibrated from recursive holdout errors."""
    output = forecast.copy()
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")

    if RECURSIVE_PREDICTIONS_PATH.exists():
        calibration = pd.read_csv(RECURSIVE_PREDICTIONS_PATH)
    else:
        calibration = pd.DataFrame()

    for target in TARGETS:
        lower_name = f"{target}_lower_{int(coverage * 100)}"
        upper_name = f"{target}_upper_{int(coverage * 100)}"
        count_name = f"{target}_band_samples"
        output[lower_name] = np.nan
        output[upper_name] = np.nan
        output[count_name] = 0
        if calibration.empty:
            continue

        zone_rows = calibration.loc[calibration["zone"] == target]
        for position in range(len(output)):
            horizon = position + 1
            errors = (
                zone_rows.loc[zone_rows["horizon_hour"] == horizon, "actual"]
                - zone_rows.loc[zone_rows["horizon_hour"] == horizon, "recursive_prediction"]
            ).abs().dropna().to_numpy()
            output.loc[position, count_name] = len(errors)
            if len(errors) < minimum_samples:
                continue
            radius = float(np.quantile(errors, coverage, method="higher"))
            output.loc[position, lower_name] = max(0.0, float(output.loc[position, target]) - radius)
            output.loc[position, upper_name] = float(output.loc[position, target]) + radius
    return output


def record_live_forecast(forecast: pd.DataFrame, location: str, timezone_name: str = "Africa/Casablanca") -> None:
    """Append one issued forecast run without changing source or processed data."""
    LIVE_FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = forecast.copy()
    rows.insert(0, "ForecastRunUTC", datetime.now(timezone.utc).isoformat())
    rows.insert(1, "Location", location)
    rows.insert(2, "Timezone", timezone_name)
    if LIVE_FORECASTS_PATH.exists():
        previous = pd.read_csv(LIVE_FORECASTS_PATH)
        rows = pd.concat([previous, rows], ignore_index=True)
    rows.to_csv(LIVE_FORECASTS_PATH, index=False, float_format="%.6f")


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
    """Upsert user-entered hourly zone actuals in a separate observations file."""
    instant = pd.Timestamp(timestamp)
    if instant.minute or instant.second or instant.microsecond:
        raise ValueError("Actual reading timestamp must be aligned to the start of an hour.")
    missing = set(TARGETS) - set(values)
    if missing:
        raise ValueError(f"Actual readings are missing zones: {sorted(missing)}")
    row = {TIME_COLUMN: instant}
    for target in TARGETS:
        value = float(values[target])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{target} must be a finite, non-negative number.")
        row[target] = value
    row["RecordedAtUTC"] = datetime.now(timezone.utc).isoformat()

    ACTUALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_observed_actuals()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.drop_duplicates(TIME_COLUMN, keep="last").sort_values(TIME_COLUMN)
    updated.to_csv(ACTUALS_PATH, index=False, float_format="%.6f")


def load_hourly_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    if not HOURLY_DATA_PATH.exists():
        raise FileNotFoundError(f"Hourly history is missing: {HOURLY_DATA_PATH}")
    if not HOURLY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hourly model is missing: {HOURLY_MODEL_PATH}")
    import joblib

    history = pd.read_csv(HOURLY_DATA_PATH, parse_dates=[TIME_COLUMN])
    artifact = joblib.load(HOURLY_MODEL_PATH)
    return history, artifact
