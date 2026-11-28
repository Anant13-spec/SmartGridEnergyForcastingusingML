from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "smart_grid_forecasters.joblib"
LEGACY_MODEL_PATH = ROOT / "models" / "load_forecaster.joblib"
TAIL_DATA_PATH = ROOT / "data" / "processed" / "recent_5min_tail.csv"
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "clean_3_zones_5min.csv"
RAW_DATA_PATH = ROOT / "data" / "df_3_zones_final.csv"

WEATHER_FIELDS = {
    "temp": "Temperature (°C)",
    "dwpt": "Dew Point (°C)",
    "rhum": "Relative Humidity (%)",
    "wdir": "Wind Direction (°)",
    "wspd": "Wind Speed (km/h)",
    "pres": "Atmospheric Pressure (hPa)",
}

TARGET_KEY_MAP = {
    "zone1": "demand_zone_1_mw",
    "zone 1": "demand_zone_1_mw",
    "demand_zone_1_mw": "demand_zone_1_mw",
    "zone2": "demand_zone_2_mw",
    "zone 2": "demand_zone_2_mw",
    "demand_zone_2_mw": "demand_zone_2_mw",
    "zone3": "demand_zone_3_mw",
    "zone 3": "demand_zone_3_mw",
    "demand_zone_3_mw": "demand_zone_3_mw",
    "total": "Power demand",
    "power demand": "Power demand",
    "power_demand": "Power demand",
}

_CACHED_ARTIFACT = None
_CACHED_HISTORY = None


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is not None:
        return _CACHED_ARTIFACT

    if model_path.exists():
        _CACHED_ARTIFACT = joblib.load(model_path)
        return _CACHED_ARTIFACT

    if LEGACY_MODEL_PATH.exists():
        legacy = joblib.load(LEGACY_MODEL_PATH)
        return {
            "targets": {
                "Power demand": {
                    "best_model": legacy.get("model"),
                    "features": legacy.get("features", []),
                    "best_model_name": legacy.get("model_name", "Trained Model"),
                }
            },
            "best_models": {
                "Power demand": {
                    "model": legacy.get("model"),
                    "features": legacy.get("features", []),
                    "model_name": legacy.get("model_name", "Trained Model"),
                }
            }
        }

    raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train_models.py first.")


def load_recent_history(limit_rows: int = 500) -> pd.DataFrame:
    global _CACHED_HISTORY
    if _CACHED_HISTORY is not None:
        return _CACHED_HISTORY

    if TAIL_DATA_PATH.exists():
        data_file = TAIL_DATA_PATH
    elif CLEAN_DATA_PATH.exists():
        data_file = CLEAN_DATA_PATH
    elif RAW_DATA_PATH.exists():
        data_file = RAW_DATA_PATH
    else:
        raise FileNotFoundError(f"Historical dataset not found at {CLEAN_DATA_PATH}.")

    df = pd.read_csv(data_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    _CACHED_HISTORY = df.tail(limit_rows).reset_index(drop=True)
    return _CACHED_HISTORY


def validate_weather(weather: Mapping[str, float]) -> dict[str, float]:
    cleaned = {}
    missing = set(WEATHER_FIELDS) - set(weather)
    if missing:
        raise ValueError(f"Missing required weather inputs: {sorted(missing)}")

    for field in WEATHER_FIELDS:
        try:
            val = float(weather[field])
        except (TypeError, ValueError):
            raise ValueError(f"Weather field '{field}' must be numeric.") from None

        if not math.isfinite(val):
            raise ValueError(f"Weather field '{field}' must be a finite number.")
        cleaned[field] = val

    if not -20.0 <= cleaned["temp"] <= 60.0:
        raise ValueError(f"Temperature {cleaned['temp']}°C is outside realistic range (-20 to 60°C).")
    if not -30.0 <= cleaned["dwpt"] <= 40.0:
        raise ValueError(f"Dew point {cleaned['dwpt']}°C is outside realistic range (-30 to 40°C).")
    if not 0.0 <= cleaned["rhum"] <= 100.0:
        raise ValueError(f"Relative humidity {cleaned['rhum']}% must be between 0% and 100%.")
    if not 0.0 <= cleaned["wdir"] <= 360.0:
        raise ValueError(f"Wind direction {cleaned['wdir']}° must be between 0° and 360°.")
    if not 0.0 <= cleaned["wspd"] <= 150.0:
        raise ValueError(f"Wind speed {cleaned['wspd']} km/h is outside realistic range (0 to 150 km/h).")
    if not 850.0 <= cleaned["pres"] <= 1100.0:
        raise ValueError(f"Pressure {cleaned['pres']} hPa is outside realistic range (850 to 1100 hPa).")

    return cleaned


def parse_forecast_date(date_text: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(date_text)
        if pd.isna(ts):
            raise ValueError
        return ts
    except Exception as exc:
        raise ValueError("Enter a valid forecast date in YYYY-MM-DD format.") from exc


def get_latest_lags(target_col: str, history_df: pd.DataFrame | None = None) -> dict[str, float]:
    if history_df is None:
        history_df = load_recent_history(500)

    prefix = target_col.lower()
    if prefix == "power demand":
        prefix = "power_demand"

    lag_1_col = f"{prefix}_lag_1"
    lag_12_col = f"{prefix}_lag_12"
    lag_288_col = f"{prefix}_lag_288"
    rolling_col = f"{prefix}_rolling_mean_1h"

    latest = history_df.iloc[-1]

    if lag_1_col in history_df.columns:
        return {
            lag_1_col: float(latest[lag_1_col]),
            lag_12_col: float(latest[lag_12_col]),
            lag_288_col: float(latest[lag_288_col]),
            rolling_col: float(latest[rolling_col]),
        }

    series = history_df[target_col].dropna().values
    if len(series) < 288:
        raise ValueError(f"At least 288 historical records required for {target_col}; only {len(series)} available.")

    return {
        lag_1_col: float(series[-1]),
        lag_12_col: float(series[-12]),
        lag_288_col: float(series[-288]),
        rolling_col: float(np.mean(series[-12:])),
    }


def predict_demand(
    forecast_datetime: str | pd.Timestamp,
    weather_inputs: Mapping[str, float],
    target: str = "Power demand",
    model_name: str | None = None,
    custom_lags: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cleaned_weather = validate_weather(weather_inputs)
    ts = pd.Timestamp(forecast_datetime)

    target_key = TARGET_KEY_MAP.get(target.strip().lower(), target)
    artifact = load_model_artifact()

    target_info = artifact.get("targets", {}).get(target_key)
    if not target_info:
        target_info = artifact.get("best_models", {}).get(target_key)
        if not target_info:
            raise ValueError(f"Unknown target '{target}'. Available targets: {list(TARGET_KEY_MAP.keys())}")

    if model_name and "models" in target_info and model_name in target_info["models"]:
        selected_model = target_info["models"][model_name]["model"]
        used_model_name = model_name
    else:
        selected_model = target_info.get("best_model") or target_info.get("model")
        used_model_name = target_info.get("best_model_name") or target_info.get("model_name", "Trained ML Model")

    hour = ts.hour
    minute = ts.minute
    dayofweek = ts.day_of_week
    month = ts.month
    is_weekend = int(dayofweek >= 5)

    calendar_features = {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "minute_sin": np.sin(2 * np.pi * minute / 60.0),
        "minute_cos": np.cos(2 * np.pi * minute / 60.0),
        "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7.0),
        "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7.0),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "is_weekend": is_weekend,
    }

    if custom_lags:
        lag_features = dict(custom_lags)
    else:
        lag_features = get_latest_lags(target_key)

    row = {**cleaned_weather, **calendar_features, **lag_features}
    feature_order = target_info["features"]

    missing_cols = set(feature_order) - set(row)
    if missing_cols:
        raise ValueError(f"Input features missing required columns: {sorted(missing_cols)}")

    X = pd.DataFrame([row])[feature_order]
    pred_raw = float(selected_model.predict(X)[0])
    pred_mw = max(0.0, pred_raw)

    prefix = target_key.lower().replace("power demand", "power_demand")
    baseline_mw = lag_features.get(f"{prefix}_lag_1", lag_features.get(f"{prefix}_lag_288", pred_mw))
    delta_mw = pred_mw - baseline_mw

    return {
        "prediction_mw": round(pred_mw, 2),
        "target": target_key,
        "display_name": target_info.get("display_name", target_key),
        "timestamp": str(ts),
        "model_used": used_model_name,
        "baseline_mw": round(baseline_mw, 2),
        "delta_mw": round(delta_mw, 2),
        "weather_inputs": cleaned_weather,
        "lag_inputs": lag_features,
    }


def predict_all_zones(
    forecast_datetime: str | pd.Timestamp,
    weather_inputs: Mapping[str, float],
    model_name: str | None = None,
) -> dict[str, Any]:
    cleaned_weather = validate_weather(weather_inputs)
    results = {}
    total_zones_sum = 0.0

    for z_key in ["demand_zone_1_mw", "demand_zone_2_mw", "demand_zone_3_mw"]:
        res = predict_demand(forecast_datetime, cleaned_weather, target=z_key, model_name=model_name)
        results[z_key] = res
        total_zones_sum += res["prediction_mw"]

    total_res = predict_demand(forecast_datetime, cleaned_weather, target="Power demand", model_name=model_name)
    results["Power demand"] = total_res
    results["Combined_Zones_Sum"] = round(total_zones_sum, 2)
    results["Difference_Total_vs_Sum"] = round(total_res["prediction_mw"] - total_zones_sum, 2)

    return results
