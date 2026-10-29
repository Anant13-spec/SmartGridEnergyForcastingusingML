"""
Preprocessing and Feature Engineering Module.
Builds leak-free time-series features (cyclical calendar harmonics, day-of-week indicators, and lags)
and handles data cleaning and chronological alignment.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import config


def clean_electricity_data(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = config.DEFAULT_TARGET_COL,
    fill_calendar_gaps: bool = True,
    impute_implausible: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset:
    - Parse dates and sort chronologically.
    - Remove duplicate dates.
    - Treat demand <= 0 or below MIN_PLAUSIBLE_DEMAND as missing, then interpolate.
    - Reindex to complete regular daily calendar sequence if daily.
    """
    data = df.copy()

    # 1. Parse date and sort
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Drop duplicate dates
    initial_rows = len(data)
    data = data.drop_duplicates(subset=[date_col], keep="first").reset_index(drop=True)
    duplicates_dropped = initial_rows - len(data)

    # 2. Reindex calendar gaps if daily
    gaps_filled_count = 0
    if fill_calendar_gaps and len(data) > 1:
        full_idx = pd.date_range(data[date_col].min(), data[date_col].max(), freq="D")
        if len(full_idx) > len(data):
            data = data.set_index(date_col).reindex(full_idx)
            gaps_filled_count = int(data.index.to_series().diff().iloc[1:].gt(pd.Timedelta(days=1)).sum())
            data.index.name = date_col
            data = data.reset_index()

    # 3. Handle target column values
    clean_target_col = f"{target_col}_CLEAN"
    data[clean_target_col] = pd.to_numeric(data[target_col], errors="coerce")

    # Identify implausible values
    implausible_mask = (data[clean_target_col] < config.MIN_PLAUSIBLE_DEMAND) | (
        data[clean_target_col] > config.MAX_PLAUSIBLE_DEMAND
    )
    implausible_count = int(implausible_mask.sum())

    if impute_implausible:
        data.loc[implausible_mask, clean_target_col] = np.nan
        data[clean_target_col] = data[clean_target_col].interpolate(method="linear").bfill().ffill()

    # Also forward/backward fill numeric weather columns if any rows were added by reindexing
    all_weather_cols = list(dict.fromkeys(config.WEATHER_COLUMNS + getattr(config, "PRIMARY_WEATHER_FEATURES", [])))
    for col in all_weather_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").interpolate(method="linear").bfill().ffill()

    cleaning_metadata = {
        "initial_rows": initial_rows,
        "clean_rows": len(data),
        "duplicates_dropped": duplicates_dropped,
        "gaps_filled_count": gaps_filled_count,
        "implausible_count": implausible_count,
        "target_col_clean": clean_target_col,
        "clean_min_demand": float(data[clean_target_col].min()),
        "clean_mean_demand": float(data[clean_target_col].mean()),
        "clean_max_demand": float(data[clean_target_col].max()),
    }

    return data, cleaning_metadata


def build_feature_pipeline(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    lags: Optional[List[int]] = None,
    include_weather: bool = False,
    weather_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Construct strictly leak-free time-series features.
    
    CRITICAL TIME-SERIES INTEGRITY RULES:
    1. Cyclical annual Fourier terms: sin_doy and cos_doy capture seasonal cycles smoothly.
    2. Discrete Day-of-Week indicators (dow_0 to dow_6) capture true weekday load profiles.
    3. Auto-regressive lags are created strictly via .shift(k) without lookahead.
    4. Row t NEVER accesses target(t) during feature construction.
    """
    if lags is None:
        lags = [1, 7]

    data = df.copy()
    if target_col not in data.columns and config.DEFAULT_TARGET_COL in data.columns:
        target_col = config.DEFAULT_TARGET_COL

    feature_cols: List[str] = []

    # 1. Cyclical Annual Seasonality (Fourier Harmonics)
    dates = pd.to_datetime(data[date_col])
    doy = dates.dt.day_of_year
    data["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    data["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    seasonal_features = ["sin_doy", "cos_doy"]
    feature_cols.extend(seasonal_features)

    # 2. Day-of-Week Indicators (0=Monday ... 6=Sunday)
    data["day_of_week"] = dates.dt.day_of_week
    dow_features = []
    for d in range(7):
        col_name = f"dow_{d}"
        data[col_name] = (data["day_of_week"] == d).astype(float)
        dow_features.append(col_name)
    feature_cols.extend(dow_features)

    # 3. Auto-regressive Lag Features (No lookahead)
    lag_feature_names = []
    for lag in lags:
        col_name = f"lag_{lag}"
        data[col_name] = data[target_col].shift(lag)
        lag_feature_names.append(col_name)
    feature_cols.extend(lag_feature_names)

    # 4. Optional Weather Features (same-day exogenous variables)
    active_weather_cols = []
    if include_weather:
        candidate_weather = weather_cols or getattr(config, "PRIMARY_WEATHER_FEATURES", ["temp", "humidity", "precip", "windspeed", "solarradiation"])
        for w_col in candidate_weather:
            if w_col in data.columns:
                active_weather_cols.append(w_col)
                feature_cols.append(w_col)

    # Drop rows that have NaN in the required feature columns (due to lag initialization)
    required_cols = [date_col, target_col] + feature_cols
    before_drop = len(data)
    valid_data = data.dropna(subset=required_cols).reset_index(drop=True)
    rows_dropped_lags = before_drop - len(valid_data)

    metadata = {
        "seasonal_features": seasonal_features,
        "dow_features": dow_features,
        "lag_features": lag_feature_names,
        "weather_features": active_weather_cols,
        "total_features": len(feature_cols),
        "feature_cols": feature_cols,
        "rows_dropped_lags": rows_dropped_lags,
        "valid_rows": len(valid_data),
        "target_col": target_col,
        "date_col": date_col,
    }

    return valid_data, feature_cols, metadata
