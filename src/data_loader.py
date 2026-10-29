"""
Data Loading and Validation Module.
Inspects, validates, and generates data quality reports for the smart grid dataset.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import config


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Safely load the raw electricity dataset.
    Never overwrites or modifies the input file.
    """
    path = filepath or config.DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    df = pd.read_csv(path)
    return df


def detect_columns(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None
) -> Tuple[str, str]:
    """
    Identify date and electricity demand columns, falling back to heuristics if needed.
    """
    columns = df.columns.tolist()
    
    # 1. Date column detection
    if date_col and date_col in columns:
        chosen_date_col = date_col
    elif config.DEFAULT_DATE_COL in columns:
        chosen_date_col = config.DEFAULT_DATE_COL
    else:
        # Heuristic search
        candidates = [c for c in columns if any(k in c.lower() for k in ["date", "time", "timestamp"])]
        if candidates:
            chosen_date_col = candidates[0]
        else:
            raise ValueError(f"Could not identify date column among: {columns}")

    # 2. Target column detection
    if target_col and target_col in columns:
        chosen_target_col = target_col
    elif config.DEFAULT_TARGET_COL in columns:
        chosen_target_col = config.DEFAULT_TARGET_COL
    else:
        candidates = [c for c in columns if any(k in c.lower() for k in ["demand", "load", "power", "mw"])]
        if candidates:
            chosen_target_col = candidates[0]
        else:
            raise ValueError(f"Could not identify electricity demand target column among: {columns}")

    return chosen_date_col, chosen_target_col


def detect_frequency(dates: pd.Series) -> Tuple[str, str, pd.Timedelta]:
    """
    Detect the empirical temporal frequency of a datetime series.
    Returns:
        (freq_label, description, median_delta)
    """
    clean_dates = dates.dropna().sort_values().drop_duplicates()
    if len(clean_dates) < 2:
        return "Unknown", "Insufficient dates to determine frequency", pd.Timedelta(0)
    
    diffs = clean_dates.diff().dropna()
    median_delta = diffs.median()
    
    if median_delta >= pd.Timedelta(hours=23) and median_delta <= pd.Timedelta(hours=25):
        return "Daily", "Daily observations (1-day interval)", median_delta
    elif median_delta >= pd.Timedelta(minutes=50) and median_delta <= pd.Timedelta(minutes=70):
        return "Hourly", "Hourly observations (1-hour interval)", median_delta
    elif median_delta >= pd.Timedelta(minutes=10) and median_delta <= pd.Timedelta(minutes=20):
        return "15-Min", "Sub-hourly 15-minute intervals", median_delta
    else:
        return f"{median_delta}", f"Custom interval (median: {median_delta})", median_delta


def inspect_data_quality(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = config.DEFAULT_TARGET_COL
) -> Dict[str, Any]:
    """
    Perform deep validation of dataset and generate a comprehensive quality report.
    """
    total_rows, total_cols = df.shape
    columns = df.columns.tolist()
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing_cells = df.isnull().sum().sum()
    
    # Check date column
    date_present = date_col in df.columns
    target_present = target_col in df.columns
    
    # Check DATE vs datetime duplicate status
    date_datetime_identical = False
    if "DATE" in df.columns and "datetime" in df.columns:
        date_datetime_identical = bool((df["DATE"].astype(str) == df["datetime"].astype(str)).all())

    # Location and weather station metadata
    locations = df["name"].dropna().unique().tolist() if "name" in df.columns else []
    stations = df["stations"].dropna().unique().tolist() if "stations" in df.columns else []
    conditions = df["conditions"].value_counts().to_dict() if "conditions" in df.columns else {}

    report: Dict[str, Any] = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": columns,
        "missing_counts": missing_counts,
        "total_missing_cells": int(total_missing_cells),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_col": date_col,
        "target_col": target_col,
        "date_present": date_present,
        "target_present": target_present,
        "date_datetime_identical": date_datetime_identical,
        "locations": locations,
        "stations_count": len(stations),
        "conditions_summary": conditions,
        "available_weather_cols": [c for c in config.WEATHER_COLUMNS if c in df.columns],
        "primary_weather_cols": [c for c in config.PRIMARY_WEATHER_FEATURES if c in df.columns],
    }
    
    if not date_present or not target_present:
        report["status"] = "ERROR: Missing required date or target columns."
        return report

    # Parse dates safely
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    null_dates = int(parsed_dates.isnull().sum())
    report["null_dates"] = null_dates
    
    valid_dates = parsed_dates.dropna().sort_values()
    if len(valid_dates) > 0:
        start_date = valid_dates.iloc[0]
        end_date = valid_dates.iloc[-1]
        freq_label, freq_desc, median_delta = detect_frequency(valid_dates)
        report["start_date"] = str(start_date.date())
        report["end_date"] = str(end_date.date())
        report["date_range_days"] = int((end_date - start_date).days)
        report["frequency_label"] = freq_label
        report["frequency_desc"] = freq_desc
        report["duplicate_dates"] = int(parsed_dates.duplicated().sum())

        # Check gaps in date sequence (for daily)
        if freq_label == "Daily":
            expected_range = pd.date_range(start_date, end_date, freq="D")
            missing_dates = expected_range.difference(pd.DatetimeIndex(valid_dates))
            report["missing_calendar_dates_count"] = len(missing_dates)
            report["missing_calendar_dates"] = [str(d.date()) for d in missing_dates]
        else:
            report["missing_calendar_dates_count"] = 0
            report["missing_calendar_dates"] = []

    # Target column analysis
    numeric_target = pd.to_numeric(df[target_col], errors="coerce")
    report["target_non_numeric_count"] = int(numeric_target.isnull().sum() - df[target_col].isnull().sum())
    
    valid_target = numeric_target.dropna()
    report["target_valid_count"] = len(valid_target)
    
    if len(valid_target) > 0:
        # Plausibility checks
        zero_or_negative = int((valid_target <= 0).sum())
        below_plausible = int((valid_target < config.MIN_PLAUSIBLE_DEMAND).sum())
        above_plausible = int((valid_target > config.MAX_PLAUSIBLE_DEMAND).sum())
        
        # Outlier detection using IQR on non-zero valid data
        clean_subset = valid_target[valid_target >= config.MIN_PLAUSIBLE_DEMAND]
        if len(clean_subset) > 0:
            q25 = clean_subset.quantile(0.25)
            q75 = clean_subset.quantile(0.75)
            iqr = q75 - q25
            lower_bound = max(0, q25 - 1.5 * iqr)
            upper_bound = q75 + 1.5 * iqr
            potential_outliers = int(((clean_subset < lower_bound) | (clean_subset > upper_bound)).sum())
        else:
            q25, q75, iqr, lower_bound, upper_bound, potential_outliers = 0, 0, 0, 0, 0, 0

        report["raw_target_summary"] = {
            "mean": float(valid_target.mean()),
            "std": float(valid_target.std()),
            "min": float(valid_target.min()),
            "min_nonzero": float(clean_subset.min()) if len(clean_subset) > 0 else 0.0,
            "q25": float(valid_target.quantile(0.25)),
            "median": float(valid_target.median()),
            "q75": float(valid_target.quantile(0.75)),
            "max": float(valid_target.max()),
        }

        report["clean_target_summary"] = {
            "mean": float(clean_subset.mean()) if len(clean_subset) > 0 else float(valid_target.mean()),
            "std": float(clean_subset.std()) if len(clean_subset) > 0 else float(valid_target.std()),
            "min": float(clean_subset.min()) if len(clean_subset) > 0 else 0.0,
            "q25": float(clean_subset.quantile(0.25)) if len(clean_subset) > 0 else 0.0,
            "median": float(clean_subset.median()) if len(clean_subset) > 0 else 0.0,
            "q75": float(clean_subset.quantile(0.75)) if len(clean_subset) > 0 else 0.0,
            "max": float(valid_target.max()),
        }

        # Backward compatibility
        report["target_summary"] = report["raw_target_summary"]
        report["zero_or_negative_count"] = zero_or_negative
        report["below_plausible_count"] = below_plausible
        report["above_plausible_count"] = above_plausible
        report["iqr_potential_outliers"] = potential_outliers
        report["iqr_bounds"] = (float(lower_bound), float(upper_bound))

        # Dates where demand <= 0
        zero_dates = df.loc[valid_target <= 0, date_col].astype(str).tolist() if zero_or_negative > 0 else []
        report["zero_demand_dates"] = zero_dates

    # Academic requirements check
    report["meets_hourly_requirement"] = (report.get("frequency_label") == "Hourly")
    total_years = report.get("date_range_days", 0) / 365.25
    report["total_years"] = round(total_years, 2)
    report["meets_duration_requirement"] = (total_years >= 4.0)
    report["limitation_note"] = config.PROJECTED_REQUIREMENT_NOTE

    return report
