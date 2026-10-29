"""
Exploratory Data Analysis (EDA) Module.
Generates statistical summaries, seasonal profiles, weekday patterns,
correlations, and distribution summaries for smart grid energy demand.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import config


def get_summary_statistics(
    df: pd.DataFrame,
    target_col: str = "POWER_DEMAND_CLEAN"
) -> Dict[str, float]:
    """
    Calculate essential summary metrics for demand.
    """
    col = target_col if target_col in df.columns else config.DEFAULT_TARGET_COL
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    
    if len(series) == 0:
        return {}

    return {
        "count": float(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "skewness": float(series.skew()),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
    }


def get_peak_and_trough_records(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = "POWER_DEMAND_CLEAN",
    n: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Retrieve top N peak demand days and lowest N trough demand days.
    """
    col = target_col if target_col in df.columns else config.DEFAULT_TARGET_COL
    cols_to_show = [date_col, col]
    for w in ["temp", "humidity", "conditions"]:
        if w in df.columns:
            cols_to_show.append(w)
            
    sorted_df = df.dropna(subset=[col]).sort_values(col, ascending=False)
    peaks = sorted_df.head(n)[cols_to_show].reset_index(drop=True)
    troughs = sorted_df.tail(n).iloc[::-1][cols_to_show].reset_index(drop=True)
    
    return {
        "peaks": peaks,
        "troughs": troughs,
    }


def get_weekday_pattern(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = "POWER_DEMAND_CLEAN"
) -> pd.DataFrame:
    """
    Compute average and standard deviation of electricity demand grouped by day of week.
    """
    col = target_col if target_col in df.columns else config.DEFAULT_TARGET_COL
    temp_df = df.copy()
    dates = pd.to_datetime(temp_df[date_col])
    temp_df["day_name"] = dates.dt.day_name()
    temp_df["day_index"] = dates.dt.day_of_week
    
    grouped = temp_df.groupby(["day_index", "day_name"])[col].agg(
        mean_demand="mean",
        std_demand="std",
        min_demand="min",
        max_demand="max",
        count="count"
    ).reset_index().sort_values("day_index")
    
    return grouped


def get_monthly_pattern(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = "POWER_DEMAND_CLEAN"
) -> pd.DataFrame:
    """
    Compute average demand grouped by calendar month.
    """
    col = target_col if target_col in df.columns else config.DEFAULT_TARGET_COL
    temp_df = df.copy()
    dates = pd.to_datetime(temp_df[date_col])
    temp_df["month_name"] = dates.dt.strftime("%B")
    temp_df["month_num"] = dates.dt.month
    
    grouped = temp_df.groupby(["month_num", "month_name"])[col].agg(
        mean_demand="mean",
        std_demand="std",
        min_demand="min",
        max_demand="max",
        count="count"
    ).reset_index().sort_values("month_num")
    
    return grouped


def get_correlation_matrix(
    df: pd.DataFrame,
    target_col: str = "POWER_DEMAND_CLEAN",
    candidate_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix between demand and numeric weather/feature columns.
    """
    col = target_col if target_col in df.columns else config.DEFAULT_TARGET_COL
    candidates = candidate_cols or [col] + config.WEATHER_COLUMNS
    numeric_cols = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if col not in numeric_cols:
        numeric_cols = [col] + numeric_cols
        
    corr = df[numeric_cols].corr()
    return corr
