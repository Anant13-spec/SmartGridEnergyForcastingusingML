"""
Forecasting Module for Smart Grid Energy Load Forecasting.
Implements explainable Linear Regression and recursive multi-step future forecasting.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import config


class LinearRegressionForecaster:
    """
    Explainable Linear Regression forecasting model.
    Fits without intercept when full set of day-of-week indicators is used,
    providing direct physical interpretability for each weekday's baseline demand.
    """

    def __init__(self, fit_intercept: bool = False):
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Linear Regression model on training features."""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate demand predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def get_feature_coefficients(self) -> pd.DataFrame:
        """
        Extract model coefficients for explainability.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Human-readable label map
        label_map = {
            "sin_doy": "Annual Seasonality (Sine)",
            "cos_doy": "Annual Seasonality (Cosine)",
            "dow_0": "Monday Baseline",
            "dow_1": "Tuesday Baseline",
            "dow_2": "Wednesday Baseline",
            "dow_3": "Thursday Baseline",
            "dow_4": "Friday Baseline",
            "dow_5": "Saturday Baseline",
            "dow_6": "Sunday Baseline",
            "lag_1": "Yesterday Demand (Lag 1)",
            "lag_7": "Last Week Demand (Lag 7)",
        }

        records = []
        for feat, coef in zip(self.feature_names, self.model.coef_):
            records.append({
                "Feature": label_map.get(feat, feat),
                "Coefficient (MW)": round(float(coef), 2),
                "Absolute Magnitude": round(abs(float(coef)), 2),
            })

        df_coef = pd.DataFrame(records).sort_values("Absolute Magnitude", ascending=False).reset_index(drop=True)
        return df_coef


# Backward compatibility alias
TimeSeriesModels = LinearRegressionForecaster


def generate_future_forecast(
    model: Any,
    feature_cols: List[str],
    historical_df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    horizon: int = config.DEFAULT_FORECAST_HORIZON,
    test_rmse: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform genuine recursive multi-step future forecasting into unobserved periods.
    
    Starting from the last observed historical date:
    For step h = 1..horizon:
    1. Determine the future timestamp (t + h).
    2. Compute exact calendar features for the future date:
       - sin_doy, cos_doy (Fourier annual cycle)
       - dow_0..dow_6 (one-hot indicator for the specific weekday)
    3. Update auto-regressive lags:
       - lag_1 uses demand from (t + h - 1)
       - lag_7 uses demand from (t + h - 7)
    4. Predict demand using the fitted Linear Regression model.
    5. Append predicted demand to the running sequence to predict subsequent steps.
    """
    if len(historical_df) < 14:
        raise ValueError("Insufficient historical observations to initialize lag window.")

    hist = historical_df.copy().sort_values(date_col).reset_index(drop=True)
    if target_col not in hist.columns:
        target_col = config.DEFAULT_TARGET_COL

    last_date = pd.to_datetime(hist[date_col].iloc[-1])
    running_demand = list(hist[target_col].astype(float))

    model_to_use = model.model if hasattr(model, "model") else model
    forecast_records = []

    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)

        # 1. Fourier annual harmonics
        f_doy = future_date.day_of_year
        sin_doy = np.sin(2 * np.pi * f_doy / 365.25)
        cos_doy = np.cos(2 * np.pi * f_doy / 365.25)

        # 2. Day of week indicators (0=Monday .. 6=Sunday)
        f_dow = future_date.day_of_week

        # 3. Recursive auto-regressive lags from running sequence
        lag_1 = running_demand[-1]
        lag_7 = running_demand[-7] if len(running_demand) >= 7 else running_demand[0]

        row_feat: Dict[str, float] = {
            "sin_doy": float(sin_doy),
            "cos_doy": float(cos_doy),
            "lag_1": float(lag_1),
            "lag_7": float(lag_7),
        }
        for d in range(7):
            row_feat[f"dow_{d}"] = 1.0 if f_dow == d else 0.0

        # Any fallback for optional columns
        for f in feature_cols:
            if f not in row_feat:
                if f in hist.columns and pd.api.types.is_numeric_dtype(hist[f]):
                    row_feat[f] = float(hist[f].tail(7).mean())
                else:
                    row_feat[f] = 0.0

        X_future = pd.DataFrame([row_feat])[feature_cols]

        # Predict future demand
        predicted_demand = float(model_to_use.predict(X_future)[0])
        predicted_demand = max(config.MIN_PLAUSIBLE_DEMAND, predicted_demand)

        # Confidence bounds based on test RMSE and horizon growth (sqrt(step))
        rmse_val = test_rmse if test_rmse is not None else 322.0
        uncertainty = 1.96 * rmse_val * np.sqrt(step)
        lower_bound = max(0.0, predicted_demand - uncertainty)
        upper_bound = predicted_demand + uncertainty

        forecast_records.append({
            "DATE": future_date.strftime("%Y-%m-%d"),
            "DAY_OF_WEEK": future_date.strftime("%A"),
            "FORECASTED_DEMAND_MW": round(predicted_demand, 2),
            "LOWER_BOUND_MW": round(lower_bound, 2),
            "UPPER_BOUND_MW": round(upper_bound, 2),
        })

        # Append to running sequence for recursive prediction of subsequent days
        running_demand.append(predicted_demand)

    df_forecast = pd.DataFrame(forecast_records)

    metadata = {
        "start_date": forecast_records[0]["DATE"],
        "end_date": forecast_records[-1]["DATE"],
        "horizon_days": horizon,
        "frequency": "Daily",
        "method": "Recursive Auto-Regressive Multi-Step Forecasting (Fourier Seasonality + Weekday Structure)",
    }

    return df_forecast, metadata
