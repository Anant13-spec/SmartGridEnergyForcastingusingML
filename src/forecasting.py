"""
Forecasting Module for Smart Grid Energy Load Forecasting.
Implements explainable Linear Regression, Random Forest, Gradient Boosting,
and recursive multi-step future forecasting with interactive weather scenario analysis.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import config


# Human-readable feature label map
FEATURE_LABEL_MAP = {
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
    "temp": "Mean Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "precip": "Rainfall / Precip (mm)",
    "windspeed": "Wind Speed (km/h)",
    "solarradiation": "Solar Radiation (W/m²)",
    "feelslike": "Feels-like Temp (°C)",
    "cloudcover": "Cloud Cover (%)",
    "sealevelpressure": "Sea Level Pressure (hPa)",
}


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

        records = []
        for feat, coef in zip(self.feature_names, self.model.coef_):
            records.append({
                "Feature": FEATURE_LABEL_MAP.get(feat, feat),
                "Raw Feature": feat,
                "Coefficient (MW)": round(float(coef), 2),
                "Absolute Magnitude": round(abs(float(coef)), 2),
            })

        df_coef = pd.DataFrame(records).sort_values("Absolute Magnitude", ascending=False).reset_index(drop=True)
        return df_coef


class RandomForestForecaster:
    """
    Random Forest time-series forecasting model.
    Provides non-linear partitioning and feature importance benchmarking.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        records = []
        for feat, imp in zip(self.feature_names, self.model.feature_importances_):
            records.append({
                "Feature": FEATURE_LABEL_MAP.get(feat, feat),
                "Raw Feature": feat,
                "Importance": round(float(imp), 4),
            })
        return pd.DataFrame(records).sort_values("Importance", ascending=False).reset_index(drop=True)


class GradientBoostingForecaster:
    """
    Gradient Boosting forecaster using HistGradientBoostingRegressor.
    High performance non-linear model with native missing value support.
    """

    def __init__(self, max_iter: int = 150, max_depth: int = 5, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)


# Backward compatibility alias
TimeSeriesModels = LinearRegressionForecaster


def generate_future_forecast(
    model: Any,
    feature_cols: List[str],
    historical_df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    horizon: int = config.DEFAULT_FORECAST_HORIZON,
    test_rmse: Optional[float] = None,
    weather_scenario: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform genuine recursive multi-step future forecasting into unobserved periods.
    Supports interactive weather scenarios (Baseline, Heatwave, Cool Front, Custom).
    
    Starting from the last observed historical date:
    For step h = 1..horizon:
    1. Determine the future timestamp (t + h).
    2. Compute exact calendar features for the future date:
       - sin_doy, cos_doy (Fourier annual cycle)
       - dow_0..dow_6 (one-hot indicator for the specific weekday)
    3. Update auto-regressive lags:
       - lag_1 uses demand from (t + h - 1)
       - lag_7 uses demand from (t + h - 7)
    4. Inject weather features (from scenario dict or recent trailing historical average).
    5. Predict demand using the fitted model.
    6. Append predicted demand to the running sequence to predict subsequent steps.
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

    # Trailing baseline weather (past 7 days average)
    trailing_weather: Dict[str, float] = {}
    for col in feature_cols:
        if col in hist.columns and pd.api.types.is_numeric_dtype(hist[col]):
            trailing_weather[col] = float(hist[col].tail(7).mean())

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

        # 4. Inject weather inputs (priority: weather_scenario > trailing_weather > 0.0)
        for f in feature_cols:
            if f not in row_feat:
                if weather_scenario is not None and f in weather_scenario:
                    row_feat[f] = float(weather_scenario[f])
                elif f in trailing_weather:
                    row_feat[f] = trailing_weather[f]
                else:
                    row_feat[f] = 0.0

        X_future = pd.DataFrame([row_feat])[feature_cols]

        # Predict future demand
        predicted_demand = float(model_to_use.predict(X_future)[0])
        predicted_demand = max(config.MIN_PLAUSIBLE_DEMAND, predicted_demand)

        # Confidence bounds based on test RMSE and horizon growth (sqrt(step))
        rmse_val = test_rmse if test_rmse is not None else 297.0
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
        "method": "Recursive Auto-Regressive Multi-Step Forecasting (Fourier Seasonality + Weekday Structure + Exogenous Weather)",
        "weather_scenario_applied": weather_scenario is not None,
    }

    return df_forecast, metadata


def simulate_scenario_comparison(
    model: Any,
    feature_cols: List[str],
    historical_df: pd.DataFrame,
    horizon: int = config.DEFAULT_FORECAST_HORIZON,
    test_rmse: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run baseline, hotter (+4°C), and cooler (-4°C) weather scenarios and produce
    a side-by-side comparative forecast with delta MW from baseline.
    """
    # 1. Baseline weather (trailing 7-day average)
    baseline_weather: Dict[str, float] = {}
    for col in config.PRIMARY_WEATHER_FEATURES:
        if col in historical_df.columns and pd.api.types.is_numeric_dtype(historical_df[col]):
            baseline_weather[col] = float(historical_df[col].tail(7).mean())

    # 2. Hotter weather (+4°C, -5% humidity, +40 W/m² solar)
    hotter_weather = baseline_weather.copy()
    hotter_weather["temp"] = hotter_weather.get("temp", 28.0) + 4.0
    hotter_weather["humidity"] = max(10.0, hotter_weather.get("humidity", 60.0) - 5.0)
    hotter_weather["precip"] = 0.0
    hotter_weather["solarradiation"] = min(400.0, hotter_weather.get("solarradiation", 200.0) + 40.0)

    # 3. Cooler weather (-4°C, +15% humidity, 25mm precip, -50 W/m² solar)
    cooler_weather = baseline_weather.copy()
    cooler_weather["temp"] = max(5.0, cooler_weather.get("temp", 28.0) - 4.0)
    cooler_weather["humidity"] = min(100.0, cooler_weather.get("humidity", 60.0) + 15.0)
    cooler_weather["precip"] = 25.0
    cooler_weather["solarradiation"] = max(10.0, cooler_weather.get("solarradiation", 200.0) - 50.0)
    cooler_weather["windspeed"] = cooler_weather.get("windspeed", 15.0) + 4.0

    df_base, _ = generate_future_forecast(
        model=model, feature_cols=feature_cols, historical_df=historical_df,
        horizon=horizon, test_rmse=test_rmse, weather_scenario=baseline_weather
    )
    df_hot, _ = generate_future_forecast(
        model=model, feature_cols=feature_cols, historical_df=historical_df,
        horizon=horizon, test_rmse=test_rmse, weather_scenario=hotter_weather
    )
    df_cool, _ = generate_future_forecast(
        model=model, feature_cols=feature_cols, historical_df=historical_df,
        horizon=horizon, test_rmse=test_rmse, weather_scenario=cooler_weather
    )

    comparison_records = []
    for i in range(len(df_base)):
        date_str = df_base.loc[i, "DATE"]
        dow_str = df_base.loc[i, "DAY_OF_WEEK"]
        base_val = df_base.loc[i, "FORECASTED_DEMAND_MW"]
        hot_val = df_hot.loc[i, "FORECASTED_DEMAND_MW"]
        cool_val = df_cool.loc[i, "FORECASTED_DEMAND_MW"]

        comparison_records.append({
            "DATE": date_str,
            "DAY_OF_WEEK": dow_str,
            "BASELINE_DEMAND_MW": base_val,
            "HOTTER_SCENARIO_MW": hot_val,
            "HOTTER_DELTA_MW": round(hot_val - base_val, 2),
            "COOLER_SCENARIO_MW": cool_val,
            "COOLER_DELTA_MW": round(cool_val - base_val, 2),
        })

    df_comparison = pd.DataFrame(comparison_records)
    scenarios_dict = {
        "Baseline": df_base,
        "Heatwave (+4°C)": df_hot,
        "Cool Front (-4°C)": df_cool,
    }
    return df_comparison, scenarios_dict

