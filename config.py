"""
Configuration module for Smart Grid Energy Load Forecasting prototype.
Defines default parameters, paths, column mappings, and grid parameters.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "df_final.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
FUTURE_FORECAST_PATH = OUTPUTS_DIR / "future_forecast.csv"

# Column names in dataset
DEFAULT_DATE_COL = "DATE"
DEFAULT_TARGET_COL = "POWER_DEMAND"
DEFAULT_DEMAND_UNIT = "MW"  # Megawatts

# Candidate weather columns in df_final.csv
WEATHER_COLUMNS = [
    "temp",
    "humidity",
    "feelslike",
    "windspeed",
    "solarradiation",
    "dew",
    "sealevelpressure",
    "cloudcover",
]

# Primary weather features verified for modeling
PRIMARY_WEATHER_FEATURES = [
    "temp",
    "humidity",
    "precip",
    "windspeed",
    "solarradiation",
]

# Additional output file paths
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
SCENARIO_FORECASTS_PATH = OUTPUTS_DIR / "scenario_forecasts.csv"
ACTUAL_VS_PRED_PLOT_PATH = OUTPUTS_DIR / "actual_vs_predicted.png"
MODEL_COMPARISON_PLOT_PATH = OUTPUTS_DIR / "model_comparison_plot.png"
RESIDUALS_PLOT_PATH = OUTPUTS_DIR / "residuals_plot.png"
SCENARIO_PLOT_PATH = OUTPUTS_DIR / "scenario_comparison.png"

# Academic / Project Specs
EXPECTED_FREQ = "D"  # Daily observations
PROJECTED_REQUIREMENT_NOTE = (
    "Note on Data Horizon: The current dataset contains daily records spanning "
    "May 2022 to September 2024 (~28 months). College/Production specifications "
    "ideally recommend 4 to 5 years of hourly data. This prototype adapts "
    "transparently to the verified daily resolution while remaining extensible "
    "to hourly data when available."
)

# Plausibility bounds (in MW)
MIN_PLAUSIBLE_DEMAND = 500.0   # Grid load below this is treated as sensor dropout / outage
MAX_PLAUSIBLE_DEMAND = 15000.0 # Upper ceiling for physical regional load

# Modeling settings
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_LAGS = [1, 7]          # Yesterday (1 day) and last week same day (7 days)
DEFAULT_ROLLING_WINDOW = 7     # Past 7 days rolling mean
DEFAULT_FORECAST_HORIZON = 7   # 7 days forward recursive forecast

# Scenario Presets and UI Slider Ranges (grounded in Delhi dataset distribution)
WEATHER_BOUNDS = {
    "temp": {"min": 5.0, "max": 50.0, "step": 0.5, "unit": "°C", "label": "Mean Temperature"},
    "humidity": {"min": 10.0, "max": 100.0, "step": 1.0, "unit": "%", "label": "Relative Humidity"},
    "precip": {"min": 0.0, "max": 100.0, "step": 0.5, "unit": "mm", "label": "Rainfall / Precipitation"},
    "windspeed": {"min": 0.0, "max": 60.0, "step": 0.5, "unit": "km/h", "label": "Wind Speed"},
    "solarradiation": {"min": 0.0, "max": 400.0, "step": 5.0, "unit": "W/m²", "label": "Solar Radiation"},
}

SCENARIO_PRESETS = {
    "Baseline": {
        "description": "Recent 7-day average weather trajectory",
        "delta": {"temp": 0.0, "humidity": 0.0, "precip": 0.0, "windspeed": 0.0, "solarradiation": 0.0},
    },
    "Heatwave (+4°C, High Solar)": {
        "description": "Extreme summer heatwave with intense sunshine and increased AC demand",
        "delta": {"temp": 4.0, "humidity": -5.0, "precip": 0.0, "windspeed": 0.0, "solarradiation": 40.0},
    },
    "Cool Front / Monsoon (-4°C, Rain)": {
        "description": "Cool front with heavy rainfall and cloud cover suppressing cooling demand",
        "delta": {"temp": -4.0, "humidity": 15.0, "precip": 25.0, "windspeed": 4.0, "solarradiation": -50.0},
    },
}

# SAFE-LOAD Architecture Readiness Configuration
SAFE_LOAD_CONFIG = {
    "nominal_capacity_mw": 9000.0,
    "warning_threshold_pct": 0.85, # 85% of nominal capacity
    "critical_threshold_pct": 0.95, # 95% of nominal capacity
}

