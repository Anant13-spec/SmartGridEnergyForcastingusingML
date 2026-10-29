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

# SAFE-LOAD Architecture Readiness Configuration
# (Thresholds can be adjusted dynamically in the future load alert module)
SAFE_LOAD_CONFIG = {
    "nominal_capacity_mw": 9000.0,
    "warning_threshold_pct": 0.85, # 85% of nominal capacity
    "critical_threshold_pct": 0.95, # 95% of nominal capacity
}
