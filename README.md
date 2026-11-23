# SmartGrid: High-Frequency Multi-Zone Energy Consumption Forecasting

SmartGrid is a machine learning electricity load forecasting platform upgraded with a comprehensive 4-year smart grid dataset (`df_3_zones_final.csv`). It provides high-frequency (5-minute and hourly) predictive intelligence for three distinct power distribution zones as well as total system grid demand.

## 1. Project Architecture & Upgraded Dataset

The system replaces the legacy single-year dataset with a 4-year smart grid dataset spanning January 2021 to December 2024:
- **Dataset File:** `data/df_3_zones_final.csv`
- **Total Observations:** 393,152 observations
- **Temporal Resolution:** Predominantly 5-minute intervals (99.7% of all time intervals)
- **Coverage:** January 2, 2021 through December 12, 2024 (48 months)
- **Demand Targets:**
  - `demand_zone_1_mw`: Distribution Zone 1 Demand (Mean: ~1,538 MW)
  - `demand_zone_2_mw`: Distribution Zone 2 Demand (Mean: ~1,456 MW)
  - `demand_zone_3_mw`: Distribution Zone 3 Demand (Mean: ~1,088 MW)
  - `Power demand`: Total Grid System Demand (Mean: ~3,961 MW, Peak: 8,632 MW)
- **Atmospheric Weather Features:**
  - `temp`: Ambient Temperature (°C)
  - `dwpt`: Dew Point Temperature (°C)
  - `rhum`: Relative Humidity (%)
  - `wdir`: Wind Direction (degrees)
  - `wspd`: Wind Speed (km/h)
  - `pres`: Sea-Level / Atmospheric Pressure (hPa)

### Three-Zone vs Total Demand Verification
Investigation confirms that `Power demand` is **not** identical to `demand_zone_1_mw + demand_zone_2_mw + demand_zone_3_mw` (mean difference: ~428 MW, correlation: 0.947). In utility grid dispatch, total power demand represents system-wide state gross drawal (including transmission losses and auxiliary/institutional load), while the three zones cover specific discom distribution areas. The platform models both individual zones and total demand independently.

## 2. Leakage-Free Feature Engineering

- **Strictly Non-Leaking Lags:** Lags are constructed strictly with backwards shifts (`.shift(1)`, `.shift(12)`, `.shift(288)`).
  - `lag_1`: 5 minutes prior
  - `lag_12`: 1 hour prior (12 x 5-minute intervals)
  - `lag_288`: 24 hours prior (288 x 5-minute intervals)
  - `rolling_mean_1h`: Past 1-hour rolling mean computed strictly as `.shift(1).rolling(12).mean()`.
- **Exclusion of Leaking Features:** The raw dataset column `moving_avg_3` was found to be computed using contemporaneous unshifted values (including the target at time $t$). It is strictly excluded from all feature sets.
- **Seasonality & Calendar:** Sine and cosine cyclical encodings for hour-of-day, minute, day-of-week, and month capture diurnal and annual demand cycles smoothly.

## 3. Evaluated Forecasting Models

Models are trained using strict forward chronological splitting (zero random shuffling):
- **Train Set (70%):** 275,004 observations (2021-01-03 to 2023-10-08)
- **Validation Set (15%):** 58,930 observations (2023-10-08 to 2024-05-12)
- **Held-Out Test Set (15%):** 58,930 observations (2024-05-12 to 2024-12-12)

### Held-Out Test Set Performance

| Target | Selected Model | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---|---|---|---|---|
| **Total Power Demand** | Linear Regression (Ridge) | 27.95 MW | 79.98 MW | **0.9966** | 0.63% |
| **Zone 1 Demand** | HistGradientBoosting / Ridge | 18.59 MW | 70.78 MW | **0.9907** | 1.05% |
| **Zone 2 Demand** | HistGradientBoosting / Ridge | 15.60 MW | 41.10 MW | **0.9941** | 0.97% |
| **Zone 3 Demand** | HistGradientBoosting / Ridge | 16.03 MW | 71.36 MW | **0.9808** | 1.23% |

*(All models, including Random Forest and XGBoost, are saved and selectable in the dashboard).*

## 4. Environment Setup

Python 3.10 or newer (tested with Python 3.14):

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Execution Pipeline

Execute the end-to-end pipeline using the commands below:

```bash
# 1. Preprocess raw dataset into clean 5-minute and continuous hourly datasets
python src/preprocess.py

# 2. Train and benchmark multi-model forecasters across all zones and total demand
python src/train_models.py

# 3. Evaluate models on held-out test data and generate visual plots
python src/evaluate.py

# 4. Run tests
pytest
```

## 6. Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

### Dashboard Tabs:
1. **🎯 User-Based Prediction:**
   - Interactive sliders and number inputs for Temperature, Humidity, Wind Speed, Dew Point, Pressure, and Wind Direction.
   - Quick operational scenario presets (Extreme Heatwave, Monsoon High Humidity, Winter Peak, High Wind).
   - Generates real ML predictions for Zone 1, Zone 2, Zone 3, and Total Demand simultaneously.
   - Compares predicted values against actual historical baseline lags.
2. **⚡ 24-Hour Horizon Forecast:**
   - 24-hour lookahead forecasting with peak-hour summaries and downloadable CSVs.
3. **📈 Historical Grid Analytics:**
   - Filter and visualize 4 years of 5-minute and hourly historical demand patterns.
4. **🧪 Model Comparison & Benchmarks:**
   - Inspect MAE, RMSE, R², and MAPE comparison tables and high-resolution Actual vs Predicted demand curves.
5. **📝 Actuals Tracking:**
   - Enter post-forecast observed meter readings and monitor operational error accumulation.
