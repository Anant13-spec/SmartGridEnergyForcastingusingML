# Smart Grid Energy Load Forecasting Using Machine Learning

An end-to-end, explainable machine learning forecasting prototype for regional electricity load forecasting, multi-model benchmarking, and interactive weather scenario simulation.

---

## 1. Project Overview & Problem Statement

Accurate electricity load forecasting is vital for power grid operators to match generation with demand, avoid brownouts/blackouts, schedule unit commitment, and integrate renewable energy.

This prototype provides an explainable machine learning pipeline that:
- **Audits and validates** the historical dataset, checking measurement units, location metadata, empirical frequency, duplicate fields, and sensor health.
- **Distinguishes raw vs. cleaned dataset statistics**: explicitly documents the raw minimum (0 MW, caused by 13 sensor dropouts) vs. the genuine physical minimum (2,444 MW) in both code and UI.
- **Engineers strictly leak-free time-series features**: cyclical Fourier harmonics (`sin_doy`, `cos_doy`), discrete weekday indicators (`dow_0`–`dow_6`), and auto-regressive lags (`lag_1`, `lag_7`).
- **Integrates meteorological predictors**: temperature (`temp`), relative humidity (`humidity`), rainfall (`precip`), wind speed (`windspeed`), and solar radiation (`solarradiation`).
- **Fairly compares multiple model architectures** on an identical chronological test period (80% train / 20% test) with and without weather:
  - Persistence Baseline ($y_t = y_{t-1}$)
  - Seasonal-Naive Baseline ($y_t = y_{t-7}$)
  - Explainable Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor (`HistGradientBoostingRegressor`)
- **Provides 5-Fold TimeSeriesSplit Cross-Validation** to assess expanding-window generalization.
- **Generates genuine post-dataset out-of-sample projections** (September 19 to 25, 2024) immediately following the verified dataset cutoff date (September 18, 2024), clearly distinguished from current live forecasting.
- **Simulates interactive weather scenarios** (Baseline, +4°C Heatwave, -4°C Cool Front, and custom parameters) through genuine model execution.
- **Features an interactive Streamlit dashboard** and 19 automated unit/integration regression tests.

---

## 2. Dataset Audit & Verified Limitations

The project utilizes `df_final.csv` (870 rows, 35 columns). Comprehensive empirical inspection reveals:

| Parameter | Verified Reality in `df_final.csv` | Initial Requirement | Handling / Engineering Decision |
| :--- | :--- | :--- | :--- |
| **Target Column** | `POWER_DEMAND` | Electricity Demand | Confirmed regional grid load (Megawatts - MW). |
| **Observation Frequency** | **Daily** ($1\text{ observation/day}$) | Hourly ($24\text{ obs/day}$) | Verified daily; no synthetic hourly rows fabricated. |
| **Date Range** | **May 1, 2022 to September 18, 2024** | 4 to 5 contiguous years | Spans ~2.4 years (~28 months, 870 rows). |
| **Geographic Location** | **Delhi, India** (`Delhi,India` / `delhi`) | Regional Grid | State Load Despatch Centre (SLDC) Delhi scale. |
| **Weather Stations** | Station VIDP (Indira Gandhi Airport) + IMD | Regional Stations | Verified station metadata aligns with regional demand. |
| **`DATE` vs `datetime`** | **Identical duplicates** (0 mismatches) | Independent fields | Confirmed duplicate fields; `DATE` used as primary index. |
| **Calendar Gaps** | 2 missing calendar dates (`2022-09-15`, `2022-09-16`) | Full daily sequence | Reindexed to complete daily sequence (872 days) and interpolated. |
| **Sensor Zero Dropouts** | 13 records recorded as $0\text{ MW}$ | Continuous positive load | Real regional grids cannot drop to 0 MW. Safely imputed via linear interpolation without altering raw CSV. |
| **Raw vs. Cleaned Minimum** | Raw: $0.0\text{ MW}$ / Cleaned: $2,444.0\text{ MW}$ | Continuous positive load | Cleaned minimum reflects lowest genuine non-zero load ($2,444\text{ MW}$ on 2023-01-29). |
| **Demand Peak** | $8,568\text{ MW}$ (2024-06-18) | Regional Grid Scale | Physically plausible regional load matching Delhi record summer peak. |

### Forecast Date Handling & Temporal Scope
- **Cutoff Date:** The dataset concludes on **September 18, 2024**.
- **Out-of-Sample Horizon:** Projections starting **September 19, 2024** represent out-of-sample forward projections immediately following the dataset cutoff.
- **Why Live Forecasting for Today Cannot Be Supported:** Because auto-regressive lags ($y_{t-1}, y_{t-7}$) require continuous telemetry, projecting from September 2024 across a 2-year unobserved gap to today would accumulate compounding recursive errors. Live forecasting requires a real-time telemetry feed from Delhi SLDC.

---

## 3. Project Architecture

```
smart-grid-prototype/
├── app.py                     # Interactive Streamlit dashboard
├── config.py                  # Global configurations, bounds, scenario presets, paths
├── requirements.txt           # Minimal dependencies
├── README.md                  # Comprehensive documentation & presentation guide
├── pytest.ini                 # Pytest configuration
├── df_final.csv               # Raw historical dataset (Preserved 100% untouched)
├── train_model.py             # CLI pipeline for training, benchmarking, and plot generation
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, empirical frequency detection & quality audits
│   ├── preprocessing.py       # Imputation & leak-free feature engineering
│   ├── forecasting.py         # Multi-model forecasters & recursive scenario simulation
│   ├── evaluation.py          # Chronological split, multi-model evaluation & TimeSeriesSplit CV
│   └── eda.py                 # Exploratory data analysis helpers
├── tests/
│   ├── test_data_loader.py    # Unit tests for loading, audit, raw vs cleaned stats, and weather alignment
│   ├── test_forecasting.py    # Unit tests for lags, multi-model suite, post-cutoff origin, and scenario sensitivity
│   └── test_app.py            # Automated Streamlit AppTest integration test
└── outputs/
    ├── model_comparison.csv   # Complete evaluation metrics table
    ├── test_predictions.csv   # Historical test predictions & residual errors
    ├── future_forecast.csv    # 7-day projected out-of-sample forecast (baseline)
    ├── scenario_forecasts.csv # Side-by-side comparative scenario forecasts
    ├── actual_vs_predicted.png# Plot of actual vs predicted demand on test period
    ├── model_comparison_plot.png # Bar chart of MAE and RMSE across models
    ├── residuals_plot.png     # Residual error histogram and timeline
    ├── scenario_comparison.png# Comparative scenario forecast trajectories
    └── ANTIGRAVITY_HANDOFF.md # Structured handoff report
```

---

## 4. Model Benchmark & Evaluated Results

All models were evaluated on the **identical chronological test set** (earliest 80% train: 692 days from `2022-05-08` to `2024-03-29`; unobserved 20% test: 173 days from `2024-03-30` to `2024-09-18`).

### Measured Test Performance

| Model Architecture | Feature Set | Weather-Aware | MAE (MW) | RMSE (MW) | $R^2$ Score | Performance Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **Calendar + Lags + Weather** | **Yes** | **218.24** | **297.56** | **0.9366** | 🥇 **Best Overall** |
| Linear Regression | Calendar + Lags | No | 248.27 | 322.06 | 0.9258 | 🥈 |
| Gradient Boosting (`HistGBR`) | Calendar + Lags + Weather | Yes | 272.58 | 379.38 | 0.8970 | 🥉 |
| Baseline (Persistence / Lag 1) | Historical Lag 1 | No | 283.97 | 363.14 | 0.9056 | Standard Baseline |
| Random Forest Regressor | Calendar + Lags + Weather | Yes | 296.32 | 401.21 | 0.8848 | Tree-based |
| Gradient Boosting (`HistGBR`) | Calendar + Lags | No | 351.53 | 463.37 | 0.8464 | Tree-based |
| Random Forest Regressor | Calendar + Lags | No | 361.50 | 472.27 | 0.8404 | Tree-based |
| Baseline (Seasonal-Naive / Lag 7) | Historical Lag 7 | No | 551.73 | 683.02 | 0.6662 | Cyclical Baseline |

*All metrics are verified from direct model runs on unobserved test data and saved to `outputs/model_comparison.csv`.*

### Quantified Impact of Weather Features
- **Linear Regression:** MAE reduced from 248.27 MW to **218.24 MW** (**-12.1% error reduction**, $R^2$ increased from 0.9258 to 0.9366).
- **Random Forest:** MAE reduced from 361.50 MW to **296.32 MW** (**-18.0% error reduction**, $R^2$ increased from 0.8404 to 0.8848).
- **Gradient Boosting:** MAE reduced from 351.53 MW to **272.58 MW** (**-22.5% error reduction**, $R^2$ increased from 0.8464 to 0.8970).

---

## 5. Weather Scenario Simulation (Post-Cutoff September 2024 Horizon)

The forecasting pipeline supports interactive scenario analysis by injecting hypothetical meteorological trajectories into the recursive forecasting loop:

| Forecast Date | Day of Week | Baseline (Trailing Weather) | Heatwave (+4°C, High Sun) | Heatwave $\Delta$ MW | Cool Front (-4°C, Rain) | Cool Front $\Delta$ MW |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2024-09-19** | Thursday | 4,995.59 MW | 5,102.41 MW | **+106.82 MW** | 4,841.93 MW | **-153.66 MW** |
| **2024-09-20** | Friday | 4,895.00 MW | 5,098.96 MW | **+203.96 MW** | 4,601.62 MW | **-293.38 MW** |
| **2024-09-21** | Saturday | 4,541.00 MW | 4,833.28 MW | **+292.28 MW** | 4,120.55 MW | **-420.45 MW** |
| **2024-09-22** | Sunday | 4,365.48 MW | 4,738.07 MW | **+372.59 MW** | 3,829.50 MW | **-535.98 MW** |
| **2024-09-23** | Monday | 4,448.37 MW | 4,894.00 MW | **+445.63 MW** | 3,807.33 MW | **-641.04 MW** |
| **2024-09-24** | Tuesday | 4,451.11 MW | 4,963.15 MW | **+512.04 MW** | 3,714.54 MW | **-736.57 MW** |
| **2024-09-25** | Wednesday | 4,443.95 MW | 5,016.37 MW | **+572.42 MW** | 3,620.51 MW | **-823.44 MW** |

*All scenario predictions invoke the actual fitted model weights. Thermal accumulation compounds recursively over consecutive hot/cool days.*

---

## 6. Execution Instructions

### Step 1: Activate Virtual Environment
```bash
cd /home/anj/Projects/smart-grid-prototype
source .venv/bin/activate
```

### Step 2: Run Automated Test Suite
```bash
.venv/bin/pytest -v
```
*Expected: 19 passed in ~8 seconds (0 errors, 0 warnings).*

### Step 3: Run Model Training & Benchmark Pipeline
```bash
.venv/bin/python train_model.py
```
*Generates all metric tables, predictions, and visualization PNGs in `outputs/`.*

### Step 4: Launch Interactive Streamlit Dashboard
```bash
.venv/bin/streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 7. Known Limitations & Future Work

1. **Daily vs Hourly Resolution:** The dataset provides daily total energy demand. Sub-daily intraday peak dispatch (e.g., afternoon 3 PM vs midnight) requires hourly telemetry.
2. **Current Telemetry Gap:** Operational out-of-sample deployment requires continuous daily telemetry to populate lag 1 and lag 7. Live forecasting for today is blocked until a real-time feed from Delhi SLDC is connected.
3. **Catastrophic Weather Shocks:** Sudden violent monsoon thunderstorms or dust storms (Andhi) that drop temperatures by 10°C in an hour cannot be forecasted days in advance without real-time meteorological radar feeds.
