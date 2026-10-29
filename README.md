# Smart Grid Energy Load Forecasting Using Machine Learning

A clean, working college project prototype for electricity demand analytics, explainable machine learning forecasting, and multi-step future projections.

---

## 1. Project Overview & Problem Statement

Modern electrical grid operations require accurate electricity load forecasting to match power generation with consumer demand, prevent outages, and maintain grid stability. 

This prototype builds an end-to-end, explainable machine learning pipeline that:
- Audits and preprocesses historical electricity demand data.
- Imputes sensor dropouts and aligns calendar discontinuities without corrupting auto-regressive structure.
- Trains an explainable **Linear Regression** model using leak-free calendar, lag, and rolling-mean features.
- Evaluates forecasting performance using strict chronological testing (80% train / 20% test).
- Benchmarks against a naive persistence baseline ($y_t = y_{t-1}$).
- Generates recursive multi-step future forecasts (default 7 days).
- Presents the analytics through an interactive Streamlit dashboard.

---

## 2. Dataset Verification & Verified Limitations

The project utilizes `df_final.csv`. Inspection and validation of the actual file reveals:

| Parameter | Verified Dataset Reality | Initial College Requirement | Status / Handling |
| :--- | :--- | :--- | :--- |
| **Observation Frequency** | **Daily** ($1\text{ record/day}$) | Hourly ($24\text{ records/day}$) | Verified daily; no synthetic hourly data fabricated. |
| **Date Range** | **May 1, 2022 to September 18, 2024** | 4 to 5 contiguous years | Spans ~2.4 years (~28 months, 870 rows). |
| **Missing Calendar Dates** | Exactly 2 dates (`2022-09-15` & `2022-09-16`) | Complete calendar sequence | Reindexed to complete daily sequence and interpolated. |
| **Sensor Zero Dropouts** | 13 records recorded as $0\text{ MW}$ | Continuous positive load | Real regional grids cannot drop to 0 MW. Safely imputed via linear interpolation. |
| **Demand Scale** | Min: $2,444\text{ MW}$ (non-zero) / Max: $8,568\text{ MW}$ | Regional Grid Scale | Physically plausible regional load. |

### Explicit Limitation Disclosure
As required for scientific accuracy: the current dataset contains **daily resolution over ~2.4 years** rather than the ideal 4–5 years of hourly data. This prototype transparently operates at the verified daily frequency, with the forecasting pipeline ready to ingest hourly data if higher-resolution feeds become available in the future.

---

## 3. Project Architecture

```
smart-grid-prototype/
├── app.py                     # Streamlit dashboard
├── config.py                  # Project paths, bounds, and parameters
├── requirements.txt           # Minimal dependencies
├── README.md                  # Project documentation & presentation guide
├── pytest.ini                 # Pytest configuration
├── df_final.csv               # Original dataset (Preserved 100% untouched)
├── legacy_backup/             # Backup of old reference scripts
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, frequency detection & quality audits
│   ├── preprocessing.py       # Imputation & leak-free feature engineering
│   ├── forecasting.py         # Linear Regression & recursive future forecaster
│   └── evaluation.py          # Chronological split & metric calculations (MAE, RMSE, R²)
├── tests/
│   ├── test_data_loader.py    # Unit tests for loading & quality validation
│   ├── test_forecasting.py    # Unit tests for leak-free lags & recursive forecast
│   └── test_app.py            # Automated Streamlit AppTest integration test
└── outputs/
    ├── test_predictions.csv   # Evaluated test period predictions & errors
    └── future_forecast.csv    # 7-day projected future forecast
```

---

## 4. Machine Learning Methodology

### Explainable Linear Regression (Primary Model)
Electricity demand is modeled using an explainable linear equation:
$$\hat{y}_t = \beta_0 + \beta_1 \cdot \text{day\_of\_week}_t + \beta_2 \cdot \text{month}_t + \beta_3 \cdot \text{is\_weekend}_t + \beta_4 \cdot y_{t-1} + \beta_5 \cdot y_{t-7} + \beta_6 \cdot \bar{y}_{t-7:t-1}$$

### Why Linear Regression?
1. **Explainability:** Each coefficient $\beta_i$ directly quantifies the physical MW change associated with each calendar day or past load state.
2. **Auto-Regressive Reliability:** High autocorrelation in power grids allows linear combinations of recent demand to track cyclical trends reliably without overfitting.

### Feature Engineering (Strictly Leak-Free)
- **`day_of_week`:** Day of the week (0=Monday, 6=Sunday).
- **`month`:** Calendar month (1 to 12).
### Feature Engineering (Strictly Leak-Free)
- **`sin_doy` & `cos_doy`:** Annual seasonal Fourier harmonics ($\sin(2\pi \cdot \text{doy} / 365.25)$ and $\cos(2\pi \cdot \text{doy} / 365.25)$) that smoothly capture the yearly temperature/climate cooling without future lookahead.
- **`dow_0` through `dow_6`:** Discrete indicator variables for each weekday (Monday through Sunday), allowing the model to capture the true physical profile of each day (e.g. Wednesday peak vs. Saturday/Sunday load drop).
- **`lag_1`:** Demand from 1 day prior ($y_{t-1}$).
- **`lag_7`:** Demand from 7 days prior ($y_{t-7}$, capturing weekly recurrence).

---

## 5. Chronological Evaluation & Real Results

### Evaluation Strategy
- **Chronological Split:** Earliest 80% used for training, final 20% for testing. Never randomly shuffled.
- **Training Period:** 692 days (`2022-05-08` to `2024-03-29`).
- **Testing Period:** 173 unobserved days (`2024-03-30` to `2024-09-18`).

### Real Evaluated Metrics

| Model | MAE (MW) | RMSE (MW) | $R^2$ Score | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Persistence / Lag 1)** | 283.97 | 363.14 | 0.9056 | Naive benchmark: tomorrow equals today |
| **Baseline (Seasonal-Naive / Lag 7)** | 551.73 | 683.02 | 0.6662 | 7-day cyclical benchmark: today equals last week |
| **Linear Regression (Harmonics + DOW + Lags)** | **248.27** | **322.06** | **0.9258** | **Best performance; beats both baselines across all metrics** |

*All metrics are calculated live from real model outputs on unobserved test data.*

---

## 6. Genuine Recursive Future Forecasting & Variance Diagnosis

### Why Did the Initial Forecast Appear Flat?
1. **Linear Month Encoding:** `month` was originally encoded as a single continuous integer (1 to 12). Because electricity demand rises in summer and drops in winter/autumn, fitting a linear line through a curved seasonal cycle yielded a slope near zero ($\beta \approx -0.54\text{ MW/month}$), blinding the model to annual seasonal cooling.
2. **Linear Day-of-Week Encoding:** `day_of_week` was an integer (0 to 6), flattening individual weekday effects into a single straight line.
3. **Rolling Mean Collinearity:** Including both `lag_1` and `rolling_mean_7` resulted in collinear coefficients, where `rolling_mean_7` carried a negative weight that acted like an artificial low-pass filter, dampening cyclical fluctuations.
4. **Conditional Expectation vs. Weather Noise:** Historical demand exhibits extreme jagged spikes (e.g., 7,000+ MW) driven by unforecasted weather shocks (heatwaves/storms). Without oracle future weather data, a statistically valid model produces the expected (mean) conditional trajectory rather than fabricating artificial noise.

### The Resolution
- Replaced linear month with cyclical Fourier harmonics (`sin_doy`, `cos_doy`).
- Replaced linear day-of-week with discrete indicator variables (`dow_0` to `dow_6`).
- Removed collinear rolling means, relying on clean `lag_1` and `lag_7`.
- Marked the historical cutoff boundary clearly on the chart (red dashed line).
- Result: Test MAE improved to **248.27 MW** ($R^2 = 0.9258$), and future multi-step predictions capture distinct, realistic weekly load cycles with 95% confidence bands.

---

## 7. Setup & Execution Instructions (Omarchy Linux / Arch Linux)

### Step 1: Open Terminal in Project Folder
```bash
cd /home/anj/Projects/smart-grid-prototype
```

### Step 2: Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```bash
pytest
```
*Expected: 12 passed in ~1.5 seconds.*

### Step 5: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 8. College Presentation Script

> *"Good morning, Professors. Today I am presenting our prototype for Smart Grid Energy Load Forecasting Using Machine Learning.*
>
> *Electricity grid supply must continually match consumer load to avoid blackouts. Our system delivers a complete, transparent forecasting pipeline.*
>
> *First, we audited the dataset: 870 daily observations from May 2022 to September 2024. We detected 13 sensor zero-demand anomalies and 2 calendar gaps, cleanly resolving them via chronological reindexing and temporal interpolation without altering the original CSV.*
>
> *Second, to prevent lookahead data leakage, all auto-regressive lags (lag-1, lag-7, rolling-mean-7) and calendar features were strictly shifted from past observations.*
>
> *Third, we split the data chronologically (earliest 80% train / final 20% test). Our explainable Linear Regression model achieved a Test MAE of 265 MW, an RMSE of 343 MW, and an $R^2$ of 0.9158, convincingly outperforming the naive persistence baseline.*
>
> *Finally, our dashboard implements a genuine recursive multi-step future forecaster that predicts the next 7 days with confidence bands. The application is tested with automated unit tests and is running live on Streamlit. Thank you."*
