# Five-Minute SmartGrid AI Walkthrough Demo

1. **Launch Dashboard:**
   ```bash
   streamlit run app.py
   ```
   The dashboard loads the 4-year smart grid dataset (393,152 observations from Jan 2021 to Dec 2024).

2. **Overview Metrics:**
   Point out the top-level KPI metrics showing 48 months of continuous coverage, 5-minute sampling resolution, four forecasting targets (Zone 1, Zone 2, Zone 3, Total Power Demand), and state-of-the-art accuracy ($R^2 = 0.9966$, MAE = 27.95 MW on 4,000 MW total load).

3. **Tab 1 · User-Based Prediction:**
   - Select the scenario preset **"Extreme Summer Heatwave"** (Temp = 44°C, Humidity = 30%, Pressure = 998 hPa).
   - Click **"Generate Real ML Prediction"**.
   - Show how the real ML model forecasts an elevated cooling load across Zone 1, Zone 2, Zone 3, and Total Demand, calculating exact delta increments above historical baseline lags.
   - Switch inference models (e.g. from Ridge to Random Forest or XGBoost) to demonstrate multi-model capability.

4. **Tab 2 · 24-Hour Horizon Forecast:**
   - Click **"Forecast Next 24 Hours"**.
   - Show the 24-hour lookahead curve, peak hours per zone, and downloadable prediction CSV.

5. **Tab 3 · Historical Grid Analytics:**
   - Select a custom date window (e.g., peak summer months June–July).
   - Demonstrate the high-frequency 5-minute intra-day peaking and compare the 3 zone load profiles.

6. **Tab 4 · Model Comparison & Benchmarks:**
   - Inspect the benchmark evaluation table across Ridge, HistGradientBoosting, Random Forest, and XGBoost on the late 2024 held-out test set.
   - Show the Actual vs Predicted demand curves in `reports/actual_vs_predicted.png` and `reports/zone_comparison.png`.

7. **Tab 5 · Actuals Tracking:**
   - Demonstrate entering verified meter readings for past hours to monitor operational error drift.
