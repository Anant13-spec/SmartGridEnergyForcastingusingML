import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of how Streamlit is launched
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

import config
from src.data_loader import load_raw_data, inspect_data_quality
from src.preprocessing import clean_electricity_data, build_feature_pipeline
from src.forecasting import LinearRegressionForecaster, generate_future_forecast
from src.evaluation import split_chronological, evaluate_forecaster

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Grid Energy Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Clean, professional energy-analytics aesthetic
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .badge-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400e;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }
    .badge-info {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        color: #065f46;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & CACHED PIPELINE
# ==========================================
@st.cache_data(show_spinner="Loading and preparing grid dataset...")
def load_and_prepare_pipeline(train_ratio: float = config.DEFAULT_TRAIN_RATIO):
    # 1. Load raw dataset (untouched)
    raw_df = load_raw_data()
    quality_report = inspect_data_quality(raw_df)

    # 2. Clean dataset (interpolate sensor dropouts & align calendar)
    clean_df, clean_meta = clean_electricity_data(raw_df, fill_calendar_gaps=True, impute_implausible=True)

    # 3. Build leak-free auto-regressive features
    feat_df, feat_cols, feat_meta = build_feature_pipeline(clean_df, include_weather=False)

    # 4. Strict chronological split
    train_df, test_df, split_meta = split_chronological(feat_df, train_ratio=train_ratio)

    # 5. Fit explainable Linear Regression model
    forecaster = LinearRegressionForecaster()
    forecaster.fit(train_df[feat_cols], train_df["POWER_DEMAND_CLEAN"])

    # 6. Evaluate on test set
    metrics_df, test_predictions = evaluate_forecaster(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feat_cols,
        model=forecaster.model,
        target_col="POWER_DEMAND_CLEAN"
    )

    coef_df = forecaster.get_feature_coefficients()

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "quality_report": quality_report,
        "train_df": train_df,
        "test_df": test_df,
        "feat_cols": feat_cols,
        "split_meta": split_meta,
        "forecaster": forecaster,
        "metrics_df": metrics_df,
        "test_predictions": test_predictions,
        "coef_df": coef_df,
    }


# Load pipeline
data = load_and_prepare_pipeline()
raw_df = data["raw_df"]
clean_df = data["clean_df"]
quality_report = data["quality_report"]
test_predictions = data["test_predictions"]
metrics_df = data["metrics_df"]
split_meta = data["split_meta"]
forecaster = data["forecaster"]
feat_cols = data["feat_cols"]


# ==========================================
# HEADER & PROJECT OVERVIEW
# ==========================================
st.title("⚡ Smart Grid Energy Load Forecasting Using Machine Learning")
st.markdown("#### *College Project Prototype | Explainable Time-Series Load Analytics*")

# Verified Dataset Limitation Disclosure
st.markdown(f"""
<div class="badge-warning">
    ⚠️ <strong>Dataset Verification & Limitation Note:</strong><br>
    The verified dataset <code>df_final.csv</code> contains <strong>{quality_report['total_rows']} daily observations</strong> 
    spanning from <strong>{quality_report.get('start_date')}</strong> to <strong>{quality_report.get('end_date')}</strong> (~{quality_report.get('total_years')} years).<br>
    While the initial project specification suggested 4 to 5 years of hourly data, this prototype 
    operates reliably on the verified daily frequency with full transparency.
</div>
""", unsafe_allow_html=True)

# Overview Metrics
target_sum = quality_report.get("target_summary", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Daily Demand", f"{target_sum.get('mean', 0):,.0f} MW")
c2.metric("Peak Demand", f"{target_sum.get('max', 0):,.0f} MW")
c3.metric("Minimum Demand", f"{target_sum.get('min', 0):,.0f} MW", delta=f"{quality_report.get('zero_or_negative_count')} zero dropouts handled", delta_color="inverse")
c4.metric("Total Records", f"{quality_report.get('total_rows', 0):,} days", delta=f"{quality_report.get('frequency_label')} Resolution")

st.markdown("---")


# ==========================================
# SECTION 1: HISTORICAL ELECTRICITY DEMAND
# ==========================================
st.subheader("1. Historical Electricity Demand Trajectory")
st.caption("Observed daily regional electricity load from May 2022 to September 2024.")

hist_chart_df = clean_df[["DATE", "POWER_DEMAND_CLEAN"]].copy()
hist_chart_df["DATE"] = pd.to_datetime(hist_chart_df["DATE"])

hist_chart = alt.Chart(hist_chart_df).mark_line(color="#0d9488", strokeWidth=1.5).encode(
    x=alt.X("DATE:T", title="Date"),
    y=alt.Y("POWER_DEMAND_CLEAN:Q", title="Demand (Megawatts - MW)"),
    tooltip=[
        alt.Tooltip("DATE:T", title="Date"),
        alt.Tooltip("POWER_DEMAND_CLEAN:Q", title="Demand (MW)", format=",.0f")
    ]
).properties(height=340).interactive()

st.altair_chart(hist_chart, use_container_width=True)

with st.expander("🔍 View Dataset Quality & Inspection Details"):
    st.markdown(f"""
    - **Total Raw Rows:** {quality_report['total_rows']} rows across {quality_report['total_columns']} columns.
    - **Temporal Frequency:** Daily observations (median interval: 1 day).
    - **Calendar Gaps:** Exactly 2 missing calendar dates identified (`{', '.join(quality_report.get('missing_calendar_dates', []))}`) and aligned.
    - **Zero Demand Dropouts:** {quality_report.get('zero_or_negative_count')} records with 0 MW load (sensor dropouts) safely imputed using time-series interpolation.
    - **Preservation Status:** Original `df_final.csv` is preserved completely untouched.
    """)
    st.dataframe(raw_df.head(10), use_container_width=True)

st.markdown("---")


# ==========================================
# SECTION 2: MODEL EVALUATION (ACTUAL VS PREDICTED)
# ==========================================
st.subheader("2. Model Evaluation on Unseen Test Period")
st.caption("Linear Regression trained on earliest 80% and evaluated chronologically on final 20% to prevent future-data leakage.")

# Split Info Banner
st.markdown(f"""
<div class="badge-info">
    ⏱️ <strong>Chronological Split Summary:</strong><br>
    • <strong>Training Set:</strong> {split_meta['train_rows']} days ({split_meta['train_start']} to {split_meta['train_end']})<br>
    • <strong>Testing Set:</strong> {split_meta['test_rows']} days ({split_meta['test_start']} to {split_meta['test_end']})<br>
    <em>Evaluation tests the model on unseen future observations with strictly no lookahead leakage.</em>
</div>
""", unsafe_allow_html=True)

# Metric Display Cards & Comparison Table
lr_match = metrics_df[metrics_df["Model"].str.contains("Linear Regression")]
lr_row = lr_match.iloc[0] if len(lr_match) > 0 else metrics_df.iloc[-1]

base_match = metrics_df[metrics_df["Model"].str.contains("Persistence")]
base_row = base_match.iloc[0] if len(base_match) > 0 else metrics_df.iloc[0]

seas_match = metrics_df[metrics_df["Model"].str.contains("Seasonal-Naive")]
seas_row = seas_match.iloc[0] if len(seas_match) > 0 else (metrics_df.iloc[1] if len(metrics_df) > 1 else base_row)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test MAE (Linear Regression)", f"{lr_row['MAE (MW)']} MW", delta=f"{round(base_row['MAE (MW)'] - lr_row['MAE (MW)'], 2)} MW better than persistence")
m2.metric("Test RMSE (Linear Regression)", f"{lr_row['RMSE (MW)']} MW", delta=f"{round(base_row['RMSE (MW)'] - lr_row['RMSE (MW)'], 2)} MW better than persistence")
m3.metric("Test R² Score", f"{lr_row['R2 Score']}", help="Proportion of variance explained by model")
m4.metric("Persistence Baseline MAE", f"{base_row['MAE (MW)']} MW", help="Naive rule: tomorrow equals today")

st.markdown("##### Model & Baseline Comparison Summary")
st.dataframe(metrics_df, use_container_width=True)

# Actual vs Predicted Graph
st.markdown("##### Actual vs. Predicted Demand (Test Period)")

test_plot_df = pd.melt(
    test_predictions,
    id_vars=["DATE"],
    value_vars=["ACTUAL_DEMAND", "PREDICTED_DEMAND"],
    var_name="Series",
    value_name="Demand_MW"
)
test_plot_df["DATE"] = pd.to_datetime(test_plot_df["DATE"])

eval_chart = alt.Chart(test_plot_df).mark_line(strokeWidth=1.8).encode(
    x=alt.X("DATE:T", title="Test Date"),
    y=alt.Y("Demand_MW:Q", title="Demand (MW)"),
    color=alt.Color(
        "Series:N",
        scale=alt.Scale(domain=["ACTUAL_DEMAND", "PREDICTED_DEMAND"], range=["#0f172a", "#0d9488"]),
        title="Series"
    ),
    tooltip=[
        alt.Tooltip("DATE:T", title="Date"),
        alt.Tooltip("Series:N", title="Type"),
        alt.Tooltip("Demand_MW:Q", title="Demand (MW)", format=",.1f")
    ]
).properties(height=360).interactive()

st.altair_chart(eval_chart, use_container_width=True)

# Explainability & Test Sample Table
col_coef, col_sample = st.columns(2)
with col_coef:
    st.markdown("##### Feature Weights (Explainability)")
    st.markdown("Coefficients showing the physical baseline load and lag effects:")
    st.dataframe(data["coef_df"], use_container_width=True)

with col_sample:
    st.markdown("##### Prediction Samples (First 10 Days of Test Period)")
    st.dataframe(test_predictions[["DATE", "ACTUAL_DEMAND", "PREDICTED_DEMAND", "ERROR"]].head(10), use_container_width=True)

# Test Predictions Download
csv_test = test_predictions.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Test Predictions (CSV)",
    data=csv_test,
    file_name="test_predictions_evaluated.csv",
    mime="text/csv"
)

st.markdown("---")


# ==========================================
# SECTION 3: FUTURE DEMAND FORECAST
# ==========================================
st.subheader("3. Genuine Future Demand Forecast")
st.caption("Multi-step forward forecast starting after the last observed date in the dataset using recursive lag updates.")

last_observed_date = pd.to_datetime(clean_df["DATE"].iloc[-1]).date()

c_ctrl1, c_ctrl2 = st.columns([1, 2])
with c_ctrl1:
    forecast_horizon = st.slider(
        "Select Forecast Horizon (Days):",
        min_value=1,
        max_value=30,
        value=config.DEFAULT_FORECAST_HORIZON,
        step=1
    )
    forecast_start = last_observed_date + pd.Timedelta(days=1)
    forecast_end = last_observed_date + pd.Timedelta(days=forecast_horizon)

    st.markdown(f"""
    - **Historical Cutoff Date:** `{last_observed_date}` (Red boundary line)
    - **Forecast Start:** `{forecast_start}`
    - **Forecast End:** `{forecast_end}`
    - **Observation Frequency:** Daily
    """)

# Generate forecast
test_rmse = float(lr_row["RMSE (MW)"])
future_df, future_meta = generate_future_forecast(
    model=forecaster.model,
    feature_cols=feat_cols,
    historical_df=clean_df,
    horizon=forecast_horizon,
    test_rmse=test_rmse
)

# Combined historical context + future forecast chart with clearly marked boundary
recent_hist = clean_df.tail(30)[["DATE", "POWER_DEMAND_CLEAN"]].copy()
recent_hist["DATE"] = pd.to_datetime(recent_hist["DATE"]).dt.strftime("%Y-%m-%d")
recent_hist["Type"] = "Historical Observed"
recent_hist = recent_hist.rename(columns={"POWER_DEMAND_CLEAN": "Demand_MW"})

future_plot = future_df[["DATE", "FORECASTED_DEMAND_MW"]].copy()
future_plot["Type"] = "Future Forecast"
future_plot = future_plot.rename(columns={"FORECASTED_DEMAND_MW": "Demand_MW"})

combined_chart_data = pd.concat([recent_hist, future_plot], ignore_index=True)
combined_chart_data["DATE"] = pd.to_datetime(combined_chart_data["DATE"])

# Boundary marker rule
boundary_df = pd.DataFrame([{"DATE": pd.to_datetime(last_observed_date), "Label": "Cutoff Boundary"}])

base_lines = alt.Chart(combined_chart_data).mark_line(strokeWidth=2.2, point=True).encode(
    x=alt.X("DATE:T", title="Date"),
    y=alt.Y("Demand_MW:Q", title="Power Demand (MW)"),
    color=alt.Color(
        "Type:N",
        scale=alt.Scale(domain=["Historical Observed", "Future Forecast"], range=["#475569", "#0d9488"]),
        title="Type"
    ),
    tooltip=[
        alt.Tooltip("DATE:T", title="Date"),
        alt.Tooltip("Type:N", title="Series"),
        alt.Tooltip("Demand_MW:Q", title="Demand (MW)", format=",.1f")
    ]
)

cutoff_rule = alt.Chart(boundary_df).mark_rule(
    color="#ef4444",
    strokeDash=[4, 4],
    strokeWidth=2.0
).encode(
    x="DATE:T"
)

cutoff_text = alt.Chart(boundary_df).mark_text(
    text="← Observed | Forecast →",
    align="left",
    dx=6,
    dy=-140,
    color="#ef4444",
    fontSize=11,
    fontWeight="bold"
).encode(
    x="DATE:T"
)

future_chart = (base_lines + cutoff_rule + cutoff_text).properties(height=380).interactive()
st.altair_chart(future_chart, use_container_width=True)

# Technical Explanation & Uncertainty Disclosure
st.info(r"""
💡 **Why the Forecast Appears Smooth Compared to Historical Extremes:**
1. **Unobserved Weather Shocks:** In historical data, extreme daily swings (such as 7,000+ MW peaks) were driven by sudden unforecasted weather events (e.g., severe heatwaves). During future multi-step forecasting, exact day-to-day weather shocks are unobserved.
2. **Conditional Expected Trajectory:** A valid statistical model produces the *expected (mean) conditional trajectory*. It captures the physical day-of-week load cycles (low on Saturday/Sunday, high mid-week) and annual seasonal transition, rather than injecting artificial random noise.
3. **Forecast Uncertainty:** The 95% confidence intervals below reflect widening uncertainty as the forecast horizon extends forward ($\pm 1.96 \cdot \text{RMSE} \cdot \sqrt{h}$).
""")

# Future Forecast Table
st.markdown("##### Forecast Results Table (with 95% Confidence Bounds)")
st.dataframe(future_df, use_container_width=True)

# Download Future Forecast
csv_future = future_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Future Forecast (CSV)",
    data=csv_future,
    file_name="future_energy_demand_forecast.csv",
    mime="text/csv"
)
