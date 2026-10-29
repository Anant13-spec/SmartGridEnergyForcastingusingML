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
from typing import Tuple, Dict, Any, List

import config
from src.data_loader import load_raw_data, inspect_data_quality
from src.preprocessing import clean_electricity_data, build_feature_pipeline
from src.forecasting import (
    LinearRegressionForecaster,
    RandomForestForecaster,
    GradientBoostingForecaster,
    generate_future_forecast,
    simulate_scenario_comparison,
    FEATURE_LABEL_MAP
)
from src.evaluation import split_chronological, evaluate_model_suite, evaluate_timeseries_cv

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Grid Energy Load Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional energy analytics aesthetic
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
    .badge-historical {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 12px 16px;
        border-radius: 6px;
        color: #334155;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & CACHED PIPELINE
# ==========================================
@st.cache_data(show_spinner="Training model suite and preparing load forecasting pipeline...")
def load_and_prepare_complete_pipeline():
    # 1. Load raw dataset
    raw_df = load_raw_data()
    quality_report = inspect_data_quality(raw_df)

    # 2. Clean dataset (interpolate sensor dropouts & align calendar)
    clean_df, clean_meta = clean_electricity_data(raw_df, fill_calendar_gaps=True, impute_implausible=True)

    # 3. Build leak-free feature pipelines (without and with weather)
    df_no_w, feat_no_w, meta_no_w = build_feature_pipeline(clean_df, include_weather=False)
    df_w, feat_w, meta_w = build_feature_pipeline(
        clean_df,
        include_weather=True,
        weather_cols=config.PRIMARY_WEATHER_FEATURES
    )

    # 4. Strict chronological split
    train_no_w, test_no_w, split_meta = split_chronological(df_no_w, train_ratio=config.DEFAULT_TRAIN_RATIO)
    train_w, test_w, _ = split_chronological(df_w, train_ratio=config.DEFAULT_TRAIN_RATIO)

    # 5. Evaluate complete model suite
    df_metrics, df_predictions, fitted_models = evaluate_model_suite(
        train_no_w=train_no_w,
        test_no_w=test_no_w,
        feat_no_w=feat_no_w,
        train_w=train_w,
        test_w=test_w,
        feat_w=feat_w,
    )

    # 6. Feature explainability
    lr_forecaster = LinearRegressionForecaster()
    lr_forecaster.model = fitted_models["lr_weather"]
    lr_forecaster.feature_names = feat_w
    lr_forecaster.is_fitted = True
    coef_df = lr_forecaster.get_feature_coefficients()

    rf_forecaster = RandomForestForecaster()
    rf_forecaster.model = fitted_models["rf_weather"]
    rf_forecaster.feature_names = feat_w
    rf_forecaster.is_fitted = True
    rf_importance_df = rf_forecaster.get_feature_importances()

    # 7. Baseline weather (trailing 7-day average from end of historical dataset)
    baseline_weather = {}
    for col in config.PRIMARY_WEATHER_FEATURES:
        if col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[col]):
            baseline_weather[col] = float(clean_df[col].tail(7).mean())

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "quality_report": quality_report,
        "clean_meta": clean_meta,
        "split_meta": split_meta,
        "df_metrics": df_metrics,
        "df_predictions": df_predictions,
        "fitted_models": fitted_models,
        "feat_w": feat_w,
        "feat_no_w": feat_no_w,
        "coef_df": coef_df,
        "rf_importance_df": rf_importance_df,
        "baseline_weather": baseline_weather,
    }


# Execute pipeline
data = load_and_prepare_complete_pipeline()
raw_df = data["raw_df"]
clean_df = data["clean_df"]
quality_report = data["quality_report"]
clean_meta = data["clean_meta"]
split_meta = data["split_meta"]
df_metrics = data["df_metrics"]
df_predictions = data["df_predictions"]
fitted_models = data["fitted_models"]
feat_w = data["feat_w"]
coef_df = data["coef_df"]
baseline_weather = data["baseline_weather"]

# Primary model for forecasting and scenarios
primary_model = fitted_models["lr_weather"]
primary_test_rmse = float(df_metrics.loc[df_metrics["Model"] == "Linear Regression (Weather-Aware)", "RMSE (MW)"].iloc[0])


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("⚡ Smart Grid Analytics")
st.sidebar.caption("Explainable Time-Series Load Forecasting")

view_selection = st.sidebar.radio(
    "Navigation Sections:",
    [
        "1. Data Overview & Audit",
        "2. Model Evaluation & Benchmark",
        "3. Post-Dataset Out-of-Sample Projection (Sept 2024)",
        "4. Weather Scenario Simulation (Sept 2024 Horizon)",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Scope & Horizon")
st.sidebar.write(f"• **Region:** {quality_report.get('locations', ['Delhi, India'])[0]}")
st.sidebar.write(f"• **Resolution:** {quality_report.get('frequency_label')} (Empirical)")
st.sidebar.write(f"• **Historical Dataset:** May 2022 – Sept 18, 2024")
st.sidebar.write(f"• **Cutoff Date:** `2024-09-18`")
st.sidebar.write(f"• **Best Model:** LR (Weather-Aware)")
st.sidebar.write(f"• **Test MAE:** {df_metrics.loc[df_metrics['Model'] == 'Linear Regression (Weather-Aware)', 'MAE (MW)'].iloc[0]} MW")
st.sidebar.write(f"• **Test R²:** {df_metrics.loc[df_metrics['Model'] == 'Linear Regression (Weather-Aware)', 'R2 Score'].iloc[0]}")


# ==========================================
# HEADER
# ==========================================
st.title("⚡ Smart Grid Energy Load Forecasting Using Machine Learning")
st.markdown("#### *Explainable Multi-Model Time-Series Analytics & Weather Scenario Simulation*")

# Verified Dataset Disclosure Banner
st.markdown(f"""
<div class="badge-historical">
    📌 <strong>Dataset Temporal Scope & Horizon Disclosure:</strong><br>
    The verified historical dataset spans <strong>{quality_report.get('start_date')} to {quality_report.get('end_date')}</strong> 
    ({clean_meta['clean_rows']} daily records) for <strong>{quality_report.get('locations', ['Delhi, India'])[0]}</strong>.<br>
    • <strong>Post-Cutoff Horizon:</strong> Projections starting <code>2024-09-19</code> represent out-of-sample forward projections immediately following the dataset cutoff.<br>
    • <strong>Current Date Limitation:</strong> Because genuine electricity load observations end on September 18, 2024, generating a live forecast for today would require bridging a multi-year gap without telemetry. To maintain scientific integrity, predictions are explicitly grounded in the verified post-cutoff horizon.
</div>
""", unsafe_allow_html=True)


# ==========================================
# SECTION 1: DATA OVERVIEW & AUDIT
# ==========================================
if view_selection == "1. Data Overview & Audit":
    st.subheader("1. Data Overview & Comprehensive Quality Audit")
    st.caption("Verification of target electricity demand, location metadata, empirical frequency, and sensor health.")

    # High-level Metrics Cards (Clearly distinguishing Cleaned vs Raw statistics)
    raw_stats = quality_report.get("raw_target_summary", {})
    clean_min = clean_meta["clean_min_demand"]
    clean_mean = clean_meta["clean_mean_demand"]
    clean_max = clean_meta["clean_max_demand"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Cleaned Average Demand",
        f"{clean_mean:,.0f} MW",
        delta=f"Raw: {raw_stats.get('mean', 0):,.0f} MW (unbiased)"
    )
    c2.metric(
        "Peak Observed Demand",
        f"{clean_max:,.0f} MW",
        delta="Genuine physical peak preserved"
    )
    c3.metric(
        "Cleaned Minimum Demand",
        f"{clean_min:,.0f} MW",
        delta=f"Raw was 0 MW ({quality_report.get('zero_or_negative_count')} dropouts imputed)",
        delta_color="normal"
    )
    c4.metric(
        "Usable Dataset Span",
        f"{clean_meta['clean_rows']:,} days",
        delta=f"Daily resolution ({quality_report['total_rows']} raw + {clean_meta['gaps_filled_count']} aligned)"
    )

    # Historical Load Chart
    st.markdown("##### Historical Electricity Demand Trajectory (Delhi Grid)")
    hist_chart_df = clean_df[["DATE", "POWER_DEMAND_CLEAN"]].copy()
    hist_chart_df["DATE"] = pd.to_datetime(hist_chart_df["DATE"])

    hist_chart = alt.Chart(hist_chart_df).mark_line(color="#0284c7", strokeWidth=1.5).encode(
        x=alt.X("DATE:T", title="Date"),
        y=alt.Y("POWER_DEMAND_CLEAN:Q", title="Demand (Megawatts - MW)"),
        tooltip=[
            alt.Tooltip("DATE:T", title="Date"),
            alt.Tooltip("POWER_DEMAND_CLEAN:Q", title="Demand (MW)", format=",.0f")
        ]
    ).properties(height=340).interactive()
    st.altair_chart(hist_chart, use_container_width=True)

    # Transparent Raw vs. Cleaned Dataset Comparison Table
    st.markdown("##### 🔬 Raw vs. Cleaned Dataset Quality Comparison")
    comparison_table_data = [
        {
            "Metric / Attribute": "Total Calendar Records",
            "Raw Dataset (df_final.csv)": f"{quality_report['total_rows']} rows",
            "Cleaned & Aligned Dataset": f"{clean_meta['clean_rows']} rows",
            "Data Quality Action & Integrity Rationale": "Reindexed 2 missing dates (2022-09-15 & 2022-09-16) to establish unbroken daily frequency."
        },
        {
            "Metric / Attribute": "Minimum Electricity Demand",
            "Raw Dataset (df_final.csv)": f"{raw_stats.get('min', 0.0):.2f} MW (Implausible 0 MW)",
            "Cleaned & Aligned Dataset": f"{clean_min:.2f} MW (Physical minimum)",
            "Data Quality Action & Integrity Rationale": "13 sensor dropouts (0 MW) safely imputed via linear interpolation. Genuine minimum is 2,444 MW."
        },
        {
            "Metric / Attribute": "Lowest Non-Zero Raw Load",
            "Raw Dataset (df_final.csv)": f"{raw_stats.get('min_nonzero', clean_min):.2f} MW (2023-01-29)",
            "Cleaned & Aligned Dataset": f"{clean_min:.2f} MW (2023-01-29)",
            "Data Quality Action & Integrity Rationale": "Preserved 100% untouched; confirms lowest true regional load is 2,444 MW."
        },
        {
            "Metric / Attribute": "Mean Electricity Demand",
            "Raw Dataset (df_final.csv)": f"{raw_stats.get('mean', 0.0):.2f} MW",
            "Cleaned & Aligned Dataset": f"{clean_mean:.2f} MW",
            "Data Quality Action & Integrity Rationale": "Eliminates artificial downward bias caused by zero-demand sensor dropouts."
        },
        {
            "Metric / Attribute": "Peak Electricity Demand",
            "Raw Dataset (df_final.csv)": f"{raw_stats.get('max', 0.0):.2f} MW (2024-06-18)",
            "Cleaned & Aligned Dataset": f"{clean_max:.2f} MW (2024-06-18)",
            "Data Quality Action & Integrity Rationale": "Preserved 100% untouched; captures Delhi's record summer 2024 heatwave peak load."
        },
        {
            "Metric / Attribute": "Zero Demand Dropout Count",
            "Raw Dataset (df_final.csv)": f"{quality_report.get('zero_or_negative_count')} records (0.0 MW)",
            "Cleaned & Aligned Dataset": "0 records",
            "Data Quality Action & Integrity Rationale": "Resolved without modifying or overwriting the original raw df_final.csv."
        },
    ]
    st.dataframe(pd.DataFrame(comparison_table_data), use_container_width=True)

    # Detailed Audit Breakdown
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("##### 🔍 Data Integrity Audit")
        st.markdown(f"""
        - **Total Raw Records:** `{quality_report['total_rows']}` rows across `{quality_report['total_columns']}` attributes.
        - **Target Column:** `{quality_report['target_col']}` (Unit: Megawatts - MW).
        - **Temporal Resolution:** `{quality_report.get('frequency_label')}` (Median delta: `{quality_report.get('frequency_desc')}`).
        - **`DATE` vs `datetime` Verification:** Exact duplicates (`Identical = {quality_report.get('date_datetime_identical')}`).
        - **Missing Calendar Gaps:** Exactly 2 missing dates identified (`{', '.join(quality_report.get('missing_calendar_dates', []))}`) and aligned.
        - **Raw CSV Preservation:** `df_final.csv` is preserved completely untouched.
        """)

    with col_a2:
        st.markdown("##### 📍 Geographic & Weather Alignment")
        st.markdown(f"""
        - **Location:** `{quality_report.get('locations')}`
        - **Weather Stations:** Station VIDP (Indira Gandhi International Airport, New Delhi) and IMD stations.
        - **Alignment Status:** Demand records and meteorological measurements genuinely correspond to Delhi's regional grid.
        - **Available Weather Fields:** `{len(quality_report.get('available_weather_cols', []))}` candidate fields.
        - **Primary Modeled Weather:** `temp`, `humidity`, `precip`, `windspeed`, `solarradiation`.
        """)

    with st.expander("📄 View Raw Dataset Sample"):
        st.dataframe(raw_df.head(15), use_container_width=True)


# ==========================================
# SECTION 2: MODEL EVALUATION & BENCHMARK
# ==========================================
elif view_selection == "2. Model Evaluation & Benchmark":
    st.subheader("2. Model Evaluation on Unseen Test Period")
    st.caption("Strict chronological 80/20 train-test evaluation comparing Baselines, Linear Regression, Random Forest, and Gradient Boosting.")

    # Chronological Split Banner
    st.markdown(f"""
    <div class="badge-info">
        ⏱️ <strong>Strict Chronological Split (No Lookahead Leakage):</strong><br>
        • <strong>Training Window:</strong> {split_meta['train_rows']} days ({split_meta['train_start']} to {split_meta['train_end']})<br>
        • <strong>Testing Window:</strong> {split_meta['test_rows']} days ({split_meta['test_start']} to {split_meta['test_end']})<br>
        All lag features and Fourier seasonal harmonics are computed strictly from historical observations.
    </div>
    """, unsafe_allow_html=True)

    # Key Comparative Metric Cards
    lr_w_row = df_metrics.loc[df_metrics["Model"] == "Linear Regression (Weather-Aware)"].iloc[0]
    lr_now_row = df_metrics.loc[df_metrics["Model"] == "Linear Regression (Demand + Calendar)"].iloc[0]
    base_row = df_metrics.loc[df_metrics["Model"] == "Baseline (Persistence / Lag 1)"].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "LR (Weather-Aware) MAE",
        f"{lr_w_row['MAE (MW)']} MW",
        delta=f"{round(lr_now_row['MAE (MW)'] - lr_w_row['MAE (MW)'], 2)} MW improvement with weather"
    )
    m2.metric(
        "LR (Weather-Aware) RMSE",
        f"{lr_w_row['RMSE (MW)']} MW",
        delta=f"{round(lr_now_row['RMSE (MW)'] - lr_w_row['RMSE (MW)'], 2)} MW improvement"
    )
    m3.metric(
        "LR (Weather-Aware) R²",
        f"{lr_w_row['R2 Score']}",
        delta=f"+{round(lr_w_row['R2 Score'] - lr_now_row['R2 Score'], 4)} R² increase"
    )
    m4.metric(
        "Persistence Baseline MAE",
        f"{base_row['MAE (MW)']} MW",
        delta=f"{round(base_row['MAE (MW)'] - lr_w_row['MAE (MW)'], 2)} MW worse than LR",
        delta_color="inverse"
    )

    # Complete Model Benchmark Table
    st.markdown("##### 📊 Complete Model Performance Comparison")
    st.dataframe(df_metrics, use_container_width=True)

    # Actual vs Predicted Graph
    st.markdown("##### Actual vs. Predicted Demand (Test Period Overlay)")
    test_plot_df = pd.melt(
        df_predictions,
        id_vars=["DATE"],
        value_vars=["ACTUAL_DEMAND", "PRED_LR_WEATHER", "PRED_LR_NO_WEATHER"],
        var_name="Model_Type",
        value_name="Demand_MW"
    )
    test_plot_df["DATE"] = pd.to_datetime(test_plot_df["DATE"])

    label_rename = {
        "ACTUAL_DEMAND": "Actual Demand",
        "PRED_LR_WEATHER": "Linear Regression (Weather-Aware)",
        "PRED_LR_NO_WEATHER": "Linear Regression (No Weather)"
    }
    test_plot_df["Model_Type"] = test_plot_df["Model_Type"].map(label_rename)

    eval_chart = alt.Chart(test_plot_df).mark_line(strokeWidth=1.8).encode(
        x=alt.X("DATE:T", title="Test Date"),
        y=alt.Y("Demand_MW:Q", title="Demand (MW)"),
        color=alt.Color(
            "Model_Type:N",
            scale=alt.Scale(
                domain=["Actual Demand", "Linear Regression (Weather-Aware)", "Linear Regression (No Weather)"],
                range=["#0f172a", "#2563eb", "#f59e0b"]
            ),
            title="Model Series"
        ),
        tooltip=[
            alt.Tooltip("DATE:T", title="Date"),
            alt.Tooltip("Model_Type:N", title="Type"),
            alt.Tooltip("Demand_MW:Q", title="Demand (MW)", format=",.1f")
        ]
    ).properties(height=360).interactive()
    st.altair_chart(eval_chart, use_container_width=True)

    # Residuals & Explainability
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.markdown("##### Residual Error Analysis (Actual − Predicted)")
        res_df = pd.DataFrame({
            "DATE": pd.to_datetime(df_predictions["DATE"]),
            "Residual_MW": df_predictions["RESIDUAL_LR_WEATHER"]
        })
        res_chart = alt.Chart(res_df).mark_bar(color="#0284c7").encode(
            x=alt.X("DATE:T", title="Test Date"),
            y=alt.Y("Residual_MW:Q", title="Residual (MW)"),
            tooltip=[alt.Tooltip("DATE:T"), alt.Tooltip("Residual_MW:Q", format=",.1f")]
        ).properties(height=280).interactive()
        st.altair_chart(res_chart, use_container_width=True)

    with c_res2:
        st.markdown("##### Explainable Feature Weights (Physical Impact)")
        st.dataframe(coef_df[["Feature", "Coefficient (MW)"]], use_container_width=True, height=280)

    # Download Button
    csv_eval = df_predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Test Predictions & Residuals (CSV)",
        data=csv_eval,
        file_name="smart_grid_test_predictions.csv",
        mime="text/csv"
    )


# ==========================================
# SECTION 3: POST-DATASET OUT-OF-SAMPLE PROJECTION
# ==========================================
elif view_selection == "3. Post-Dataset Out-of-Sample Projection (Sept 2024)":
    st.subheader("3. Post-Dataset Out-of-Sample Projection (September 2024)")
    st.caption("Recursive multi-step load projection originating immediately after the historical cutoff date (2024-09-18).")

    last_observed_date = pd.to_datetime(clean_df["DATE"].iloc[-1]).date()

    # Clear Operational Scope Banner
    st.markdown(f"""
    <div class="badge-historical">
        📅 <strong>Forecast Origin & Temporal Context:</strong><br>
        • <strong>Historical Dataset Cutoff:</strong> <code>{last_observed_date}</code> (Red dashed boundary marker).<br>
        • <strong>Projection Horizon:</strong> <code>{last_observed_date + pd.Timedelta(days=1)}</code> onward.<br>
        • <strong>Why Predictions Begin in September 2024 (Not Today):</strong> This auto-regressive model requires observed previous-day demand (<code>lag_1</code>) and same-day last-week demand (<code>lag_7</code>). Because grid observations end on September 18, 2024, actual load for recent days is unobserved. Recursively predicting forward across an unobserved multi-year gap would accumulate compounding errors.<br>
        • <strong>Live Forecasting Requirement:</strong> Live forecasting for today requires real-time telemetry from Delhi SLDC.
    </div>
    """, unsafe_allow_html=True)

    c_fc1, c_fc2 = st.columns([1, 2])
    with c_fc1:
        horizon_days = st.slider(
            "Forecast Horizon (Days):",
            min_value=1,
            max_value=30,
            value=config.DEFAULT_FORECAST_HORIZON,
            step=1
        )
        forecast_start_date = last_observed_date + pd.Timedelta(days=1)
        forecast_end_date = last_observed_date + pd.Timedelta(days=horizon_days)

        st.markdown(f"""
        - **Historical Cutoff:** `{last_observed_date}` (Red boundary line)
        - **Projection Start:** `{forecast_start_date}`
        - **Projection End:** `{forecast_end_date}`
        - **Confidence Level:** 95% ($\\pm 1.96 \\cdot \\text{{RMSE}} \\cdot \\sqrt{{h}}$)
        """)

    # Generate baseline future forecast
    df_future, future_meta = generate_future_forecast(
        model=primary_model,
        feature_cols=feat_w,
        historical_df=clean_df,
        horizon=horizon_days,
        test_rmse=primary_test_rmse,
    )

    # Plot historical tail + future forecast
    recent_hist = clean_df.tail(30)[["DATE", "POWER_DEMAND_CLEAN"]].copy()
    recent_hist["DATE"] = pd.to_datetime(recent_hist["DATE"]).dt.strftime("%Y-%m-%d")
    recent_hist["Type"] = "Historical Observed"
    recent_hist = recent_hist.rename(columns={"POWER_DEMAND_CLEAN": "Demand_MW"})

    future_plot = df_future[["DATE", "FORECASTED_DEMAND_MW"]].copy()
    future_plot["Type"] = "Post-Cutoff Projection"
    future_plot = future_plot.rename(columns={"FORECASTED_DEMAND_MW": "Demand_MW"})

    combined_chart_data = pd.concat([recent_hist, future_plot], ignore_index=True)
    combined_chart_data["DATE"] = pd.to_datetime(combined_chart_data["DATE"])

    boundary_df = pd.DataFrame([{"DATE": pd.to_datetime(last_observed_date), "Label": "Cutoff Boundary"}])

    base_lines = alt.Chart(combined_chart_data).mark_line(strokeWidth=2.2, point=True).encode(
        x=alt.X("DATE:T", title="Date"),
        y=alt.Y("Demand_MW:Q", title="Electricity Demand (MW)"),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=["Historical Observed", "Post-Cutoff Projection"], range=["#475569", "#2563eb"]),
            title="Series Type"
        ),
        tooltip=[
            alt.Tooltip("DATE:T", title="Date"),
            alt.Tooltip("Type:N", title="Type"),
            alt.Tooltip("Demand_MW:Q", title="Demand (MW)", format=",.1f")
        ]
    )

    cutoff_rule = alt.Chart(boundary_df).mark_rule(
        color="#ef4444",
        strokeDash=[4, 4],
        strokeWidth=2.0
    ).encode(x="DATE:T")

    cutoff_text = alt.Chart(boundary_df).mark_text(
        text="← Observed | Post-Cutoff Projection →",
        align="left",
        dx=6,
        dy=-140,
        color="#ef4444",
        fontSize=11,
        fontWeight="bold"
    ).encode(x="DATE:T")

    st.altair_chart((base_lines + cutoff_rule + cutoff_text).properties(height=380).interactive(), use_container_width=True)

    st.markdown("##### Projected Demand Table (Post-Cutoff September 2024 Horizon)")
    st.dataframe(df_future, use_container_width=True)

    csv_fut = df_future.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Post-Cutoff Forecast (CSV)",
        data=csv_fut,
        file_name="smart_grid_post_cutoff_forecast.csv",
        mime="text/csv"
    )


# ==========================================
# SECTION 4: WEATHER SCENARIO SIMULATION
# ==========================================
elif view_selection == "4. Weather Scenario Simulation (Sept 2024 Horizon)":
    st.subheader("4. Interactive Weather Scenario Simulation (September 2024 Horizon)")
    st.caption("Evaluate how modified meteorological conditions perturb out-of-sample demand via genuine model invocation.")

    st.markdown("""
    <div class="badge-info">
        🌡️ <strong>Genuine Model Execution:</strong><br>
        All scenario predictions invoke the actual trained Linear Regression (Weather-Aware) model weights.<br>
        <em>Scenarios reflect statistical estimates conditional on the specified meteorological inputs over the 7-day post-cutoff horizon (2024-09-19 to 2024-09-25).</em>
    </div>
    """, unsafe_allow_html=True)

    # View Mode Toggle: Single Scenario vs All 3 Compared
    view_mode = st.radio(
        "Display Mode:",
        ["Compare Individual Scenario against Baseline", "View All 3 Standard Scenarios Simultaneously (Baseline vs Heatwave vs Cool Front)"],
        horizontal=True
    )

    # Precompute Baseline 7-Day Forecast
    df_base_fc, _ = generate_future_forecast(
        model=primary_model,
        feature_cols=feat_w,
        historical_df=clean_df,
        horizon=config.DEFAULT_FORECAST_HORIZON,
        test_rmse=primary_test_rmse,
        weather_scenario=baseline_weather
    )

    if view_mode == "View All 3 Standard Scenarios Simultaneously (Baseline vs Heatwave vs Cool Front)":
        st.markdown("##### 📊 Multi-Scenario Overview (7-Day Post-Cutoff Horizon)")
        df_all_scenarios, _ = simulate_scenario_comparison(
            model=primary_model,
            feature_cols=feat_w,
            historical_df=clean_df,
            horizon=config.DEFAULT_FORECAST_HORIZON,
            test_rmse=primary_test_rmse,
        )

        # Plot all 3 curves together
        plot_rows = []
        for _, row in df_all_scenarios.iterrows():
            plot_rows.append({"DATE": row["DATE"], "Scenario": "Baseline (Trailing Weather)", "Demand_MW": row["BASELINE_DEMAND_MW"]})
            plot_rows.append({"DATE": row["DATE"], "Scenario": "Heatwave (+4°C, High Solar)", "Demand_MW": row["HOTTER_SCENARIO_MW"]})
            plot_rows.append({"DATE": row["DATE"], "Scenario": "Cool Front (-4°C, Rain)", "Demand_MW": row["COOLER_SCENARIO_MW"]})
        
        plot_multi_df = pd.DataFrame(plot_rows)
        plot_multi_df["DATE"] = pd.to_datetime(plot_multi_df["DATE"])

        multi_chart = alt.Chart(plot_multi_df).mark_line(strokeWidth=2.2, point=True).encode(
            x=alt.X("DATE:T", title="Forecast Date"),
            y=alt.Y("Demand_MW:Q", title="Demand (MW)"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(
                    domain=["Baseline (Trailing Weather)", "Heatwave (+4°C, High Solar)", "Cool Front (-4°C, Rain)"],
                    range=["#2563eb", "#dc2626", "#16a34a"]
                ),
                title="Scenario"
            ),
            tooltip=[alt.Tooltip("DATE:T"), alt.Tooltip("Scenario:N"), alt.Tooltip("Demand_MW:Q", format=",.1f")]
        ).properties(height=380).interactive()
        st.altair_chart(multi_chart, use_container_width=True)

        st.markdown("##### Comparative Scenario Forecast Table")
        st.dataframe(df_all_scenarios, use_container_width=True)

    else:
        # Individual Scenario Selection
        scenario_mode = st.selectbox(
            "Choose Scenario Configuration:",
            ["Baseline (Recent Weather Trajectory)", "Heatwave (+4°C, High Solar)", "Cool Front / Monsoon (-4°C, Rain)", "Custom Parameter Controls"]
        )

        active_weather = baseline_weather.copy()

        if scenario_mode == "Heatwave (+4°C, High Solar)":
            active_weather["temp"] = baseline_weather["temp"] + 4.0
            active_weather["humidity"] = max(10.0, baseline_weather["humidity"] - 5.0)
            active_weather["precip"] = 0.0
            active_weather["solarradiation"] = min(400.0, baseline_weather["solarradiation"] + 40.0)
        elif scenario_mode == "Cool Front / Monsoon (-4°C, Rain)":
            active_weather["temp"] = max(5.0, baseline_weather["temp"] - 4.0)
            active_weather["humidity"] = min(100.0, baseline_weather["humidity"] + 15.0)
            active_weather["precip"] = 25.0
            active_weather["solarradiation"] = max(10.0, baseline_weather["solarradiation"] - 50.0)
            active_weather["windspeed"] = baseline_weather["windspeed"] + 4.0
        elif scenario_mode == "Custom Parameter Controls":
            st.markdown("##### 🎛️ Adjust Scenario Inputs")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                active_weather["temp"] = st.slider("Mean Temperature (°C)", 10.0, 48.0, float(baseline_weather["temp"]), 0.5)
                active_weather["humidity"] = st.slider("Relative Humidity (%)", 10.0, 100.0, float(baseline_weather["humidity"]), 1.0)
            with sc2:
                active_weather["precip"] = st.slider("Rainfall / Precip (mm)", 0.0, 80.0, float(baseline_weather["precip"]), 1.0)
                active_weather["windspeed"] = st.slider("Wind Speed (km/h)", 0.0, 50.0, float(baseline_weather["windspeed"]), 0.5)
            with sc3:
                active_weather["solarradiation"] = st.slider("Solar Radiation (W/m²)", 10.0, 350.0, float(baseline_weather["solarradiation"]), 5.0)

        # Generate Scenario Forecast
        df_scen_fc, _ = generate_future_forecast(
            model=primary_model,
            feature_cols=feat_w,
            historical_df=clean_df,
            horizon=config.DEFAULT_FORECAST_HORIZON,
            test_rmse=primary_test_rmse,
            weather_scenario=active_weather
        )

        # Display Scenario Input Verification Table
        st.markdown("##### 🔍 Scenario Meteorological Inputs (Passed to Model)")
        inputs_table = [
            {
                "Weather Factor": "Mean Temperature",
                "Baseline Value": f"{baseline_weather['temp']:.1f} °C",
                "Scenario Value": f"{active_weather['temp']:.1f} °C",
                "Difference (Δ)": f"{active_weather['temp'] - baseline_weather['temp']:+.1f} °C",
                "Physical Effect on Grid": "Positive load response (+7.53 MW/°C cooling load)"
            },
            {
                "Weather Factor": "Relative Humidity",
                "Baseline Value": f"{baseline_weather['humidity']:.1f} %",
                "Scenario Value": f"{active_weather['humidity']:.1f} %",
                "Difference (Δ)": f"{active_weather['humidity'] - baseline_weather['humidity']:+.1f} %",
                "Physical Effect on Grid": "Monsoon cloud cover / latent cooling effect (-0.38 MW/%)"
            },
            {
                "Weather Factor": "Rainfall / Precipitation",
                "Baseline Value": f"{baseline_weather['precip']:.1f} mm",
                "Scenario Value": f"{active_weather['precip']:.1f} mm",
                "Difference (Δ)": f"{active_weather['precip'] - baseline_weather['precip']:+.1f} mm",
                "Physical Effect on Grid": "Rain drops ambient temperature and suppresses load (-2.65 MW/mm)"
            },
            {
                "Weather Factor": "Wind Speed",
                "Baseline Value": f"{baseline_weather['windspeed']:.1f} km/h",
                "Scenario Value": f"{active_weather['windspeed']:.1f} km/h",
                "Difference (Δ)": f"{active_weather['windspeed'] - baseline_weather['windspeed']:+.1f} km/h",
                "Physical Effect on Grid": "Natural breeze cooling ventilation (-8.73 MW/km/h)"
            },
            {
                "Weather Factor": "Solar Radiation",
                "Baseline Value": f"{baseline_weather['solarradiation']:.1f} W/m²",
                "Scenario Value": f"{active_weather['solarradiation']:.1f} W/m²",
                "Difference (Δ)": f"{active_weather['solarradiation'] - baseline_weather['solarradiation']:+.1f} W/m²",
                "Physical Effect on Grid": "Direct solar heat gain (+1.06 MW per W/m²)"
            },
        ]
        st.dataframe(pd.DataFrame(inputs_table), use_container_width=True)

        # Scenario Summary Metrics
        avg_base_load = df_base_fc["FORECASTED_DEMAND_MW"].mean()
        avg_scen_load = df_scen_fc["FORECASTED_DEMAND_MW"].mean()
        delta_avg_load = avg_scen_load - avg_base_load
        pct_change = (delta_avg_load / avg_base_load) * 100 if avg_base_load > 0 else 0.0

        col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
        col_sm1.metric(
            "Simulated Temperature",
            f"{active_weather['temp']:.1f} °C",
            delta=f"{active_weather['temp'] - baseline_weather['temp']:+.1f} °C vs baseline"
        )
        col_sm2.metric("Projected 7-Day Avg Demand", f"{avg_scen_load:,.0f} MW")
        col_sm3.metric(
            "Difference from Baseline (Δ)",
            f"{delta_avg_load:+,.1f} MW",
            delta=f"{pct_change:+.2f}%" if abs(delta_avg_load) > 0.01 else "Reference Baseline"
        )
        col_sm4.metric("Simulated Rainfall", f"{active_weather['precip']:.1f} mm")

        # Visual Chart
        st.markdown(f"##### 📈 Scenario Forecast Comparison: {scenario_mode}")
        if scenario_mode == "Baseline (Recent Weather Trajectory)":
            st.info("✅ **Baseline Mode Active:** Displaying unperturbed reference demand curve under trailing weather ($26.9^\\circ\\text{C}$, $13.0\\text{ mm}$ rain). No delta applied.")
            plot_base_only = df_base_fc[["DATE", "FORECASTED_DEMAND_MW"]].copy()
            plot_base_only["DATE"] = pd.to_datetime(plot_base_only["DATE"])
            chart_base = alt.Chart(plot_base_only).mark_line(color="#2563eb", strokeWidth=2.2, point=True).encode(
                x=alt.X("DATE:T", title="Forecast Date"),
                y=alt.Y("FORECASTED_DEMAND_MW:Q", title="Demand (MW)"),
                tooltip=[alt.Tooltip("DATE:T"), alt.Tooltip("FORECASTED_DEMAND_MW:Q", format=",.1f")]
            ).properties(height=360).interactive()
            st.altair_chart(chart_base, use_container_width=True)
        else:
            plot_df_base = df_base_fc[["DATE", "FORECASTED_DEMAND_MW"]].copy()
            plot_df_base["Scenario"] = "Baseline (Trailing Weather)"

            plot_df_scen = df_scen_fc[["DATE", "FORECASTED_DEMAND_MW"]].copy()
            plot_df_scen["Scenario"] = f"Scenario ({scenario_mode})"

            comp_plot_data = pd.concat([plot_df_base, plot_df_scen], ignore_index=True)
            comp_plot_data["DATE"] = pd.to_datetime(comp_plot_data["DATE"])

            scen_color = "#dc2626" if delta_avg_load > 0 else "#16a34a"
            scen_chart = alt.Chart(comp_plot_data).mark_line(strokeWidth=2.2, point=True).encode(
                x=alt.X("DATE:T", title="Forecast Date"),
                y=alt.Y("FORECASTED_DEMAND_MW:Q", title="Demand (MW)"),
                color=alt.Color(
                    "Scenario:N",
                    scale=alt.Scale(
                        domain=["Baseline (Trailing Weather)", f"Scenario ({scenario_mode})"],
                        range=["#2563eb", scen_color]
                    ),
                    title="Series"
                ),
                tooltip=[alt.Tooltip("DATE:T"), alt.Tooltip("Scenario:N"), alt.Tooltip("FORECASTED_DEMAND_MW:Q", format=",.1f")]
            ).properties(height=360).interactive()
            st.altair_chart(scen_chart, use_container_width=True)

        # Detailed Table of Day-by-Day Delta
        comp_table = []
        for i in range(len(df_base_fc)):
            d_str = df_base_fc.loc[i, "DATE"]
            dow_str = df_base_fc.loc[i, "DAY_OF_WEEK"]
            b_val = df_base_fc.loc[i, "FORECASTED_DEMAND_MW"]
            s_val = df_scen_fc.loc[i, "FORECASTED_DEMAND_MW"]
            comp_table.append({
                "DATE": d_str,
                "DAY_OF_WEEK": dow_str,
                "BASELINE_DEMAND_MW": b_val,
                "SCENARIO_DEMAND_MW": s_val,
                "DELTA_DEMAND_MW": round(s_val - b_val, 2),
                "PERCENT_CHANGE": round(((s_val - b_val) / b_val) * 100, 2) if b_val > 0 else 0.0,
            })
        st.markdown("##### Day-by-Day Scenario Comparison Table")
        st.dataframe(pd.DataFrame(comp_table), use_container_width=True)
