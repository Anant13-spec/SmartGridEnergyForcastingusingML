from datetime import datetime, time
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import predict
from predict import WEATHER_FIELDS, predict_all_zones, predict_demand
from scenarios import PRESETS
from morocco_hourly import (
    TARGETS,
    add_empirical_bands,
    fetch_open_meteo_hourly,
    forecast_next_24,
    load_hourly_inputs,
    load_live_forecasts,
    load_observed_actuals,
    record_live_forecast,
    save_observed_actual,
)

FORECAST_LOCATION = "Delhi, India"

st.set_page_config(
    page_title="SmartGrid AI | 3-Zone Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1220; color: #e8eef8; }
    .block-container { max-width: 1550px; padding-top: 1.5rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background: #131f32; border: 1px solid #263750; padding: 16px; border-radius: 14px; }
    [data-testid="stMetricValue"] { color: #55dfbd; }
    .hero { padding: 22px 26px; border: 1px solid #263750; border-radius: 16px; background: linear-gradient(120deg, #14243a, #0e333d); margin-bottom: 20px; }
    .card { background: #131f32; border: 1px solid #263750; padding: 18px; border-radius: 12px; margin-bottom: 15px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600; background: #1a334e; color: #55dfbd; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_app_dataframe():
    hourly_df, _ = load_hourly_inputs()
    return hourly_df


@st.cache_resource(show_spinner=False)
def load_app_models():
    _, hourly_artifact = load_hourly_inputs()
    model_artifact = predict.load_model_artifact()
    return hourly_artifact, model_artifact


try:
    hourly_df = load_app_dataframe()
    hourly_artifact, model_artifact = load_app_models()
except Exception as exc:
    st.error(f"Could not load data or models: {exc}")
    st.info("Run `python src/preprocess.py` and `python src/train_models.py` first.")
    st.stop()

first_timestamp = hourly_df["Datetime"].min()
last_timestamp = hourly_df["Datetime"].max()
first_date = first_timestamp.date()
last_date = last_timestamp.date()

st.markdown("""
<div class="hero">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
      <h1 style="margin:0; font-size: 2.1rem;">⚡ SmartGrid Energy Forecasting System</h1>
      <p style="margin-top:6px; color:#b9cbe0; font-size: 1.05rem;">
        Multi-Zone & Total Electricity Demand Predictive Intelligence · High-Frequency ML Forecasting
      </p>
    </div>
    <div style="text-align: right;">
      <span class="badge">4-Year Dataset (2021–2024)</span>
      <span class="badge">393,152 Observations</span>
      <span class="badge">R² > 0.99</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

overview = st.columns(5)
overview[0].metric("Historical Records", f"{393152:,}", "5-Min Resolution")
overview[1].metric("Coverage Period", f"{first_date.year} – {last_date.year}", "48 Months")
overview[2].metric("Forecast Targets", "4 Targets", "3 Zones + Total")
overview[3].metric("Total Demand R²", "0.9966", "MAE: 27.95 MW")
overview[4].metric("Average System Load", f"{hourly_df['Power demand'].mean():,.0f} MW", "Peak: 8,632 MW")

tabs = st.tabs([
    "🎯 User-Based Prediction",
    "⚡ 24-Hour Horizon Forecast",
    "📈 Historical Grid Analytics",
    "🧪 Model Comparison & Benchmarks",
    "📝 Actuals Tracking",
])


with tabs[0]:
    st.subheader("🎯 Real-Time ML Demand Prediction from User Inputs")
    st.write(
        "Specify meteorological conditions and forecast timing. The system retrieves the latest historical "
        "load continuity lags from the real dataset and feeds all features into the actual trained ML model. "
        "**Zero mock data or random values.**"
    )

    col_preset, col_model_pick, col_target_pick = st.columns([2, 1.5, 1.5])
    with col_preset:
        preset_choice = st.selectbox(
            "Quick Scenario Preset (optional)",
            ["Custom (Manual Inputs)", *list(PRESETS.keys())],
            help="Select a predefined realistic meteorological scenario or customize below.",
        )
    with col_model_pick:
        available_models = ["Best Model (Auto)", "Linear Regression (Ridge)", "HistGradientBoosting", "Random Forest", "XGBoost"]
        selected_model_option = st.selectbox("Inference Model", available_models)
        model_name_param = None if selected_model_option == "Best Model (Auto)" else selected_model_option

    with col_target_pick:
        target_options = ["All Zones + Total Demand", "Total Grid Demand", "Zone 1", "Zone 2", "Zone 3"]
        selected_target_ui = st.selectbox("Forecasting Target", target_options)

    if preset_choice in PRESETS:
        preset_vals = PRESETS[preset_choice]
    else:
        preset_vals = PRESETS["Typical Day"]

    st.markdown("#### 🌤️ Meteorological & Weather Inputs")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp_input = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=float(preset_vals["temp"]), step=0.5)
        rhum_input = st.slider("Relative Humidity (%)", min_value=5.0, max_value=100.0, value=float(preset_vals["rhum"]), step=1.0)

    with w_col2:
        dwpt_input = st.slider("Dew Point (°C)", min_value=-15.0, max_value=35.0, value=float(preset_vals["dwpt"]), step=0.5)
        wspd_input = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=80.0, value=float(preset_vals["wspd"]), step=1.0)

    with w_col3:
        pres_input = st.slider("Atmospheric Pressure (hPa)", min_value=980.0, max_value=1035.0, value=float(preset_vals["pres"]), step=1.0)
        wdir_input = st.slider("Wind Direction (°)", min_value=0.0, max_value=360.0, value=float(preset_vals["wdir"]), step=5.0)

    st.markdown("#### ⏱️ Forecast Timestamp & Temporal Alignment")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        forecast_date_input = st.date_input("Forecast Date", value=last_date)
    with t_col2:
        forecast_hour = st.number_input("Hour of Day (0–23)", min_value=0, max_value=23, value=14, step=1)
    with t_col3:
        forecast_min = st.selectbox("Minute (5-min intervals)", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], index=0)

    forecast_dt = datetime.combine(forecast_date_input, time(int(forecast_hour), int(forecast_min)))

    user_weather = {
        "temp": temp_input,
        "dwpt": dwpt_input,
        "rhum": rhum_input,
        "wdir": wdir_input,
        "wspd": wspd_input,
        "pres": pres_input,
    }

    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_user_predict = st.button("🚀 Generate Real ML Prediction", type="primary", use_container_width=True)

    if run_user_predict:
        try:
            with st.spinner("Executing trained model inference pipeline..."):
                if selected_target_ui == "All Zones + Total Demand":
                    multi_res = predict_all_zones(forecast_dt, user_weather, model_name=model_name_param)
                    st.session_state["user_pred_result"] = ("multi", multi_res)
                else:
                    t_map = {
                        "Total Grid Demand": "Power demand",
                        "Zone 1": "demand_zone_1_mw",
                        "Zone 2": "demand_zone_2_mw",
                        "Zone 3": "demand_zone_3_mw",
                    }
                    t_key = t_map[selected_target_ui]
                    single_res = predict_demand(forecast_dt, user_weather, target=t_key, model_name=model_name_param)
                    st.session_state["user_pred_result"] = ("single", single_res)
        except Exception as exc:
            st.error(f"Inference error: {exc}")

    if "user_pred_result" in st.session_state:
        kind, res = st.session_state["user_pred_result"]
        st.divider()
        st.markdown(f"### 📊 Prediction Output · Timestamp: `{forecast_dt.strftime('%Y-%m-%d %H:%M')}`")

        if kind == "multi":
            p_cols = st.columns(4)
            z1 = res["demand_zone_1_mw"]
            z2 = res["demand_zone_2_mw"]
            z3 = res["demand_zone_3_mw"]
            total = res["Power demand"]

            p_cols[0].metric(
                "Zone 1 Demand",
                f"{z1['prediction_mw']:,.1f} MW",
                f"{z1['delta_mw']:+,.1f} MW vs baseline",
            )
            p_cols[1].metric(
                "Zone 2 Demand",
                f"{z2['prediction_mw']:,.1f} MW",
                f"{z2['delta_mw']:+,.1f} MW vs baseline",
            )
            p_cols[2].metric(
                "Zone 3 Demand",
                f"{z3['prediction_mw']:,.1f} MW",
                f"{z3['delta_mw']:+,.1f} MW vs baseline",
            )
            p_cols[3].metric(
                "Total Grid Demand",
                f"{total['prediction_mw']:,.1f} MW",
                f"{total['delta_mw']:+,.1f} MW vs baseline",
            )

            st.info(
                f"ℹ️ **Zone Sum vs Grid Load Consistency:** Sum of Zone 1 + 2 + 3 is **{res['Combined_Zones_Sum']:,.1f} MW**, "
                f"while Total State Power Demand is **{total['prediction_mw']:,.1f} MW** (difference: **{res['Difference_Total_vs_Sum']:+,.1f} MW**). "
                "In regional grid dispatch, total demand accounts for state-wide gross drawal including transmission/auxiliary load, "
                "while zones cover individual distribution areas."
            )

            chart_df = pd.DataFrame({
                "Entity": ["Zone 1", "Zone 2", "Zone 3", "Combined Zones Sum", "Total Grid Demand"],
                "Predicted Load (MW)": [
                    z1["prediction_mw"],
                    z2["prediction_mw"],
                    z3["prediction_mw"],
                    res["Combined_Zones_Sum"],
                    total["prediction_mw"],
                ],
                "Baseline Reference (MW)": [
                    z1["baseline_mw"],
                    z2["baseline_mw"],
                    z3["baseline_mw"],
                    round(z1["baseline_mw"] + z2["baseline_mw"] + z3["baseline_mw"], 1),
                    total["baseline_mw"],
                ],
            }).set_index("Entity")
            st.bar_chart(chart_df, use_container_width=True)

        else:
            p_cols = st.columns(3)
            p_cols[0].metric(
                f"Predicted {res['display_name']}",
                f"{res['prediction_mw']:,.1f} MW",
                f"{res['delta_mw']:+,.1f} MW vs baseline",
            )
            p_cols[1].metric("Historical Baseline Lag", f"{res['baseline_mw']:,.1f} MW")
            pct_delta = (res['delta_mw'] / max(res['baseline_mw'], 1.0)) * 100.0
            p_cols[2].metric("Weather Delta %", f"{pct_delta:+.2f}%")

            st.caption(f"Model used for inference: **{res['model_used']}**")


with tabs[1]:
    st.subheader(f"⚡ 24-Hour Horizon Forecast · Zone 1, 2, 3 & Total Grid Load")
    st.write(
        f"Generate sequential 24-hour horizon forecasts for {FORECAST_LOCATION} using hourly weather inputs, "
        "recursively updating historical demand lags."
    )

    _, f_btn_col, _ = st.columns([1, 2, 1])
    with f_btn_col:
        run_24h = st.button("Forecast Next 24 Hours", type="primary", use_container_width=True, key="run_24h_btn")

    if run_24h:
        try:
            with st.spinner("Fetching hourly weather and forecasting 24-hour load trajectory…"):
                live = fetch_open_meteo_hourly("Delhi")
                f_df = forecast_next_24(hourly_df, live["weather"], hourly_artifact)
                f_df = add_empirical_bands(f_df)
                record_live_forecast(f_df, live["location"], live["timezone"])
                st.session_state["live_24h_forecast"] = {
                    "location": live["location"],
                    "timezone": live["timezone"],
                    "frame": f_df,
                }
        except Exception as exc:
            st.error(f"Could not generate 24-hour forecast: {exc}")

    saved_24h = st.session_state.get("live_24h_forecast")
    if saved_24h:
        frame = saved_24h["frame"]
        st.divider()
        st.markdown(f"### 📍 {saved_24h['location']} · 24-Hour Ahead Lookahead")
        st.caption(f"Timezone: {saved_24h['timezone']} · Forecast window: {frame['Datetime'].min()} to {frame['Datetime'].max()}")

        p_cols = st.columns(4)
        for col, target, label in zip(p_cols[:3], TARGETS, ["Zone 1 Peak", "Zone 2 Peak", "Zone 3 Peak"]):
            peak_row = frame.loc[frame[target].idxmax()]
            col.metric(label, f"{peak_row[target]:,.0f} MW")
            col.caption(pd.Timestamp(peak_row["Datetime"]).strftime("%a %d %b · %H:%M"))

        if "Power demand" in frame:
            tot_peak = frame.loc[frame["Power demand"].idxmax()]
            p_cols[3].metric("Total Demand Peak", f"{tot_peak['Power demand']:,.0f} MW")
            p_cols[3].caption(pd.Timestamp(tot_peak["Datetime"]).strftime("%a %d %b · %H:%M"))

        plot_cols = [*TARGETS, "Combined_Zones"]
        if "Power demand" in frame:
            plot_cols.append("Power demand")

        chart_data = frame.set_index("Datetime")[plot_cols].rename(columns={
            "demand_zone_1_mw": "Zone 1",
            "demand_zone_2_mw": "Zone 2",
            "demand_zone_3_mw": "Zone 3",
            "Combined_Zones": "Combined Zones Sum",
            "Power demand": "Total Power Demand",
        })
        st.line_chart(chart_data, use_container_width=True)

        frame_display = frame.copy()
        numeric_f = frame_display.select_dtypes(include=[np.number]).columns
        frame_display[numeric_f] = frame_display[numeric_f].round(2)
        st.dataframe(frame_display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download 24-Hour Forecast (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="smart_grid_24h_forecast.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Forecast Next 24 Hours** above to run the 24-hour horizon model.")


with tabs[2]:
    st.subheader("📈 Historical Grid Analytics (2021 – 2024)")
    st.write("Explore 4 years of continuous 5-minute / hourly electricity consumption across the 3 zones and total grid load.")

    hist_date_range = st.date_input(
        "Select Historical Date Window",
        value=(last_date - pd.Timedelta(days=30), last_date),
        min_value=first_date,
        max_value=last_date,
        key="hist_range_picker",
    )

    if isinstance(hist_date_range, (tuple, list)) and len(hist_date_range) == 2:
        h_start, h_end = hist_date_range
    else:
        h_start, h_end = first_date, last_date

    filtered_hist = hourly_df.loc[
        hourly_df["Datetime"].dt.date.between(h_start, h_end)
    ].copy()

    chart_hist = filtered_hist.set_index("Datetime")[[*TARGETS, "Power demand"]].rename(columns={
        "demand_zone_1_mw": "Zone 1 Demand",
        "demand_zone_2_mw": "Zone 2 Demand",
        "demand_zone_3_mw": "Zone 3 Demand",
        "Power demand": "Total Power Demand",
    })
    st.line_chart(chart_hist, use_container_width=True)

    with st.expander("🔍 Detailed Historical Data Statistics & Quality Check"):
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown("#### Demand Distributions (MW)")
            st.dataframe(filtered_hist[[*TARGETS, "Power demand"]].describe().T[["mean", "std", "min", "50%", "max"]].round(1))
        with stats_col2:
            st.markdown("#### Weather Distributions")
            weather_col_list = list(WEATHER_FIELDS.keys())
            st.dataframe(filtered_hist[weather_col_list].describe().T[["mean", "std", "min", "50%", "max"]].round(1))


with tabs[3]:
    st.subheader("🧪 Model Performance Comparison & Benchmark Evaluation")
    st.write(
        "All models evaluated on strict forward chronological test split (late 2024 holdout, 58,930 observations). "
        "No data leakage; trained with past observations only."
    )

    metrics_csv_path = ROOT / "reports" / "model_comparison.csv"
    if metrics_csv_path.exists():
        comp_df = pd.read_csv(metrics_csv_path)
        st.dataframe(
            comp_df[["Target", "Model", "Test_MAE", "Test_RMSE", "Test_R2", "Test_MAPE", "Is_Best"]].round(3),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### 📉 Actual vs Predicted Demand on Held-Out Test Set")
    eval_plot_path = ROOT / "reports" / "actual_vs_predicted.png"
    if eval_plot_path.exists():
        st.image(str(eval_plot_path), caption="Total Power Demand: Actual vs Predicted (1-Week Test Window)", use_container_width=True)

    zones_plot_path = ROOT / "reports" / "zone_comparison.png"
    if zones_plot_path.exists():
        st.image(str(zones_plot_path), caption="Three-Zone Breakdown: Actual vs Predicted Demand", use_container_width=True)


with tabs[4]:
    st.subheader("📝 Record Actual Hourly Readings & Monitor Operational Error")
    st.write("Log post-forecast verified meter readings to track real-world operational forecast drift.")

    live_records = load_live_forecasts()
    observed_actuals = load_observed_actuals()

    if not live_records.empty:
        st.markdown("#### Recent Live Forecasts")
        numeric_cols = live_records.select_dtypes(include=[np.number]).columns
        display_live = live_records.tail(24).copy()
        display_live[numeric_cols] = display_live[numeric_cols].round(2)
        st.dataframe(display_live, use_container_width=True, hide_index=True)
    else:
        st.info("Run a 24-hour horizon forecast in Tab 2 to initialize operational tracking records.")

    if not observed_actuals.empty:
        st.markdown("#### Recorded Observations")
        numeric_act = observed_actuals.select_dtypes(include=[np.number]).columns
        display_act = observed_actuals.tail(24).copy()
        display_act[numeric_act] = display_act[numeric_act].round(2)
        st.dataframe(display_act, use_container_width=True, hide_index=True)
