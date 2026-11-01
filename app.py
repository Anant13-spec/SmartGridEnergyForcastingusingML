from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from morocco_hourly import (  # noqa: E402
    TARGETS,
    add_empirical_bands,
    fetch_open_meteo_hourly,
    forecast_next_24,
    load_live_forecasts,
    load_observed_actuals,
    load_hourly_inputs,
    record_live_forecast,
    save_observed_actual,
)

FORECAST_LOCATION = "Tetouan, Morocco"


st.set_page_config(
    page_title="GridWise | Morocco Hourly Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1220; color: #e8eef8; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background: #131f32; border: 1px solid #263750; padding: 16px; border-radius: 14px; }
    [data-testid="stMetricValue"] { color: #55dfbd; }
    .hero { padding: 22px 24px; border: 1px solid #263750; border-radius: 16px; background: linear-gradient(120deg, #14243a, #102c32); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)


try:
    hourly_df, hourly_artifact = load_hourly_inputs()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"The Morocco hourly dataset or model is unavailable: {exc}")
    st.info(
        "Prepare the hourly data with `python src/preprocess_morocco.py`, "
        "then train the hourly models with `python src/train_morocco_hourly.py`."
    )
    st.stop()

required_columns = {"Datetime", "SourceReadings", *TARGETS}
missing_columns = required_columns - set(hourly_df.columns)
if missing_columns:
    st.error(f"The processed Morocco file is missing columns: {sorted(missing_columns)}")
    st.stop()
hourly_df["Datetime"] = pd.to_datetime(hourly_df["Datetime"], errors="coerce")
for target in TARGETS:
    hourly_df[target] = pd.to_numeric(hourly_df[target], errors="coerce")
if hourly_df[["Datetime", *TARGETS]].isna().any().any():
    st.error("The processed Morocco file has invalid timestamps or zone readings. Re-run preprocessing.")
    st.stop()
hourly_df = hourly_df.sort_values("Datetime").reset_index(drop=True)
if hourly_df["Datetime"].duplicated().any():
    st.error("The processed Morocco file has duplicate hourly timestamps. Re-run preprocessing.")
    st.stop()
if hourly_df["Datetime"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
    st.error("The processed Morocco file has gaps between hours. Re-run preprocessing.")
    st.stop()
if hourly_df.empty:
    st.error("The Morocco hourly dataset has no valid rows.")
    st.stop()

first_timestamp = hourly_df["Datetime"].min()
last_timestamp = hourly_df["Datetime"].max()
first_date = first_timestamp.date()
last_date = last_timestamp.date()


@st.cache_data(ttl=900, show_spinner=False)
def cached_hourly_weather(city: str):
    return fetch_open_meteo_hourly(city)


st.markdown("""
<div class="hero">
  <h1 style="margin:0;">⚡ GridWise</h1>
  <p style="margin-top:8px; color:#b9cbe0;">Morocco · Hourly multi-zone energy forecasting</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype: demand history covers 2017 only. Zone readings are "
    "hourly averages of the source values; the source does not confirm that "
    "these are hourly energy totals. Forecasts are experimental."
)

overview = st.columns(4)
overview[0].metric("Hourly records", f"{len(hourly_df):,}")
overview[1].metric("History starts", str(first_timestamp))
overview[2].metric("History ends", str(last_timestamp))
overview[3].metric("Forecast targets", "3 zones")

forecast_tab, history_tab, accuracy_tab, actuals_tab = st.tabs([
    "⚡ 24-hour Zone Forecast",
    "📈 Hourly History",
    "🧪 Accuracy & Backtests",
    "📝 Actuals Tracking",
])


with forecast_tab:
    st.subheader("Next 24 hours · Zone 1, Zone 2, Zone 3")
    st.write(
        "Fetch hourly weather for Tetouan and forecast its three zones "
        "recursively, feeding each predicted hour into the next."
    )
    st.caption("Location is fixed to Tetouan because the demand history contains only Tetouan zones.")

    _, button_col, _ = st.columns([1, 2, 1])
    with button_col:
        st.write("")
        run_forecast = st.button(
            "Forecast next 24 hours",
            type="primary",
            use_container_width=True,
            key="morocco_run_forecast",
        )

    if run_forecast:
        try:
            with st.spinner("Fetching hourly weather and forecasting the three zones…"):
                live = cached_hourly_weather(FORECAST_LOCATION)
                forecast = forecast_next_24(hourly_df, live["weather"], hourly_artifact)
                forecast = add_empirical_bands(forecast)
                record_live_forecast(forecast, live["location"], live["timezone"])
            st.session_state["morocco_forecast"] = {
                "location": live["location"],
                "latitude": live["latitude"],
                "longitude": live["longitude"],
                "timezone": live["timezone"],
                "frame": forecast,
            }
        except requests.RequestException as exc:
            st.error(f"Could not retrieve hourly weather. Check your connection and retry. Details: {exc}")
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            st.error(f"Could not generate the forecast: {exc}")

    saved_forecast = st.session_state.get("morocco_forecast")
    if saved_forecast:
        frame = saved_forecast["frame"]
        first = frame.iloc[0]
        st.divider()
        st.markdown(f"### 📍 {saved_forecast['location']}")
        st.caption(
            f"Timezone: {saved_forecast['timezone']} · "
            f"Coordinates: {saved_forecast['latitude']:.3f}, {saved_forecast['longitude']:.3f} · "
            f"Forecast window: {frame['Datetime'].min()} to {frame['Datetime'].max()}"
        )

        metric_cols = st.columns(4)
        labels = ["Zone 1", "Zone 2", "Zone 3"]
        for col, label, target in zip(metric_cols[:3], labels, TARGETS):
            spec = hourly_artifact["models"][target]
            method = "previous day" if spec["kind"] == "seasonal_naive" else "weather-aware"
            col.metric(f"{label} · {method}", f"{first[target]:,.0f}")
        metric_cols[3].metric("Combined · next hour", f"{first['Combined_Zones']:,.0f}")

        weather_model_zones = [
            f"Zone {index + 1}"
            for index, target in enumerate(TARGETS)
            if hourly_artifact["models"][target]["kind"] == "hist_gradient_boosting"
        ]
        baseline_model_zones = [
            f"Zone {index + 1}"
            for index, target in enumerate(TARGETS)
            if hourly_artifact["models"][target]["kind"] == "seasonal_naive"
        ]
        st.caption(
            f"Weather-aware models: {', '.join(weather_model_zones) or 'none'}. "
            f"Previous-day baselines: {', '.join(baseline_model_zones) or 'none'}. "
            "Zone values are hourly mean readings; the combined line is their "
            "per-hour sum, not hourly energy."
        )

        st.markdown("#### Peak-hour summary")
        peak_cols = st.columns(4)
        for col, label, column in zip(peak_cols[:3], ["Zone 1", "Zone 2", "Zone 3"], TARGETS):
            peak = frame.loc[frame[column].idxmax()]
            col.metric(f"{label} peak", f"{peak[column]:,.0f}")
            col.caption(pd.Timestamp(peak["Datetime"]).strftime("%a %d %b · %H:%M"))
        combined_peak = frame.loc[frame["Combined_Zones"].idxmax()]
        peak_cols[3].metric("Combined peak", f"{combined_peak['Combined_Zones']:,.0f}")
        peak_cols[3].caption(pd.Timestamp(combined_peak["Datetime"]).strftime("%a %d %b · %H:%M"))

        band_columns = [f"{target}_lower_90" for target in TARGETS] + [f"{target}_upper_90" for target in TARGETS]
        bands_ready = all(column in frame and frame[column].notna().all() for column in band_columns)
        if bands_ready:
            st.caption(
                "The table includes empirical 90% error bands calibrated by "
                "forecast horizon from the recursive holdout. They use 2017 "
                "observed-weather errors and do not include weather-forecast uncertainty."
            )
        else:
            st.warning("Forecast error bands are unavailable: the backtest has too few calibration errors.")
        chart_columns = [*TARGETS, "Combined_Zones"]
        chart = frame.set_index("Datetime")[chart_columns].rename(columns={
            TARGETS[0]: "Zone 1",
            TARGETS[1]: "Zone 2",
            TARGETS[2]: "Zone 3",
            "Combined_Zones": "Combined zones",
        })
        st.line_chart(chart, use_container_width=True)

        if bands_ready:
            band_zone = st.selectbox(
                "Zone to inspect with its empirical 90% range",
                TARGETS,
                format_func=lambda target: target.replace("PowerConsumption_", ""),
                key="forecast_band_zone",
            )
            lower_name = f"{band_zone}_lower_90"
            upper_name = f"{band_zone}_upper_90"
            band_chart = frame.set_index("Datetime")[[band_zone, lower_name, upper_name]].rename(columns={
                band_zone: "Forecast",
                lower_name: "Lower 90% band",
                upper_name: "Upper 90% band",
            })
            st.line_chart(band_chart, use_container_width=True)

        display = frame[[
            "Datetime",
            "Temperature",
            "Humidity",
            "WindSpeed",
            "GeneralDiffuseFlows",
            "DiffuseFlows",
            *TARGETS,
            *([f"{target}_lower_90" for target in TARGETS] if bands_ready else []),
            *([f"{target}_upper_90" for target in TARGETS] if bands_ready else []),
            "Combined_Zones",
        ]].rename(columns={
            "Temperature": "Temperature (°C)",
            "Humidity": "Humidity (%)",
            "WindSpeed": "Wind speed (km/h)",
            "GeneralDiffuseFlows": "General diffuse flow proxy (W/m²)",
            "DiffuseFlows": "Diffuse flow proxy (W/m²)",
            TARGETS[0]: "Zone 1",
            TARGETS[1]: "Zone 2",
            TARGETS[2]: "Zone 3",
            f"{TARGETS[0]}_lower_90": "Zone 1 lower 90%",
            f"{TARGETS[1]}_lower_90": "Zone 2 lower 90%",
            f"{TARGETS[2]}_lower_90": "Zone 3 lower 90%",
            f"{TARGETS[0]}_upper_90": "Zone 1 upper 90%",
            f"{TARGETS[1]}_upper_90": "Zone 2 upper 90%",
            f"{TARGETS[2]}_upper_90": "Zone 3 upper 90%",
            "Combined_Zones": "Combined zones",
        })
        st.dataframe(display.round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "Download 24-hour forecast CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="morocco_hourly_24h_forecast.csv",
            mime="text/csv",
        )
        st.info(
            "Temperature, humidity, and wind speed use Open-Meteo forecast "
            "fields. Open-Meteo shortwave and diffuse solar radiation are used "
            "as proxies for the source's `GeneralDiffuseFlows` and `DiffuseFlows`; "
            "their source units are unspecified. This mapping is approximate."
        )
    else:
        st.info("Click **Forecast next 24 hours** to fetch Tetouan weather and generate the forecast.")


with history_tab:
    st.subheader("Explore hourly zone history")
    st.write("Browse the processed Morocco dataset used to train the hourly models.")
    date_range = st.date_input(
        "Date range",
        value=(first_date, last_date),
        min_value=first_date,
        max_value=last_date,
        key="morocco_history_dates",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = first_date, last_date

    visible = hourly_df.loc[
        hourly_df["Datetime"].dt.date.between(start_date, end_date)
    ].copy()
    selected = visible.set_index("Datetime")[[*TARGETS]].rename(columns={
        TARGETS[0]: "Zone 1",
        TARGETS[1]: "Zone 2",
        TARGETS[2]: "Zone 3",
    })
    st.line_chart(selected, use_container_width=True)
    st.caption("Each point is the mean of the source readings grouped into that clock hour.")

    with st.expander("Data coverage and quality"):
        quality = pd.DataFrame([
            {"Check": "Hourly rows", "Result": f"{len(hourly_df):,}"},
            {"Check": "Missing values in displayed fields", "Result": str(int(hourly_df[["Datetime", *TARGETS]].isna().sum().sum()))},
            {"Check": "Duplicate timestamps", "Result": str(int(hourly_df["Datetime"].duplicated().sum()))},
            {"Check": "Hours with six 10-minute readings", "Result": str(int((hourly_df["SourceReadings"] == 6).sum()))},
        ])
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.dataframe(
            visible[["Datetime", "SourceReadings", *TARGETS]].tail(100).round(2),
            hide_index=True,
            use_container_width=True,
        )


with accuracy_tab:
    st.subheader("One-step holdout accuracy")
    st.write("Compare held-out predictions with observed hourly readings.")
    metrics_path = ROOT / "reports" / "morocco_hourly_model_comparison.csv"
    holdout_path = ROOT / "reports" / "morocco_hourly_holdout_predictions.csv"
    recursive_path = ROOT / "reports" / "morocco_hourly_recursive_24h_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics.round(3), hide_index=True, use_container_width=True)
    else:
        st.info("Run `python src/train_morocco_hourly.py` to create the model comparison report.")

    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path, parse_dates=["Datetime"])
        selected_zone = st.selectbox("Zone to inspect", TARGETS, format_func=lambda target: target.replace("PowerConsumption_", ""))
        actual_column = f"{selected_zone}_actual"
        predicted_column = f"{selected_zone}_predicted"
        holdout_plot = holdout.set_index("Datetime")[[actual_column, predicted_column]].rename(columns={
            actual_column: "Actual",
            predicted_column: "Predicted",
        })
        st.line_chart(holdout_plot, use_container_width=True)
        holdout_errors = (holdout[actual_column] - holdout[predicted_column]).abs()
        st.caption(f"Holdout MAE for this selected model: {holdout_errors.mean():,.2f} source units.")

    if recursive_path.exists():
        st.subheader("Recursive 24-hour backtest")
        st.caption(
            "This evaluation uses observed weather during the holdout, so it "
            "isolates recursive demand behavior rather than weather forecast error."
        )
        recursive = pd.read_csv(recursive_path)
        st.dataframe(recursive.round(3), hide_index=True, use_container_width=True)
        recursive_zone = st.selectbox(
            "Zone for error-by-horizon chart",
            TARGETS,
            format_func=lambda target: target.replace("PowerConsumption_", ""),
            key="recursive_zone",
        )
        zone_errors = recursive.loc[recursive["zone"] == recursive_zone]
        horizon_mae = zone_errors.pivot(index="horizon_hours", columns="model", values="MAE")
        st.bar_chart(horizon_mae, use_container_width=True)
    else:
        st.info("Run `python src/evaluate_morocco_recursive.py` to create the recursive backtest report.")

    st.markdown(
        "Source dataset: [UCI Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2B). "
        "Weather forecast: [Open-Meteo](https://open-meteo.com/en/docs)."
    )


with actuals_tab:
    st.subheader("Record actual hourly readings")
    st.write(
        "When a forecasted hour has passed, enter its observed zone readings. "
        "Actuals are stored separately in `data/observed/morocco_actuals.csv`; "
        "the source and processed datasets are not changed."
    )
    live_forecasts = load_live_forecasts()
    actuals = load_observed_actuals()

    if not live_forecasts.empty:
        latest_forecasts = (
            live_forecasts.sort_values("ForecastRunUTC")
            .drop_duplicates("Datetime", keep="last")
        )
        timezone_name = (
            latest_forecasts["Timezone"].dropna().iloc[-1]
            if "Timezone" in latest_forecasts and latest_forecasts["Timezone"].notna().any()
            else "Africa/Casablanca"
        )
        local_now = pd.Timestamp.now(tz=ZoneInfo(timezone_name)).tz_localize(None).floor("h")
        latest_forecasts = latest_forecasts.loc[
            pd.to_datetime(latest_forecasts["Datetime"]) <= local_now
        ]
        known_actual_times = set(pd.to_datetime(actuals["Datetime"])) if not actuals.empty else set()
        unreported = latest_forecasts.loc[
            ~pd.to_datetime(latest_forecasts["Datetime"]).isin(known_actual_times)
        ].sort_values("Datetime", ascending=False)
        if latest_forecasts.empty:
            st.info("Forecast hours are still in the future. Return after they pass to enter actual readings.")
        elif not unreported.empty:
            options = [pd.Timestamp(value) for value in unreported["Datetime"].head(1000)]
            with st.form("morocco_actual_entry"):
                chosen_time = st.selectbox(
                    "Forecast hour to report",
                    options,
                    format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M"),
                )
                entry_cols = st.columns(3)
                readings = {
                    target: entry_cols[index].number_input(
                        target.replace("PowerConsumption_", "") + " actual",
                        min_value=0.0,
                        step=100.0,
                        key=f"actual_{index}",
                    )
                    for index, target in enumerate(TARGETS)
                }
                submitted = st.form_submit_button("Save actual readings", type="primary")
            if submitted:
                try:
                    save_observed_actual(chosen_time, readings)
                    st.success("Actual readings saved and linked to the latest forecast for that hour.")
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Could not save actual readings: {exc}")
        else:
            st.info("All saved forecast hours already have actual readings.")
    else:
        st.info("Generate a forecast first. Its 24 hourly timestamps will then appear here for later actual entry.")

    if not actuals.empty and not live_forecasts.empty:
        latest_forecasts = (
            live_forecasts.sort_values("ForecastRunUTC")
            .drop_duplicates("Datetime", keep="last")
        )
        latest_forecasts = latest_forecasts[["Datetime", *TARGETS]].rename(
            columns={target: f"{target}_forecast" for target in TARGETS}
        )
        comparisons = actuals.merge(latest_forecasts, on="Datetime", how="inner")
        st.subheader("Recorded forecast accuracy")
        if not comparisons.empty:
            score_rows = []
            for target in TARGETS:
                actual_values = comparisons[target].astype(float)
                forecast_values = comparisons[f"{target}_forecast"].astype(float)
                score_rows.append({
                    "Zone": target.replace("PowerConsumption_", ""),
                    "Actual hours": len(comparisons),
                    "MAE": float((actual_values - forecast_values).abs().mean()),
                    "RMSE": float(((actual_values - forecast_values) ** 2).mean() ** 0.5),
                })
            st.dataframe(pd.DataFrame(score_rows).round(2), hide_index=True, use_container_width=True)
            compare_zone = st.selectbox(
                "Recorded actuals vs saved forecast",
                TARGETS,
                format_func=lambda target: target.replace("PowerConsumption_", ""),
                key="actual_compare_zone",
            )
            chart = comparisons.set_index("Datetime")[[compare_zone, f"{compare_zone}_forecast"]].rename(columns={
                compare_zone: "Actual",
                f"{compare_zone}_forecast": "Saved forecast",
            })
            st.line_chart(chart, use_container_width=True)
        else:
            st.info("No recorded actual timestamps match the saved forecast hours yet.")
        st.dataframe(actuals.sort_values("Datetime", ascending=False), hide_index=True, use_container_width=True)
        st.download_button(
            "Download recorded actuals CSV",
            data=actuals.to_csv(index=False).encode("utf-8"),
            file_name="morocco_actual_readings.csv",
            mime="text/csv",
        )
