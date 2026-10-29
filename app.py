
from datetime import date
from pathlib import Path
import sys

import joblib
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# PATHS AND EXISTING MODEL
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import WEATHER_FIELDS, validate_weather, predict_demand

DATA_PATH = ROOT / "df_final.csv"
MODEL_PATH = ROOT / "models" / "load_forecaster.joblib"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="GridWise | Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #e8eef8;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: #131f32;
        border: 1px solid #263750;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #55dfbd;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bbd3;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background: #0e9f83;
        color: white;
        border: none;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .hero {
        padding: 22px 24px;
        border: 1px solid #263750;
        border-radius: 16px;
        background: linear-gradient(120deg, #14243a, #102c32);
        margin-bottom: 20px;
    }

    .muted {
        color: #a9bbd3;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# WEATHER PRESETS
# --------------------------------------------------
PRESETS = {
    "Typical": {
        "temp": 30, "tempmax": 35, "tempmin": 25,
        "feelslike": 33, "humidity": 60, "dew": 20,
        "precip": 0, "precipprob": 10, "windspeed": 12,
        "windgust": 20, "sealevelpressure": 1010,
        "cloudcover": 30, "visibility": 10,
        "solarradiation": 200, "solarenergy": 15, "uvindex": 5,
    },
    "Hot": {
        "temp": 38, "tempmax": 42, "tempmin": 34,
        "feelslike": 40, "humidity": 35, "dew": 18,
        "precip": 0, "precipprob": 5, "windspeed": 15,
        "windgust": 25, "sealevelpressure": 1005,
        "cloudcover": 15, "visibility": 10,
        "solarradiation": 500, "solarenergy": 25, "uvindex": 8,
    },
    "Hot and humid": {
        "temp": 34, "tempmax": 38, "tempmin": 30,
        "feelslike": 39, "humidity": 80, "dew": 29,
        "precip": 1, "precipprob": 40, "windspeed": 10,
        "windgust": 18, "sealevelpressure": 1008,
        "cloudcover": 65, "visibility": 8,
        "solarradiation": 250, "solarenergy": 18, "uvindex": 6,
    },
    "Rainy": {
        "temp": 24, "tempmax": 27, "tempmin": 21,
        "feelslike": 24, "humidity": 85, "dew": 22,
        "precip": 12, "precipprob": 80, "windspeed": 18,
        "windgust": 30, "sealevelpressure": 1002,
        "cloudcover": 90, "visibility": 6,
        "solarradiation": 80, "solarenergy": 5, "uvindex": 1,
    },
}

RANGES = {
    "temp": (-50, 60),
    "tempmax": (-50, 70),
    "tempmin": (-60, 60),
    "feelslike": (-60, 70),
    "humidity": (0, 100),
    "dew": (-80, 60),
    "precip": (0, 1000),
    "precipprob": (0, 100),
    "windspeed": (0, 300),
    "windgust": (0, 400),
    "sealevelpressure": (800, 1100),
    "cloudcover": (0, 100),
    "visibility": (0, 100),
    "solarradiation": (0, 2000),
    "solarenergy": (0, 100),
    "uvindex": (0, 15),
}

# --------------------------------------------------
# DATA AND MODEL CHECKS
# --------------------------------------------------
if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Dataset or trained model is missing. "
        "Check df_final.csv and models/load_forecaster.joblib."
    )
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)

    historical_df = pd.read_csv(DATA_PATH)
    historical_df["DATE"] = pd.to_datetime(
        historical_df["DATE"], errors="coerce"
    )
    historical_df = historical_df.dropna(subset=["DATE"])

    if historical_df.empty:
        raise ValueError("The dataset contains no valid dates.")

    last_date = historical_df["DATE"].max().date()

except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()

# --------------------------------------------------
# AUTOMATIC CURRENT WEATHER
# --------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_current_weather(city):
    """Fetch current conditions from Open-Meteo for a named city."""

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 5,
            "language": "en",
            "format": "json",
        },
        timeout=15,
    )
    geo_response.raise_for_status()

    locations = geo_response.json().get("results", [])

    if not locations:
        raise ValueError(
            f"No matching city found for '{city}'. "
            "Try including the state or country."
        )

    # Use the best geocoding match returned by the provider.
    place = locations[0]

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,dew_point_2m,"
                "wind_speed_10m,wind_gusts_10m,pressure_msl,"
                "cloud_cover,visibility,shortwave_radiation,uv_index"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,shortwave_radiation_sum"
            ),
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    current = data.get("current", {})
    daily = data.get("daily", {})

    def read_value(section, key, index=None):
        raw = section.get(key)

        if isinstance(raw, list):
            if index is None:
                index = 0
            if len(raw) <= index:
                raise ValueError(f"Missing weather value: {key}")
            raw = raw[index]

        if raw is None:
            raise ValueError(f"Weather provider omitted {key}")

        return float(raw)

    weather = {
        "temp": read_value(current, "temperature_2m"),
        "tempmax": read_value(daily, "temperature_2m_max"),
        "tempmin": read_value(daily, "temperature_2m_min"),
        "feelslike": read_value(current, "apparent_temperature"),
        "humidity": read_value(current, "relative_humidity_2m"),
        "dew": read_value(current, "dew_point_2m"),
        "precip": read_value(current, "precipitation"),
        "precipprob": read_value(
            daily, "precipitation_probability_max"
        ),
        "windspeed": read_value(current, "wind_speed_10m"),
        "windgust": read_value(current, "wind_gusts_10m"),
        "sealevelpressure": read_value(current, "pressure_msl"),
        "cloudcover": read_value(current, "cloud_cover"),
        "visibility": read_value(current, "visibility") / 1000,
        "solarradiation": read_value(current, "shortwave_radiation"),
        "solarenergy": read_value(daily, "shortwave_radiation_sum"),
        "uvindex": read_value(current, "uv_index"),
    }

    return {
        "city": place["name"],
        "country": place.get("country", ""),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "observed_at": current.get("time"),
        "timezone": data.get("timezone", ""),
        "weather": weather,
    }

# --------------------------------------------------
# PREDICTION HELPERS
# --------------------------------------------------
def get_prediction(target_date, weather):
    validate_weather(weather)

    return predict_demand(str(target_date), weather)


def save_report(frame, filename):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT_DIR / filename, index=False)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1 style="margin:0;">⚡ GridWise</h1>
    <p style="margin-top:8px; color:#b9cbe0;">
        Smart Grid · Weather-aware energy demand forecasting
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    f"Prototype limitation: historical demand ends on {last_date}. "
    "Live weather can be fetched, but demand history is not current. "
    "Predictions are experimental and are not validated operational forecasts."
)

info1, info2, info3 = st.columns(3)
info1.metric("Historical data through", str(last_date))
info2.metric("Weather source", "Open-Meteo")
info3.metric("Model", artifact.get("model_name", "Saved model"))

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
weather_tab, scenario_tab = st.tabs([
    "🌦️ Current Weather",
    "📊 Scenario Lab",
])

# ==================================================
# TAB 1: CURRENT WEATHER
# ==================================================
with weather_tab:
    st.subheader("Current weather → demand prediction")
    st.write(
        "Select a city. GridWise will retrieve the latest available "
        "weather conditions and use them as inputs to the saved model."
    )

    city_col, button_col = st.columns([3, 1])

    with city_col:
        city = st.text_input(
            "City or town",
            value="Delhi",
            key="current_weather_city",
            placeholder="e.g. Delhi, Mumbai, Pune",
        )

    with button_col:
        st.write("")
        st.write("")
        fetch_button = st.button(
            "Get weather & predict",
            type="primary",
            use_container_width=True,
            key="fetch_current",
        )

    if fetch_button:
        if not city.strip():
            st.error("Enter a city name first.")
        else:
            try:
                with st.spinner("Fetching current weather..."):
                    live = fetch_current_weather(city.strip())

                weather = live["weather"]
                observed_at = live["observed_at"]

                if not observed_at:
                    raise ValueError(
                        "The weather provider did not return an observation time."
                    )

                # The date belongs to the weather location's timezone.
                weather_date = pd.Timestamp(observed_at).date()

                validate_weather(weather)

                if weather_date <= last_date:
                    st.error(
                        f"The retrieved weather date is {weather_date}, "
                        f"but the demand dataset ends on {last_date}. "
                        "The current-weather prediction cannot be evaluated "
                        "as a future-date prediction with this dataset."
                    )
                else:
                    with st.spinner("Generating demand prediction..."):
                        prediction = get_prediction(
                            weather_date, weather
                        )

                    st.session_state["live_weather_result"] = {
                        "location": (
                            f'{live["city"]}, {live["country"]}'
                        ),
                        "observed_at": observed_at,
                        "weather_date": str(weather_date),
                        "weather": weather,
                        "prediction": prediction,
                        "timezone": live["timezone"],
                    }

            except requests.RequestException as exc:
                st.error(
                    "Unable to contact the weather service. "
                    f"Check your internet connection and retry. Details: {exc}"
                )
            except (ValueError, KeyError) as exc:
                st.error(f"Could not generate prediction: {exc}")

    live_result = st.session_state.get("live_weather_result")

    if live_result:
        st.divider()

        st.markdown(
            f"### 📍 {live_result['location']}"
        )
        st.caption(
            f"Weather timestamp: {live_result['observed_at']} · "
            f"Timezone: {live_result['timezone']}"
        )

        weather = live_result["weather"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f'{weather["temp"]:.1f} °C')
        c2.metric("Feels like", f'{weather["feelslike"]:.1f} °C')
        c3.metric("Humidity", f'{weather["humidity"]:.0f}%')
        c4.metric("Wind speed", f'{weather["windspeed"]:.1f} km/h')

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Cloud cover", f'{weather["cloudcover"]:.0f}%')
        c6.metric("Precipitation", f'{weather["precip"]:.1f} mm')
        c7.metric("Solar radiation", f'{weather["solarradiation"]:.0f} W/m²')
        c8.metric("UV index", f'{weather["uvindex"]:.1f}')

        st.divider()

        st.subheader("Predicted electricity demand")

        st.metric(
            "Model prediction",
            f'{live_result["prediction"]:,.2f}',
        )

        st.caption(
            "This prediction uses retrieved weather but historical demand "
            "features from the saved dataset. It should not be interpreted "
            "as a validated current operational forecast."
        )

        with st.expander("Inspect all model weather inputs"):
            st.dataframe(
                pd.DataFrame(
                    list(weather.items()),
                    columns=["Feature", "Value"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Clear current result"):
            st.session_state.pop("live_weather_result", None)
            st.rerun()

    else:
        st.info(
            "No weather retrieved yet. Enter a city and click "
            "**Get weather & predict**."
        )

# ==================================================
# TAB 2: SCENARIO LAB
# ==================================================
with scenario_tab:
    st.subheader("Compare weather scenarios")
    st.write(
        "Compare the model's predictions under typical, hot, "
        "hot-and-humid, and rainy hypothetical conditions."
    )

    forecast_date = st.date_input(
        "Scenario date",
        value=max(date.today(), last_date) + pd.Timedelta(days=1),
        min_value=last_date + pd.Timedelta(days=1),
        key="scenario_date",
    )

    st.caption(
        "These presets are hypothetical, not actual or forecast weather."
    )

    if st.button(
        "Run scenario comparison",
        type="primary",
        key="compare_scenarios",
    ):
        rows = []
        errors = []

        with st.spinner("Comparing scenarios..."):
            for name, weather in PRESETS.items():
                try:
                    prediction = get_prediction(
                        forecast_date, weather.copy()
                    )

                    rows.append({
                        "Scenario": name,
                        "Temperature (°C)": weather["temp"],
                        "Humidity (%)": weather["humidity"],
                        "Predicted demand": round(prediction, 2),
                    })

                except (ValueError, KeyError) as exc:
                    errors.append(f"{name}: {exc}")

        if errors:
            for error in errors:
                st.error(error)

        if rows:
            results = pd.DataFrame(rows)
            baseline = results.iloc[0]["Predicted demand"]

            results["Change from typical"] = (
                results["Predicted demand"] - baseline
            ).round(2)

            st.session_state["scenario_results"] = results

    results = st.session_state.get("scenario_results")

    if results is not None:
        st.divider()

        metric_cols = st.columns(len(results))

        for col, (_, row) in zip(
            metric_cols, results.iterrows()
        ):
            col.metric(
                row["Scenario"],
                f'{row["Predicted demand"]:,.2f}',
                delta=f'{row["Change from typical"]:+,.2f}',
            )

        st.subheader("Predicted demand comparison")

        st.bar_chart(
            results.set_index("Scenario")[["Predicted demand"]]
        )

        with st.expander("View full comparison table"):
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Export comparison CSV"):
            save_report(
                results,
                "dashboard_scenario_comparison.csv",
            )
            st.success(
                "Saved to reports/dashboard_scenario_comparison.csv"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "GridWise · Academic prototype · Weather supplied by Open-Meteo. "
    "The dashboard does not automatically collect current electricity demand. "
    "Recent demand data and further chronological validation are required "
    "before operational use."
)
