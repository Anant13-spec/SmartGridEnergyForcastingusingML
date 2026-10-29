from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "df_final.csv"
MODEL_PATH = ROOT / "models" / "load_forecaster.joblib"

WEATHER_FIELDS = {
    "temp": "Average temperature (°C)",
    "tempmax": "Maximum temperature (°C)",
    "tempmin": "Minimum temperature (°C)",
    "feelslike": "Feels-like temperature (°C)",
    "humidity": "Humidity (%)",
    "dew": "Dew point (°C)",
    "precip": "Precipitation (mm)",
    "precipprob": "Precipitation probability (%)",
    "windspeed": "Wind speed (km/h)",
    "windgust": "Wind gust (km/h)",
    "sealevelpressure": "Sea-level pressure (hPa)",
    "cloudcover": "Cloud cover (%)",
    "visibility": "Visibility (km)",
    "solarradiation": "Solar radiation (W/m²)",
    "solarenergy": "Solar energy (MJ/m²)",
    "uvindex": "UV index",
}


def ask_number(label):
    while True:
        try:
            value = float(input(f"{label}: "))
            if not pd.notna(value):
                raise ValueError
            return value
        except ValueError:
            print("Please enter a valid number.")


def parse_forecast_date(date_text):
    """Parse a real calendar date supplied in YYYY-MM-DD format."""
    try:
        return pd.to_datetime(date_text, format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Enter a valid forecast date in YYYY-MM-DD format."
        ) from error


def validate_weather(weather):
    errors = []

    if not -50 <= weather["temp"] <= 60:
        errors.append("Average temperature must be between -50 and 60 °C.")

    if not -60 <= weather["tempmin"] <= 60:
        errors.append("Minimum temperature is outside the allowed range.")

    if not -50 <= weather["tempmax"] <= 70:
        errors.append("Maximum temperature is outside the allowed range.")

    if not (
        weather["tempmin"] <= weather["temp"] <= weather["tempmax"]
    ):
        errors.append(
            "Temperature must satisfy: minimum <= average <= maximum."
        )

    if not 0 <= weather["humidity"] <= 100:
        errors.append("Humidity must be between 0 and 100%.")

    if not 0 <= weather["precipprob"] <= 100:
        errors.append("Precipitation probability must be between 0 and 100%.")

    if not 0 <= weather["cloudcover"] <= 100:
        errors.append("Cloud cover must be between 0 and 100%.")

    for field in (
        "precip",
        "windspeed",
        "windgust",
        "visibility",
        "solarradiation",
        "solarenergy",
    ):
        if weather[field] < 0:
            errors.append(f"{field} cannot be negative.")

    if not 800 <= weather["sealevelpressure"] <= 1100:
        errors.append("Sea-level pressure must be between 800 and 1100 hPa.")

    if not 0 <= weather["uvindex"] <= 15:
        errors.append("UV index must be between 0 and 15.")

    if errors:
        raise ValueError("\n".join(errors))


def predict_demand(date, weather):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["POWER_DEMAND"] = pd.to_numeric(
        df["POWER_DEMAND"], errors="coerce"
    )

    df = (
        df.dropna(subset=["DATE", "POWER_DEMAND"])
        .sort_values("DATE")
        .drop_duplicates("DATE")
        .reset_index(drop=True)
    )

    target_date = pd.Timestamp(date)

    if target_date <= df["DATE"].max():
        raise ValueError(
            f"Choose a date after {df['DATE'].max().date()}."
        )

    demand = df["POWER_DEMAND"].where(
        df["POWER_DEMAND"] > 0
    ).dropna()

    if len(demand) < 7:
        raise ValueError("At least seven positive demand readings are needed.")

    history = demand.tolist()

    x = {
        "year": target_date.year,
        "month": target_date.month,
        "day_of_year": target_date.day_of_year,
        "day_of_week": target_date.day_of_week,
        "is_weekend": int(target_date.day_of_week >= 5),
        "demand_lag_1": history[-1],
        "demand_lag_7": history[-7],
        "demand_mean_7": sum(history[-7:]) / 7,
    }

    x.update(weather)

    missing = [feature for feature in features if feature not in x]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = pd.DataFrame([x])[features]
    prediction = max(0.0, float(model.predict(X)[0]))

    return prediction


if __name__ == "__main__":
    print("\nSMART GRID — WEATHER SCENARIO PREDICTION")
    print("----------------------------------------")
    print("Enter the weather values for your chosen forecast date.")
    print("Use consistent units; do not leave weather values blank.\n")

    date = input("Forecast date (YYYY-MM-DD): ").strip()

    weather = {}
    for field, description in WEATHER_FIELDS.items():
        weather[field] = ask_number(description)

    try:
        parsed_date = parse_forecast_date(date)
        validate_weather(weather)
        result = predict_demand(parsed_date, weather)

        print("\n========== FORECAST RESULT ==========")
        print("Forecast date:", parsed_date.date())
        print(f"Predicted electricity demand: {result:.2f}")
        print("Weather inputs: user-supplied")
        print(
            "\nIMPORTANT: Demand history comes from the saved dataset, "
            "which ends in 2024. This is a scenario demonstration, "
            "not a validated live forecast."
        )
    except (ValueError, KeyError) as error:
        print("\nPrediction could not be completed:", error)
        