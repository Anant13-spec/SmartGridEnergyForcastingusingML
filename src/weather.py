"""Open-Meteo response mapping for the model's daily weather inputs."""


def map_open_meteo_weather(current, daily):
    def value(section, key, index=None):
        raw = section.get(key)
        if isinstance(raw, list):
            raw = raw[0 if index is None else index] if len(raw) > (0 if index is None else index) else None
        if raw is None:
            raise ValueError(f"Weather provider omitted {key}")
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Weather provider returned a non-numeric {key}") from None

    return {
        "temp": value(current, "temperature_2m"),
        "tempmax": value(daily, "temperature_2m_max"),
        "tempmin": value(daily, "temperature_2m_min"),
        "feelslike": value(current, "apparent_temperature"),
        "humidity": value(current, "relative_humidity_2m"),
        "dew": value(current, "dew_point_2m"),
        "precip": value(current, "precipitation"),
        "precipprob": value(daily, "precipitation_probability_max"),
        "windspeed": value(current, "wind_speed_10m"),
        "windgust": value(current, "wind_gusts_10m"),
        "sealevelpressure": value(current, "pressure_msl"),
        "cloudcover": value(current, "cloud_cover"),
        "visibility": value(current, "visibility") / 1000.0,
        "solarradiation": value(current, "shortwave_radiation"),
        "solarenergy": value(daily, "shortwave_radiation_sum"),
        "uvindex": value(current, "uv_index"),
    }
