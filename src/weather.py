import math
from typing import Any, Mapping


def map_open_meteo_weather(current: Mapping[str, Any], hourly: Mapping[str, Any] | None = None) -> dict[str, float]:
    def extract_val(data: Mapping[str, Any], key: str, fallback_key: str | None = None) -> float:
        val = data.get(key)
        if val is None and fallback_key:
            val = data.get(fallback_key)
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if val is None:
            raise ValueError(f"Weather provider response omitted required field: {key}")
        try:
            return float(val)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Weather provider returned non-numeric value for {key}: {val}") from err

    temp = extract_val(current, "temperature_2m")
    rhum = extract_val(current, "relative_humidity_2m")

    if "dew_point_2m" in current and current["dew_point_2m"] is not None:
        dwpt = extract_val(current, "dew_point_2m")
    else:
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + (math.log(max(rhum, 1.0) / 100.0))
        dwpt = (b * alpha) / (a - alpha)

    wdir = extract_val(current, "wind_direction_10m", fallback_key="winddirection_10m")
    wspd = extract_val(current, "wind_speed_10m", fallback_key="windspeed_10m")
    pres = extract_val(current, "surface_pressure", fallback_key="pressure_msl")

    return {
        "temp": temp,
        "dwpt": round(dwpt, 2),
        "rhum": rhum,
        "wdir": wdir,
        "wspd": wspd,
        "pres": pres,
    }
