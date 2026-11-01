import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import WEATHER_FIELDS, validate_weather
from scenarios import PRESETS, SCENARIO_OVERRIDES, TYPICAL
from weather import map_open_meteo_weather


def test_open_meteo_mapping_populates_all_model_weather_features_and_converts_visibility():
    current = {
        "temperature_2m": 28, "apparent_temperature": 30,
        "relative_humidity_2m": 70, "dew_point_2m": 22,
        "precipitation": 0.2, "wind_speed_10m": 12,
        "wind_gusts_10m": 18, "pressure_msl": 1008,
        "cloud_cover": 40, "visibility": 8500,
        "shortwave_radiation": 250, "uv_index": 4,
    }
    daily = {
        "temperature_2m_max": [32], "temperature_2m_min": [24],
        "precipitation_probability_max": [30],
        "shortwave_radiation_sum": [18],
    }
    weather = map_open_meteo_weather(current, daily)
    assert set(weather) == set(WEATHER_FIELDS)
    assert weather["visibility"] == pytest.approx(8.5)
    validate_weather(weather)


def test_weather_validation_rejects_missing_nonfinite_and_inconsistent_values():
    weather = PRESETS["Typical"].copy()
    validate_weather(weather)
    with pytest.raises(ValueError, match="Missing weather"):
        validate_weather({"temp": 20})
    weather = PRESETS["Typical"].copy()
    weather["humidity"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_weather(weather)
    weather = PRESETS["Typical"].copy()
    weather["temp"] = 5
    with pytest.raises(ValueError, match="minimum <= average"):
        validate_weather(weather)


def test_all_scenario_presets_match_model_schema_and_pass_validation():
    assert set(PRESETS) == {"Typical", "Hot", "Hot & Humid", "Rainy"}
    for weather in PRESETS.values():
        assert set(weather) == set(WEATHER_FIELDS)
        validate_weather(weather.copy())


def test_scenarios_only_override_their_declared_weather_fields():
    for name, overrides in SCENARIO_OVERRIDES.items():
        changed = {key for key in WEATHER_FIELDS if PRESETS[name][key] != TYPICAL[key]}
        assert changed == set(overrides)
