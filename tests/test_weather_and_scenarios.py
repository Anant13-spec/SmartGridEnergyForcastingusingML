import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import WEATHER_FIELDS, validate_weather
from scenarios import PRESETS, SCENARIO_OVERRIDES, TYPICAL
from weather import map_open_meteo_weather


def test_open_meteo_mapping_populates_all_six_weather_features():
    current = {
        "temperature_2m": 32.5,
        "dew_point_2m": 22.0,
        "relative_humidity_2m": 65.0,
        "wind_speed_10m": 12.0,
        "wind_direction_10m": 190.0,
        "surface_pressure": 1006.0,
    }
    weather = map_open_meteo_weather(current)
    assert set(weather) == set(WEATHER_FIELDS)
    assert weather["temp"] == 32.5
    assert weather["dwpt"] == 22.0
    assert weather["rhum"] == 65.0
    validate_weather(weather)


def test_weather_validation_rejects_missing_and_nonfinite():
    weather = TYPICAL.copy()
    validate_weather(weather)

    with pytest.raises(ValueError, match="Missing required weather"):
        validate_weather({"temp": 25.0})

    weather_nan = TYPICAL.copy()
    weather_nan["rhum"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_weather(weather_nan)


def test_all_scenario_presets_match_model_schema_and_pass_validation():
    for name, weather in PRESETS.items():
        assert set(weather) == set(WEATHER_FIELDS), f"Scenario '{name}' does not match WEATHER_FIELDS"
        validate_weather(weather.copy())


def test_scenarios_only_override_their_declared_weather_fields():
    for name, overrides in SCENARIO_OVERRIDES.items():
        scenario = PRESETS[name]
        changed = {key for key in WEATHER_FIELDS if scenario[key] != TYPICAL[key]}
        assert changed == set(overrides), f"Scenario '{name}' modified undeclared fields"
