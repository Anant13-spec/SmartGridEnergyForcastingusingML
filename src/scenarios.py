"""Hypothetical weather presets, with explicit changes from a typical day."""

from copy import deepcopy

TYPICAL = {
    "temp": 30, "tempmax": 35, "tempmin": 25, "feelslike": 33,
    "humidity": 60, "dew": 20, "precip": 0, "precipprob": 10,
    "windspeed": 12, "windgust": 20, "sealevelpressure": 1010,
    "cloudcover": 30, "visibility": 10, "solarradiation": 200,
    "solarenergy": 15, "uvindex": 5,
}

SCENARIO_OVERRIDES = {
    "Hot": {
        "temp": 38, "tempmax": 42, "tempmin": 34, "feelslike": 40,
    },
    "Hot & Humid": {
        "temp": 34, "tempmax": 38, "tempmin": 30, "feelslike": 39,
        "humidity": 80, "dew": 29,
    },
    "Rainy": {
        "temp": 24, "tempmax": 27, "tempmin": 21, "feelslike": 24,
        "humidity": 85, "dew": 22, "precip": 12, "precipprob": 80,
        "windspeed": 18, "windgust": 30, "sealevelpressure": 1002,
        "cloudcover": 90, "visibility": 6, "solarradiation": 80,
        "solarenergy": 5, "uvindex": 1,
    },
}


def build_scenarios():
    scenarios = {"Typical": deepcopy(TYPICAL)}
    for name, overrides in SCENARIO_OVERRIDES.items():
        scenarios[name] = {**deepcopy(TYPICAL), **overrides}
    return scenarios


PRESETS = build_scenarios()
