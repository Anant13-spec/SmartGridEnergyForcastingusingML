from copy import deepcopy

TYPICAL = {
    "temp": 28.0,
    "dwpt": 16.0,
    "rhum": 55.0,
    "wdir": 180.0,
    "wspd": 8.0,
    "pres": 1008.0,
}

SCENARIO_OVERRIDES = {
    "Extreme Summer Heatwave": {
        "temp": 44.0,
        "dwpt": 22.0,
        "rhum": 30.0,
        "wdir": 270.0,
        "wspd": 15.0,
        "pres": 998.0,
    },
    "Monsoon High Humidity": {
        "temp": 32.0,
        "dwpt": 28.0,
        "rhum": 85.0,
        "wdir": 120.0,
        "wspd": 18.0,
        "pres": 1002.0,
    },
    "Winter Morning Peak": {
        "temp": 8.0,
        "dwpt": 5.0,
        "rhum": 80.0,
        "wdir": 310.0,
        "wspd": 5.0,
        "pres": 1018.0,
    },
    "High Wind Stiff Breeze": {
        "temp": 24.0,
        "dwpt": 14.0,
        "rhum": 50.0,
        "wdir": 220.0,
        "wspd": 35.0,
        "pres": 1005.0,
    },
}


def build_scenarios() -> dict[str, dict[str, float]]:
    scenarios = {"Typical Day": deepcopy(TYPICAL)}
    for name, overrides in SCENARIO_OVERRIDES.items():
        scenario = deepcopy(TYPICAL)
        scenario.update(overrides)
        scenarios[name] = scenario
    return scenarios


PRESETS = build_scenarios()
