
from pathlib import Path
import copy

import pandas as pd

from predict import WEATHER_FIELDS, validate_weather, predict_demand

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "reports" / "scenario_comparison.csv"


def main():
    print("\nSMART GRID — WEATHER SCENARIO COMPARISON")
    print("----------------------------------------")

    date = input("Forecast date (YYYY-MM-DD): ").strip()
    parsed_date = pd.to_datetime(date, errors="raise")

    base_weather = {}

    print("\nEnter the baseline weather conditions:")
    for field, description in WEATHER_FIELDS.items():
        while True:
            try:
                value = float(input(f"{description}: "))
                if not pd.notna(value):
                    raise ValueError
                base_weather[field] = value
                break
            except ValueError:
                print("Enter a valid number.")

    # Construct three comparable hypothetical scenarios.
    scenarios = {
        "Moderate (baseline)": copy.deepcopy(base_weather),
        "Hot": copy.deepcopy(base_weather),
        "Hot and humid": copy.deepcopy(base_weather),
    }

    for name in ["Hot", "Hot and humid"]:
        weather = scenarios[name]
        for field in ["temp", "tempmax", "tempmin", "feelslike"]:
            weather[field] += 5

    # In the hot-and-humid scenario, increase humidity
    # and dew point as explicit hypothetical assumptions.
    scenarios["Hot and humid"]["humidity"] = min(
        100, base_weather["humidity"] + 15
    )
    scenarios["Hot and humid"]["dew"] += 3

    results = []

    for name, weather in scenarios.items():
        try:
            validate_weather(weather)
            prediction = predict_demand(parsed_date, weather)

            results.append({
                "scenario": name,
                "date": parsed_date.date(),
                "temperature_C": weather["temp"],
                "humidity_percent": weather["humidity"],
                "predicted_demand": round(prediction, 2),
                "change_from_baseline": None,
            })

        except (ValueError, KeyError) as error:
            print(f"\nScenario '{name}' rejected: {error}")
            print("Correct the baseline inputs and try again.")
            return

    baseline = results[0]["predicted_demand"]

    for result in results:
        result["change_from_baseline"] = round(
            result["predicted_demand"] - baseline, 2
        )

    comparison = pd.DataFrame(results)

    print("\n========== SCENARIO RESULTS ==========")
    print(comparison.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved comparison: {OUTPUT_PATH}")
    print("\nScenario assumptions:")
    print("- Hot: temperature fields increased by 5°C.")
    print("- Hot and humid: same temperature increase,")
    print("  humidity increased by 15 percentage points,")
    print("  and dew point increased by 3°C.")
    print("\nThese are hypothetical scenarios, not live forecasts.")
    print("Historical demand data ends in 2024.")


if __name__ == "__main__":
    main()
