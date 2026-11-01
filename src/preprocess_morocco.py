"""Prepare the Morocco 10-minute power dataset for hourly forecasting.

Aggregation policy:
* Weather/flow readings are averaged within each clock hour.
* Zone readings are also averaged within each hour. The data dictionary calls
  them kWh, but does not state whether a row is interval energy or a point
  load reading. A mean is therefore used as an hourly representative reading;
  it is not presented as an hourly energy total. If the source confirms that
  each row is energy consumed during its preceding 10-minute interval, change
  the zone aggregation to sum before interpreting the result as hourly kWh.

The raw CSV is never modified. By default, outputs are written to
data/processed/hourly_powerconsumption.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "powerconsumption.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "hourly_powerconsumption.csv"
DATETIME = "Datetime"
WEATHER_COLUMNS = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
]
ZONE_COLUMNS = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]


def prepare(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(input_path)
    required = [DATETIME, *WEATHER_COLUMNS, *ZONE_COLUMNS]
    absent = [column for column in required if column not in raw.columns]
    if absent:
        raise ValueError(f"Input is missing required columns: {absent}")

    source_rows = len(raw)
    source_missing = raw[required].isna().sum()
    raw[DATETIME] = pd.to_datetime(raw[DATETIME], errors="coerce")
    if raw[DATETIME].isna().any():
        raise ValueError(f"Found {int(raw[DATETIME].isna().sum())} invalid timestamps")
    raw = raw.sort_values(DATETIME, kind="stable").reset_index(drop=True)
    duplicate_rows = int(raw.duplicated().sum())
    duplicate_timestamps = int(raw[DATETIME].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"Found {duplicate_timestamps} duplicate timestamps")

    deltas = raw[DATETIME].diff().dropna()
    interval_counts = deltas.value_counts()
    modal_interval = interval_counts.index[0] if len(interval_counts) else pd.NaT
    nonmodal = int((deltas != modal_interval).sum()) if len(deltas) else 0

    # Use pandas' left-closed clock-hour bins: a reading at 10:00 belongs to
    # the 10:00–10:59 hour, as do readings at 10:10 through 10:50.
    hourly = (
        raw.set_index(DATETIME)[[*WEATHER_COLUMNS, *ZONE_COLUMNS]]
        .resample("h", label="left", closed="left")
        .agg({column: "mean" for column in [*WEATHER_COLUMNS, *ZONE_COLUMNS]})
    )
    counts = raw.set_index(DATETIME).resample("h").size()
    hourly.insert(0, "SourceReadings", counts.astype("int64"))
    hourly.index.name = DATETIME
    hourly = hourly.dropna(subset=[*WEATHER_COLUMNS, *ZONE_COLUMNS])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, float_format="%.8f", date_format="%Y-%m-%d %H:%M:%S")

    summary: dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": source_rows,
        "input_columns": list(raw.columns),
        "date_start": str(raw[DATETIME].min()),
        "date_end": str(raw[DATETIME].max()),
        "missing_values_by_column": {k: int(v) for k, v in source_missing.items()},
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "modal_interval": str(modal_interval),
        "interval_counts": {str(k): int(v) for k, v in interval_counts.items()},
        "intervals_different_from_mode": nonmodal,
        "hourly_rows": len(hourly),
        "hourly_start": str(hourly.index.min()) if len(hourly) else None,
        "hourly_end": str(hourly.index.max()) if len(hourly) else None,
        "hours_with_six_source_readings": int((hourly["SourceReadings"] == 6).sum()),
        "hours_with_other_source_reading_count": int((hourly["SourceReadings"] != 6).sum()),
        "non_hourly_output_intervals": int(
            (hourly.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).sum()
        ),
        "hourly_timestamps_chronological": bool(hourly.index.is_monotonic_increasing),
        "aggregation": {
            "weather_and_diffuse_flow_columns": "arithmetic mean of timestamped readings per clock hour",
            "zone_columns": "arithmetic mean of timestamped readings per clock hour; dictionary says kWh but interval-vs-point semantics are unspecified, so this is not claimed as hourly energy",
        },
    }
    return hourly, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hourly, summary = prepare(args.input, args.output)
    print("Morocco dataset preprocessing summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if not hourly.index.is_monotonic_increasing:
        raise RuntimeError("Hourly output timestamps are not chronological")
    if hourly.empty:
        raise RuntimeError("Hourly output is empty")


if __name__ == "__main__":
    main()
