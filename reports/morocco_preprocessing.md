# Morocco dataset: Phase 1 preprocessing

Run `python src/preprocess_morocco.py` from the project root. It reads
`data/powerconsumption.csv` and writes
`data/processed/hourly_powerconsumption.csv`; it does not modify the raw file.
The script prints a source and output summary each time it runs.

## Source inspection

- 52,416 rows and 9 columns: `Datetime`, five weather/flow fields, and three
  zone consumption fields.
- Coverage: 2017-01-01 00:00:00 through 2017-12-30 23:50:00.
- All 52,415 timestamp differences are exactly 10 minutes.
- No missing values, duplicate rows, or duplicate timestamps.
- The data dictionary describes temperature as Celsius and wind speed as
  km/h. It labels the three zone values kWh, but does not say whether each
  value is energy over the preceding interval or a load reading at that time.
  Its diffuse-flow descriptions are retained as supplied; their physical
  interpretation is not clarified by the dictionary.

## Hourly aggregation

The output has 8,736 chronological hourly rows from 2017-01-01 00:00 through
2017-12-30 23:00, with six source readings in every hour. Weather and diffuse
flow values are arithmetic means within each clock hour. Zone columns are also
hourly arithmetic means, and `SourceReadings` records the contributing row
count. These zone values are representative hourly readings in the source's
documented units; they are not represented as hourly energy totals. Summing
would only be valid if the source confirms each 10-minute value is interval
energy. The published dictionary alone does not establish that, so the mean
avoids multiplying each displayed value by six without evidence.

The source ends on December 30, so the output does not invent records for
December 31. The script and output are Phase 1 preparation only; no model was
trained or existing project model replaced.
