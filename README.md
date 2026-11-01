# GridWise: Morocco Hourly Multi-Zone Forecasting

GridWise prepares the UCI Power Consumption of Tetouan City dataset, trains a
separate hourly forecasting model for each of its three zones, evaluates
recursive 24-hour forecasts, and provides a Streamlit dashboard using this
workflow.

## Data and units

The original 10-minute source file is `data/powerconsumption.csv`. The
preprocessing script writes the separate hourly file to
`data/processed/hourly_powerconsumption.csv`; it never overwrites the source.
Weather and diffuse-flow observations are averaged within each clock hour.
Zone readings are also averaged because the dictionary does not establish
whether each reading is interval energy or a point load. The hourly targets
are therefore representative mean readings, not verified hourly energy
totals. See `reports/morocco_preprocessing.md` for the inspection and
aggregation details.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Prepare, train, and evaluate

From the project directory:

```bash
python src/preprocess_morocco.py
python src/train_morocco_hourly.py
python src/evaluate_morocco_recursive.py
```

These commands write the hourly model separately as
`models/morocco_hourly_zone_forecasters.joblib` and write reports under
`reports/morocco_hourly_*.csv`. The daily model artifacts and original
historical dataset remain in the project but are not loaded by the dashboard
or used by the Morocco hourly workflow.

The Phase 2 holdout comparison is in
`reports/morocco_hourly_model_comparison.csv`; Phase 3 horizon metrics and
predictions are in `reports/morocco_hourly_recursive_24h_metrics.csv` and
`reports/morocco_hourly_recursive_24h_predictions.csv`. The backtest uses
observed holdout weather, so it does not measure error in future weather
forecasts.

## Run the Morocco dashboard

```bash
streamlit run app.py
```

The dashboard contains four pages:

- **24-hour Zone Forecast**: fetch hourly Open-Meteo weather for Tetouan, show
  peak hours and empirical forecast ranges, and download the weather and zone
  predictions.
- **Hourly History**: explore processed hourly zone readings and data quality.
- **Accuracy & Backtests**: compare predicted and actual holdout readings and
  inspect errors by recursive forecast horizon.
- **Actuals Tracking**: record later readings against saved forecasts and
  monitor errors as actual observations accumulate.

Open-Meteo supplies temperature, humidity, and wind speed. Its shortwave and
diffuse radiation fields are used as proxies for the source's two diffuse-flow
features because the source metadata does not specify their units or exact
measurement definitions. The dashboard discloses this approximation. Live
weather access requires an internet connection.

## Scope and limitations

The source spans 2017-01-01 through 2017-12-30 and is not current demand data.
The forecast is an academic prototype, not an operational grid forecast. The
zone unit semantics and diffuse-flow measurement semantics need confirmation
before treating its outputs as physical energy or a validated utility forecast.
Empirical ranges use a small 2017 holdout with observed weather and exclude
weather forecast errors. New actual readings are stored separately in
`data/observed/morocco_actuals.csv`.

Dataset source: [UCI Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2B).
