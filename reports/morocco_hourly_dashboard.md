# Morocco hourly forecast dashboard: Phase 4

The dashboard fixes the forecast location to Tetouan, requests hourly weather,
and recursively forecasts the three zone readings. It shows peak-hour
summaries, empirical error bands, and a downloadable table. Separate tabs show
hourly history, holdout accuracy by zone and horizon, and manual actual-reading
entry matched against saved forecasts.

Weather inputs are mapped as follows:

| Model input | Open-Meteo hourly field | Note |
|---|---|---|
| `Temperature` | `temperature_2m` | °C |
| `Humidity` | `relative_humidity_2m` | % |
| `WindSpeed` | `wind_speed_10m` | Requested in km/h |
| `GeneralDiffuseFlows` | `shortwave_radiation` | Solar-radiation proxy |
| `DiffuseFlows` | `diffuse_radiation` | Solar-radiation proxy |

The official source page calls these last two dataset fields “general diffuse
flows” and “diffuse flows” but does not state their units. The supplied
dictionary describes them in unclear terms. Open-Meteo radiation values are
therefore disclosed as proxies in the app; this feature mapping may differ
from the original measurement process. Open-Meteo documents shortwave and
diffuse radiation as hourly solar-radiation fields and supports requesting
wind speed in km/h. See [Open-Meteo forecast API documentation](https://open-meteo.com/en/docs)
and the [UCI dataset page](https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2B).

The forecasts are experimental. Demand training data ends in 2017, zone
readings are hourly averages rather than verified hourly energy. Empirical
90% bands use 72 recursive holdout errors per zone and forecast horizon with
observed weather; they exclude weather-forecast error and are not guaranteed
coverage. Future weather may differ from what occurs. The combined value is
the sum of the three predicted zone readings for each hour; it is not a
measured hourly energy total. Saved forecasts and manual actual readings are
stored separately from source and processed data.
