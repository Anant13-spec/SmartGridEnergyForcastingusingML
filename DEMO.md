# Five-minute GridWise demo · Morocco hourly forecasting

1. Start `streamlit run app.py`. The app opens on the Morocco hourly workflow.
2. Show the coverage at the top: 8,736 hourly rows from 2017-01-01 through
   2017-12-30. Explain that zone values are hourly means of the source records,
   not confirmed hourly energy totals.
3. In **24-hour Zone Forecast**, click **Forecast next 24 hours**. Show the
   per-zone peak hours, 90% empirical error bands, all weather inputs, and the
   downloadable table. Open-Meteo radiation fields are disclosed proxies for
   the source diffuse-flow fields.
4. In **Hourly History**, select a shorter date range and inspect the hourly
   zone patterns and six-reading-per-hour quality check.
5. In **Accuracy & Backtests**, compare one-step predictions to the holdout and
   inspect how recursive error changes by forecast horizon. Explain that the
   recursive backtest uses observed holdout weather.
6. In **Actuals Tracking**, enter a measured reading for a forecasted hour to
   show live forecast error monitoring. The entered observations are stored in
   a separate file.
7. Close with the limits: demand history is from 2017 only, zone units are
   ambiguous, empirical ranges use a small same-year holdout, and forecasts
   are not operational utility forecasts.
