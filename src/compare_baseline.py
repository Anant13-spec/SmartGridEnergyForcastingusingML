
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
pred_path = ROOT / "reports" / "rolling_forecast_predictions.csv"
data_path = ROOT / "df_final.csv"

pred = pd.read_csv(pred_path, parse_dates=["DATE"])
df = pd.read_csv(data_path, parse_dates=["DATE"])

df = (
    df.sort_values("DATE")
    .drop_duplicates("DATE")
    .set_index("DATE")
)

pred = pred.merge(
    df[["POWER_DEMAND"]].rename(
        columns={"POWER_DEMAND": "actual_from_source"}
    ),
    left_on="DATE",
    right_index=True,
    how="left",
)

pred["previous_day_demand"] = pred["DATE"].map(
    df["POWER_DEMAND"].shift(1)
)

# Exclude records with zero or missing actual demand.
# Also require a valid previous-day demand for the baseline.
valid = (
    pred["actual_demand"].gt(0)
    & pred["actual_from_source"].gt(0)
    & pred["previous_day_demand"].gt(0)
    & pred["predicted_demand"].notna()
)

scored = pred.loc[valid].copy()

if scored.empty:
    raise ValueError("No valid observations available for comparison.")

for name, values in [
    ("Weather-aware recursive model", scored["predicted_demand"]),
    ("Previous-day baseline", scored["previous_day_demand"]),
]:
    mae = mean_absolute_error(scored["actual_demand"], values)
    rmse = np.sqrt(
        mean_squared_error(scored["actual_demand"], values)
    )
    print(f"{name}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

print(f"\nMatched evaluation records: {len(scored)}")
print("Comparison saved nowhere; original files are unchanged.")
