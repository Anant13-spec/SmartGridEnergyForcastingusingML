
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "df_final.csv")
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df["POWER_DEMAND"] = pd.to_numeric(
    df["POWER_DEMAND"], errors="coerce"
)

df = (
    df.dropna(subset=["DATE", "POWER_DEMAND"])
    .sort_values("DATE")
    .drop_duplicates("DATE")
    .reset_index(drop=True)
)

demand = df["POWER_DEMAND"].where(df["POWER_DEMAND"] > 0).ffill()

df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month
df["day_of_year"] = df["DATE"].dt.dayofyear
df["day_of_week"] = df["DATE"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["demand_lag_1"] = demand.shift(1)
df["demand_lag_7"] = demand.shift(7)
df["demand_mean_7"] = demand.shift(1).rolling(7).mean()

artifact = joblib.load(ROOT / "models" / "load_forecaster.joblib")
model = artifact["model"]
features = artifact["features"]

# Match the chronological split used during training.
split = int(len(df) * 0.8)
test = df.iloc[split:].copy()

valid = test["POWER_DEMAND"].gt(0) & test[features].notna().any(axis=1)
test = test.loc[valid].copy()

X = test[features]
actual = test["POWER_DEMAND"].to_numpy()
predicted = model.predict(X)

test["predicted_demand"] = predicted
test["absolute_error"] = np.abs(actual - predicted)

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"Test records: {len(test)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("\nSample predictions:")
print(
    test[["DATE", "POWER_DEMAND", "predicted_demand", "absolute_error"]]
    .head(15)
    .round(2)
    .to_string(index=False)
)

test.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(test["DATE"], actual, label="Actual demand")
plt.plot(test["DATE"], predicted, label="Predicted demand")
plt.xlabel("Date")
plt.ylabel("POWER_DEMAND")
plt.title("Actual vs Predicted Electricity Demand")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "reports" / "actual_vs_predicted.png", dpi=150)
print("\nSaved graph: reports/actual_vs_predicted.png")
