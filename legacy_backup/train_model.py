
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("df_final.csv")
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

df = (
    df.dropna(subset=["DATE", "POWER_DEMAND"])
      .sort_values("DATE")
      .drop_duplicates("DATE")
      .reset_index(drop=True)
)

# Calendar features
df["day_of_year"] = df["DATE"].dt.dayofyear
df["day_of_week"] = df["DATE"].dt.dayofweek
df["month"] = df["DATE"].dt.month

# Previous day's observed demand
df["previous_day_demand"] = df["POWER_DEMAND"].shift(1)

df = df.dropna().reset_index(drop=True)

features = [
    "day_of_year",
    "day_of_week",
    "month",
    "previous_day_demand",
]

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

model = LinearRegression()
model.fit(train[features], train["POWER_DEMAND"])

predictions = model.predict(test[features])

mae = mean_absolute_error(test["POWER_DEMAND"], predictions)
rmse = np.sqrt(
    mean_squared_error(test["POWER_DEMAND"], predictions)
)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

results = pd.DataFrame({
    "DATE": test["DATE"].values,
    "ACTUAL_DEMAND": test["POWER_DEMAND"].values,
    "PREDICTED_DEMAND": predictions,
})

print("\nFirst 10 actual vs predicted values:")
print(results.head(10).round(2).to_string(index=False))

results.to_csv("test_predictions.csv", index=False)
print("\nSaved test_predictions.csv")

