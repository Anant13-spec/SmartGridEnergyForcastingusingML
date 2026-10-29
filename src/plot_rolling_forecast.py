
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports" / "rolling_forecast_predictions.csv"
OUTPUT = ROOT / "reports" / "rolling_forecast_graph.png"

df = pd.read_csv(INPUT, parse_dates=["DATE"])
df = df.sort_values("DATE")

plt.figure(figsize=(14, 6))
plt.plot(
    df["DATE"], df["actual_demand"],
    label="Actual demand", linewidth=1.6
)
plt.plot(
    df["DATE"], df["predicted_demand"],
    label="Predicted demand", linewidth=1.6
)

plt.title("Rolling Seven-Day Electricity Demand Forecast")
plt.xlabel("Date")
plt.ylabel("POWER_DEMAND")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=160)
plt.close()

print(f"Graph saved to: {OUTPUT}")
