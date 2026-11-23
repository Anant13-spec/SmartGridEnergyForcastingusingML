from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TEST_PREDS_PATH = ROOT / "reports" / "test_predictions.csv"
PLOT_TOTAL_PATH = ROOT / "reports" / "actual_vs_predicted.png"
PLOT_ZONES_PATH = ROOT / "reports" / "zone_comparison.png"


def run_evaluation() -> pd.DataFrame:
    if not TEST_PREDS_PATH.exists():
        from train_models import train_and_evaluate
        train_and_evaluate()

    print(f"Reading test predictions from {TEST_PREDS_PATH}...")
    df = pd.read_csv(TEST_PREDS_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    targets = [
        ("Power demand", "Total Power Demand", "Power demand_pred_Linear_Regression_(Ridge)"),
        ("demand_zone_1_mw", "Zone 1 Demand", "demand_zone_1_mw_pred_Linear_Regression_(Ridge)"),
        ("demand_zone_2_mw", "Zone 2 Demand", "demand_zone_2_mw_pred_Linear_Regression_(Ridge)"),
        ("demand_zone_3_mw", "Zone 3 Demand", "demand_zone_3_mw_pred_Linear_Regression_(Ridge)"),
    ]

    metrics_rows = []

    for target_col, name, pred_col in targets:
        actual_col = f"{target_col}_actual"
        if actual_col not in df.columns or pred_col not in df.columns:
            matching_preds = [c for c in df.columns if c.startswith(f"{target_col}_pred_")]
            if matching_preds:
                pred_col = matching_preds[0]
            else:
                continue

        actual = df[actual_col].to_numpy()
        predicted = df[pred_col].to_numpy()

        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2 = float(r2_score(actual, predicted))
        mape = float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1.0))) * 100.0)

        metrics_rows.append({
            "Target": name,
            "Target_Column": target_col,
            "Selected_Model": pred_col.replace(f"{target_col}_pred_", "").replace("_", " "),
            "Test_Records": len(actual),
            "MAE_MW": round(mae, 2),
            "RMSE_MW": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE_percent": round(mape, 2),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    print(metrics_df.to_string(index=False))

    sample_window = df.tail(2016).copy()
    plt.figure(figsize=(14, 6))
    plt.plot(sample_window["datetime"], sample_window["Power demand_actual"], label="Actual Total Demand (MW)", color="#1f77b4", alpha=0.8, linewidth=1.5)
    plt.plot(sample_window["datetime"], sample_window["Power demand_pred_Linear_Regression_(Ridge)"], label="Predicted Demand (Ridge)", color="#ff7f0e", linestyle="--", alpha=0.85, linewidth=1.5)
    plt.title("Total Grid Power Demand: Actual vs Predicted (1-Week Test Window - 5-Min Intervals)", fontsize=14, fontweight="bold")
    plt.xlabel("Datetime", fontsize=11)
    plt.ylabel("Power Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_TOTAL_PATH, dpi=180)
    plt.close()
    print(f"Saved total demand plot to {PLOT_TOTAL_PATH}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    zone_info = [
        ("demand_zone_1_mw", "Zone 1 Demand", "#2ca02c"),
        ("demand_zone_2_mw", "Zone 2 Demand", "#d62728"),
        ("demand_zone_3_mw", "Zone 3 Demand", "#9467bd"),
    ]
    for idx, (z_col, z_name, color) in enumerate(zone_info):
        ax = axes[idx]
        actual = sample_window[f"{z_col}_actual"]
        pred = sample_window[f"{z_col}_pred_Linear_Regression_(Ridge)"]
        ax.plot(sample_window["datetime"], actual, label=f"Actual {z_name}", color=color, alpha=0.8, linewidth=1.3)
        ax.plot(sample_window["datetime"], pred, label=f"Predicted {z_name}", color="black", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.set_ylabel("Demand (MW)", fontsize=10)
        ax.set_title(z_name, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    plt.xlabel("Datetime", fontsize=11)
    plt.suptitle("Three-Zone Electricity Demand: Actual vs Predicted (5-Min Observations)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_ZONES_PATH, dpi=180)
    plt.close()
    print(f"Saved 3-zone comparison plot to {PLOT_ZONES_PATH}")

    return metrics_df


if __name__ == "__main__":
    run_evaluation()
