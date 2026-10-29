"""
Complete Model Training, Evaluation, and Plot Generation Pipeline
for Smart Grid Energy Load Forecasting Prototype.

Trains multiple models (Baselines, Linear Regression, Random Forest, Gradient Boosting),
fairly compares models with and without weather, runs TimeSeriesSplit cross validation,
generates multi-step recursive forecasts with weather scenarios, and exports
all metrics, predictions, and visualization plots to outputs/.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for headless plot generation
import matplotlib.pyplot as plt

import config
from src.data_loader import load_raw_data, inspect_data_quality
from src.preprocessing import clean_electricity_data, build_feature_pipeline
from src.forecasting import (
    LinearRegressionForecaster,
    RandomForestForecaster,
    GradientBoostingForecaster,
    generate_future_forecast,
    simulate_scenario_comparison,
)
from src.evaluation import split_chronological, evaluate_model_suite, evaluate_timeseries_cv


def run_pipeline():
    print("=" * 70)
    print(" SMART GRID ENERGY LOAD FORECASTING — TRAINING & BENCHMARK PIPELINE")
    print("=" * 70)

    # 1. Load and Audit Dataset
    print("\n[Step 1/6] Loading and auditing raw electricity dataset...")
    raw_df = load_raw_data()
    quality_report = inspect_data_quality(raw_df)
    print(f" -> Raw shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f" -> Date range: {quality_report.get('start_date')} to {quality_report.get('end_date')}")
    print(f" -> Empirical Frequency: {quality_report.get('frequency_label')} ({quality_report.get('frequency_desc')})")
    print(f" -> Zero/Dropout Demand records: {quality_report.get('zero_or_negative_count')}")
    print(f" -> Missing calendar dates: {quality_report.get('missing_calendar_dates_count')}")
    print(f" -> DATE == datetime duplicate: {quality_report.get('date_datetime_identical')}")
    print(f" -> Location: {quality_report.get('locations')}")

    # 2. Clean and Preprocess
    print("\n[Step 2/6] Cleaning dataset and engineering leak-free features...")
    clean_df, clean_meta = clean_electricity_data(raw_df, fill_calendar_gaps=True, impute_implausible=True)
    print(f" -> Cleaned rows: {clean_meta['clean_rows']} (imputed {clean_meta['implausible_count']} implausible values)")

    # Feature pipelines
    df_no_w, feat_no_w, meta_no_w = build_feature_pipeline(clean_df, include_weather=False)
    df_w, feat_w, meta_w = build_feature_pipeline(
        clean_df,
        include_weather=True,
        weather_cols=config.PRIMARY_WEATHER_FEATURES
    )
    print(f" -> Feature set without weather ({len(feat_no_w)} features): {feat_no_w}")
    print(f" -> Feature set with weather ({len(feat_w)} features): {feat_w}")

    # 3. Chronological Train-Test Split (80% train / 20% test)
    print("\n[Step 3/6] Splitting data chronologically (80% train / 20% test)...")
    train_no_w, test_no_w, split_meta = split_chronological(df_no_w, train_ratio=config.DEFAULT_TRAIN_RATIO)
    train_w, test_w, _ = split_chronological(df_w, train_ratio=config.DEFAULT_TRAIN_RATIO)
    print(f" -> Train: {split_meta['train_rows']} days ({split_meta['train_start']} to {split_meta['train_end']})")
    print(f" -> Test:  {split_meta['test_rows']} days ({split_meta['test_start']} to {split_meta['test_end']})")

    # 4. Multi-Model Evaluation Suite
    print("\n[Step 4/6] Training and benchmarking complete model suite...")
    df_metrics, df_predictions, fitted_models = evaluate_model_suite(
        train_no_w=train_no_w,
        test_no_w=test_no_w,
        feat_no_w=feat_no_w,
        train_w=train_w,
        test_w=test_w,
        feat_w=feat_w,
    )

    print("\n" + "=" * 70)
    print(" MODEL PERFORMANCE BENCHMARK (Unobserved Test Period)")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print(f"\nSaved metrics table to: {config.MODEL_COMPARISON_PATH}")
    print(f"Saved test predictions and residuals to: {config.TEST_PREDICTIONS_PATH}")

    # TimeSeriesSplit Cross-Validation
    print("\n[Step 4b] Evaluating 5-Fold TimeSeriesSplit Cross Validation...")
    cv_results = evaluate_timeseries_cv(df_w, feat_w, n_splits=5)
    print(cv_results.to_string(index=False))
    print(f"Mean CV MAE: {cv_results['MAE (MW)'].mean():.2f} MW, Mean RMSE: {cv_results['RMSE (MW)'].mean():.2f} MW")

    # 5. Future Forecasting & Weather Scenarios
    print("\n[Step 5/6] Generating 7-day recursive future forecasts & scenarios...")
    lr_weather_model = fitted_models["lr_weather"]
    test_rmse = float(df_metrics.loc[df_metrics["Model"] == "Linear Regression (Weather-Aware)", "RMSE (MW)"].iloc[0])

    df_future, future_meta = generate_future_forecast(
        model=lr_weather_model,
        feature_cols=feat_w,
        historical_df=clean_df,
        horizon=config.DEFAULT_FORECAST_HORIZON,
        test_rmse=test_rmse,
    )
    df_future.to_csv(config.FUTURE_FORECAST_PATH, index=False)
    print(f"Saved baseline future forecast to: {config.FUTURE_FORECAST_PATH}")

    df_scenario_comp, scenarios_dict = simulate_scenario_comparison(
        model=lr_weather_model,
        feature_cols=feat_w,
        historical_df=clean_df,
        horizon=config.DEFAULT_FORECAST_HORIZON,
        test_rmse=test_rmse,
    )
    df_scenario_comp.to_csv(config.SCENARIO_FORECASTS_PATH, index=False)
    print(f"Saved scenario comparison forecasts to: {config.SCENARIO_FORECASTS_PATH}")
    print("\n7-Day Future Weather Scenarios:")
    print(df_scenario_comp.to_string(index=False))

    # 6. Generate Publication-Quality Visualizations
    print("\n[Step 6/6] Generating evaluation and scenario plots...")
    generate_evaluation_plots(df_predictions, df_metrics, df_scenario_comp)

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def generate_evaluation_plots(df_predictions: pd.DataFrame, df_metrics: pd.DataFrame, df_scenarios: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    dates = pd.to_datetime(df_predictions["DATE"])

    # Plot 1: Actual vs Predicted Demand on Test Set
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df_predictions["ACTUAL_DEMAND"], label="Actual Demand", color="#0f172a", linewidth=2.0)
    plt.plot(dates, df_predictions["PRED_LR_WEATHER"], label="Linear Regression (Weather-Aware)", color="#2563eb", linewidth=1.8, linestyle="--")
    plt.plot(dates, df_predictions["PERSISTENCE_BASELINE"], label="Persistence Baseline", color="#94a3b8", linewidth=1.2, alpha=0.7)
    plt.title("Actual vs Predicted Electricity Demand — Chronological Test Period", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Electricity Demand (MW)", fontsize=11)
    plt.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(config.ACTUAL_VS_PRED_PLOT_PATH, dpi=180)
    plt.close()
    print(f" -> Saved actual vs predicted plot to: {config.ACTUAL_VS_PRED_PLOT_PATH}")

    # Plot 2: Model Performance Comparison Bar Chart (MAE & RMSE)
    plt.figure(figsize=(12, 6))
    models_sorted = df_metrics.sort_values("MAE (MW)")
    x = np.arange(len(models_sorted))
    width = 0.35

    plt.bar(x - width/2, models_sorted["MAE (MW)"], width, label="MAE (MW)", color="#3b82f6", alpha=0.9)
    plt.bar(x + width/2, models_sorted["RMSE (MW)"], width, label="RMSE (MW)", color="#f59e0b", alpha=0.9)

    plt.xticks(x, models_sorted["Model"], rotation=25, ha="right", fontsize=9)
    plt.ylabel("Error (Megawatts)", fontsize=11)
    plt.title("Forecasting Error Comparison Across Models (Lower is Better)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(config.MODEL_COMPARISON_PLOT_PATH, dpi=180)
    plt.close()
    print(f" -> Saved model comparison plot to: {config.MODEL_COMPARISON_PLOT_PATH}")

    # Plot 3: Residual Error Distribution and Timeline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    residuals = df_predictions["RESIDUAL_LR_WEATHER"]
    
    # Histogram of residuals
    ax1.hist(residuals, bins=25, color="#0284c7", edgecolor="white", alpha=0.85)
    ax1.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    ax1.set_title("Residual Error Distribution (LR Weather-Aware)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual (Actual - Predicted MW)", fontsize=10)
    ax1.set_ylabel("Frequency (Days)", fontsize=10)

    # Time series of residuals
    ax2.plot(dates, residuals, color="#0284c7", linewidth=1.2)
    ax2.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    ax2.set_title("Residual Timeline Over Unobserved Test Period", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Residual (MW)", fontsize=10)

    plt.tight_layout()
    plt.savefig(config.RESIDUALS_PLOT_PATH, dpi=180)
    plt.close()
    print(f" -> Saved residuals plot to: {config.RESIDUALS_PLOT_PATH}")

    # Plot 4: Scenario Forecast Trajectory Comparison
    plt.figure(figsize=(10, 5))
    scen_dates = pd.to_datetime(df_scenarios["DATE"])
    plt.plot(scen_dates, df_scenarios["BASELINE_DEMAND_MW"], marker="o", label="Baseline (Trailing Weather)", color="#2563eb", linewidth=2.2)
    plt.plot(scen_dates, df_scenarios["HOTTER_SCENARIO_MW"], marker="s", label="Heatwave (+4°C)", color="#dc2626", linewidth=2.0, linestyle="--")
    plt.plot(scen_dates, df_scenarios["COOLER_SCENARIO_MW"], marker="^", label="Cool Front (-4°C, Rain)", color="#16a34a", linewidth=2.0, linestyle="-.")
    plt.title("7-Day Out-of-Sample Scenario Forecasts (Real Model Invocation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Forecast Date", fontsize=11)
    plt.ylabel("Projected Demand (MW)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(config.SCENARIO_PLOT_PATH, dpi=180)
    plt.close()
    print(f" -> Saved scenario comparison plot to: {config.SCENARIO_PLOT_PATH}")


if __name__ == "__main__":
    run_pipeline()
