"""
Evaluation Module for Time-Series Forecasting.
Handles strict chronological splitting, metric calculations (MAE, RMSE, R2),
baseline benchmarking (Persistence and 7-day Seasonal-Naive), and prediction export.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def split_chronological(
    df: pd.DataFrame,
    date_col: str = config.DEFAULT_DATE_COL,
    train_ratio: float = config.DEFAULT_TRAIN_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform strict chronological train/test split.
    Guarantees train dates strictly precede test dates to prevent future-data leakage.
    """
    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    n = len(sorted_df)
    split_idx = int(n * train_ratio)

    train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    metadata = {
        "total_rows": n,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_ratio": train_ratio,
        "train_start": str(train_df[date_col].iloc[0].date() if hasattr(train_df[date_col].iloc[0], "date") else train_df[date_col].iloc[0]),
        "train_end": str(train_df[date_col].iloc[-1].date() if hasattr(train_df[date_col].iloc[-1], "date") else train_df[date_col].iloc[-1]),
        "test_start": str(test_df[date_col].iloc[0].date() if hasattr(test_df[date_col].iloc[0], "date") else test_df[date_col].iloc[0]),
        "test_end": str(test_df[date_col].iloc[-1].date() if hasattr(test_df[date_col].iloc[-1], "date") else test_df[date_col].iloc[-1]),
    }

    return train_df, test_df, metadata


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression"
) -> Dict[str, Any]:
    """
    Calculate MAE, RMSE, and R2 scores.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "Model": model_name,
        "MAE (MW)": round(mae, 2),
        "RMSE (MW)": round(rmse, 2),
        "R2 Score": round(r2, 4),
    }


def evaluate_forecaster(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model: Any,
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    date_col: str = config.DEFAULT_DATE_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate Linear Regression against:
    1. Persistence Baseline (Lag 1: y_t = y_{t-1})
    2. Seasonal-Naive Baseline (Lag 7: y_t = y_{t-7})
    """
    y_test = test_df[target_col].values

    # 1. Persistence Baseline (y_t = y_{t-1})
    if "lag_1" in test_df.columns:
        y_pred_lag1 = test_df["lag_1"].values
    else:
        y_pred_lag1 = np.roll(y_test, 1)
        y_pred_lag1[0] = train_df[target_col].iloc[-1]
    m_lag1 = compute_metrics(y_test, y_pred_lag1, "Baseline (Persistence / Lag 1)")

    # 2. 7-Day Seasonal-Naive Baseline (y_t = y_{t-7})
    if "lag_7" in test_df.columns:
        y_pred_lag7 = test_df["lag_7"].values
    else:
        y_pred_lag7 = np.roll(y_test, 7)
        y_pred_lag7[:7] = train_df[target_col].iloc[-7:]
    m_lag7 = compute_metrics(y_test, y_pred_lag7, "Baseline (Seasonal-Naive / Lag 7)")

    # 3. Linear Regression Model Prediction
    model_to_use = model.model if hasattr(model, "model") else model
    X_test = test_df[feature_cols]
    y_pred_lr = model_to_use.predict(X_test)
    m_lr = compute_metrics(y_test, y_pred_lr, "Linear Regression")

    df_metrics = pd.DataFrame([m_lag1, m_lag7, m_lr])

    # Table of actual vs predicted values
    df_predictions = pd.DataFrame({
        "DATE": test_df[date_col].values,
        "ACTUAL_DEMAND": np.round(y_test, 2),
        "PREDICTED_DEMAND": np.round(y_pred_lr, 2),
        "PERSISTENCE_BASELINE": np.round(y_pred_lag1, 2),
        "SEASONAL_BASELINE": np.round(y_pred_lag7, 2),
        "ERROR": np.round(y_test - y_pred_lr, 2),
    })

    # Save to outputs
    df_predictions.to_csv(config.TEST_PREDICTIONS_PATH, index=False)

    return df_metrics, df_predictions


# Backward compatibility alias
evaluate_all_models = evaluate_forecaster
