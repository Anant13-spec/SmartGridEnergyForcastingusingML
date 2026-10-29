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


def evaluate_model_suite(
    train_no_w: pd.DataFrame,
    test_no_w: pd.DataFrame,
    feat_no_w: List[str],
    train_w: pd.DataFrame,
    test_w: pd.DataFrame,
    feat_w: List[str],
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    date_col: str = config.DEFAULT_DATE_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train and fairly compare the complete model suite on identical chronological test periods:
    1. Persistence Baseline (Lag 1)
    2. Seasonal-Naive Baseline (Lag 7)
    3. Linear Regression (Demand + Calendar)
    4. Linear Regression (Weather-Aware)
    5. Random Forest (Demand + Calendar)
    6. Random Forest (Weather-Aware)
    7. Gradient Boosting (Demand + Calendar)
    8. Gradient Boosting (Weather-Aware)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

    y_train_no_w = train_no_w[target_col].values
    y_test = test_no_w[target_col].values
    y_train_w = train_w[target_col].values

    # 1. Baselines
    pred_persist = test_no_w["lag_1"].values
    m_persist = compute_metrics(y_test, pred_persist, "Baseline (Persistence / Lag 1)")
    m_persist["Features"] = "Historical Lag 1"
    m_persist["Weather Aware"] = "No"

    pred_snaive = test_no_w["lag_7"].values
    m_snaive = compute_metrics(y_test, pred_snaive, "Baseline (Seasonal-Naive / Lag 7)")
    m_snaive["Features"] = "Historical Lag 7"
    m_snaive["Weather Aware"] = "No"

    # 2. Linear Regression (No Weather vs Weather-Aware)
    lr_no_w = LinearRegression(fit_intercept=False).fit(train_no_w[feat_no_w], y_train_no_w)
    pred_lr_no_w = lr_no_w.predict(test_no_w[feat_no_w])
    m_lr_no_w = compute_metrics(y_test, pred_lr_no_w, "Linear Regression (Demand + Calendar)")
    m_lr_no_w["Features"] = "Calendar + Lags"
    m_lr_no_w["Weather Aware"] = "No"

    lr_w = LinearRegression(fit_intercept=False).fit(train_w[feat_w], y_train_w)
    pred_lr_w = lr_w.predict(test_w[feat_w])
    m_lr_w = compute_metrics(y_test, pred_lr_w, "Linear Regression (Weather-Aware)")
    m_lr_w["Features"] = "Calendar + Lags + Weather"
    m_lr_w["Weather Aware"] = "Yes"

    # 3. Random Forest (No Weather vs Weather-Aware)
    rf_no_w = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(train_no_w[feat_no_w], y_train_no_w)
    pred_rf_no_w = rf_no_w.predict(test_no_w[feat_no_w])
    m_rf_no_w = compute_metrics(y_test, pred_rf_no_w, "Random Forest (Demand + Calendar)")
    m_rf_no_w["Features"] = "Calendar + Lags"
    m_rf_no_w["Weather Aware"] = "No"

    rf_w = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(train_w[feat_w], y_train_w)
    pred_rf_w = rf_w.predict(test_w[feat_w])
    m_rf_w = compute_metrics(y_test, pred_rf_w, "Random Forest (Weather-Aware)")
    m_rf_w["Features"] = "Calendar + Lags + Weather"
    m_rf_w["Weather Aware"] = "Yes"

    # 4. Gradient Boosting (No Weather vs Weather-Aware)
    gbr_no_w = HistGradientBoostingRegressor(max_iter=150, max_depth=5, min_samples_leaf=5, random_state=42).fit(train_no_w[feat_no_w], y_train_no_w)
    pred_gbr_no_w = gbr_no_w.predict(test_no_w[feat_no_w])
    m_gbr_no_w = compute_metrics(y_test, pred_gbr_no_w, "Gradient Boosting (Demand + Calendar)")
    m_gbr_no_w["Features"] = "Calendar + Lags"
    m_gbr_no_w["Weather Aware"] = "No"

    gbr_w = HistGradientBoostingRegressor(max_iter=150, max_depth=5, min_samples_leaf=5, random_state=42).fit(train_w[feat_w], y_train_w)
    pred_gbr_w = gbr_w.predict(test_w[feat_w])
    m_gbr_w = compute_metrics(y_test, pred_gbr_w, "Gradient Boosting (Weather-Aware)")
    m_gbr_w["Features"] = "Calendar + Lags + Weather"
    m_gbr_w["Weather Aware"] = "Yes"

    metrics_list = [
        m_persist,
        m_snaive,
        m_lr_no_w,
        m_lr_w,
        m_rf_no_w,
        m_rf_w,
        m_gbr_no_w,
        m_gbr_w,
    ]
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    # Detailed test predictions and residuals
    df_predictions = pd.DataFrame({
        "DATE": test_w[date_col].values,
        "ACTUAL_DEMAND": np.round(y_test, 2),
        "PRED_LR_WEATHER": np.round(pred_lr_w, 2),
        "PRED_LR_NO_WEATHER": np.round(pred_lr_no_w, 2),
        "PRED_RF_WEATHER": np.round(pred_rf_w, 2),
        "PRED_GBR_WEATHER": np.round(pred_gbr_w, 2),
        "PERSISTENCE_BASELINE": np.round(pred_persist, 2),
        "SEASONAL_BASELINE": np.round(pred_snaive, 2),
        "RESIDUAL_LR_WEATHER": np.round(y_test - pred_lr_w, 2),
        "RESIDUAL_LR_NO_WEATHER": np.round(y_test - pred_lr_no_w, 2),
    })
    df_predictions.to_csv(config.TEST_PREDICTIONS_PATH, index=False)

    fitted_models = {
        "lr_weather": lr_w,
        "lr_no_weather": lr_no_w,
        "rf_weather": rf_w,
        "rf_no_weather": rf_no_w,
        "gbr_weather": gbr_w,
        "gbr_no_weather": gbr_no_w,
    }

    return df_metrics, df_predictions, fitted_models


def evaluate_timeseries_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = f"{config.DEFAULT_TARGET_COL}_CLEAN",
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform expanding-window TimeSeriesSplit cross validation.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LinearRegression

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    
    fold = 1
    for train_idx, val_idx in tscv.split(df):
        tr = df.iloc[train_idx]
        va = df.iloc[val_idx]
        
        m = LinearRegression(fit_intercept=False).fit(tr[feature_cols], tr[target_col])
        preds = m.predict(va[feature_cols])
        
        mae = mean_absolute_error(va[target_col], preds)
        rmse = float(np.sqrt(mean_squared_error(va[target_col], preds)))
        r2 = r2_score(va[target_col], preds)
        
        records.append({
            "Fold": fold,
            "Train Rows": len(tr),
            "Val Rows": len(va),
            "Val Start": str(va["DATE"].iloc[0].date() if hasattr(va["DATE"].iloc[0], "date") else va["DATE"].iloc[0]),
            "Val End": str(va["DATE"].iloc[-1].date() if hasattr(va["DATE"].iloc[-1], "date") else va["DATE"].iloc[-1]),
            "MAE (MW)": round(mae, 2),
            "RMSE (MW)": round(rmse, 2),
            "R2 Score": round(r2, 4),
        })
        fold += 1

    return pd.DataFrame(records)


# Backward compatibility alias
evaluate_all_models = evaluate_forecaster

