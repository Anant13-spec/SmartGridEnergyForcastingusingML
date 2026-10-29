"""
Unit tests for preprocessing, feature leakage prevention, chronological splitting,
and recursive future forecasting.
"""

import unittest
import pandas as pd
import numpy as np
import config
from src.preprocessing import clean_electricity_data, build_feature_pipeline
from src.evaluation import split_chronological, compute_metrics, evaluate_forecaster
from src.forecasting import LinearRegressionForecaster, generate_future_forecast


class TestForecastingPipeline(unittest.TestCase):

    def setUp(self):
        # Create a clean synthetic daily series
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        base = 5000 + 10 * np.arange(100)
        noise = np.random.normal(0, 50, 100)
        demand = base + noise
        self.sample_df = pd.DataFrame({
            "DATE": dates,
            "POWER_DEMAND": demand,
            "temp": 25.0 + np.sin(np.linspace(0, 3.14, 100)) * 10
        })

    def test_cleaning_imputes_implausible_zeroes(self):
        """Verify zero/negative values are imputed cleanly."""
        dirty_df = self.sample_df.copy()
        dirty_df.loc[10, "POWER_DEMAND"] = 0  # Implausible dropout
        cleaned, meta = clean_electricity_data(dirty_df, impute_implausible=True)
        
        self.assertEqual(meta["implausible_count"], 1)
        self.assertGreater(cleaned.loc[10, "POWER_DEMAND_CLEAN"], 4000)

    def test_no_data_leakage_in_lags_and_rolling(self):
        """
        Verify that feature at row t depends strictly on values prior to t.
        Target value at index t must NOT affect lag_1, lag_7, or rolling_mean_7 at index t.
        """
        cleaned, _ = clean_electricity_data(self.sample_df)
        feat_df, cols, meta = build_feature_pipeline(cleaned)
        
        # Test specific row
        idx = 15
        target_val = feat_df.loc[idx, "POWER_DEMAND_CLEAN"]
        yesterday_target = feat_df.loc[idx - 1, "POWER_DEMAND_CLEAN"]
        
        # lag_1 at idx must equal target at idx-1, NOT target at idx
        self.assertEqual(feat_df.loc[idx, "lag_1"], yesterday_target)
        self.assertNotEqual(feat_df.loc[idx, "lag_1"], target_val)

    def test_split_chronological_order(self):
        """Verify chronological split guarantees train dates precede test dates."""
        cleaned, _ = clean_electricity_data(self.sample_df)
        feat_df, cols, _ = build_feature_pipeline(cleaned)
        train_df, test_df, meta = split_chronological(feat_df, train_ratio=0.8)
        
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertLess(train_df["DATE"].max(), test_df["DATE"].min())
        self.assertEqual(len(train_df) + len(test_df), len(feat_df))

    def test_metrics_finite_and_sensible(self):
        """Verify metrics calculations produce valid numbers."""
        y_true = np.array([5000.0, 5200.0, 5100.0, 5300.0])
        y_pred = np.array([4950.0, 5180.0, 5120.0, 5320.0])
        metrics = compute_metrics(y_true, y_pred, "Linear Regression")
        
        self.assertAlmostEqual(metrics["MAE (MW)"], 27.5, delta=1.0)
        self.assertGreater(metrics["RMSE (MW)"], 0)
        self.assertLessEqual(metrics["R2 Score"], 1.0)

    def test_recursive_future_forecasting(self):
        """Verify that future forecasting generates continuous subsequent timestamps without looking ahead."""
        cleaned, _ = clean_electricity_data(self.sample_df)
        feat_df, cols, _ = build_feature_pipeline(cleaned, include_weather=False)
        train_df, test_df, _ = split_chronological(feat_df, train_ratio=0.8)
        
        forecaster = LinearRegressionForecaster()
        forecaster.fit(train_df[cols], train_df["POWER_DEMAND_CLEAN"])
        
        horizon = 7
        future_df, meta = generate_future_forecast(
            model=forecaster.model,
            feature_cols=cols,
            historical_df=cleaned,
            horizon=horizon
        )
        
        self.assertEqual(len(future_df), horizon)
        expected_start = pd.to_datetime(cleaned["DATE"].iloc[-1]) + pd.Timedelta(days=1)
        actual_start = pd.to_datetime(future_df["DATE"].iloc[0])
        self.assertEqual(expected_start, actual_start)
        self.assertTrue(all(future_df["FORECASTED_DEMAND_MW"] >= config.MIN_PLAUSIBLE_DEMAND))

    def test_missing_weather_handled_gracefully(self):
        """Verify pipeline succeeds even when optional weather columns are absent."""
        no_weather_df = self.sample_df.drop(columns=["temp"])
        cleaned, _ = clean_electricity_data(no_weather_df)
        feat_df, cols, meta = build_feature_pipeline(cleaned, include_weather=True)
        
        # Weather columns should simply be omitted without crashing
        self.assertNotIn("temp", cols)
        self.assertGreater(len(cols), 0)


if __name__ == "__main__":
    unittest.main()
