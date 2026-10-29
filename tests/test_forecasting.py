"""
Unit tests for preprocessing, feature leakage prevention, chronological splitting,
multi-model benchmarking, recursive forecasting, and weather scenario simulation.
"""

import unittest
import pandas as pd
import numpy as np
import config
from src.preprocessing import clean_electricity_data, build_feature_pipeline
from src.evaluation import split_chronological, compute_metrics, evaluate_model_suite
from src.forecasting import (
    LinearRegressionForecaster,
    RandomForestForecaster,
    GradientBoostingForecaster,
    generate_future_forecast,
    simulate_scenario_comparison,
)


class TestForecastingPipeline(unittest.TestCase):

    def setUp(self):
        # Create a clean synthetic daily series
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(42)
        base = 5000 + 5 * np.arange(120)
        noise = np.random.normal(0, 30, 120)
        demand = base + noise
        self.sample_df = pd.DataFrame({
            "DATE": dates,
            "POWER_DEMAND": demand,
            "temp": 25.0 + np.sin(np.linspace(0, 3.14, 120)) * 10,
            "humidity": 60.0 + np.cos(np.linspace(0, 3.14, 120)) * 15,
            "precip": np.random.choice([0.0, 5.0, 15.0], size=120, p=[0.8, 0.15, 0.05]),
            "windspeed": 15.0 + np.random.normal(0, 3, 120),
            "solarradiation": 200.0 + np.random.normal(0, 20, 120),
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
        Target value at index t must NOT affect lag_1, lag_7 at index t.
        """
        cleaned, _ = clean_electricity_data(self.sample_df)
        feat_df, cols, meta = build_feature_pipeline(cleaned)
        
        idx = 15
        target_val = feat_df.loc[idx, "POWER_DEMAND_CLEAN"]
        yesterday_target = feat_df.loc[idx - 1, "POWER_DEMAND_CLEAN"]
        
        # lag_1 at idx must equal target at idx-1, NOT target at idx
        self.assertEqual(feat_df.loc[idx, "lag_1"], yesterday_target)
        self.assertNotEqual(feat_df.loc[idx, "lag_1"], target_val)

    def test_split_chronological_order(self):
        """Verify chronological split guarantees train dates strictly precede test dates."""
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
        """Verify future forecasting generates continuous subsequent timestamps without lookahead."""
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

    def test_multi_model_suite_evaluation(self):
        """Verify complete model benchmark suite runs and computes metrics for all 8 configurations."""
        cleaned, _ = clean_electricity_data(self.sample_df)
        df_no_w, feat_no_w, _ = build_feature_pipeline(cleaned, include_weather=False)
        df_w, feat_w, _ = build_feature_pipeline(cleaned, include_weather=True, weather_cols=config.PRIMARY_WEATHER_FEATURES)
        
        tr_no_w, te_no_w, _ = split_chronological(df_no_w, train_ratio=0.8)
        tr_w, te_w, _ = split_chronological(df_w, train_ratio=0.8)
        
        metrics_df, preds_df, fitted_models = evaluate_model_suite(
            train_no_w=tr_no_w,
            test_no_w=te_no_w,
            feat_no_w=feat_no_w,
            train_w=tr_w,
            test_w=te_w,
            feat_w=feat_w,
        )
        
        self.assertEqual(len(metrics_df), 8)
        self.assertIn("Linear Regression (Weather-Aware)", metrics_df["Model"].values)
        self.assertIn("Random Forest (Weather-Aware)", metrics_df["Model"].values)
        self.assertIn("Gradient Boosting (Weather-Aware)", metrics_df["Model"].values)
        self.assertTrue(all(metrics_df["MAE (MW)"] > 0))
        self.assertTrue(all(metrics_df["RMSE (MW)"] > 0))

    def test_scenario_forecasting_sensitivity(self):
        """Verify scenario forecast responds dynamically when weather parameters are modified."""
        cleaned, _ = clean_electricity_data(self.sample_df)
        df_w, feat_w, _ = build_feature_pipeline(cleaned, include_weather=True, weather_cols=config.PRIMARY_WEATHER_FEATURES)
        tr_w, te_w, _ = split_chronological(df_w, train_ratio=0.8)
        
        model = LinearRegressionForecaster().fit(tr_w[feat_w], tr_w["POWER_DEMAND_CLEAN"])
        
        df_comparison, scenarios_dict = simulate_scenario_comparison(
            model=model,
            feature_cols=feat_w,
            historical_df=cleaned,
            horizon=7,
        )
        
        self.assertEqual(len(df_comparison), 7)
        self.assertIn("BASELINE_DEMAND_MW", df_comparison.columns)
        self.assertIn("HOTTER_SCENARIO_MW", df_comparison.columns)
        self.assertIn("COOLER_SCENARIO_MW", df_comparison.columns)
        self.assertIn("HOTTER_DELTA_MW", df_comparison.columns)
        self.assertIn("COOLER_DELTA_MW", df_comparison.columns)
        
        # Verify delta values are computed
        self.assertTrue((df_comparison["HOTTER_DELTA_MW"] != 0).any())

    def test_missing_weather_handled_gracefully(self):
        """Verify pipeline succeeds even when weather columns are absent."""
        no_weather_df = self.sample_df.drop(columns=["temp", "humidity", "precip", "windspeed", "solarradiation"])
        cleaned, _ = clean_electricity_data(no_weather_df)
        feat_df, cols, meta = build_feature_pipeline(cleaned, include_weather=True)
        
        # Weather columns omitted gracefully without crashing
        self.assertNotIn("temp", cols)
        self.assertGreater(len(cols), 0)

    def test_post_cutoff_forecast_origin(self):
        """Verify out-of-sample projections strictly begin immediately following the dataset cutoff date."""
        from src.data_loader import load_raw_data
        raw_df = load_raw_data()
        cleaned, _ = clean_electricity_data(raw_df)
        df_w, feat_w, _ = build_feature_pipeline(cleaned, include_weather=True, weather_cols=config.PRIMARY_WEATHER_FEATURES)
        tr_w, te_w, _ = split_chronological(df_w, train_ratio=0.8)
        
        model = LinearRegressionForecaster().fit(tr_w[feat_w], tr_w["POWER_DEMAND_CLEAN"])
        
        horizon = 7
        fc_df, meta = generate_future_forecast(
            model=model,
            feature_cols=feat_w,
            historical_df=cleaned,
            horizon=horizon,
        )
        
        last_date = pd.to_datetime(cleaned["DATE"].iloc[-1])
        first_fc_date = pd.to_datetime(fc_df["DATE"].iloc[0])
        
        # Must start exactly 1 day after dataset ends
        self.assertEqual(first_fc_date, last_date + pd.Timedelta(days=1))
        self.assertEqual(len(fc_df), horizon)
        self.assertEqual(meta["start_date"], "2024-09-19")

    def test_scenario_independent_invocation(self):
        """Verify Baseline, Heatwave, and Cool Front scenarios update predictions independently with correct signs."""
        from src.data_loader import load_raw_data
        raw_df = load_raw_data()
        cleaned, _ = clean_electricity_data(raw_df)
        df_w, feat_w, _ = build_feature_pipeline(cleaned, include_weather=True, weather_cols=config.PRIMARY_WEATHER_FEATURES)
        tr_w, te_w, _ = split_chronological(df_w, train_ratio=0.8)
        model = LinearRegressionForecaster().fit(tr_w[feat_w], tr_w["POWER_DEMAND_CLEAN"])
        
        base_weather = cleaned[config.PRIMARY_WEATHER_FEATURES].tail(7).mean().to_dict()
        
        # 1. Baseline
        fc_base, _ = generate_future_forecast(model, feat_w, cleaned, weather_scenario=base_weather)
        
        # 2. Heatwave (+4 C)
        hot_weather = base_weather.copy()
        hot_weather["temp"] += 4.0
        hot_weather["humidity"] -= 5.0
        hot_weather["precip"] = 0.0
        hot_weather["solarradiation"] += 40.0
        fc_hot, _ = generate_future_forecast(model, feat_w, cleaned, weather_scenario=hot_weather)
        
        # 3. Cool Front (-4 C, 25mm rain)
        cool_weather = base_weather.copy()
        cool_weather["temp"] -= 4.0
        cool_weather["humidity"] += 15.0
        cool_weather["precip"] = 25.0
        cool_weather["solarradiation"] -= 50.0
        cool_weather["windspeed"] += 4.0
        fc_cool, _ = generate_future_forecast(model, feat_w, cleaned, weather_scenario=cool_weather)
        
        # Verify Heatwave increases demand for every day
        for i in range(len(fc_base)):
            self.assertGreater(fc_hot.loc[i, "FORECASTED_DEMAND_MW"], fc_base.loc[i, "FORECASTED_DEMAND_MW"])
            self.assertLess(fc_cool.loc[i, "FORECASTED_DEMAND_MW"], fc_base.loc[i, "FORECASTED_DEMAND_MW"])
        
        # Verify 7-day average deltas
        delta_hot = fc_hot["FORECASTED_DEMAND_MW"].mean() - fc_base["FORECASTED_DEMAND_MW"].mean()
        delta_cool = fc_cool["FORECASTED_DEMAND_MW"].mean() - fc_base["FORECASTED_DEMAND_MW"].mean()
        self.assertGreater(delta_hot, 100.0)
        self.assertLess(delta_cool, -100.0)


if __name__ == "__main__":
    unittest.main()

