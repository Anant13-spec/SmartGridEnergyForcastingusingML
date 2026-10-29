"""
Unit tests for data loader, column detection, quality validation,
and weather data alignment.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.data_loader import (
    load_raw_data,
    detect_columns,
    detect_frequency,
    inspect_data_quality,
)


class TestDataLoader(unittest.TestCase):

    def test_load_raw_data_exists(self):
        """Verify that df_final.csv loads without error and preserves original rows."""
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 800)
        self.assertIn("DATE", df.columns)
        self.assertIn("POWER_DEMAND", df.columns)

    def test_detect_columns(self):
        """Test column detection heuristic and fallback."""
        df_sample = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "grid_load_mw": [5000, 5200]
        })
        d_col, t_col = detect_columns(df_sample)
        self.assertEqual(d_col, "timestamp")
        self.assertEqual(t_col, "grid_load_mw")

    def test_detect_frequency_daily(self):
        """Test daily frequency detection."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        label, desc, delta = detect_frequency(pd.Series(dates))
        self.assertEqual(label, "Daily")
        self.assertEqual(delta, pd.Timedelta(days=1))

    def test_detect_frequency_hourly(self):
        """Test hourly frequency detection."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        label, desc, delta = detect_frequency(pd.Series(dates))
        self.assertEqual(label, "Hourly")
        self.assertEqual(delta, pd.Timedelta(hours=1))

    def test_inspect_data_quality_report(self):
        """Test data quality report structure, key metrics, and calendar checks."""
        df = load_raw_data()
        report = inspect_data_quality(df)
        
        self.assertTrue(report["date_present"])
        self.assertTrue(report["target_present"])
        self.assertEqual(report["frequency_label"], "Daily")
        self.assertIn("target_summary", report)
        self.assertGreaterEqual(report["zero_or_negative_count"], 0)
        self.assertIn("missing_calendar_dates", report)
        self.assertFalse(report["meets_hourly_requirement"])  # Discloses daily limitation

    def test_date_vs_datetime_duplicate_check(self):
        """Verify audit correctly flags DATE and datetime as duplicate fields."""
        df = load_raw_data()
        report = inspect_data_quality(df)
        self.assertIn("date_datetime_identical", report)
        self.assertTrue(report["date_datetime_identical"])

    def test_weather_data_alignment(self):
        """Verify that weather records align with target demand location (Delhi)."""
        df = load_raw_data()
        report = inspect_data_quality(df)
        self.assertIn("primary_weather_cols", report)
        for col in ["temp", "humidity", "precip", "windspeed", "solarradiation"]:
            self.assertIn(col, report["primary_weather_cols"])
        self.assertTrue(any("delhi" in str(loc).lower() for loc in report.get("locations", [])))

    def test_raw_vs_clean_target_statistics(self):
        """Verify data quality reporting accurately distinguishes raw vs cleaned minimum statistics."""
        df = load_raw_data()
        report = inspect_data_quality(df)
        
        self.assertIn("raw_target_summary", report)
        self.assertIn("clean_target_summary", report)
        
        # Raw stats include the 13 sensor zero dropouts
        self.assertEqual(report["raw_target_summary"]["min"], 0.0)
        self.assertEqual(report["raw_target_summary"]["min_nonzero"], 2444.0)
        
        # Clean stats reflect genuine physical regional minimum load
        self.assertEqual(report["clean_target_summary"]["min"], 2444.0)
        self.assertGreaterEqual(report["clean_target_summary"]["min"], config.MIN_PLAUSIBLE_DEMAND)
        self.assertGreater(report["clean_target_summary"]["mean"], report["raw_target_summary"]["mean"])


if __name__ == "__main__":
    unittest.main()

