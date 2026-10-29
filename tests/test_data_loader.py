"""
Unit tests for data loader, column detection, and quality validation.
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
        """Test data quality report structure and key metrics."""
        df = load_raw_data()
        report = inspect_data_quality(df)
        
        self.assertTrue(report["date_present"])
        self.assertTrue(report["target_present"])
        self.assertEqual(report["frequency_label"], "Daily")
        self.assertIn("target_summary", report)
        self.assertGreaterEqual(report["zero_or_negative_count"], 0)
        self.assertIn("missing_calendar_dates", report)
        self.assertFalse(report["meets_hourly_requirement"])  # Discloses daily limitation


if __name__ == "__main__":
    unittest.main()
