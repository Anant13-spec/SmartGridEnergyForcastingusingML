import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict
from scenarios import TYPICAL


def test_forecast_date_parser_is_strict():
    assert str(predict.parse_forecast_date("2026-10-31").date()) == "2026-10-31"
    for bad in ("2026-02-30", "not-a-date", ""):
        with pytest.raises(ValueError):
            predict.parse_forecast_date(bad)


def test_saved_artifact_contains_weather_and_lag_features():
    root = Path(__file__).resolve().parents[1]
    artifact = predict.joblib.load(root / "models" / "smart_grid_forecasters.joblib")
    assert "targets" in artifact
    for target in ["demand_zone_1_mw", "demand_zone_2_mw", "demand_zone_3_mw", "Power demand"]:
        assert target in artifact["targets"]
        feat = artifact["targets"][target]["features"]
        for w in predict.WEATHER_FIELDS:
            assert w in feat


def test_prediction_produces_real_numeric_output_for_total():
    weather = TYPICAL.copy()
    res = predict.predict_demand("2024-12-15 14:00:00", weather, target="Power demand")
    assert "prediction_mw" in res
    assert isinstance(res["prediction_mw"], float)
    assert res["prediction_mw"] > 0
    assert "delta_mw" in res
    assert "baseline_mw" in res


def test_prediction_all_three_zones_and_total():
    weather = TYPICAL.copy()
    res = predict.predict_all_zones("2024-12-15 14:00:00", weather)
    assert "demand_zone_1_mw" in res
    assert "demand_zone_2_mw" in res
    assert "demand_zone_3_mw" in res
    assert "Power demand" in res
    assert "Combined_Zones_Sum" in res
    assert res["demand_zone_1_mw"]["prediction_mw"] > 0
    assert res["demand_zone_2_mw"]["prediction_mw"] > 0
    assert res["demand_zone_3_mw"]["prediction_mw"] > 0
    assert res["Power demand"]["prediction_mw"] > 0


def test_weather_validation_rejects_missing_inputs():
    with pytest.raises(ValueError, match="Missing required weather inputs"):
        predict.predict_demand("2024-12-15 14:00:00", {"temp": 28.0})


def test_weather_validation_rejects_out_of_bounds():
    bad_weather = TYPICAL.copy()
    bad_weather["temp"] = 150.0
    with pytest.raises(ValueError, match="outside realistic range"):
        predict.validate_weather(bad_weather)

    bad_rhum = TYPICAL.copy()
    bad_rhum["rhum"] = 120.0
    with pytest.raises(ValueError, match="between 0% and 100%"):
        predict.validate_weather(bad_rhum)


def test_specific_model_selection_and_routing():
    weather = TYPICAL.copy()
    for model_name in ["Linear Regression (Ridge)", "Random Forest", "HistGradientBoosting", "XGBoost"]:
        res = predict.predict_demand("2024-12-15 14:00:00", weather, target="Power demand", model_name=model_name)
        assert res["model_used"] == model_name
        assert res["prediction_mw"] > 0


def test_prediction_changes_with_weather():
    w_cool = {**TYPICAL, "temp": 12.0}
    w_hot = {**TYPICAL, "temp": 45.0}
    res_cool = predict.predict_demand("2024-12-15 14:00:00", w_cool, target="demand_zone_2_mw", model_name="HistGradientBoosting")
    res_hot = predict.predict_demand("2024-12-15 14:00:00", w_hot, target="demand_zone_2_mw", model_name="HistGradientBoosting")
    assert res_hot["prediction_mw"] != res_cool["prediction_mw"]
    assert res_hot["prediction_mw"] > res_cool["prediction_mw"]
