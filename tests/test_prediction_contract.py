import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict


def test_forecast_date_parser_is_strict():
    assert str(predict.parse_forecast_date("2026-10-31").date()) == "2026-10-31"
    for bad in ("2026-02-30", "31-10-2026", "not-a-date"):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            predict.parse_forecast_date(bad)


def test_saved_artifact_feature_order_contains_all_sixteen_weather_inputs():
    root = Path(__file__).resolve().parents[1]
    artifact = predict.joblib.load(root / "models" / "load_forecaster.joblib")
    expected_weather = list(predict.WEATHER_FIELDS)
    assert artifact["features"][:16] == expected_weather
    assert artifact["model"].n_features_in_ == len(artifact["features"])


class CaptureModel:
    def predict(self, frame):
        self.frame = frame.copy()
        return [123.0]


def test_prediction_uses_same_chronological_lag_rows_as_training(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    demands = [100, 110, 120, 130, 140, 150, 160, 0, 180]
    pd.DataFrame({"DATE": dates, "POWER_DEMAND": demands}).to_csv(
        tmp_path / "demand.csv", index=False
    )
    model = CaptureModel()
    features = ["demand_lag_1", "demand_lag_7", "demand_mean_7", "temp"]
    monkeypatch.setattr(predict, "DATA_PATH", tmp_path / "demand.csv")
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "unused.joblib")
    monkeypatch.setattr(predict.joblib, "load", lambda _: {"model": model, "features": features})
    weather = {key: value for key, value in {
        "temp": 20, "tempmax": 25, "tempmin": 15, "feelslike": 20,
        "humidity": 50, "dew": 10, "precip": 0, "precipprob": 0,
        "windspeed": 4, "windgust": 6, "sealevelpressure": 1010,
        "cloudcover": 10, "visibility": 10, "solarradiation": 200,
        "solarenergy": 15, "uvindex": 3,
    }.items()}
    assert predict.predict_demand("2024-01-10", weather) == 123.0
    assert model.frame.iloc[0]["demand_lag_1"] == 180
    assert model.frame.iloc[0]["demand_lag_7"] == 120
    assert model.frame.iloc[0]["demand_mean_7"] != model.frame.iloc[0]["demand_mean_7"]


def test_prediction_rejects_date_inside_historical_period(tmp_path, monkeypatch):
    pd.DataFrame({"DATE": ["2024-01-01", "2024-01-02"], "POWER_DEMAND": [1, 2]}).to_csv(
        tmp_path / "demand.csv", index=False
    )
    monkeypatch.setattr(predict, "DATA_PATH", tmp_path / "demand.csv")
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "unused.joblib")
    with pytest.raises(ValueError, match="after 2024-01-02"):
        predict.predict_demand("2024-01-02", {})
