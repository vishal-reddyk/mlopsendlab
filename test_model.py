
import os
import json
import joblib

def test_model_exists():
    assert os.path.exists("artifacts/model.joblib")

def test_metrics_exist():
    assert os.path.exists("artifacts/metrics.json")

def test_model_prediction():
    model = joblib.load("artifacts/model.joblib")
    result = model.predict([[0.05] * 10])
    assert len(result) == 1

def test_model_performance():
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)
    assert metrics["r2_score"] >= 0.30
