"""Smoke tests for the API contract. Run with: python -m pytest -q"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def sample_payload():
    return json.loads((ROOT / "artifacts" / "sample_request.json").read_text(encoding="utf-8"))


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["feature_count"] == 13
    assert len(response.json()["classes"]) == 3


def test_prediction_endpoint(client):
    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in [0, 1, 2]
    assert body["class_label"] in ["class_0", "class_1", "class_2"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5


def test_prediction_rejects_missing_features(client):
    response = client.post("/predict", json={"alcohol": 14.2})
    assert response.status_code == 422
