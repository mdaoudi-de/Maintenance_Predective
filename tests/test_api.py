"""Tests de l'API FastAPI via le client de test (sans serveur réseau)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from hydraulic_valve import config

client = TestClient(app)
_MODEL_AVAILABLE = config.MODEL_PATH.exists()


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "model_missing"}


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="Modèle absent")
def test_predict_endpoint_ok():
    r = client.get("/predict/1")
    assert r.status_code == 200
    body = r.json()
    assert body["cycle"] == 1
    assert body["prediction"] in (0, 1)
    assert body["label"] in {"Optimale", "Non optimale"}


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="Modèle absent")
def test_predict_endpoint_out_of_range():
    r = client.get("/predict/0")
    assert r.status_code == 422  # cycle hors limites


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="Modèle absent")
def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_name" in r.json()
    assert len(r.json()["feature_names"]) == 48
