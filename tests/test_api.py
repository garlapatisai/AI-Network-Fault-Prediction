"""Tests for the Flask API."""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app
from backend.database import init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with a temporary database."""
    import config
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    init_db()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_valid(self, client):
        payload = {
            "node_id": "node_001",
            "traffic_volume": 650,
            "latency_ms": 95,
            "packet_loss_pct": 8.2,
            "signal_strength_dbm": -82,
            "error_rate": 0.18,
            "cpu_utilisation": 78,
            "memory_utilisation": 65,
        }
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "fault_probability" in data
        assert "is_fault" in data
        assert "severity" in data

    def test_predict_empty_body(self, client):
        r = client.post("/predict", data="{}", content_type="application/json")
        assert r.status_code == 400  # empty body is rejected


class TestHistoryEndpoint:
    def test_history(self, client):
        r = client.get("/history")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)


class TestAlertsEndpoint:
    def test_alerts(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)
