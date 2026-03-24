"""
Flask REST API for the AI Network Fault Prediction System.

Endpoints
---------
GET  /health    – Service health check
POST /predict   – Predict fault from network metrics
GET  /alerts    – List recent alerts
GET  /history   – Historical predictions
GET  /metrics   – Model performance metrics
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import config
from backend.database import (
    init_db,
    insert_prediction,
    get_recent_predictions,
    get_alerts,
    resolve_alert,
    get_prediction_stats,
)
from backend.alerts import maybe_create_alert
from models.predict import get_predictor
from models.anomaly import get_detector

app = Flask(__name__)
CORS(app)


@app.before_request
def _ensure_db():
    init_db()


# ── Root ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "AI Network Fault Prediction API",
        "status": "running",
        "endpoints": {
            "GET  /":        "This overview",
            "GET  /health":  "Service health check",
            "POST /predict": "Predict fault from network metrics",
            "GET  /alerts":  "List recent alerts",
            "GET  /history": "Historical predictions",
            "GET  /metrics": "Model performance metrics",
        },
    })


# ── Health ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": get_predictor().model_name})


# ── Predict ──────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect JSON body with network metrics, e.g.:
    {
      "node_id": "node_001",
      "traffic_volume": 650,
      "latency_ms": 95,
      "packet_loss_pct": 8.2,
      "signal_strength_dbm": -82,
      "error_rate": 0.18,
      "cpu_utilisation": 78,
      "memory_utilisation": 65
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    node_id = data.pop("node_id", "unknown")

    # Fault prediction
    predictor = get_predictor()
    result = predictor.predict(data)

    # Anomaly detection
    try:
        detector = get_detector()
        anomaly = detector.score(data)
        result.update(anomaly)
    except Exception:
        result["anomaly_score"] = None
        result["is_anomaly"] = None

    # Persist
    insert_prediction(
        node_id,
        result["fault_probability"],
        result["is_fault"],
        result["severity"],
        result["model"],
        json.dumps(data),
    )

    # Maybe create alert
    alert = maybe_create_alert(node_id, result["fault_probability"], result["severity"])
    result["alert"] = alert
    result["node_id"] = node_id

    return jsonify(result)


# ── Alerts ───────────────────────────────────────────────────────────────────

@app.route("/alerts", methods=["GET"])
def alerts():
    resolved = request.args.get("resolved")
    if resolved is not None:
        resolved = int(resolved)
    limit = int(request.args.get("limit", 100))
    return jsonify(get_alerts(resolved=resolved, limit=limit))


@app.route("/alerts/<int:alert_id>/resolve", methods=["POST"])
def resolve(alert_id):
    resolve_alert(alert_id)
    return jsonify({"status": "resolved", "alert_id": alert_id})


# ── History ──────────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_recent_predictions(limit=limit))


# ── Model Metrics ────────────────────────────────────────────────────────────

@app.route("/metrics", methods=["GET"])
def metrics():
    meta_path = os.path.join(config.MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "No model metadata found. Train models first."}), 404

    with open(meta_path) as f:
        meta = json.load(f)

    stats = get_prediction_stats()
    return jsonify({**meta, **stats})


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    init_db()
    print(f"🚀  API server starting on http://{config.API_HOST}:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)


if __name__ == "__main__":
    main()
