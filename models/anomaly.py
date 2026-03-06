"""
Anomaly detection using Isolation Forest.
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AnomalyDetector:
    """Isolation-Forest-based anomaly detector for network metrics."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        self.scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler.joblib"))
        model_path = os.path.join(config.MODEL_DIR, "anomaly_detector.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def fit(self, X: np.ndarray):
        """Fit the Isolation Forest on training data (already scaled)."""
        self.model = IsolationForest(
            contamination=config.ANOMALY_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        joblib.dump(self.model, os.path.join(config.MODEL_DIR, "anomaly_detector.joblib"))
        print("✅  Anomaly detector fitted and saved.")

    def score(self, features_dict: dict) -> dict:
        """Return anomaly score for a single observation."""
        feature_names = joblib.load(os.path.join(config.MODEL_DIR, "feature_names.joblib"))
        vec = np.array([[features_dict.get(c, 0.0) for c in feature_names]])
        vec = self.scaler.transform(vec)

        raw_score = float(self.model.decision_function(vec)[0])
        is_anomaly = bool(self.model.predict(vec)[0] == -1)

        return {
            "anomaly_score": round(raw_score, 4),
            "is_anomaly": is_anomaly,
        }


_detector = None


def get_detector() -> AnomalyDetector:
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector
