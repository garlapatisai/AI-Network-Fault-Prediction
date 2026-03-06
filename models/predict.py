"""
Model inference module.

Loads the best saved model and exposes a predict() function for the API layer.
"""
import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FaultPredictor:
    """Wraps the best trained model for inference."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None
        self._load()

    def _load(self):
        meta_path = os.path.join(config.MODEL_DIR, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        self.model_name = meta["best_model"]
        self.feature_names = meta["feature_cols"]

        # Load scaler
        self.scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler.joblib"))

        # Load model
        if self.model_name in ("random_forest", "xgboost"):
            model_file = f"{self.model_name}.joblib"
            self.model = joblib.load(os.path.join(config.MODEL_DIR, model_file))
        else:
            # For LSTM, fall back to XGBoost for API serving (simpler)
            self.model = joblib.load(os.path.join(config.MODEL_DIR, "xgboost.joblib"))
            self.model_name = "xgboost (fallback from lstm)"

    def predict(self, features_dict: dict) -> dict:
        """
        Predict fault probability from a features dictionary.

        Parameters
        ----------
        features_dict : dict
            Keys should include the base feature columns. Missing rolling /
            time features will be filled with zeros.

        Returns
        -------
        dict with keys: fault_probability, is_fault, severity, model
        """
        # Build feature vector in the correct order
        vec = []
        for col in self.feature_names:
            vec.append(features_dict.get(col, 0.0))

        X = np.array([vec])
        X = self.scaler.transform(X)

        prob = float(self.model.predict_proba(X)[0][1])
        is_fault = prob >= config.FAULT_PROB_THRESHOLD

        severity = "normal"
        for level, (lo, hi) in config.SEVERITY_LEVELS.items():
            if lo <= prob < hi:
                severity = level
                break

        return {
            "fault_probability": round(prob, 4),
            "is_fault": bool(is_fault),
            "severity": severity,
            "model": self.model_name,
        }


# Singleton for reuse
_predictor = None


def get_predictor() -> FaultPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FaultPredictor()
    return _predictor
