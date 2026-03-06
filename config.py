"""
Central configuration for the AI Network Fault Prediction System.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "network_logs.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
DB_PATH = os.path.join(BASE_DIR, "api", "network_faults.db")

# ── Dataset Generation ───────────────────────────────────────────────────────
NUM_SAMPLES = 50_000
NUM_NODES = 20
FAULT_RATIO = 0.15  # ~15 % of samples are faults

# ── Feature Columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "traffic_volume",
    "latency_ms",
    "packet_loss_pct",
    "signal_strength_dbm",
    "error_rate",
    "cpu_utilisation",
    "memory_utilisation",
]

ROLLING_FEATURES = ["latency_ms", "packet_loss_pct", "error_rate"]
ROLLING_WINDOW = 5

TARGET_COL = "fault"

# ── Model Hyper-parameters ───────────────────────────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

LSTM_PARAMS = {
    "input_size": None,  # set dynamically
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.2,
    "epochs": 3,
    "batch_size": 512,
    "learning_rate": 0.001,
    "sequence_length": 5,
}

# ── Anomaly Detection ────────────────────────────────────────────────────────
ANOMALY_CONTAMINATION = 0.1

# ── Alert Thresholds ─────────────────────────────────────────────────────────
FAULT_PROB_THRESHOLD = 0.5
SEVERITY_LEVELS = {
    "low": (0.5, 0.7),
    "medium": (0.7, 0.85),
    "high": (0.85, 0.95),
    "critical": (0.95, 1.01),
}

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5050

# ── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_PORT = 8501
