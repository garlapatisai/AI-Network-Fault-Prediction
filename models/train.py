"""
Train and evaluate fault-prediction models.

Models:
  1. Random Forest (scikit-learn)
  2. XGBoost
  3. LSTM (PyTorch)

The best model (by F1-score) is saved to models/saved/.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_splits():
    """Load the processed train / val / test CSVs."""
    train = pd.read_csv(config.TRAIN_DATA_PATH)
    val = pd.read_csv(config.VAL_DATA_PATH)
    test = pd.read_csv(config.TEST_DATA_PATH)

    feature_cols = [c for c in train.columns if c != config.TARGET_COL]

    X_train, y_train = train[feature_cols].values, train[config.TARGET_COL].values
    X_val, y_val = val[feature_cols].values, val[config.TARGET_COL].values
    X_test, y_test = test[feature_cols].values, test[config.TARGET_COL].values

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def _evaluate(name: str, y_true, y_pred):
    """Print metrics and return a dict."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal', 'Fault'])}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:\n{cm}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 1: Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n🌲  Training Random Forest …")
    clf = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _evaluate("Random Forest", y_test, y_pred)
    return clf, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 2: XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n🚀  Training XGBoost …")
    # calculate scale_pos_weight for imbalanced dataset
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {**config.XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, verbose=False)
    y_pred = clf.predict(X_test)
    metrics = _evaluate("XGBoost", y_test, y_pred)
    return clf, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 3: LSTM (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        return self.fc(out).squeeze(-1)


def _create_sequences(X, y, seq_len):
    """Reshape flat arrays into overlapping sequences for the LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test):
    print("\n🧠  Training LSTM …")
    p = config.LSTM_PARAMS
    seq_len = p["sequence_length"]

    # Limit data for CPU training speed
    max_train = min(2000, len(X_train))
    X_tr_sub, y_tr_sub = X_train[:max_train], y_train[:max_train]
    max_test = min(1000, len(X_test))
    X_te_sub, y_te_sub = X_test[:max_test], y_test[:max_test]
    print(f"    Using {max_train} train / {max_test} test samples for LSTM")

    X_tr_seq, y_tr_seq = _create_sequences(X_tr_sub, y_tr_sub, seq_len)
    X_te_seq, y_te_seq = _create_sequences(X_te_sub, y_te_sub, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(X_tr_seq, dtype=torch.float32),
        torch.tensor(y_tr_seq, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True)

    input_size = X_tr_sub.shape[1]
    model = LSTMClassifier(input_size, p["hidden_size"], p["num_layers"], p["dropout"]).to(device)

    # handle class imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

    for epoch in range(p["epochs"]):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{p['epochs']}  loss={total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        logits = model(X_te_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)

    metrics = _evaluate("LSTM", y_te_seq, y_pred)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = _load_splits()
    print(f"📊  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"    Features: {len(feature_cols)}")

    results = {}

    # 1. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = rf_metrics

    # 2. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results["xgboost"] = xgb_metrics

    # 3. LSTM
    lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_test, y_test)
    results["lstm"] = lstm_metrics

    # ── Pick best model by F1 ────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n🏆  Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if best_name == "random_forest":
        joblib.dump(rf_model, os.path.join(config.MODEL_DIR, "best_model.joblib"))
    elif best_name == "xgboost":
        joblib.dump(xgb_model, os.path.join(config.MODEL_DIR, "best_model.joblib"))
    else:
        torch.save(lstm_model.state_dict(), os.path.join(config.MODEL_DIR, "best_model.pt"))

    # Always save RF and XGB for the API (easier to serve)
    joblib.dump(rf_model, os.path.join(config.MODEL_DIR, "random_forest.joblib"))
    joblib.dump(xgb_model, os.path.join(config.MODEL_DIR, "xgboost.joblib"))
    torch.save(lstm_model.state_dict(), os.path.join(config.MODEL_DIR, "lstm.pt"))

    # Save metadata
    meta = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "results": {k: {m: round(v, 4) for m, v in v_dict.items()} for k, v_dict in results.items()},
    }
    with open(os.path.join(config.MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅  Models saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
