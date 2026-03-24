"""
SQLite database helper for storing predictions and alerts.
"""
import os
import sys
import sqlite3
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DB_PATH = config.DB_PATH


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            node_id     TEXT,
            fault_prob  REAL,
            is_fault    INTEGER,
            severity    TEXT,
            model       TEXT,
            features    TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            node_id     TEXT,
            severity    TEXT,
            message     TEXT,
            fault_prob  REAL,
            resolved    INTEGER DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()


def insert_prediction(node_id, fault_prob, is_fault, severity, model, features_json):
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (timestamp, node_id, fault_prob, is_fault, severity, model, features) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), node_id, fault_prob, int(is_fault), severity, model, features_json),
    )
    conn.commit()
    conn.close()


def insert_alert(node_id, severity, message, fault_prob):
    conn = get_connection()
    conn.execute(
        "INSERT INTO alerts (timestamp, node_id, severity, message, fault_prob) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), node_id, severity, message, fault_prob),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_alerts(resolved=None, limit=100):
    conn = get_connection()
    if resolved is not None:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE resolved = ? ORDER BY id DESC LIMIT ?",
            (int(resolved), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_alert(alert_id):
    conn = get_connection()
    conn.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def get_prediction_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    faults = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_fault = 1").fetchone()[0]
    conn.close()
    return {"total_predictions": total, "total_faults": faults}
