"""
Alert generation logic.

Determines severity and creates alert records when fault probability
exceeds the configured threshold.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.database import insert_alert


def maybe_create_alert(node_id: str, fault_prob: float, severity: str) -> dict | None:
    """
    Create an alert if the prediction crosses the fault threshold.

    Returns the alert dict if one was created, else None.
    """
    if fault_prob < config.FAULT_PROB_THRESHOLD:
        return None

    messages = {
        "low": f"⚠️  Node {node_id}: elevated fault risk ({fault_prob:.1%})",
        "medium": f"🟠  Node {node_id}: moderate fault risk ({fault_prob:.1%}). Investigate soon.",
        "high": f"🔴  Node {node_id}: HIGH fault risk ({fault_prob:.1%}). Immediate attention required.",
        "critical": f"🚨  Node {node_id}: CRITICAL fault imminent ({fault_prob:.1%})! Take action NOW.",
    }
    message = messages.get(severity, messages["low"])

    insert_alert(node_id, severity, message, fault_prob)

    return {
        "node_id": node_id,
        "severity": severity,
        "message": message,
        "fault_probability": fault_prob,
    }
