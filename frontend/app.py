"""
Streamlit Dashboard — AI Network Fault Prediction System

Multi-page app:
  1. Network Status   – live gauges & time-series
  2. Fault Predictions – recent predictions table
  3. Alerts           – active / resolved alerts
  4. Model Performance – accuracy, precision, recall, F1, confusion matrix
  5. Historical Analysis – fault timeline
"""
# ── Python 3.10+ compatibility fix for dask / xarray ─────────────────────────
import collections
import collections.abc
for attr in ("Hashable", "Mapping", "MutableMapping", "MutableSet", "Callable"):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

import config

API_BASE = f"http://127.0.0.1:{config.API_PORT}"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Network Fault Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               border-radius: 12px; padding: 16px; border: 1px solid #30363d;}
    .stAlert {border-radius: 10px;}
    h1 {background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800 !important;}
    .severity-critical {color: #ff4444; font-weight: bold;}
    .severity-high {color: #ff8800; font-weight: bold;}
    .severity-medium {color: #ffcc00; font-weight: bold;}
    .severity-low {color: #44ff44; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📡 Network Fault AI")
page = st.sidebar.radio(
    "Navigate",
    ["🖥️ Network Status", "🔮 Fault Predictions", "🚨 Alerts", "📊 Model Performance", "📈 Historical Analysis"],
)

# ── Helper: call API ────────────────────────────────────────────────────────


def api_get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(endpoint: str, data: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 1: Network Status
# ═══════════════════════════════════════════════════════════════════════════════

def page_network_status():
    st.title("🖥️ Real-Time Network Status")

    # Load a sample of raw data for visualisation
    raw_path = config.RAW_DATA_PATH
    if not os.path.exists(raw_path):
        st.warning("No raw data found. Run `python run.py generate` first.")
        return

    df = pd.read_csv(raw_path, parse_dates=["timestamp"])

    # Node selector
    nodes = sorted(df["node_id"].unique())
    selected_node = st.sidebar.selectbox("Select Node", nodes)
    node_df = df[df["node_id"] == selected_node].tail(200)

    # KPI row
    latest = node_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📶 Signal Strength", f"{latest['signal_strength_dbm']:.1f} dBm")
    col2.metric("⏱️ Latency", f"{latest['latency_ms']:.1f} ms")
    col3.metric("📦 Packet Loss", f"{latest['packet_loss_pct']:.2f} %")
    col4.metric("📊 Traffic Volume", f"{latest['traffic_volume']:.0f}")

    st.markdown("---")

    # Time series charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Latency (ms)", "Packet Loss (%)", "Signal Strength (dBm)", "Traffic Volume"),
        vertical_spacing=0.12,
    )
    colors = ["#00d2ff", "#ff6b6b", "#feca57", "#54a0ff"]

    for i, (col, color) in enumerate(
        zip(["latency_ms", "packet_loss_pct", "signal_strength_dbm", "traffic_volume"], colors)
    ):
        r, c = divmod(i, 2)
        fig.add_trace(
            go.Scatter(x=node_df["timestamp"], y=node_df[col], mode="lines", name=col,
                       line=dict(color=color, width=2)),
            row=r + 1, col=c + 1,
        )
        # Highlight fault zones
        fault_df = node_df[node_df["fault"] == 1]
        if not fault_df.empty:
            fig.add_trace(
                go.Scatter(x=fault_df["timestamp"], y=fault_df[col], mode="markers",
                           name="Fault", marker=dict(color="red", size=6, symbol="x"),
                           showlegend=(i == 0)),
                row=r + 1, col=c + 1,
            )

    fig.update_layout(
        height=600, template="plotly_dark",
        title_text=f"Network Metrics — {selected_node}",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Gauge charts
    st.subheader("System Resource Utilisation")
    gc1, gc2 = st.columns(2)
    for container, metric, val in [
        (gc1, "CPU Utilisation", latest["cpu_utilisation"]),
        (gc2, "Memory Utilisation", latest["memory_utilisation"]),
    ]:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": metric},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00d2ff"},
                "steps": [
                    {"range": [0, 50], "color": "#1a1a2e"},
                    {"range": [50, 80], "color": "#2d2d44"},
                    {"range": [80, 100], "color": "#4a1a1a"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 85},
            },
        ))
        gauge.update_layout(height=280, template="plotly_dark")
        container.plotly_chart(gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 2: Fault Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def page_fault_predictions():
    st.title("🔮 Fault Predictions")

    # Manual prediction form
    st.subheader("Submit Network Metrics for Prediction")
    with st.form("predict_form"):
        fc1, fc2, fc3 = st.columns(3)
        node_id = fc1.text_input("Node ID", value="node_001")
        traffic = fc1.number_input("Traffic Volume", value=500.0)
        latency = fc2.number_input("Latency (ms)", value=20.0)
        pkt_loss = fc2.number_input("Packet Loss (%)", value=0.5)
        signal = fc3.number_input("Signal Strength (dBm)", value=-65.0)
        error_rt = fc3.number_input("Error Rate", value=0.02)
        cpu = fc1.number_input("CPU Utilisation (%)", value=45.0)
        mem = fc2.number_input("Memory Utilisation (%)", value=50.0)
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        payload = {
            "node_id": node_id,
            "traffic_volume": traffic,
            "latency_ms": latency,
            "packet_loss_pct": pkt_loss,
            "signal_strength_dbm": signal,
            "error_rate": error_rt,
            "cpu_utilisation": cpu,
            "memory_utilisation": mem,
        }
        result = api_post("/predict", payload)
        if result:
            severity_color = {
                "normal": "green", "low": "blue", "medium": "orange",
                "high": "red", "critical": "red",
            }
            color = severity_color.get(result.get("severity", ""), "gray")
            st.markdown(f"### Result: :{color}[{result.get('severity', 'N/A').upper()}]")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Fault Probability", f"{result.get('fault_probability', 0):.2%}")
            rc2.metric("Is Fault?", "⚠️ YES" if result.get("is_fault") else "✅ NO")
            rc3.metric("Anomaly?", "⚠️ YES" if result.get("is_anomaly") else "✅ NO")

            if result.get("alert"):
                st.error(result["alert"]["message"])
        else:
            st.error("Could not reach the API. Make sure it is running (`python run.py api`).")

    # Recent predictions
    st.markdown("---")
    st.subheader("Recent Predictions")
    preds = api_get("/history", {"limit": 50})
    if preds:
        df = pd.DataFrame(preds)
        if not df.empty:
            st.dataframe(
                df[["timestamp", "node_id", "fault_prob", "severity", "model"]].style.applymap(
                    lambda v: f"color: {'red' if v in ('high','critical') else 'orange' if v == 'medium' else ''}"
                    if isinstance(v, str) else "",
                    subset=["severity"],
                ),
                use_container_width=True,
            )
        else:
            st.info("No predictions yet.")
    else:
        st.info("API unreachable — showing no data.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 3: Alerts
# ═══════════════════════════════════════════════════════════════════════════════

def page_alerts():
    st.title("🚨 Alerts")

    tab_active, tab_resolved = st.tabs(["🔴 Active Alerts", "✅ Resolved Alerts"])

    with tab_active:
        alerts = api_get("/alerts", {"resolved": 0, "limit": 50})
        if alerts:
            for a in alerts:
                sev = a.get("severity", "low")
                icon = {"critical": "🚨", "high": "🔴", "medium": "🟠", "low": "⚠️"}.get(sev, "ℹ️")
                with st.container():
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"**{icon} [{sev.upper()}]** — {a.get('message', '')}")
                    c1.caption(f"Node: {a.get('node_id')} | Prob: {a.get('fault_prob', 0):.2%} | {a.get('timestamp', '')}")
                    if c2.button("Resolve", key=f"resolve_{a['id']}"):
                        api_post(f"/alerts/{a['id']}/resolve", {})
                        st.rerun()
                    st.divider()
        else:
            st.success("No active alerts! 🎉")

    with tab_resolved:
        resolved = api_get("/alerts", {"resolved": 1, "limit": 50})
        if resolved:
            df = pd.DataFrame(resolved)
            st.dataframe(df[["timestamp", "node_id", "severity", "message", "fault_prob"]], use_container_width=True)
        else:
            st.info("No resolved alerts yet.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 4: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.title("📊 Model Performance")

    meta_path = os.path.join(config.MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        st.warning("No model metadata found. Train models first with `python run.py train`.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    best = meta.get("best_model", "")
    st.success(f"🏆 Best model: **{best}**")

    results = meta.get("results", {})
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Model"

    st.subheader("Comparison Table")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1a5e3a"), use_container_width=True)

    # Bar chart
    fig = go.Figure()
    for metric in ["accuracy", "precision", "recall", "f1"]:
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=list(results.keys()),
            y=[results[m][metric] for m in results],
        ))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        title="Model Comparison", yaxis_title="Score",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model (from test data)
    st.subheader("Test Set — Confusion Matrix (best model)")
    test_path = config.TEST_DATA_PATH
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        y_true = test_df[config.TARGET_COL].values

        # Re-run predictions on test set using saved model
        import joblib as _jl
        model_file = os.path.join(config.MODEL_DIR, f"{best}.joblib") if best in ("random_forest", "xgboost") else os.path.join(config.MODEL_DIR, "xgboost.joblib")
        if os.path.exists(model_file):
            model = _jl.load(model_file)
            feature_cols = [c for c in test_df.columns if c != config.TARGET_COL]
            y_pred = model.predict(test_df[feature_cols].values)
            from sklearn.metrics import confusion_matrix as _cm
            cm = _cm(y_true, y_pred)

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Fault"],
                y=["Normal", "Fault"],
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig_cm.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Live stats
    stats = api_get("/metrics")
    if stats:
        st.subheader("Live Prediction Stats")
        sc1, sc2 = st.columns(2)
        sc1.metric("Total Predictions", stats.get("total_predictions", 0))
        sc2.metric("Faults Detected", stats.get("total_faults", 0))


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 5: Historical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def page_historical():
    st.title("📈 Historical Analysis")

    raw_path = config.RAW_DATA_PATH
    if not os.path.exists(raw_path):
        st.warning("No raw data found.")
        return

    df = pd.read_csv(raw_path, parse_dates=["timestamp"])

    # Date range filter
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    d1, d2 = st.columns(2)
    start = d1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end = d2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    filtered = df[mask]

    st.metric("Records in range", f"{len(filtered):,}")

    # Fault timeline
    st.subheader("Fault Timeline")
    fault_counts = (
        filtered.set_index("timestamp")
        .resample("1h")["fault"]
        .sum()
        .reset_index()
    )
    fig = px.area(
        fault_counts, x="timestamp", y="fault",
        title="Hourly Fault Count",
        labels={"fault": "Faults", "timestamp": "Time"},
        color_discrete_sequence=["#ff6b6b"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Per-node fault distribution
    st.subheader("Faults per Node")
    node_faults = filtered.groupby("node_id")["fault"].sum().sort_values(ascending=False).reset_index()
    fig2 = px.bar(
        node_faults, x="node_id", y="fault",
        color="fault", color_continuous_scale="reds",
        title="Fault Count by Node",
    )
    fig2.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation")
    numeric_cols = config.FEATURE_COLS + [config.TARGET_COL]
    corr = filtered[numeric_cols].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
    )
    fig3.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig3, use_container_width=True)


# ── Route pages ──────────────────────────────────────────────────────────────
pages = {
    "🖥️ Network Status": page_network_status,
    "🔮 Fault Predictions": page_fault_predictions,
    "🚨 Alerts": page_alerts,
    "📊 Model Performance": page_model_performance,
    "📈 Historical Analysis": page_historical,
}
pages[page]()
