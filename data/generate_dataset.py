"""
Generate a realistic synthetic telecom network dataset.

The dataset simulates network monitoring data from multiple nodes with
correlated anomaly patterns that precede faults.
"""
import os
import sys
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_dataset(
    num_samples: int = config.NUM_SAMPLES,
    num_nodes: int = config.NUM_NODES,
    fault_ratio: float = config.FAULT_RATIO,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic telecom network logs with fault labels."""
    rng = np.random.RandomState(seed)

    # --- timestamps (5-minute intervals) ---
    samples_per_node = num_samples // num_nodes
    total = samples_per_node * num_nodes
    base_time = pd.Timestamp("2024-01-01")
    timestamps = []
    node_ids = []
    for nid in range(1, num_nodes + 1):
        ts = pd.date_range(base_time, periods=samples_per_node, freq="5min")
        timestamps.extend(ts)
        node_ids.extend([f"node_{nid:03d}"] * samples_per_node)

    # --- normal baseline metrics ---
    traffic_volume = rng.normal(500, 120, total).clip(50)
    latency_ms = rng.normal(20, 5, total).clip(1)
    packet_loss_pct = rng.exponential(0.5, total).clip(0, 100)
    signal_strength_dbm = rng.normal(-65, 8, total).clip(-120, -30)
    error_rate = rng.exponential(0.02, total).clip(0, 1)
    cpu_utilisation = rng.normal(45, 15, total).clip(0, 100)
    memory_utilisation = rng.normal(50, 12, total).clip(0, 100)

    # --- inject correlated anomaly windows that precede faults ---
    fault = np.zeros(total, dtype=int)
    num_fault_windows = int(total * fault_ratio / 6)  # each window ~6 samples

    for _ in range(num_fault_windows):
        start = rng.randint(0, total - 8)
        window = slice(start, start + 6)
        # degrade metrics in the window
        latency_ms[window] += rng.normal(80, 20, 6).clip(0)
        packet_loss_pct[window] += rng.normal(8, 3, 6).clip(0)
        error_rate[window] += rng.normal(0.15, 0.05, 6).clip(0)
        cpu_utilisation[window] += rng.normal(25, 8, 6).clip(0)
        traffic_volume[window] *= rng.uniform(1.5, 2.5, 6)
        signal_strength_dbm[window] -= rng.normal(15, 5, 6).clip(0)
        # fault fires near the end of the degradation window
        fault[start + 4 : start + 6] = 1

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "node_id": node_ids,
            "traffic_volume": np.round(traffic_volume, 2),
            "latency_ms": np.round(latency_ms, 2),
            "packet_loss_pct": np.round(packet_loss_pct, 4),
            "signal_strength_dbm": np.round(signal_strength_dbm, 2),
            "error_rate": np.round(error_rate, 4),
            "cpu_utilisation": np.round(cpu_utilisation, 2),
            "memory_utilisation": np.round(memory_utilisation, 2),
            "fault": fault,
        }
    )
    return df


def main():
    print("🔧  Generating synthetic telecom dataset …")
    df = generate_dataset()

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)

    fault_pct = df["fault"].mean() * 100
    print(f"✅  Saved {len(df):,} rows to {config.RAW_DATA_PATH}")
    print(f"    Nodes: {df['node_id'].nunique()}")
    print(f"    Fault ratio: {fault_pct:.1f} %")
    print(f"    Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
