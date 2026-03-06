"""Tests for the data pipeline."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.generate_dataset import generate_dataset
from data.preprocess import load_raw_data, add_rolling_features, add_time_features, preprocess


class TestDataGeneration:
    def test_shape_and_columns(self):
        df = generate_dataset(num_samples=1000, num_nodes=5)
        assert len(df) == 1000
        expected = [
            "timestamp", "node_id", "traffic_volume", "latency_ms",
            "packet_loss_pct", "signal_strength_dbm", "error_rate",
            "cpu_utilisation", "memory_utilisation", "fault",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_fault_label_binary(self):
        df = generate_dataset(num_samples=1000, num_nodes=5)
        assert set(df["fault"].unique()).issubset({0, 1})

    def test_no_nans(self):
        df = generate_dataset(num_samples=1000, num_nodes=5)
        assert df.isna().sum().sum() == 0

    def test_node_count(self):
        df = generate_dataset(num_samples=1000, num_nodes=5)
        assert df["node_id"].nunique() == 5


class TestPreprocessing:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # Generate a small dataset and save as raw
        df = generate_dataset(num_samples=500, num_nodes=2)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        self.raw_path = raw_dir / "network_logs.csv"
        df.to_csv(self.raw_path, index=False)
        self.df = df

    def test_rolling_features(self):
        df = add_rolling_features(self.df.copy())
        for col in config.ROLLING_FEATURES:
            assert f"{col}_rolling_mean" in df.columns
            assert f"{col}_rolling_std" in df.columns

    def test_time_features(self):
        df = add_time_features(self.df.copy())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in df.columns

    def test_preprocess_output(self):
        X_train, X_val, X_test, y_train, y_val, y_test, feat = preprocess(self.df.copy())
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert not np.isnan(X_train).any()
        assert set(np.unique(y_test)).issubset({0, 1})
