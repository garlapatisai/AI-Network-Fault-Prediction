#!/usr/bin/env python3
"""
CLI entry point for the AI Network Fault Prediction System.

Usage
-----
    python run.py generate     – Generate synthetic telecom dataset
    python run.py preprocess   – Preprocess data & create train/val/test splits
    python run.py train        – Train & evaluate ML models
    python run.py backend      – Start the Flask REST API
    python run.py frontend     – Start the Streamlit dashboard
    python run.py all          – Run generate → preprocess → train (no servers)
"""
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "generate": "data/generate_dataset.py",
    "preprocess": "data/preprocess.py",
    "train": "models/train.py",
}


def run_script(name):
    script = os.path.join(BASE_DIR, COMMANDS[name])
    print(f"\n{'═' * 60}")
    print(f"  Running: {name}")
    print(f"{'═' * 60}\n")
    subprocess.run([sys.executable, script], check=True)


def start_api():
    script = os.path.join(BASE_DIR, "backend", "app.py")
    print("🚀  Starting Flask API …")
    subprocess.run([sys.executable, script])


def start_dashboard():
    script = os.path.join(BASE_DIR, "frontend", "app.py")
    print("📊  Starting Streamlit Dashboard …")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script, "--server.port", "8501"])


def print_help():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "all":
        for step in ("generate", "preprocess", "train"):
            run_script(step)
    elif cmd in COMMANDS:
        run_script(cmd)
    elif cmd == "backend":
        start_api()
    elif cmd == "frontend":
        start_dashboard()
    elif cmd in ("-h", "--help", "help"):
        print_help()
    else:
        print(f"❌  Unknown command: {cmd}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
