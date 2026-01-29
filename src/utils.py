import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_tradeoff(df, out_path: str, title: str):
    """
    df must contain columns: model, accuracy, latency_ms
    """
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.scatter(df["latency_ms"], df["accuracy"])
    for _, r in df.iterrows():
        plt.text(r["latency_ms"], r["accuracy"], r["model"])
    plt.xlabel("Latency (ms/sample)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
