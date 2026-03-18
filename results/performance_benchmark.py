"""Helpers for saving runtime scaling metrics and plots."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from config import RESULTS_DIR


def write_performance_metrics(rows: list[dict[str, float]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "performance_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["debris_count", "runtime_seconds"])
        for row in rows:
            writer.writerow([int(row["debris_count"]), float(row["runtime_seconds"])])
    return csv_path


def plot_runtime_scaling(rows: list[dict[str, float]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / "runtime_scaling.png"
    x = [int(row["debris_count"]) for row in rows]
    y = [float(row["runtime_seconds"]) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Debris Objects")
    plt.ylabel("Runtime (s)")
    plt.title("Propagation Runtime Scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path
