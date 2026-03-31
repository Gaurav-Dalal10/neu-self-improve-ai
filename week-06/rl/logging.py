"""Simple CSV logger and matplotlib learning-curve plotter."""

import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class CSVLogger:
    """Accumulate dicts and flush to a single CSV at the end."""

    def __init__(self, path: str):
        self.path = path
        self.rows: List[Dict[str, float]] = []

    def log(self, row: Dict[str, float]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        keys = sorted(set().union(*(r.keys() for r in self.rows)))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def plot_curve(csv_path: str, png_path: str) -> None:
    """Plot episodic return vs update from a metrics CSV."""
    df = pd.read_csv(csv_path)
    col = "charts/episodic_return"
    if col not in df.columns:
        return
    y = df[col].ffill()
    x = df["update"] if "update" in df.columns else range(len(y))
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("update")
    plt.ylabel("avg episodic return")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
