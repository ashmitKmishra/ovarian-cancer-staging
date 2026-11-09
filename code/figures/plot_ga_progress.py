"""Validation accuracy of every hyperparameter candidate the genetic algorithm evaluated.

Reads results/original_ga_search_history.csv (transcribed from results/original_training_log.txt).
For a new run, pass --history runs/latest/ga_search_history.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import BLUE, BLUES, INK_SECONDARY, ORANGE, RESULTS_DIR, SURFACE, plt, save

p = argparse.ArgumentParser()
p.add_argument("--history", type=Path, default=RESULTS_DIR / "original_ga_search_history.csv")
args = p.parse_args()

df = pd.read_csv(args.history)
summary = df.groupby("generation")["val_accuracy"].agg(["max", "mean"])

fig, ax = plt.subplots(figsize=(7.5, 4.5))
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.06, 0.06, len(df))
ax.scatter(df["generation"] + jitter, df["val_accuracy"], s=46, color=BLUES[1],
           edgecolor=SURFACE, linewidth=1.5, zorder=2,
           label="Each candidate")
ax.plot(summary.index, summary["max"], color=BLUE, marker="o", markersize=8, zorder=3, label="Best in generation")
ax.plot(summary.index, summary["mean"], color=ORANGE, marker="o", markersize=8, linestyle="--", zorder=3,
        label="Generation average")
for g, row in summary.iterrows():
    ax.annotate(f"{row['max']:.2f}%", (g, row["max"]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9.5, color=INK_SECONDARY)
    ax.annotate(f"{row['mean']:.2f}%", (g, row["mean"]), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=9.5, color=INK_SECONDARY)

ax.set_xticks(summary.index)
ax.set_xlim(0.6, summary.index.max() + 0.5)
ax.set_ylim(40, 102)
ax.set_xlabel("Generation")
ax.set_ylabel("Validation accuracy after 5 epochs (%)")
ax.set_title("Genetic algorithm hyperparameter search")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "ga_progress.png")
