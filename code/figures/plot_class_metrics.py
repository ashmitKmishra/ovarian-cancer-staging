"""Precision, recall and F1 score for each stage on the test set.

Computed from results/confusion_matrix.csv, which code/evaluate.py writes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import BLUE, BLUES, CLASS_NAMES, INK_SECONDARY, ORANGE, RESULTS_DIR, plt, save

p = argparse.ArgumentParser()
p.add_argument("--matrix", type=Path, default=RESULTS_DIR / "confusion_matrix.csv")
args = p.parse_args()

cm = pd.read_csv(args.matrix, index_col=0).to_numpy().astype(float)
tp = np.diag(cm)
precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
support = cm.sum(axis=1).astype(int)

# A dot plot rather than bars: every score is between 0.9 and 1.0, and bars would have to
# start at zero, which hides the differences.
metrics = [("Precision", precision, BLUE, "o"), ("Recall", recall, ORANGE, "s"), ("F1 score", f1, BLUES[5], "D")]
y = np.arange(len(CLASS_NAMES))[::-1]
offsets = [0.2, 0, -0.2]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for (name, values, color, marker), dy in zip(metrics, offsets):
    ax.scatter(values, y + dy, s=60, color=color, marker=marker, edgecolor="white", linewidth=1.2,
               zorder=3, label=name)
    for v, yy in zip(values, y + dy):
        ax.annotate(f"{v:.2f}", (v, yy), textcoords="offset points", xytext=(-9, 0), ha="right", va="center",
                    fontsize=9, color=INK_SECONDARY)

ax.set_yticks(y, [f"{n}  (n = {s})" for n, s in zip(CLASS_NAMES, support)])
ax.set_xlim(0.84, 1.015)
ax.set_xlabel("Score (1.00 = perfect)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.set_title("Per-stage precision, recall and F1 on the 126 test images")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
fig.tight_layout()
save(fig, "class_metrics.png")
