"""Confusion matrix of the test-set predictions.

Reads results/confusion_matrix.csv, which code/evaluate.py writes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from style import BLUES, CLASS_NAMES, INK, RESULTS_DIR, plt, save

p = argparse.ArgumentParser()
p.add_argument("--matrix", type=Path, default=RESULTS_DIR / "confusion_matrix.csv")
args = p.parse_args()

cm = pd.read_csv(args.matrix, index_col=0).to_numpy()
cmap = LinearSegmentedColormap.from_list("blues", ["#f5f9fe"] + BLUES)

fig, ax = plt.subplots(figsize=(6.8, 5.8))
im = ax.imshow(cm, cmap=cmap, vmin=0)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Each cell: the count, plus the share of that true stage's images
row_totals = cm.sum(axis=1, keepdims=True)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        dark_cell = cm[i, j] > cm.max() * 0.55
        color = "white" if dark_cell else INK
        weight = "bold" if i == j else "normal"
        ax.text(j, i - 0.08, str(cm[i, j]), ha="center", va="center", fontsize=14, color=color, fontweight=weight)
        if cm[i, j]:
            ax.text(j, i + 0.24, f"{100 * cm[i, j] / row_totals[i, 0]:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=color)

ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=0)
ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
ax.set_xlabel("Predicted stage")
ax.set_ylabel("True stage")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.tick_params(length=0)
correct = int(np.trace(cm))
ax.set_title(f"Test set: {correct} of {cm.sum()} images correct ({100 * correct / cm.sum():.2f}%)", pad=36)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of images")
fig.tight_layout()
save(fig, "confusion_matrix.png")
