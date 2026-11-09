"""Loss and accuracy per epoch for the final training run.

Reads results/original_final_training_history.csv (transcribed from results/original_training_log.txt).
Gaps are epochs whose values were not captured in the saved console log.
For a new run, pass --history runs/latest/training_history.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from style import BLUE, INK_SECONDARY, ORANGE, RESULTS_DIR, plt, save

p = argparse.ArgumentParser()
p.add_argument("--history", type=Path, default=RESULTS_DIR / "original_final_training_history.csv")
args = p.parse_args()

df = pd.read_csv(args.history)
best = df.loc[df["val_accuracy"].idxmax()]

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, metric, title, ylabel in [
    (ax_loss, "loss", "Loss", "Cross-entropy loss"),
    (ax_acc, "accuracy", "Accuracy", "Accuracy (%)"),
]:
    for split, color, marker in [("train", ORANGE, "o"), ("val", BLUE, "o")]:
        # Missing epochs stay as gaps in the line rather than being bridged
        ax.plot(df["epoch"], df[f"{split}_{metric}"], color=color, marker=marker,
                label="Training" if split == "train" else "Validation")
    ax.axvline(best["epoch"], color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, int(df["epoch"].max()) + 1))

ax_acc.annotate(f"Best model kept\n(epoch {int(best['epoch'])}, {best['val_accuracy']:.2f}%)",
                xy=(best["epoch"], best["val_accuracy"]), xytext=(best["epoch"] + 0.4, 84),
                color=INK_SECONDARY, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.8))
ax_loss.legend(loc="upper right")
fig.suptitle("Final training run (ResNet-101, GA-selected hyperparameters)", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
save(fig, "training_curves.png")
