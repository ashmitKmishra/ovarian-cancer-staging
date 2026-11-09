"""Number of images per stage in the training, validation and test sets.

Uses data/train.csv and data/test.csv and the same 80/20 stratified split as code/train.py
(random_state=42), so the images themselves are not needed.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from style import BLUE, BLUES, CLASS_NAMES, DATA_DIR, INK_SECONDARY, ORANGE, RESULTS_DIR, plt, save

train_val = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
train, valid = train_test_split(train_val, test_size=0.2, stratify=train_val["label"], random_state=42)

counts = pd.DataFrame({
    "Training": train["label"].value_counts(),
    "Validation": valid["label"].value_counts(),
    "Test": test["label"].value_counts(),
}).sort_index().fillna(0).astype(int)
counts.index = CLASS_NAMES
counts["Total"] = counts.sum(axis=1)
counts.loc["All stages"] = counts.sum()
counts.to_csv(RESULTS_DIR / "dataset_split.csv")
print(counts)

plot = counts.drop(index="All stages").drop(columns="Total")
x = np.arange(len(plot))
width = 0.26
fig, ax = plt.subplots(figsize=(9, 4.4))
for k, (split, color) in enumerate(zip(plot.columns, [BLUE, ORANGE, BLUES[5]])):
    bars = ax.bar(x + (k - 1) * width, plot[split], width - 0.03, color=color, label=split)
    ax.bar_label(bars, padding=2, fontsize=9, color=INK_SECONDARY)

ax.set_xticks(x, plot.index)
ax.set_ylabel("Number of images")
ax.grid(axis="x", visible=False)
ax.set_title(f"Images per stage ({counts.loc['All stages', 'Total']:,} in total)")
ax.legend(loc="upper right")
fig.tight_layout()
save(fig, "dataset_split.png")
