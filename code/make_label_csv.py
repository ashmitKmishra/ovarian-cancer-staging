"""Build a label CSV (image filename, stage label) from a folder of images.

The microscopy images are named by stage, for example "Ov_C (12).jpg" or "Ov_III (4).jpg":
    Ov_C -> 0 (normal/benign control)   Ov_I -> 1   Ov_II -> 2   Ov_III -> 3   Ov_IV -> 4

This is how data/train.csv and data/test.csv were produced.

Usage (from the repository root):
    python code/make_label_csv.py data/images/train_val data/train.csv
    python code/make_label_csv.py data/images/test data/test.csv
"""

import argparse
from pathlib import Path

import pandas as pd

PREFIX_TO_LABEL = {"Ov_C": 0, "Ov_I": 1, "Ov_II": 2, "Ov_III": 3, "Ov_IV": 4}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_dir", type=Path)
    p.add_argument("output_csv", type=Path)
    args = p.parse_args()

    rows, skipped = [], []
    for path in sorted(args.image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        prefix = path.name.split(" ")[0]
        if prefix in PREFIX_TO_LABEL:
            rows.append({"image": path.name, "label": PREFIX_TO_LABEL[prefix]})
        else:
            skipped.append(path.name)

    df = pd.DataFrame(rows, columns=["image", "label"])
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print("Images per label:", df["label"].value_counts().sort_index().to_dict())
    if skipped:
        print(f"Skipped {len(skipped)} files with an unrecognised name, e.g. {skipped[:3]}")


if __name__ == "__main__":
    main()
