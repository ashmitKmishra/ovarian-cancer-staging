"""Evaluate a trained model on the held-out test set.

Usage (from the repository root):
    python code/evaluate.py
    python code/evaluate.py --model runs/latest/best_ovarian_model.pth --output-dir runs/latest

Writes to the output directory (default: results/):
    test_results.txt       accuracy, per-class precision / recall / F1, confusion matrix
    confusion_matrix.csv   5 x 5 counts (rows = true stage, columns = predicted stage)
    test_predictions.csv   one row per test image with its true and predicted stage
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    CLASS_NAMES, DATA_DIR, MODEL_DIR, NUM_CLASSES, PAPER_HYPERPARAMETERS, RESULTS_DIR,
    OvarianDataset, build_model, eval_transform, load_labels,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=MODEL_DIR / "best_ovarian_model.pth")
    p.add_argument("--test-csv", type=Path, default=DATA_DIR / "test.csv")
    p.add_argument("--image-dir", type=Path, default=DATA_DIR / "images" / "test")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--dropout-rate", type=float, default=PAPER_HYPERPARAMETERS["dropout_rate"],
                   help="must match the model being loaded (it does not affect predictions)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    test_df = load_labels(args.test_csv)
    loader = DataLoader(OvarianDataset(test_df, args.image_dir, eval_transform()),
                        batch_size=args.batch_size, shuffle=False)

    model = build_model(args.dropout_rate, pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.to(device).eval()

    preds = []
    with torch.no_grad():
        for images, _ in tqdm(loader, unit="batch", desc="Testing"):
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    y_true = test_df["label"].to_numpy()
    y_pred = np.array(preds)
    labels = list(range(NUM_CLASSES))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=CLASS_NAMES, columns=CLASS_NAMES)

    summary = (
        f"Model: {args.model.name}\n"
        f"Test images: {len(y_true)}\n"
        f"Accuracy: {100 * acc:.2f}% ({int((y_true == y_pred).sum())}/{len(y_true)} correct)\n\n"
        f"Per-class metrics:\n{report}\n"
        f"Confusion matrix (rows = true stage, columns = predicted stage):\n{cm.to_string()}\n"
    )
    print("\n" + summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_results.txt").write_text(summary)
    cm.to_csv(args.output_dir / "confusion_matrix.csv")
    pd.DataFrame({
        "image": test_df["image"],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
    }).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
