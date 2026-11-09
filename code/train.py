"""Train the ovarian cancer stage classifier.

Two stages, as described in the paper:
  1. A genetic algorithm (GA) searches for the learning rate, dropout rate and weight decay.
     Each candidate is scored by training an ImageNet-pretrained ResNet-101 for a few epochs
     and measuring validation accuracy.
  2. A final model is trained with the best hyperparameters for up to 20 epochs, with
     learning-rate reduction on plateau and early stopping. The checkpoint with the highest
     validation accuracy is kept.

Usage (from the repository root):
    python code/train.py                # full run: GA search + final training
    python code/train.py --skip-ga      # final training only, using the paper's hyperparameters
    python code/train.py --subset 100 --generations 1 --population-size 2 --ga-epochs 1 --epochs 1
                                        # a quick check that everything runs

New runs are written to runs/latest/ so they never overwrite the published model in model/.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (
    DATA_DIR, PAPER_HYPERPARAMETERS, REPO_ROOT, OvarianDataset, build_model,
    eval_transform, load_labels, train_transform,
)

HYPERPARAMETER_BOUNDS = {
    "learning_rate": (1e-5, 1e-3),
    "dropout_rate": (0.3, 0.7),
    "weight_decay": (1e-6, 1e-4),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", type=Path, default=DATA_DIR / "train.csv")
    p.add_argument("--image-dir", type=Path, default=DATA_DIR / "images" / "train_val")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "latest")
    p.add_argument("--skip-ga", action="store_true", help="skip the GA and use the paper's hyperparameters")
    p.add_argument("--population-size", type=int, default=5)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--ga-epochs", type=int, default=5, help="training epochs per GA candidate")
    p.add_argument("--epochs", type=int, default=20, help="maximum epochs for the final model")
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=None, help="use only N images (for a quick test)")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training), tqdm(loader, unit="batch", desc=desc, leave=False) as bar:
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
            bar.set_postfix(loss=f"{total_loss / seen:.4f}", acc=f"{100 * correct / seen:.2f}%")
    return total_loss / seen, 100 * correct / seen


def make_optimizer(model, hp):
    return optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])


def clip(hp):
    return {k: min(max(v, HYPERPARAMETER_BOUNDS[k][0]), HYPERPARAMETER_BOUNDS[k][1]) for k, v in hp.items()}


def genetic_search(args, train_loader, valid_loader, criterion, device, history_rows):
    """Tournament selection, blend crossover and Gaussian mutation over three hyperparameters."""
    population = [
        {k: random.uniform(lo, hi) for k, (lo, hi) in HYPERPARAMETER_BOUNDS.items()}
        for _ in range(args.population_size)
    ]
    best_hp, best_fitness = None, -1.0

    for gen in range(1, args.generations + 1):
        print(f"\nGeneration {gen}/{args.generations}")
        fitness = []
        for i, hp in enumerate(population, start=1):
            print(f"  Candidate {i}/{len(population)}: {hp}")
            model = build_model(hp["dropout_rate"]).to(device)
            optimizer = make_optimizer(model, hp)
            for epoch in range(1, args.ga_epochs + 1):
                run_epoch(model, train_loader, criterion, device, optimizer, desc=f"    epoch {epoch}")
            val_loss, val_acc = run_epoch(model, valid_loader, criterion, device, desc="    validating")
            print(f"    validation loss {val_loss:.4f}, accuracy {val_acc:.2f}%")
            fitness.append(val_acc)
            history_rows.append({"generation": gen, "candidate": i, **hp,
                                 "val_loss": val_loss, "val_accuracy": val_acc})
            if val_acc > best_fitness:
                best_hp, best_fitness = dict(hp), val_acc

        if gen == args.generations:
            break

        # Tournament selection (size 3, or the whole population if smaller)
        k = min(3, len(population))
        parents = [population[max(random.sample(range(len(population)), k), key=lambda j: fitness[j])]
                   for _ in range(args.population_size)]

        # Blend crossover
        children = []
        for i in range(0, args.population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % args.population_size]
            alpha = {key: random.uniform(0, 1) for key in HYPERPARAMETER_BOUNDS}
            children.append(clip({key: alpha[key] * p1[key] + (1 - alpha[key]) * p2[key] for key in p1}))
            children.append(clip({key: alpha[key] * p2[key] + (1 - alpha[key]) * p1[key] for key in p1}))
        children = children[:args.population_size]

        # Gaussian mutation
        for child in children:
            if random.random() < args.mutation_rate:
                for key, (lo, hi) in HYPERPARAMETER_BOUNDS.items():
                    child[key] += random.gauss(0, 0.01) * (hi - lo)
                child.update(clip(child))
        population = children

    print(f"\nBest hyperparameters (validation accuracy {best_fitness:.2f}%): {best_hp}")
    return best_hp


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    df = load_labels(args.train_csv)
    if args.subset:
        df = df.sample(n=min(args.subset, len(df)), random_state=args.seed)

    # 80/20 stratified split into training and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print("Training images per class:  ", train_df["label"].value_counts().sort_index().to_dict())
    print("Validation images per class:", valid_df["label"].value_counts().sort_index().to_dict())

    train_ds = OvarianDataset(train_df, args.image_dir, train_transform())
    valid_ds = OvarianDataset(valid_df, args.image_dir, eval_transform())

    # Class imbalance: inverse-frequency weights for both the sampler and the loss
    labels = train_df["label"].values
    classes = np.unique(labels)
    class_weights = torch.tensor(compute_class_weight("balanced", classes=classes, y=labels), dtype=torch.float)
    sample_weights = [class_weights[np.searchsorted(classes, y)] for y in labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    ga_rows = []
    if args.skip_ga:
        best_hp = dict(PAPER_HYPERPARAMETERS)
        print(f"Skipping GA; using the paper's hyperparameters: {best_hp}")
    else:
        best_hp = genetic_search(args, train_loader, valid_loader, criterion, device, ga_rows)
        write_csv(args.output_dir / "ga_search_history.csv", ga_rows)
    (args.output_dir / "hyperparameters.json").write_text(json.dumps(best_hp, indent=2))

    # Final training with the selected hyperparameters
    model = build_model(best_hp["dropout_rate"]).to(device)
    optimizer = make_optimizer(model, best_hp)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)
    checkpoint = args.output_dir / "best_ovarian_model.pth"

    best_acc, epochs_without_improvement, history = 0.0, 0, []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                          desc=f"Epoch {epoch}/{args.epochs}")
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, device, desc="validating")
        scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
              f"validation loss {val_loss:.4f}, acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, epochs_without_improvement = val_acc, 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  new best model saved to {checkpoint}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping: no improvement for {args.early_stopping_patience} epochs.")
                break

    write_csv(args.output_dir / "training_history.csv", history)
    print(f"\nDone. Best validation accuracy {best_acc:.2f}%. Outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
