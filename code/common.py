"""Pieces shared by train.py and evaluate.py: paths, the dataset, image transforms and the model."""

from pathlib import Path

import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

NUM_CLASSES = 5
CLASS_NAMES = ["Normal (0)", "Stage I", "Stage II", "Stage III", "Stage IV"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hyperparameters selected by the genetic algorithm in the original run
# (see results/original_training_log.txt, "Best hyperparameters").
PAPER_HYPERPARAMETERS = {
    "learning_rate": 0.00020027035366007752,
    "dropout_rate": 0.30315513401405514,
    "weight_decay": 3.724157568008995e-05,
}


class OvarianDataset(Dataset):
    """Reads (image filename, label) rows from a CSV and loads the images from a folder."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(int)
    return df


def train_transform():
    """Random augmentation applied to training images only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform():
    """Deterministic preprocessing for validation and test images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(dropout_rate: float, pretrained: bool = True) -> nn.Module:
    """ResNet-101 with its final layer replaced by Dropout -> Linear(2048, 5)."""
    weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet101(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model
