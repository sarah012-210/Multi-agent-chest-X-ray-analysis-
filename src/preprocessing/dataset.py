"""
PyTorch Dataset for CheXpert.

Reads train.csv / valid.csv (as shipped by the Kaggle "ashery/chexpert"
dataset), applies the chosen uncertain-label policy, and returns
(image_tensor, label_vector) pairs.
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CHEXPERT_LABELS, IMAGE_SIZE, UNCERTAIN_POLICY


def _resolve_image_path(csv_path: str, dataset_root: str) -> str:
    """
    The CSV 'Path' column looks like 'CheXpert-v1.0-small/train/patient.../view1.jpg'.
    This joins it against the root folder where the images actually live,
    handling the common case where the CSV's leading folder name differs
    from the extracted folder name.
    """
    candidate = os.path.join(dataset_root, csv_path)
    if os.path.exists(candidate):
        return candidate

    # Fallback: strip the first path component (dataset version folder name)
    # and try again, since kagglehub sometimes flattens that top folder.
    parts = csv_path.split("/")
    if len(parts) > 1:
        stripped = os.path.join(dataset_root, *parts[1:])
        if os.path.exists(stripped):
            return stripped

    return candidate  # let it fail loudly later if truly missing


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


class CheXpertDataset(Dataset):
    def __init__(self, csv_path: str, dataset_root: str, train: bool = True,
                 frontal_only: bool = True, uncertain_policy: str = UNCERTAIN_POLICY):
        self.dataset_root = dataset_root
        self.train = train
        self.uncertain_policy = uncertain_policy
        self.transform = build_transforms(train=train)

        df = pd.read_csv(csv_path)

        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        for label in CHEXPERT_LABELS:
            if label not in df.columns:
                df[label] = 0.0

        df[CHEXPERT_LABELS] = df[CHEXPERT_LABELS].fillna(0.0)
        self.df = df

    def __len__(self):
        return len(self.df)

    def _apply_uncertain_policy(self, labels: np.ndarray) -> np.ndarray:
        if self.uncertain_policy == "ones":
            labels[labels == -1] = 1.0
        elif self.uncertain_policy == "zeros":
            labels[labels == -1] = 0.0
        else:  # "ignore" -> keep -1, caller/loss must mask it out
            pass
        return labels

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_image_path(row["Path"], self.dataset_root)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = row[CHEXPERT_LABELS].values.astype(np.float32)
        labels = self._apply_uncertain_policy(labels)

        return image, torch.tensor(labels, dtype=torch.float32)
