"""Data loading utilities for CIFAR-100 and CIFAR-100-C.

Handles train/val split, standard augmentation, normalization,
and corrupted test-set loaders.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional

from config import DATA_CFG


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return standard CIFAR-100 transforms.

    Args:
        train: If True, apply random crop and horizontal flip.
    """
    mean = DATA_CFG.mean
    std = DATA_CFG.std
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transform


def get_dataloaders(
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    quick_test: bool = False,
    quick_test_subset: int = 500,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders for CIFAR-100.

    The 50k official training set is split into 45k train / 5k val.
    Validation is held out strictly for calibration and threshold search.

    Returns:
        train_loader, val_loader, test_loader
    """
    bs = batch_size if batch_size is not None else DATA_CFG.batch_size
    nw = num_workers if num_workers is not None else DATA_CFG.num_workers

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    os.makedirs("./data", exist_ok=True)
    full_train = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )

    # Deterministic split for reproducibility
    n_total = len(full_train)
    n_train = int(n_total * DATA_CFG.train_val_split)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    if quick_test:
        train_indices = train_indices[:quick_test_subset]
        val_indices = val_indices[:quick_test_subset // 5]

    train_set = Subset(full_train, train_indices)

    # Validation uses test-time transforms (no augmentation)
    val_base = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=test_transform
    )
    val_set = Subset(val_base, val_indices)

    test_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )
    if quick_test:
        test_set = Subset(test_set, list(range(quick_test_subset // 5)))

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class CIFAR100C(Dataset):
    """Lazy-loading wrapper for CIFAR-100-C corrupted images.

    CIFAR-100-C is available from:
    https://github.com/hendrycks/robustness
    Expected directory structure:
        ./data/CIFAR-100-C/
            labels.npy
            <corruption_type>.npy   (e.g., fog.npy, gaussian_noise.npy)

    Each .npy file contains 50,000 images (10k per severity x 5 severities).
    Severities are interleaved: first 10k = severity 1, next 10k = severity 2, etc.
    """

    _cache = {}  # class-level cache to avoid reloading

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        transform: Optional[transforms.Compose] = None,
        root: str = "./data/CIFAR-100-C",
    ):
        if not 1 <= severity <= 5:
            raise ValueError("severity must be in [1, 5]")
        self.corruption_type = corruption_type
        self.severity = severity
        self.root = root
        self.transform = transform if transform is not None else get_transforms(train=False)

        cache_key = f"{corruption_type}_{severity}"
        if cache_key not in CIFAR100C._cache:
            data_path = os.path.join(root, f"{corruption_type}.npy")
            labels_path = os.path.join(root, "labels.npy")

            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"CIFAR-100-C data not found at {data_path}.\n"
                    "Please download from https://github.com/hendrycks/robustness "
                    "and place in ./data/CIFAR-100-C/"
                )

            all_data = np.load(data_path)  # shape (50000, 32, 32, 3)
            all_labels = np.load(labels_path)  # shape (50000,)

            start = (severity - 1) * 10000
            end = severity * 10000
            self.data = all_data[start:end]
            self.labels = all_labels[start:end].astype(np.int64)
            CIFAR100C._cache[cache_key] = (self.data, self.labels)
        else:
            self.data, self.labels = CIFAR100C._cache[cache_key]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        label = self.labels[idx]
        # Convert numpy HWC uint8 -> PIL Image for torchvision transforms
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_corrupted_loader(
    corruption_type: str,
    severity: int,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Return a DataLoader for a specific CIFAR-100-C corruption and severity."""
    bs = batch_size if batch_size is not None else DATA_CFG.batch_size
    nw = num_workers if num_workers is not None else DATA_CFG.num_workers
    dataset = CIFAR100C(corruption_type=corruption_type, severity=severity)
    return DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )


if __name__ == "__main__":
    # Sanity check
    print("Loading CIFAR-100...")
    train_loader, val_loader, test_loader = get_dataloaders()
    for imgs, labels in train_loader:
        print(f"Train batch: {imgs.shape}, labels: {labels.shape}")
        break
    for imgs, labels in val_loader:
        print(f"Val batch:   {imgs.shape}, labels: {labels.shape}")
        break
    for imgs, labels in test_loader:
        print(f"Test batch:  {imgs.shape}, labels: {labels.shape}")
        break
    print("Data loaders OK.")
