"""CIFAR-10 loading and preprocessing.

Two output formats are exposed:

- ``load_cifar10_numpy`` returns flat float32 arrays in [0, 1], ready for the
  convex baselines (logistic regression, linear SVM) where we treat each image
  as a single 3072-dimensional vector.
- ``load_cifar10_torch`` returns torch DataLoaders with standard CIFAR-10
  normalization and optional augmentation, suitable for the CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
IMG_SHAPE = (3, 32, 32)
FLAT_DIM = 3 * 32 * 32


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_cifar10_numpy(
    flatten: bool = True,
    normalize: bool = True,
    subset: int | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 as numpy arrays.

    Parameters
    ----------
    flatten : bool
        If True, images are returned as (N, 3072) vectors. Otherwise (N, 3, 32, 32).
    normalize : bool
        If True, pixels are mapped to [0, 1] and channel-normalized.
    subset : int or None
        Optional cap on the number of training samples (useful for quick experiments).
    seed : int
        RNG seed for subset sampling.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    _ensure_data_dir()
    train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)

    X_train = train.data.astype(np.float32) / 255.0
    y_train = np.asarray(train.targets, dtype=np.int64)
    X_test = test.data.astype(np.float32) / 255.0
    y_test = np.asarray(test.targets, dtype=np.int64)

    if normalize:
        mean = np.array(CIFAR10_MEAN, dtype=np.float32)
        std = np.array(CIFAR10_STD, dtype=np.float32)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    # (N, H, W, C) -> (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    if subset is not None and subset < X_train.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_train.shape[0], size=subset, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test


def load_cifar10_torch(
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for the CNN experiments."""
    _ensure_data_dir()

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_tf = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        if augment
        else transforms.Compose([transforms.ToTensor(), normalize])
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
