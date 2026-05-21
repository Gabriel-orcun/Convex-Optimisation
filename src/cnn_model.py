"""Small CNN for CIFAR-10 — the non-convex side of our comparison.

The architecture is intentionally modest (≈200k parameters): the goal of this
project is **not** to chase state-of-the-art accuracy, but to obtain a
non-convex loss landscape on which we can replay the same optimizers compared
on the convex baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Three conv blocks + FC head.

    Conv-BN-ReLU x2 -> MaxPool -> Conv-BN-ReLU x2 -> MaxPool -> Conv-BN-ReLU -> AvgPool -> FC.

    Batch normalization is included to keep training stable across optimizers,
    so that any differences we observe come from the *optimizer*, not from
    initialization or normalization tricks.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 8x8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                      # 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.flatten(z, 1)
        return self.classifier(z)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
