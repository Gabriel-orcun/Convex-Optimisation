"""Plotting and small utilities shared across notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_run(name: str) -> dict:
    path = EXPERIMENTS_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_loss_curves(runs: Dict[str, dict], log_y: bool = True, ax=None):
    """Plot training loss vs step/epoch for several runs side-by-side.

    Accepts both convex-model histories (key 'losses') and CNN histories
    (key 'train_loss').
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, payload in runs.items():
        h = payload["history"]
        if "losses" in h:                 # convex
            ax.plot(h["step_idx"], h["losses"], label=name)
            ax.set_xlabel("optimizer step")
        else:                              # CNN
            ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=name)
            ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_grad_norms(runs: Dict[str, dict], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, payload in runs.items():
        h = payload["history"]
        if "grad_norms" in h:
            ax.plot(h["step_idx"], h["grad_norms"], label=name)
            ax.set_xlabel("optimizer step")
        else:
            ax.plot(range(1, len(h["grad_norm"]) + 1), h["grad_norm"], label=name)
            ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_accuracy(runs: Dict[str, dict], split: str = "test", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    key = f"{split}_acc"
    for name, payload in runs.items():
        h = payload["history"]
        if key in h:
            ax.plot(range(1, len(h[key]) + 1), h[key], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{split} accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def summarize(runs: Iterable[str]) -> Dict[str, dict]:
    """Load runs by name and return a dict suitable for the plot helpers."""
    return {name: load_run(name) for name in runs}
