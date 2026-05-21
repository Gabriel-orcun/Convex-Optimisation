"""Training entry-point and CNN training loop.

Run from the project root:

    python -m src.train --model logistic --optimizer adam --epochs 30
    python -m src.train --model svm      --optimizer momentum --epochs 30
    python -m src.train --model cnn      --optimizer sgd  --epochs 30 --momentum 0.9

Results are written to ``experiments/<run_name>/`` as a JSON file with the
recorded metrics so that notebooks can load and plot them uniformly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .cnn_model import SmallCNN, num_params
from .convex_models import (
    logistic_loss_and_grad,
    svm_loss_and_grad,
    train_linear_model,
)
from .data_loader import NUM_CLASSES, load_cifar10_numpy, load_cifar10_torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


# ---------------------------------------------------------------------------
# CNN training loop  (torch.optim — we observe, we don't reimplement here)
# ---------------------------------------------------------------------------


@dataclass
class CNNHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    epoch_time: List[float] = field(default_factory=list)


def _build_torch_optimizer(name: str, params, lr: float, momentum: float, weight_decay: float):
    name = name.lower()
    if name in {"sgd", "gd"}:
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def _grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


def _epoch_metrics(model: nn.Module, loader, device, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(loss_fn(logits, y).item()) * x.size(0)
            total_correct += int((logits.argmax(1) == y).sum().item())
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def train_cnn(
    *,
    optimizer: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    augment: bool,
    seed: int,
    device: str,
) -> CNNHistory:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = load_cifar10_torch(
        batch_size=batch_size, augment=augment, num_workers=2
    )

    model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    print(f"CNN parameters: {num_params(model):,}")

    opt = _build_torch_optimizer(optimizer, model.parameters(), lr, momentum, weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = CNNHistory()
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss, running_correct, n = 0.0, 0, 0
        running_grad = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            running_grad += _grad_global_norm(model)
            opt.step()

            running_loss += float(loss.item()) * x.size(0)
            running_correct += int((logits.argmax(1) == y).sum().item())
            n += x.size(0)

        epoch_time = time.time() - t0
        train_loss = running_loss / n
        train_acc = running_correct / n
        test_loss, test_acc = _epoch_metrics(model, test_loader, device, loss_fn)
        avg_grad = running_grad / max(1, len(train_loader))

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.test_loss.append(test_loss)
        history.test_acc.append(test_acc)
        history.grad_norm.append(avg_grad)
        history.epoch_time.append(epoch_time)

        print(
            f"[{optimizer:>8s}] epoch {epoch + 1:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
            f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
            f"|grad|={avg_grad:.3f}  ({epoch_time:.1f}s)"
        )
    return history


# ---------------------------------------------------------------------------
# Convex-model entry-point
# ---------------------------------------------------------------------------


def train_convex(
    *,
    model: str,
    optimizer: str,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    subset: int | None,
    seed: int,
) -> Dict[str, list]:
    X_train, y_train, X_test, y_test = load_cifar10_numpy(
        flatten=True, normalize=True, subset=subset, seed=seed
    )
    if model == "logistic":
        loss_fn = logistic_loss_and_grad
    elif model == "svm":
        loss_fn = svm_loss_and_grad
    else:
        raise ValueError(f"unknown convex model {model!r}")

    _, hist = train_linear_model(
        X_train,
        y_train,
        X_test,
        y_test,
        loss_fn=loss_fn,
        num_classes=NUM_CLASSES,
        optimizer=optimizer,
        optim_kwargs={"lr": lr},
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        seed=seed,
    )
    return hist.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _save_history(run_name: str, payload: dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPERIMENTS_DIR / f"{run_name}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    p.add_argument("--model", choices=["logistic", "svm", "cnn"], required=True)
    p.add_argument("--optimizer", default="sgd",
                   choices=["gd", "sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="learning rate (sane default per optimizer)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4, help="CNN only")
    p.add_argument("--l2", type=float, default=1e-4, help="convex models only")
    p.add_argument("--subset", type=int, default=None, help="cap training samples (convex models)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    default_lrs = {"adam": 1e-3, "rmsprop": 1e-3, "adagrad": 1e-2,
                   "nesterov": 1e-2, "momentum": 1e-2, "sgd": 1e-2, "gd": 1e-2}
    lr = args.lr if args.lr is not None else default_lrs[args.optimizer]
    run_name = args.run_name or f"{args.model}_{args.optimizer}_lr{lr}_bs{args.batch_size}_ep{args.epochs}"

    if args.model == "cnn":
        hist = train_cnn(
            optimizer=args.optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            augment=not args.no_augment,
            seed=args.seed,
            device=args.device,
        )
        payload = {"config": vars(args), "lr": lr, "history": asdict(hist)}
    else:
        hist = train_convex(
            model=args.model,
            optimizer=args.optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr,
            l2=args.l2,
            subset=args.subset,
            seed=args.seed,
        )
        payload = {"config": vars(args), "lr": lr, "history": hist}

    out = _save_history(run_name, payload)
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
