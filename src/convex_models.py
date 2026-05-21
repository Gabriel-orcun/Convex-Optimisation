"""Convex baselines: multinomial logistic regression and multiclass linear SVM.

Both losses are convex in the parameters. We provide:

  * an analytical loss and gradient — this lets us plug in any of the numpy
    optimizers from :mod:`src.optimizers` and inspect convergence in a setting
    where the course's theory is fully applicable;
  * a training routine that records loss / accuracy / gradient norm at every
    step, for plotting and comparison.

Both models use a single weight matrix ``W`` of shape (D+1, K) where the last
row of the input is augmented with a constant 1 to absorb the bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .optimizers import build_optimizer

Array = np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _augment_bias(X: Array) -> Array:
    """Append a column of 1s so the bias is folded into W."""
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def _one_hot(y: Array, num_classes: int) -> Array:
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def accuracy(W: Array, X: Array, y: Array) -> float:
    scores = _augment_bias(X) @ W
    preds = scores.argmax(axis=1)
    return float((preds == y).mean())


# ---------------------------------------------------------------------------
# Multinomial logistic regression  (softmax + cross-entropy + L2)
# ---------------------------------------------------------------------------


def _softmax(z: Array) -> Array:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logistic_loss_and_grad(
    W: Array, X_aug: Array, Y_onehot: Array, l2: float
) -> Tuple[float, Array]:
    """L = -1/N * sum log p_{y_i}(x_i)  +  l2/2 * ||W||^2 .

    L is convex in W (sum of a log-sum-exp composed with a linear map, plus a
    quadratic regularizer). Its gradient is provided in closed form.
    """
    N = X_aug.shape[0]
    scores = X_aug @ W
    P = _softmax(scores)
    log_lik = -np.log(np.clip(P[np.arange(N), Y_onehot.argmax(axis=1)], 1e-12, 1.0)).mean()
    reg = 0.5 * l2 * float(np.sum(W * W))
    loss = log_lik + reg

    grad = X_aug.T @ (P - Y_onehot) / N + l2 * W
    return float(loss), grad


# ---------------------------------------------------------------------------
# Multiclass linear SVM  (Crammer–Singer style, squared hinge)
# ---------------------------------------------------------------------------


def svm_loss_and_grad(
    W: Array, X_aug: Array, y: Array, l2: float, margin: float = 1.0
) -> Tuple[float, Array]:
    """One-vs-rest squared hinge loss:

        L(W) = 1/N * sum_i sum_{j != y_i} max(0, m + s_j - s_{y_i})^2 + l2/2 ||W||^2

    Squared hinge is convex and differentiable everywhere (unlike the regular
    hinge), which makes the gradient well-defined for vanilla GD.
    """
    N = X_aug.shape[0]
    scores = X_aug @ W                                 # (N, K)
    correct = scores[np.arange(N), y][:, None]         # (N, 1)
    margins = np.maximum(0.0, scores - correct + margin)
    margins[np.arange(N), y] = 0.0

    loss = float((margins ** 2).sum() / N) + 0.5 * l2 * float(np.sum(W * W))

    # gradient
    coef = 2.0 * margins                               # (N, K)
    coef[np.arange(N), y] -= coef.sum(axis=1)
    grad = X_aug.T @ coef / N + l2 * W
    return loss, grad


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------


LossFn = Callable[[Array, Array, Array, float], Tuple[float, Array]]


@dataclass
class TrainHistory:
    losses: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    step_idx: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, list]:
        return {
            "losses": self.losses,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "grad_norms": self.grad_norms,
            "step_idx": self.step_idx,
        }


def train_linear_model(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    *,
    loss_fn: LossFn,
    num_classes: int,
    optimizer: str = "adam",
    optim_kwargs: dict | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    l2: float = 1e-4,
    eval_every: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[Array, TrainHistory]:
    """Mini-batch SGD-style training for any convex linear model.

    The same routine works for logistic regression and linear SVM by swapping
    ``loss_fn``. Set ``batch_size`` to the full training size to recover full
    (deterministic) gradient descent.
    """
    rng = np.random.default_rng(seed)

    X_train_aug = _augment_bias(X_train)
    X_test_aug = _augment_bias(X_test)
    D_aug, K = X_train_aug.shape[1], num_classes

    # Y_onehot only used by softmax — SVM uses raw y, so pass both
    Y_onehot = _one_hot(y_train, num_classes)

    W = rng.standard_normal((D_aug, K)).astype(np.float32) * 0.01
    opt = build_optimizer(optimizer, shape=W.shape, **(optim_kwargs or {}))

    history = TrainHistory()
    step = 0
    N = X_train.shape[0]

    for epoch in range(epochs):
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            Xb = X_train_aug[idx]
            if loss_fn is svm_loss_and_grad:
                loss, grad = loss_fn(W, Xb, y_train[idx], l2)
            else:
                loss, grad = loss_fn(W, Xb, Y_onehot[idx], l2)
            W = opt.step(W, grad)

            if step % eval_every == 0:
                history.losses.append(loss)
                history.grad_norms.append(float(np.linalg.norm(grad)))
                history.step_idx.append(step)
            step += 1

        tr_acc = accuracy(W, X_train, y_train)
        te_acc = accuracy(W, X_test, y_test)
        history.train_acc.append(tr_acc)
        history.test_acc.append(te_acc)
        if verbose:
            print(
                f"[{optimizer:>8s}] epoch {epoch + 1:3d}/{epochs}  "
                f"loss={loss:.4f}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}"
            )

    return W, history
