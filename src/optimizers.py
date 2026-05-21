"""Hand-written first-order optimizers operating on numpy arrays.

These mirror the standard formulations used in the course and in PyTorch, but
are written from scratch so that we can inspect, plot, and reason about them
when training the convex models (logistic regression, linear SVM).

Each optimizer exposes the same interface:

    opt = Optimizer(params_shape, **hyperparams)
    new_params = opt.step(params, grad)

where ``params`` and ``grad`` are numpy arrays of identical shape.

For the CNN experiments we rely on ``torch.optim`` to avoid re-implementing
auto-diff machinery — the goal in that setting is empirical observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

Array = np.ndarray
Shape = Tuple[int, ...]


@dataclass
class GD:
    """Vanilla (batch or full) gradient descent.

    Update:  x_{k+1} = x_k - lr * g_k
    """

    shape: Shape
    lr: float = 1e-2

    def step(self, params: Array, grad: Array) -> Array:
        return params - self.lr * grad


@dataclass
class SGDMomentum:
    """Heavy-ball / Polyak momentum.

    v_{k+1} = mu * v_k + g_k
    x_{k+1} = x_k - lr * v_{k+1}
    """

    shape: Shape
    lr: float = 1e-2
    momentum: float = 0.9
    velocity: Array = field(init=False)

    def __post_init__(self):
        self.velocity = np.zeros(self.shape, dtype=np.float32)

    def step(self, params: Array, grad: Array) -> Array:
        self.velocity = self.momentum * self.velocity + grad
        return params - self.lr * self.velocity


@dataclass
class Nesterov:
    """Nesterov accelerated gradient (NAG).

    Look-ahead form:
        v_{k+1} = mu * v_k - lr * g(x_k + mu * v_k)
        x_{k+1} = x_k + v_{k+1}

    Because we don't have control over where the gradient is evaluated from
    inside the optimizer (the caller passes ``grad``), we use the equivalent
    "PyTorch" reformulation that only changes the parameter update.
    """

    shape: Shape
    lr: float = 1e-2
    momentum: float = 0.9
    velocity: Array = field(init=False)

    def __post_init__(self):
        self.velocity = np.zeros(self.shape, dtype=np.float32)

    def step(self, params: Array, grad: Array) -> Array:
        prev_v = self.velocity
        self.velocity = self.momentum * self.velocity + grad
        return params - self.lr * (grad + self.momentum * self.velocity - self.momentum * prev_v)


@dataclass
class AdaGrad:
    """Per-parameter learning rate, accumulating squared gradients.

    G_{k+1} = G_k + g_k**2
    x_{k+1} = x_k - lr * g_k / (sqrt(G_{k+1}) + eps)
    """

    shape: Shape
    lr: float = 1e-2
    eps: float = 1e-8
    accum: Array = field(init=False)

    def __post_init__(self):
        self.accum = np.zeros(self.shape, dtype=np.float32)

    def step(self, params: Array, grad: Array) -> Array:
        self.accum += grad * grad
        return params - self.lr * grad / (np.sqrt(self.accum) + self.eps)


@dataclass
class RMSProp:
    """Running average of squared gradients (Tieleman & Hinton).

    E[g^2]_k = rho * E[g^2]_{k-1} + (1 - rho) * g_k**2
    x_{k+1}  = x_k - lr * g_k / (sqrt(E[g^2]_k) + eps)
    """

    shape: Shape
    lr: float = 1e-3
    rho: float = 0.9
    eps: float = 1e-8
    avg_sq: Array = field(init=False)

    def __post_init__(self):
        self.avg_sq = np.zeros(self.shape, dtype=np.float32)

    def step(self, params: Array, grad: Array) -> Array:
        self.avg_sq = self.rho * self.avg_sq + (1.0 - self.rho) * grad * grad
        return params - self.lr * grad / (np.sqrt(self.avg_sq) + self.eps)


@dataclass
class Adam:
    """Adam (Kingma & Ba 2014) with bias correction.

    m_k = b1 * m_{k-1} + (1-b1) g_k
    v_k = b2 * v_{k-1} + (1-b2) g_k**2
    m_hat = m_k / (1 - b1**t), v_hat = v_k / (1 - b2**t)
    x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + eps)
    """

    shape: Shape
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: Array = field(init=False)
    v: Array = field(init=False)
    t: int = field(default=0, init=False)

    def __post_init__(self):
        self.m = np.zeros(self.shape, dtype=np.float32)
        self.v = np.zeros(self.shape, dtype=np.float32)

    def step(self, params: Array, grad: Array) -> Array:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "gd": GD,
    "sgd": GD,  # SGD = GD on a minibatch
    "momentum": SGDMomentum,
    "nesterov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
}


def build_optimizer(name: str, shape: Shape, **kwargs):
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose one of {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](shape=shape, **kwargs)
