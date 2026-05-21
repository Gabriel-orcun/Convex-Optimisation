# Convex Optimisation — Project

Group project for the EFREI Convex Optimization course (M. Léonard Benedetti, 2026).

**Subject A** — applied optimization on a concrete ML problem.
**Dataset** — [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
**Focus** — comparison of first- and second-order optimization methods (SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam, L-BFGS) on **both** a convex setting (logistic regression / linear SVM) and a non-convex one (small CNN).

**Deadline**: Sunday 2026-06-07, 23:59 CEST.

## Team

| Member  | Branch    |
|---------|-----------|
| Ilyan   | `Ilyann`  |
| Alexan  | `Alexan`  |
| Gabriel | `Gabriel` |
| Thi-Tho | `ThiTho`  |

Each member works on their personal branch and opens a PR to `main` when their part is ready.

## Repository layout

```
.
├── data/                  # CIFAR-10 binaries (gitignored, auto-downloaded)
├── src/                   # Python source
│   ├── data_loader.py     # CIFAR-10 loading + preprocessing
│   ├── optimizers.py      # Hand-written optimizers (SGD, momentum, ...)
│   ├── convex_models.py   # Logistic regression, linear SVM
│   ├── cnn_model.py       # Small CNN
│   ├── train.py           # Training loops + metrics logging
│   └── utils.py           # Plotting + helpers
├── notebooks/             # Jupyter experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_convex_baseline.ipynb
│   └── 03_nonconvex_cnn.ipynb
├── experiments/           # Saved metrics + plots (gitignored)
├── report/                # LaTeX report
│   ├── main.tex
│   └── references.bib
├── requirements.txt
└── README.md
```

## Approach — two regimes

The course is about **convex** optimization, so we structure the work around two complementary settings:

### 1. Convex regime — logistic regression & linear SVM

Loss is strictly convex → the course's theory applies directly. We compare convergence rates
of GD, SGD, SGD+momentum, Nesterov, AdaGrad, RMSProp, Adam, and L-BFGS, and confront empirical
behaviour with theoretical bounds (O(1/k), O(1/k²) for Nesterov, quadratic local rate for Newton).

### 2. Non-convex regime — small CNN

Same optimizers, same dataset, but the loss landscape is non-convex. We observe what changes,
discuss why first-order methods still work in practice (over-parametrization, benign non-convexity,
saddle-point escape) and why 2nd-order methods are rare at scale.

This split is the backbone of both the code and the report.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically the first time you call `load_cifar10()`.

## Running an experiment

```bash
python -m src.train --model logistic --optimizer adam --epochs 30
python -m src.train --model cnn      --optimizer sgd  --epochs 30 --momentum 0.9
```

Results (loss curves, metrics, plots) are written to `experiments/<run-name>/`.
