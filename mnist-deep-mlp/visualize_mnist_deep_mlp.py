"""Static visualizations for mnist-deep-mlp.

Trains the model for a small number of epochs and dumps four PNGs to
`viz/`:

  * training_curves.png        — train loss / err and test error vs epoch
  * weights_layer1.png         — first 64 hidden units' incoming weights as 28x28 receptive fields
  * augmentation_samples.png   — original digits next to several augmented versions
  * test_predictions.png       — sample test images with predicted / true labels (mistakes red)

Run:
  python3 visualize_mnist_deep_mlp.py --seed 0 --epochs 6 --outdir viz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_deep_mlp import (
    DeepMLP,
    TrainConfig,
    augment_batch,
    load_mnist,
    train,
    _flatten,
)


def plot_training_curves(history: dict, out: Path) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "-o", label="train loss", color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, [e * 100 for e in history["train_err"]], "-o", label="train err %", color="#1f77b4")
    ax.plot(epochs, [e * 100 for e in history["test_err"]], "-s", label="test err %", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error %")
    ax.set_title("Train vs test error")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_layer1_weights(model: DeepMLP, out: Path) -> None:
    W = model.W[0]  # (784, hidden)
    n_show = min(64, W.shape[1])
    cols = 8
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n_show:
            continue
        w = W[:, i].reshape(28, 28)
        # Center & normalize for display.
        v = max(abs(w.min()), abs(w.max())) + 1e-8
        ax.imshow(w, vmin=-v, vmax=v, cmap="RdBu_r")
    fig.suptitle(
        f"First {n_show} hidden-unit receptive fields (W^(1)), "
        f"layer sizes = {model.layer_sizes}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_augmentation_samples(train_x: np.ndarray, out: Path, n_originals: int = 6, n_aug: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed + 99)
    idx = rng.choice(train_x.shape[0], size=n_originals, replace=False)
    originals = train_x[idx]

    fig, axes = plt.subplots(n_originals, n_aug + 1, figsize=(1.2 * (n_aug + 1), 1.2 * n_originals))
    if n_originals == 1:
        axes = axes.reshape(1, -1)

    for r in range(n_originals):
        axes[r, 0].imshow(originals[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("original" if r == 0 else "", fontsize=8)
        axes[r, 0].axis("off")

    for c in range(1, n_aug + 1):
        aug_rng = np.random.default_rng(seed + 100 + c)
        aug = augment_batch(originals, aug_rng)
        for r in range(n_originals):
            axes[r, c].imshow(aug[r], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"aug {c}" if r == 0 else "", fontsize=8)
            axes[r, c].axis("off")

    fig.suptitle("On-the-fly augmentation: affine + elastic deformation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_test_predictions(model: DeepMLP, test_x: np.ndarray, test_y: np.ndarray, out: Path,
                          n_correct: int = 16, n_wrong: int = 16, seed: int = 0) -> None:
    preds = model.predict(_flatten(test_x))
    rng = np.random.default_rng(seed + 7)

    correct = np.where(preds == test_y)[0]
    wrong = np.where(preds != test_y)[0]
    n_wrong = min(n_wrong, wrong.size)
    n_correct = min(n_correct, correct.size)

    sel_correct = rng.choice(correct, size=n_correct, replace=False) if n_correct else np.array([], dtype=int)
    sel_wrong = wrong if n_wrong == wrong.size else rng.choice(wrong, size=n_wrong, replace=False)

    cols = 8
    rows_top = (n_correct + cols - 1) // cols
    rows_bot = (n_wrong + cols - 1) // cols
    rows = rows_top + rows_bot + 2  # +2 for header rows

    fig = plt.figure(figsize=(cols, rows * 1.05))
    # Header text rows (axes that just hold a label).
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.1)

    def show_row(start: int, indices: np.ndarray, title: str) -> int:
        head_ax = fig.add_subplot(gs[start, :])
        head_ax.axis("off")
        head_ax.set_title(title, fontsize=11, loc="left")
        for k, idx in enumerate(indices):
            r = start + 1 + k // cols
            c = k % cols
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(test_x[idx], cmap="gray", vmin=0, vmax=1)
            color = "tab:green" if preds[idx] == test_y[idx] else "tab:red"
            ax.set_title(f"pred {preds[idx]} | true {test_y[idx]}", fontsize=7, color=color)
            ax.axis("off")
        return start + 1 + (len(indices) + cols - 1) // cols

    next_row = show_row(0, sel_correct, "Correct predictions")
    show_row(next_row, sel_wrong, f"Misclassified ({wrong.size} of {test_y.size} total)")

    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6,
                   help="short run for visualizations (full run is in mnist_deep_mlp.py)")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        seed=args.seed,
        hidden_sizes=tuple(args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=200,
        cache_dir=args.cache_dir,
    )
    result = train(cfg)
    model: DeepMLP = result["model"]
    history = result["history"]

    # Re-load data once for the static plots (cheap; the file is already on disk).
    train_x, train_y, test_x, test_y = load_mnist(
        Path(args.cache_dir) if args.cache_dir else None
    )

    plot_training_curves(history, out / "training_curves.png")
    plot_layer1_weights(model, out / "weights_layer1.png")
    plot_augmentation_samples(train_x, out / "augmentation_samples.png", seed=args.seed)
    plot_test_predictions(model, test_x, test_y, out / "test_predictions.png", seed=args.seed)


if __name__ == "__main__":
    main()
