"""Static visualizations for mcdnn-image-bench.

Reads `viz/history.json` and `viz/weights.npz` produced by
`mcdnn_image_bench.py` and renders four static PNGs into `viz/`:

  - training_curves.png    : train loss / train acc / test err vs epoch
  - confusion_matrix.png   : 10x10 test-set confusion matrix (final epoch)
  - first_layer_weights.png: 64 random columns of W0 reshaped to 28x28
  - misclassified.png      : 24 misclassified test images with (true, pred)

Usage:
    python3 visualize_mcdnn_image_bench.py --seed 0 --out-dir viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcdnn_image_bench import MLP, load_mnist


def _load_weights(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def _make_model_from_weights(weights: dict[str, np.ndarray]) -> MLP:
    """Reconstruct an MLP and copy weights in (no training)."""
    n_layers = sum(1 for k in weights if k.startswith("W"))
    sizes = [weights["W0"].shape[0]]
    for i in range(n_layers):
        sizes.append(weights[f"W{i}"].shape[1])
    rng = np.random.default_rng(0)
    model = MLP(rng, sizes=tuple(sizes))
    for k, v in weights.items():
        model.params[k] = v.astype(np.float32)
    return model


def plot_training_curves(history: dict, out: Path) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "-o", color="tab:blue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train cross-entropy")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "-o", color="tab:green", label="train")
    ax.plot(epochs, history["test_acc"], "-s", color="tab:red", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Train / test accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.9, 1.005)

    ax = axes[2]
    ax.plot(epochs, [e * 100 for e in history["test_err"]], "-o", color="tab:purple")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test error (%)")
    ax.set_title("Test error")
    ax.grid(alpha=0.3)
    # mark LR decay if visible
    if "lr" in history and len(set(history["lr"])) > 1:
        for i in range(1, len(history["lr"])):
            if history["lr"][i] != history["lr"][i - 1]:
                ax.axvline(epochs[i], color="gray", linestyle="--", alpha=0.6)
                ax.text(
                    epochs[i],
                    max(history["test_err"]) * 100 * 0.9,
                    "lr decay",
                    rotation=90,
                    fontsize=8,
                    color="gray",
                )

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion(confusion: list[list[int]], out: Path) -> None:
    cm = np.array(confusion, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 6))
    # log-scale colormap so off-diagonals are visible despite huge diagonal
    cm_show = np.log10(cm + 1)
    im = ax.imshow(cm_show, cmap="Blues")
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if cm_show[i, j] > cm_show.max() * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title("Test-set confusion matrix (log10 scale)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_first_layer_weights(W0: np.ndarray, out: Path, n: int = 64) -> None:
    rng = np.random.default_rng(0)
    cols = rng.choice(W0.shape[1], size=n, replace=False)
    grid = int(np.sqrt(n))
    fig, axes = plt.subplots(grid, grid, figsize=(8, 8))
    for k, ax in enumerate(axes.flat):
        w = W0[:, cols[k]].reshape(28, 28)
        # symmetric color range so + and - weights are visible
        vmax = float(np.abs(w).max())
        ax.imshow(w, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("First-layer weights — 64 random hidden units (seismic, ±max)", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_misclassified(model: MLP, x: np.ndarray, y: np.ndarray, out: Path,
                       n: int = 24) -> None:
    pred = model.predict(x)
    wrong = np.where(pred != y)[0]
    rng = np.random.default_rng(0)
    if len(wrong) == 0:
        return
    sel = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    grid_h, grid_w = 4, 6
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.4, grid_h * 1.6))
    for k, ax in enumerate(axes.flat):
        if k >= len(sel):
            ax.axis("off")
            continue
        i = sel[k]
        ax.imshow(x[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={int(y[i])} p={int(pred[i])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Misclassified test images "
        f"(showing {min(n, len(wrong))} of {len(wrong)} total errors)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "viz")
    args = ap.parse_args()

    out = args.out_dir
    history_path = out / "history.json"
    weights_path = out / "weights.npz"
    if not history_path.exists() or not weights_path.exists():
        raise SystemExit(
            f"missing inputs: run `python3 mcdnn_image_bench.py --seed {args.seed}` first."
        )

    with open(history_path) as f:
        result = json.load(f)
    weights = _load_weights(weights_path)

    print("rendering training curves ...", flush=True)
    plot_training_curves(result["history"], out / "training_curves.png")

    print("rendering confusion matrix ...", flush=True)
    plot_confusion(result["confusion"], out / "confusion_matrix.png")

    print("rendering first-layer weights ...", flush=True)
    plot_first_layer_weights(weights["W0"], out / "first_layer_weights.png")

    print("rendering misclassified examples ...", flush=True)
    data = load_mnist()
    model = _make_model_from_weights(weights)
    plot_misclassified(model, data["x_test"], data["y_test"], out / "misclassified.png")

    print(f"\nwrote 4 PNGs to {out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
