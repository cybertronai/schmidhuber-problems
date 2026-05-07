"""Render mnist_deep_mlp.gif — first-layer receptive fields evolving across training,
plus a small training-curve sub-axis so the same animation shows both
"what's being learned" (the filters) and "how well" (the metrics).

Run:
  python3 make_mnist_deep_mlp_gif.py --seed 0 --epochs 8 --fps 3
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_deep_mlp import (
    DeepMLP,
    augment_batch,
    load_mnist,
    _flatten,
)


def _snapshot(
    model: DeepMLP,
    epoch_idx: int,
    total_epochs: int,
    history: dict,
    n_filters: int,
) -> np.ndarray:
    """Render one frame; return RGB uint8 ndarray."""
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)

    # Filter grid.
    grid_ax = fig.add_subplot(gs[0, 0])
    grid_ax.axis("off")
    grid_ax.set_title(
        f"Layer-1 receptive fields  (epoch {epoch_idx}/{total_epochs})",
        fontsize=10,
    )
    cols = 8
    rows = max(1, n_filters // cols)
    W = model.W[0]  # (784, hidden)
    n_show = min(n_filters, W.shape[1])
    canvas = np.zeros((rows * 28 + (rows - 1), cols * 28 + (cols - 1)), dtype=np.float32)
    canvas[:] = 0.0
    for k in range(n_show):
        r = k // cols
        c = k % cols
        w = W[:, k].reshape(28, 28)
        v = max(abs(w.min()), abs(w.max())) + 1e-8
        wn = w / v  # in [-1, 1]
        canvas[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = wn
    grid_ax.imshow(canvas, vmin=-1, vmax=1, cmap="RdBu_r")

    # Loss / error curve.
    curve_ax = fig.add_subplot(gs[0, 1])
    eps = history["epoch"][:epoch_idx]
    if eps:
        curve_ax.plot(eps, [e * 100 for e in history["train_err"][:epoch_idx]],
                      "-o", color="#1f77b4", label="train err %")
        curve_ax.plot(eps, [e * 100 for e in history["test_err"][:epoch_idx]],
                      "-s", color="#d62728", label="test err %")
    curve_ax.set_xlim(0, total_epochs + 0.5)
    max_err = max(history["train_err"]) if history["train_err"] else 0.2
    curve_ax.set_ylim(0, max(20.0, max_err * 110))
    curve_ax.set_xlabel("epoch")
    curve_ax.set_ylabel("error %")
    curve_ax.set_title("Train vs test error", fontsize=10)
    curve_ax.grid(alpha=0.3)
    curve_ax.legend(loc="upper right", fontsize=8)
    if eps and not np.isnan(history["test_err"][epoch_idx - 1]):
        te = history["test_err"][epoch_idx - 1] * 100
        curve_ax.annotate(f"test err {te:.2f}%",
                          xy=(eps[-1], te),
                          xytext=(0.45, 0.6), textcoords="axes fraction",
                          fontsize=10, color="#d62728")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def train_with_snapshots(
    seed: int,
    epochs: int,
    hidden_sizes: tuple[int, ...],
    batch_size: int,
    n_filters: int,
    cache_dir: Path | None,
) -> tuple[list[np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    train_x, train_y, test_x, test_y = load_mnist(cache_dir)
    layer_sizes = [28 * 28, *hidden_sizes, 10]
    model = DeepMLP(layer_sizes, rng)

    history: dict = {"epoch": [], "train_loss": [], "train_err": [], "test_err": [],
                     "lr": [], "wallclock_s": []}
    frames: list[np.ndarray] = []
    # Initial frame (epoch 0).
    frames.append(_snapshot(model, 0, epochs, history, n_filters))

    aug_rng = np.random.default_rng(seed + 1)
    perm_rng = np.random.default_rng(seed + 2)
    lr = 0.05
    momentum = 0.9
    weight_decay = 1e-5
    for epoch in range(epochs):
        order = perm_rng.permutation(train_x.shape[0])
        running_loss = 0.0
        running_n = 0
        running_err = 0
        for i in range(0, train_x.shape[0], batch_size):
            idx = order[i:i + batch_size]
            xb = augment_batch(train_x[idx], aug_rng)
            yb = train_y[idx]
            xb_flat = _flatten(xb)
            logits, acts = model.forward(xb_flat)
            dlogits, loss = model.softmax_xent_grad(logits, yb)
            dWs, dbs = model.backward(acts, dlogits)
            model.sgd_step(dWs, dbs, lr, momentum, weight_decay)
            running_loss += loss * xb.shape[0]
            running_n += xb.shape[0]
            running_err += int((logits.argmax(axis=1) != yb).sum())
        train_loss = running_loss / running_n
        train_err = running_err / running_n
        preds = model.predict(_flatten(test_x))
        test_err = float((preds != test_y).mean())
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_err"].append(train_err)
        history["test_err"].append(test_err)
        history["lr"].append(lr)
        history["wallclock_s"].append(0.0)
        print(f"  epoch {epoch+1}/{epochs}  test_err {test_err*100:.2f}%", flush=True)

        frames.append(_snapshot(model, epoch + 1, epochs, history, n_filters))
        lr *= 0.95
    return frames, history


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-filters", type=int, default=64)
    p.add_argument("--out", type=str, default="mnist_deep_mlp.gif")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    frames, _hist = train_with_snapshots(
        seed=args.seed,
        epochs=args.epochs,
        hidden_sizes=tuple(args.hidden),
        batch_size=args.batch_size,
        n_filters=args.n_filters,
        cache_dir=cache_dir,
    )

    # Hold the final frame longer.
    frames = frames + [frames[-1]] * 4

    import imageio.v2 as imageio
    imageio.mimsave(args.out, frames, duration=1.0 / max(1, args.fps), loop=0)
    print(f"  wrote {args.out}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
