"""Generate `mcdnn_image_bench.gif` showing training dynamics.

Approach: re-train (a slightly slimmer model so the GIF run stays under
~30 s) and snapshot per-epoch (a) test error, (b) 16 first-layer-weight
filters, (c) confusion matrix on a 1k sample. Each frame is one epoch.

Usage:
    python3 make_mcdnn_image_bench_gif.py --seed 0
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# matplotlib's PillowWriter handles GIF assembly without an extra dep
from matplotlib.animation import PillowWriter

from mcdnn_image_bench import MLP, load_mnist


def train_with_snapshots(
    seed: int,
    epochs: int,
    sizes: tuple[int, ...],
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_decay_epoch: int,
    lr_decay_factor: float,
):
    rng = np.random.default_rng(seed)
    data = load_mnist()
    Xtr, Ytr = data["x_train"], data["y_train"]
    Xte, Yte = data["x_test"], data["y_test"]
    model = MLP(rng, sizes=sizes)

    snapshots = []
    n = Xtr.shape[0]
    cur_lr = lr
    for epoch in range(epochs):
        if epoch == lr_decay_epoch:
            cur_lr *= lr_decay_factor
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = Xtr[idx]
            yb = Ytr[idx]
            _, cache = model.forward(xb)
            loss, grads = model.backward(cache, yb, weight_decay=weight_decay)
            model.step(grads, lr=cur_lr, momentum=momentum, nesterov=True)
            epoch_loss += loss * xb.shape[0]
        # eval
        pred = model.predict(Xte)
        test_err = float((pred != Yte).mean())
        # confusion on 1k random samples (cheap, deterministic)
        rng_eval = np.random.default_rng(0)
        sub = rng_eval.choice(len(Yte), size=1000, replace=False)
        cm = np.zeros((10, 10), dtype=np.int64)
        for t_, p_ in zip(Yte[sub], pred[sub]):
            cm[t_, p_] += 1
        snapshots.append(
            {
                "epoch": epoch,
                "lr": cur_lr,
                "train_loss": epoch_loss / n,
                "test_err": test_err,
                "W0": model.params["W0"].copy(),
                "confusion_1k": cm,
            }
        )
        print(
            f"  epoch {epoch:2d}  loss={epoch_loss / n:.4f}  "
            f"test_err={test_err * 100:.2f}%",
            flush=True,
        )
    return snapshots


def render_frame(snap: dict, all_test_err: list[float], cols: np.ndarray, ax_grid):
    """Render one GIF frame into the supplied axes grid."""
    ax_curve, ax_filters, ax_cm = ax_grid

    # left: test error curve up to current epoch
    ax_curve.clear()
    eps = list(range(len(all_test_err)))
    ax_curve.plot(eps, [e * 100 for e in all_test_err], "-o", color="tab:purple")
    ax_curve.scatter([snap["epoch"]], [snap["test_err"] * 100], s=80,
                     color="tab:red", zorder=5)
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("test error (%)")
    ax_curve.set_title(
        f"Epoch {snap['epoch']}  "
        f"test err {snap['test_err'] * 100:.2f}%  lr={snap['lr']:.4f}"
    )
    ax_curve.grid(alpha=0.3)
    ax_curve.set_xlim(-0.5, len(all_test_err) - 0.5)
    if all_test_err:
        ymax = max(all_test_err) * 100 * 1.05
        ax_curve.set_ylim(0, ymax)

    # middle: 4x4 filters from W0 (fixed columns across frames)
    ax_filters.clear()
    W0 = snap["W0"]
    n_filt = 16
    grid = 4
    big = np.zeros((28 * grid, 28 * grid), dtype=np.float32)
    for k in range(n_filt):
        w = W0[:, cols[k]].reshape(28, 28)
        # contrast-normalize per filter so the GIF is readable
        vmax = float(np.abs(w).max())
        if vmax < 1e-8:
            vmax = 1e-8
        w_norm = w / vmax
        r = (k // grid) * 28
        c = (k % grid) * 28
        big[r : r + 28, c : c + 28] = w_norm
    ax_filters.imshow(big, cmap="seismic", vmin=-1, vmax=1)
    ax_filters.set_xticks([])
    ax_filters.set_yticks([])
    ax_filters.set_title("First-layer filters (16 fixed units)")

    # right: confusion matrix on 1k sample
    ax_cm.clear()
    cm = snap["confusion_1k"]
    cm_show = np.log10(cm + 1)
    ax_cm.imshow(cm_show, cmap="Blues", vmin=0, vmax=np.log10(150))
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_xticks(range(10))
    ax_cm.set_yticks(range(10))
    ax_cm.set_title("Confusion (1k subsample)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "mcdnn_image_bench.gif")
    args = ap.parse_args()

    sizes = (784, 256, 128, 10)  # slimmer for the GIF run
    print(
        f"GIF training run (slimmer): seed={args.seed} epochs={args.epochs} "
        f"sizes={sizes}",
        flush=True,
    )
    t0 = time.time()
    snaps = train_with_snapshots(
        seed=args.seed,
        epochs=args.epochs,
        sizes=sizes,
        batch_size=128,
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        lr_decay_epoch=args.epochs // 2 + 1,
        lr_decay_factor=0.2,
    )
    print(f"  GIF training: {time.time() - t0:.1f}s", flush=True)

    # pick 16 W0 columns once so the GIF tracks the same units
    rng = np.random.default_rng(0)
    cols = rng.choice(snaps[0]["W0"].shape[1], size=16, replace=False)
    test_errs = [s["test_err"] for s in snaps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    writer = PillowWriter(fps=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  writing GIF -> {args.out}", flush=True)
    with writer.saving(fig, str(args.out), dpi=90):
        for i, snap in enumerate(snaps):
            render_frame(snap, test_errs[: i + 1], cols, axes)
            fig.tight_layout()
            writer.grab_frame()
        # hold the final frame for a beat
        for _ in range(3):
            writer.grab_frame()
    plt.close(fig)
    size_kb = args.out.stat().st_size / 1024
    print(f"  GIF size: {size_kb:.0f} KB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
