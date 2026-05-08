"""Render em_segmentation_isbi.gif: prediction-map evolution across training.

Each frame shows three synced panels for the first test image:
  * raw input + GT membrane overlay
  * MLP probability map at the current epoch
  * thresholded prediction (at the prior-matching threshold)

Plus a small training-curve sub-axis on the right tracking test AUC.

Run:
  python3 make_em_segmentation_isbi_gif.py --seed 0 --epochs 10 --fps 3
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from em_segmentation_isbi import (
    MLP,
    TrainConfig,
    edge_baseline_score,
    evaluate_full_image,
    evaluate_subsampled,
    evaluate_edge_baseline,
    make_dataset,
    prior_matching_threshold,
    sample_balanced_patches,
)


def _frame(image: np.ndarray, mask: np.ndarray, prob: np.ndarray, thr: float,
           epoch: int, total_epochs: int, history: dict, edge_auc: float,
           ) -> np.ndarray:
    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.15)

    # Panel 1: input + GT overlay.
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="#22ee44", linewidths=0.6)
    ax.set_title("input + GT", fontsize=10)
    ax.axis("off")

    # Panel 2: MLP probability.
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(prob, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"MLP prob  (epoch {epoch}/{total_epochs})", fontsize=10)
    ax.axis("off")

    # Panel 3: thresholded prediction.
    ax = fig.add_subplot(gs[0, 2])
    pred = (prob >= thr).astype(np.float32)
    ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"pred @thr {thr:.2f}", fontsize=10)
    ax.axis("off")

    # Panel 4: AUC curve.
    ax = fig.add_subplot(gs[0, 3])
    eps = history["epoch"][:epoch]
    if eps:
        ax.plot(eps, history["test_auc_sub"][:epoch], "-s",
                color="#d62728", label="MLP test AUC")
    ax.axhline(edge_auc, ls="--", color="gray",
               label=f"edge ({edge_auc:.3f})")
    ax.set_xlim(0, total_epochs + 0.5)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test ROC AUC")
    ax.set_title("Test AUC", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def train_with_snapshots(cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    train_imgs, train_masks, test_imgs, test_masks = make_dataset(
        cfg.n_train_images, cfg.n_test_images,
        cfg.image_h, cfg.image_w, cfg.n_cells, cfg.seed,
    )
    layer_sizes = (cfg.patch * cfg.patch, *cfg.hidden_sizes, 1)
    model = MLP.make(layer_sizes, rng)

    edge_acc, edge_auc = evaluate_edge_baseline(test_imgs, test_masks)
    train_pos = float(train_masks.mean())
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_acc_sub": [], "test_auc_sub": [], "wallclock_s": []}
    eval_rng = np.random.default_rng(cfg.seed + 99)

    frames = []

    # Initial frame at epoch 0 (random init).
    prob_map, _, _, thr, _ = evaluate_full_image(
        model, test_imgs[0], test_masks[0], cfg.patch, target_pos_frac=train_pos,
    )
    frames.append(_frame(test_imgs[0], test_masks[0], prob_map, thr,
                          0, cfg.epochs, history, edge_auc))

    lr = cfg.lr
    for epoch in range(1, cfg.epochs + 1):
        Xtr, ytr = sample_balanced_patches(
            train_imgs, train_masks, cfg.patches_per_epoch, cfg.patch, rng,
        )
        order = rng.permutation(Xtr.shape[0])
        Xtr = Xtr[order]
        ytr = ytr[order]

        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for i in range(0, Xtr.shape[0], cfg.batch_size):
            xb = Xtr[i:i + cfg.batch_size]
            yb = ytr[i:i + cfg.batch_size]
            p, acts = model.forward(xb)
            eps_ = 1e-7
            loss = -float(np.mean(yb * np.log(p + eps_) + (1 - yb) * np.log(1 - p + eps_)))
            running_loss += loss * yb.shape[0]
            running_n += yb.shape[0]
            running_correct += int(((p >= 0.5) == (yb >= 0.5)).sum())
            dWs, dbs = model.backward(acts, yb)
            model.sgd_step(dWs, dbs, lr, cfg.momentum, cfg.weight_decay)
        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)
        test_acc, test_auc = evaluate_subsampled(
            model, test_imgs, test_masks, cfg.patch,
            cfg.eval_pixels_per_image * test_imgs.shape[0], eval_rng,
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc_sub"].append(test_acc)
        history["test_auc_sub"].append(test_auc)
        history["wallclock_s"].append(0.0)
        print(f"  epoch {epoch}/{cfg.epochs}  test_acc {test_acc*100:.2f}%  test_AUC {test_auc:.4f}",
              flush=True)

        prob_map, _, _, thr, _ = evaluate_full_image(
            model, test_imgs[0], test_masks[0], cfg.patch,
            target_pos_frac=train_pos,
        )
        frames.append(_frame(test_imgs[0], test_masks[0], prob_map, thr,
                              epoch, cfg.epochs, history, edge_auc))
        lr *= cfg.lr_decay

    return frames, history, edge_auc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", type=str, default="em_segmentation_isbi.gif")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--patches-per-epoch", type=int, default=4096)
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed, epochs=args.epochs,
        patches_per_epoch=args.patches_per_epoch,
    )
    frames, history, edge_auc = train_with_snapshots(cfg)
    # Hold final frame longer.
    frames = frames + [frames[-1]] * 4
    imageio.mimsave(args.out, frames, duration=1.0 / max(1, args.fps), loop=0)
    print(f"  wrote {args.out}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
