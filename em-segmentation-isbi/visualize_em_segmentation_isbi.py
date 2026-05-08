"""Static visualizations for em-segmentation-isbi.

Trains the model with the headline config and writes four PNGs to
``viz/``:

  * training_curves.png     -- train loss / train acc / test acc / test AUC
  * dataset_samples.png     -- raw synthetic Voronoi-EM image, ground-truth
                               membrane mask, and Sobel-baseline edge score
  * predictions.png         -- side-by-side: input | GT mask | model prob
                               map | thresholded model | edge baseline,
                               for several test images
  * roc_comparison.png      -- ROC curves: MLP pixel classifier vs the
                               Sobel+inverted-intensity edge baseline

Run:
  python3 visualize_em_segmentation_isbi.py --seed 0 --epochs 12 --outdir viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from em_segmentation_isbi import (
    TrainConfig,
    edge_baseline_score,
    evaluate_full_image,
    prior_matching_threshold,
    roc_auc,
    train,
)


def plot_training_curves(history: dict, edge: dict, out: Path) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "-o", color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Train loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], "-o",
            color="#1f77b4", label="train (50/50 patches)")
    ax.plot(epochs, [a * 100 for a in history["test_acc_sub"]], "-s",
            color="#d62728", label="test (50/50 patches)")
    ax.axhline(edge["pixel_acc"] * 100, ls="--", color="gray",
               label=f"edge baseline ({edge['pixel_acc']*100:.1f}%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("pixel accuracy %")
    ax.set_title("Accuracy on balanced patches")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["test_auc_sub"], "-s", color="#d62728",
            label="test AUC")
    ax.axhline(edge["auc"], ls="--", color="gray",
               label=f"edge baseline ({edge['auc']:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Test AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_dataset_samples(images: np.ndarray, masks: np.ndarray, out: Path,
                          n_show: int = 4) -> None:
    n_show = min(n_show, images.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3.0 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_show):
        axes[i, 0].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("input (synthetic Voronoi EM)" if i == 0 else "")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("ground-truth membrane" if i == 0 else "")
        axes[i, 1].axis("off")
        edge = edge_baseline_score(images[i])
        axes[i, 2].imshow(edge, cmap="magma")
        axes[i, 2].set_title("Sobel + inv-intensity" if i == 0 else "")
        axes[i, 2].axis("off")
    fig.suptitle(
        "Synthetic Voronoi-EM dataset (substituted for ISBI 2012 stack)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_predictions(model, test_imgs: np.ndarray, test_masks: np.ndarray,
                     train_pos_frac: float, patch: int, out: Path,
                     n_show: int = 4) -> None:
    n_show = min(n_show, test_imgs.shape[0])
    fig, axes = plt.subplots(n_show, 5, figsize=(15, 3.0 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_show):
        prob_map, _, auc, thr, acc_prior = evaluate_full_image(
            model, test_imgs[i], test_masks[i], patch,
            target_pos_frac=train_pos_frac,
        )
        thresholded = (prob_map >= thr).astype(np.float32)
        edge = edge_baseline_score(test_imgs[i])

        axes[i, 0].imshow(test_imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("input" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(test_masks[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("GT membrane" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("MLP prob map" if i == 0 else "")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(thresholded, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title(f"MLP @prior thr  ({acc_prior*100:.1f}%, AUC {auc:.3f})"
                              if i == 0 else f"acc {acc_prior*100:.1f}%, AUC {auc:.3f}")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(edge, cmap="magma")
        axes[i, 4].set_title("edge baseline" if i == 0 else "")
        axes[i, 4].axis("off")

    fig.suptitle("Test-set predictions: input vs GT vs MLP vs edge baseline",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_roc_comparison(model, test_imgs: np.ndarray, test_masks: np.ndarray,
                         patch: int, out: Path) -> None:
    # Stack ALL pixels across all test images for both methods.
    mlp_scores = []
    edge_scores = []
    labels = []
    for i in range(test_imgs.shape[0]):
        prob_map, _, _, _, _ = evaluate_full_image(
            model, test_imgs[i], test_masks[i], patch,
            target_pos_frac=test_masks[i].mean(),
        )
        mlp_scores.append(prob_map.reshape(-1))
        edge_scores.append(edge_baseline_score(test_imgs[i]).reshape(-1))
        labels.append(test_masks[i].reshape(-1).astype(np.float32))
    mlp_scores = np.concatenate(mlp_scores)
    edge_scores = np.concatenate(edge_scores)
    labels = np.concatenate(labels)
    auc_mlp, fpr_m, tpr_m = roc_auc(mlp_scores, labels)
    auc_edge, fpr_e, tpr_e = roc_auc(edge_scores, labels)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    ax.plot(fpr_m, tpr_m, color="#1f77b4", lw=2,
            label=f"MLP pixel classifier   AUC={auc_mlp:.3f}")
    ax.plot(fpr_e, tpr_e, color="#d62728", lw=2,
            label=f"edge baseline         AUC={auc_edge:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("ROC: MLP pixel classifier vs Sobel+intensity edge baseline\n"
                 f"({test_imgs.shape[0]} test images, every pixel scored)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs)
    result = train(cfg)

    train_imgs = result["train_imgs"]
    train_masks = result["train_masks"]
    test_imgs = result["test_imgs"]
    test_masks = result["test_masks"]
    train_pos_frac = float(train_masks.mean())

    plot_training_curves(result["history"], result["edge_baseline"],
                          out / "training_curves.png")
    plot_dataset_samples(train_imgs, train_masks, out / "dataset_samples.png")
    plot_predictions(result["model"], test_imgs, test_masks, train_pos_frac,
                     cfg.patch, out / "predictions.png")
    plot_roc_comparison(result["model"], test_imgs, test_masks, cfg.patch,
                         out / "roc_comparison.png")


if __name__ == "__main__":
    main()
