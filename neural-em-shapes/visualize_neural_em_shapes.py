"""Static visualisations for neural-em-shapes.

Reads run.json and writes PNGs to viz/:

  1. learning_curves.png      train + test loss per epoch
  2. nmi_curve.png            test NMI per epoch (with peak marker)
  3. slot_assignments_em.png  HEADLINE: K=3 slot responsibilities per
                              EM iteration on 4 held-out images
  4. slot_reconstructions.png per-slot mu (final iteration) for the
                              same 4 images
  5. dataset_examples.png     6 random samples from the training
                              distribution + their ground-truth masks
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _load_viz(npz_path):
    return np.load(npz_path)


def plot_learning_curves(run, out):
    h = run["history"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(h["epoch"], h["train_loss"], "o-", color="#1f77b4",
            label="train loss (sum over T iters)")
    ax2 = ax.twinx()
    ax2.plot(h["epoch"], h["test_loss"], "s--", color="#d62728",
             label="test loss (final iter)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="#1f77b4")
    ax2.set_ylabel("test loss", color="#d62728")
    ax.set_title("N-EM training: mixture neg log-likelihood")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_nmi_curve(run, out):
    h = run["history"]
    best = run["best"]
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(h["epoch"], h["test_nmi"], "o-", color="#2ca02c", label="test NMI")
    plt.axvline(best["epoch"], color="grey", linestyle=":",
                label=f"best @ epoch {best['epoch']} (NMI={best['test_nmi']:.3f})")
    plt.axhline(1.0 / 3.0, color="lightgrey", linestyle="--",
                label="chance ≈ 0.33 (3 ground truth shapes)")
    plt.xlabel("epoch")
    plt.ylabel("NMI(slot, shape) over fg pixels, per image")
    plt.ylim(0, 1)
    plt.title("Slot-binding emerges, then partially collapses")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_slot_assignments_em(run, vz, out, n_images=4):
    """Headline: gamma_{b,k,i} for each EM iteration on 4 held-out images.

    Uses the BEST checkpoint's snapshot (highest NMI) -- at later
    training the slots collapse a bit so the headline picture lives
    here.
    """
    canvas = run["config"]["canvas"]
    K = run["config"]["K"]
    T = run["config"]["T"]
    epoch = int(vz["best_epoch"])

    x = vz["viz_x"]  # (B, D)
    mask = vz["viz_m"]
    gamma_per_iter = vz["best_gamma"]  # (T, B, K, D)
    mu_per_iter = vz["best_mu"]

    n_images = min(n_images, x.shape[0])
    cols = 1 + T  # input + T iters
    rows = n_images
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    if rows == 1:
        axes = axes[None, :]

    # Build per-pixel RGB from gamma:
    # red = slot 0, green = slot 1, blue = slot 2 (additive blend by gamma).
    palette = np.array([
        [0.95, 0.30, 0.30],  # red
        [0.20, 0.80, 0.20],  # green
        [0.30, 0.50, 0.95],  # blue
    ])
    if K > 3:
        # extend deterministically
        rng = np.random.default_rng(0)
        extra = rng.uniform(0.2, 0.95, size=(K - 3, 3))
        palette = np.concatenate([palette, extra], axis=0)
    palette = palette[:K]

    for i in range(n_images):
        # column 0: input
        ax = axes[i, 0]
        ax.imshow(x[i].reshape(canvas, canvas), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("input x")
        ax.set_ylabel(f"img {i}", fontsize=9)

        # columns 1..T: gamma at each iteration, hard-assignment colour
        for t in range(T):
            ax = axes[i, 1 + t]
            g = gamma_per_iter[t][i]  # (K, D)
            # Hard slot assignment per pixel; foreground pixels coloured by
            # their argmax slot, background stays white.
            argmax = g.argmax(axis=0)  # (D,)
            rgb = palette[argmax]      # (D, 3)
            fg_w = x[i][:, None]
            rgb_img = 1.0 - fg_w * (1.0 - rgb)
            ax.imshow(rgb_img.reshape(canvas, canvas, 3))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"iter {t}")
    fig.suptitle(
        f"Per-iteration slot responsibilities  "
        f"(best checkpoint @ epoch {epoch}, K={K}, T={T})",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_slot_reconstructions(run, vz, out, n_images=4):
    canvas = run["config"]["canvas"]
    K = run["config"]["K"]
    epoch = int(vz["best_epoch"])

    x = vz["viz_x"]
    mu_per_iter = vz["best_mu"]
    gamma_per_iter = vz["best_gamma"]

    mu_T = mu_per_iter[-1]  # (B, K, D)
    gamma_T = gamma_per_iter[-1]

    n_images = min(n_images, x.shape[0])
    cols = 2 + K  # input + K slots + composite
    rows = n_images
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    if rows == 1:
        axes = axes[None, :]

    for i in range(n_images):
        ax = axes[i, 0]
        ax.imshow(x[i].reshape(canvas, canvas), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("input x")
        ax.set_ylabel(f"img {i}", fontsize=9)

        for k in range(K):
            ax = axes[i, 1 + k]
            ax.imshow(mu_T[i, k].reshape(canvas, canvas), cmap="gray_r",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"slot {k}: μ")

        # Composite: gamma-weighted mu, summed -> mixture mean
        mix = (gamma_T[i] * mu_T[i]).sum(axis=0)
        ax = axes[i, 1 + K]
        ax.imshow(mix.reshape(canvas, canvas), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("Σ_k γ μ")

    fig.suptitle(f"Per-slot reconstructions (final iter, epoch {epoch})",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_dataset_examples(run, vz, out, n=6):
    canvas = run["config"]["canvas"]
    x = vz["viz_x"]
    mask = vz["viz_m"]
    n = min(n, x.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2))

    cmap_mask = matplotlib.colors.ListedColormap(
        [(1.0, 1.0, 1.0),  # bg
         (0.95, 0.30, 0.30),
         (0.20, 0.80, 0.20),
         (0.30, 0.50, 0.95)]
    )

    for i in range(n):
        ax = axes[0, i]
        ax.imshow(x[i].reshape(canvas, canvas), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("input", fontsize=10)

        ax = axes[1, i]
        ax.imshow(mask[i].reshape(canvas, canvas), cmap=cmap_mask,
                  vmin=0, vmax=3)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("ground-truth\nshape labels", fontsize=10)

    fig.suptitle("Static-shapes dataset: 24×24 binary canvas, 3 random shapes")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    run_path = os.path.join(HERE, "run.json")
    npz_path = os.path.join(HERE, "run_viz.npz")
    if not os.path.exists(run_path) or not os.path.exists(npz_path):
        raise SystemExit("run.json / run_viz.npz missing — run neural_em_shapes.py first")
    run = _load(run_path)
    vz = _load_viz(npz_path)

    plot_learning_curves(run, os.path.join(VIZ, "learning_curves.png"))
    plot_nmi_curve(run, os.path.join(VIZ, "nmi_curve.png"))
    plot_slot_assignments_em(run, vz, os.path.join(VIZ, "slot_assignments_em.png"))
    plot_slot_reconstructions(run, vz, os.path.join(VIZ, "slot_reconstructions.png"))
    plot_dataset_examples(run, vz, os.path.join(VIZ, "dataset_examples.png"))

    print("wrote 5 PNGs to viz/", flush=True)


if __name__ == "__main__":
    main()
