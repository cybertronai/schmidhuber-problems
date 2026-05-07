"""Static visualisations for semilinear-pm-image-patches.

Outputs to viz/:
    sample_images.png   - 6 example synthetic source images (1/f + bars)
    sample_patches.png  - 8x8 grid of raw patches before / after ZCA
    init_filters.png    - random-init encoder rows reshaped as patches
    final_filters.png   - trained encoder rows: oriented edge / Gabor atlas
    training_curves.png - per-step L_pred + mean code kurtosis
    fft_atlas.png       - 2-D Fourier magnitude of each trained filter
    kurtosis_hist.png   - per-unit excess kurtosis (random init vs trained)
    pca_baseline.png    - PCA-only filters on the same data, for contrast
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from semilinear_pm_image_patches import (
    code_kurtosis,
    filter_orientation_metrics,
    init_params,
    make_dataset,
    train,
    zca_whiten,
)


def filter_atlas(W: np.ndarray, patch_size: int, ncols: int = None) -> np.ndarray:
    """Lay out filters in a grid; per-filter [-1, 1] normalisation for
    comparable contrast across cells."""
    M = W.shape[0]
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(M)))
    nrows = int(np.ceil(M / ncols))
    pad = 1
    cell = patch_size + pad
    out = np.full((nrows * cell + pad, ncols * cell + pad), 0.5)
    for i in range(M):
        f = W[i].reshape(patch_size, patch_size)
        f = f / (np.max(np.abs(f)) + 1e-12)  # per-filter contrast
        f = (f + 1.0) / 2.0
        r, c = divmod(i, ncols)
        y0 = pad + r * cell
        x0 = pad + c * cell
        out[y0:y0 + patch_size, x0:x0 + patch_size] = f
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hidden", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--n-patches", type=int, default=30000)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr-e", type=float, default=0.05)
    parser.add_argument("--lr-p", type=float, default=0.05)
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--n-bars", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Capture the random-init encoder for the "before" image, then train.
    rng_init = np.random.default_rng(args.seed)
    raw_init, _ = make_dataset(rng_init, n_images=args.n_images,
                               patch_size=args.patch_size,
                               n_patches=args.n_patches, n_bars=args.n_bars)
    X_init, _ = zca_whiten(raw_init, eps=1e-2)
    rng2 = np.random.default_rng(args.seed)
    # Mirror the train()-internal rng order: dataset uses entropy first, then
    # init_params. Re-derive that exact split.
    _ = make_dataset(rng2, n_images=args.n_images, patch_size=args.patch_size,
                     n_patches=args.n_patches, n_bars=args.n_bars)
    W_init, _ = init_params(rng2, args.patch_size ** 2, args.n_hidden)

    res = train(
        seed=args.seed, n_hidden=args.n_hidden, patch_size=args.patch_size,
        n_patches=args.n_patches, n_steps=args.n_steps, batch=args.batch,
        lr_e=args.lr_e, lr_p=args.lr_p, n_images=args.n_images,
        n_bars=args.n_bars,
    )
    W = res["W"]
    images = res["images"]
    raw_patches = res["raw_patches"]
    X = res["X"]
    history = res["history"]

    # 1. Sample source images.
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"img {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Synthetic natural-image-statistics: 1/f noise + random oriented bars")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_images.png"), dpi=110)
    plt.close(fig)

    # 2. Raw vs whitened patches.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    n_show = 64
    grid_raw = filter_atlas(raw_patches[:n_show], args.patch_size, ncols=8)
    grid_white = filter_atlas(X[:n_show], args.patch_size, ncols=8)
    ax1.imshow(grid_raw, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("raw patches (DC-removed)")
    ax1.axis("off")
    ax2.imshow(grid_white, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("after ZCA whitening")
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_patches.png"), dpi=110)
    plt.close(fig)

    # 3. Random-init encoder rows.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(filter_atlas(W_init, args.patch_size), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"encoder rows W_i, RANDOM init (M={args.n_hidden})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "init_filters.png"), dpi=110)
    plt.close(fig)

    # 4. Trained encoder rows.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(filter_atlas(W, args.patch_size), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"encoder rows W_i, TRAINED ({args.n_steps} steps PM)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "final_filters.png"), dpi=110)
    plt.close(fig)

    # 5. Training curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["step"], history["L_pred"], lw=0.8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L_pred (predictor squared error)")
    axes[0].set_title("Predictability loss (encoder ascends, predictor descends)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["step"], history["y_kurt_mean"], lw=0.8, color="tab:orange")
    axes[1].axhline(0.0, color="k", lw=0.5, ls="--", label="Gaussian")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean excess kurtosis of code")
    axes[1].set_title("Code sparseness (higher = more Gabor-like)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=110)
    plt.close(fig)

    # 6. FFT atlas: each filter's 2-D power spectrum.
    fft_grid = np.empty_like(W)
    for i in range(W.shape[0]):
        f = W[i].reshape(args.patch_size, args.patch_size)
        F = np.fft.fftshift(np.fft.fft2(f))
        m = np.abs(F)
        fft_grid[i] = (m / (m.max() + 1e-12)).flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(filter_atlas(fft_grid, args.patch_size), cmap="magma", vmin=0, vmax=1)
    ax.set_title("Filter Fourier magnitudes (peak = orientation+freq)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fft_atlas.png"), dpi=110)
    plt.close(fig)

    # 7. Kurtosis histograms.
    k_init = code_kurtosis(W_init, X)
    k_final = code_kurtosis(W, X)
    y_init = X @ W_init.T
    y_final = X @ W.T
    init_ks = np.array([float((y_init[:, i] ** 4).mean() / ((y_init[:, i] ** 2).mean() + 1e-12) ** 2 - 3.0)
                        for i in range(W.shape[0])])
    final_ks = np.array([float((y_final[:, i] ** 4).mean() / ((y_final[:, i] ** 2).mean() + 1e-12) ** 2 - 3.0)
                         for i in range(W.shape[0])])
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(init_ks.min(), final_ks.min()),
                       max(init_ks.max(), final_ks.max()), 24)
    ax.hist(init_ks, bins=bins, alpha=0.55, label=f"random init (mean={init_ks.mean():.2f})", color="tab:gray")
    ax.hist(final_ks, bins=bins, alpha=0.7,
            label=f"trained (mean={final_ks.mean():.2f})", color="tab:blue")
    ax.axvline(0.0, color="k", lw=0.5, ls="--", label="Gaussian")
    ax.set_xlabel("excess kurtosis of W_i^T x")
    ax.set_ylabel("# units")
    ax.set_title("Per-unit code kurtosis: random projection vs PM-trained")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "kurtosis_hist.png"), dpi=110)
    plt.close(fig)

    # 8. PCA baseline filters on the same data.
    cov = X.T @ X / X.shape[0]
    _, _, Vt = np.linalg.svd(cov)
    W_pca = Vt[:args.n_hidden]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(filter_atlas(W_pca, args.patch_size), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"PCA top-{args.n_hidden} eigenvectors (NOT oriented)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "pca_baseline.png"), dpi=110)
    plt.close(fig)

    # Print a small summary.
    metrics = filter_orientation_metrics(W, args.patch_size)
    concs = np.array([m["concentration"] for m in metrics])
    print(f"oriented filters (conc > 0.5): {(concs > 0.5).sum()} / {args.n_hidden}")
    print(f"oriented filters (conc > 0.4): {(concs > 0.4).sum()} / {args.n_hidden}")
    print(f"mean kurtosis: random={k_init['y_kurtosis_mean']:.2f}, "
          f"trained={k_final['y_kurtosis_mean']:.2f}")
    print(f"wrote 8 PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
