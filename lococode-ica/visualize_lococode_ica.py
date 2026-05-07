"""
Static visualizations for the trained LOCOCODE-ICA experiment.

Outputs (in `viz/`):
  training_curves.png       reconstruction MSE, mean |H|, mean kurtosis,
                            Amari distance over epochs (LOCOCODE only)
  amari_comparison.png      bar chart: LOCOCODE vs PCA vs FastICA Amari
  hidden_distributions.png  histograms of one hidden unit's activity for
                            LOCOCODE, PCA, FastICA against a Laplace ref
  recovered_demixers.png    Hinton-style heatmaps of |W @ A| for each method
                            (perfect demixing -> permutation matrix)
  source_recovery.png       cross-correlation matrices |corr(S, H_method)|
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from lococode_ica import run_one_seed, _kurtosis


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ensure_outdir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _abs_corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return |corr(A_i, B_j)| (k x k) for column-pairs."""
    A_c = A - A.mean(0, keepdims=True)
    B_c = B - B.mean(0, keepdims=True)
    A_n = A_c / (A_c.std(0, keepdims=True) + 1e-12)
    B_n = B_c / (B_c.std(0, keepdims=True) + 1e-12)
    return np.abs(A_n.T @ B_n) / A.shape[0]


def _greedy_permute(M: np.ndarray):
    """Greedy permutation that puts large entries on the diagonal of M."""
    M = np.abs(M).copy()
    k = M.shape[0]
    row_perm = np.arange(k)
    col_perm = np.arange(k)
    used_rows = set()
    used_cols = set()
    pairs = []
    for _ in range(k):
        best = -1.0
        bi = bj = -1
        for i in range(k):
            if i in used_rows: continue
            for j in range(k):
                if j in used_cols: continue
                if M[i, j] > best:
                    best = M[i, j]
                    bi, bj = i, j
        pairs.append((bi, bj))
        used_rows.add(bi); used_cols.add(bj)
    pairs.sort(key=lambda p: p[1])
    row_perm = np.array([p[0] for p in pairs])
    return row_perm


# --------------------------------------------------------------------------
# Plot 1: training curves
# --------------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    e = history["epoch"]

    ax = axes[0, 0]
    ax.plot(e, history["recon_loss"], color="#1f77b4", lw=1.5)
    ax.set_title("whitened reconstruction MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(e, history["act_l1"], color="#2ca02c", lw=1.5)
    ax.set_title(r"mean $|H|$ (sparsity penalty target)")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\langle |h| \rangle$")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(e, history["kurtosis_mean"], color="#d62728", lw=1.5)
    ax.axhline(0.0, color="grey", lw=0.7, ls="--", label="Gaussian")
    ax.axhline(3.0, color="purple", lw=0.7, ls="--", label="Laplace")
    ax.set_title("mean excess kurtosis of hidden codes")
    ax.set_xlabel("epoch"); ax.set_ylabel("kurtosis"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if not np.isnan(history["amari"][0]):
        ax.plot(e, history["amari"], color="#9467bd", lw=1.5)
    ax.set_title("Amari distance to true mixing")
    ax.set_xlabel("epoch"); ax.set_ylabel("Amari"); ax.grid(alpha=0.3)

    fig.suptitle("LOCOCODE-ICA training dynamics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 2: Amari comparison
# --------------------------------------------------------------------------

def plot_amari_comparison(metrics: dict, out_path: str):
    methods = ["lococode", "pca", "fastica"]
    labels = ["LOCOCODE\n(L1 + tied AE)", "PCA\n(2nd order)", "FastICA\n(tanh fp)"]
    amaris = [metrics[m]["amari"] for m in methods]
    kurts = [metrics[m]["kurtosis_mean"] for m in methods]
    colors = ["#9467bd", "#7f7f7f", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=120)
    ax = axes[0]
    bars = ax.bar(labels, amaris, color=colors)
    for bar, v in zip(bars, amaris):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Amari distance (lower = better)")
    ax.set_title("source separation quality")
    ax.set_ylim(0, max(amaris) * 1.25)

    ax = axes[1]
    bars = ax.bar(labels, kurts, color=colors)
    for bar, v in zip(bars, kurts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(0.0, color="grey", lw=0.7, ls="--", label="Gaussian = 0")
    ax.axhline(3.0, color="purple", lw=0.7, ls="--", label="Laplace = 3")
    ax.set_ylabel("mean excess kurtosis")
    ax.set_title("super-Gaussian hidden codes")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(min(0, min(kurts) - 0.5), 3.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 3: hidden distribution histograms
# --------------------------------------------------------------------------

def plot_hidden_distributions(out: dict, out_path: str):
    methods = [
        ("LOCOCODE", out["H_loco"], "#9467bd"),
        ("PCA", out["H_pca"], "#7f7f7f"),
        ("FastICA", out["H_ica"], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=120, sharey=True)
    bins = np.linspace(-5, 5, 60)

    # Reference Laplace pdf
    xs = np.linspace(-5, 5, 200)
    laplace = 0.5 * np.exp(-np.abs(xs))
    gauss = 1.0 / np.sqrt(2 * np.pi) * np.exp(-xs ** 2 / 2)

    for ax, (name, H, color) in zip(axes, methods):
        # Pick the unit with max kurtosis as the most "ICA-like"
        kurt = _kurtosis(H)
        idx = int(np.argmax(kurt))
        h_unit = H[:, idx]
        h_unit = (h_unit - h_unit.mean()) / (h_unit.std() + 1e-12)
        ax.hist(h_unit, bins=bins, density=True, color=color, alpha=0.6,
                edgecolor="black", linewidth=0.4,
                label=f"{name}\n(unit {idx}, k={kurt[idx]:.2f})")
        ax.plot(xs, laplace, color="purple", lw=1.4, ls="--", label="Laplace")
        ax.plot(xs, gauss, color="grey", lw=1.0, ls=":", label="Gaussian")
        ax.set_xlim(-5, 5)
        ax.set_xlabel("standardised activation")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")

    fig.suptitle("hidden-unit activation density (most-kurtotic unit per method)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 4: recovered demixers (Hinton-style)
# --------------------------------------------------------------------------

def plot_recovered_demixers(out: dict, out_path: str):
    A = out["A"]
    methods = [
        ("LOCOCODE", out["W_loco_xspace"], "#9467bd"),
        ("PCA", out["W_pca"], "#7f7f7f"),
        ("FastICA", out["W_ica"], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), dpi=120)
    for ax, (name, W, color) in zip(axes, methods):
        P = np.abs(W @ A)
        # Normalize each row to its max so a perfect permutation is all 1s
        # along one column per row.
        P = P / (P.max(axis=1, keepdims=True) + 1e-12)
        # Greedy permute rows so the diagonal is largest where possible.
        rp = _greedy_permute(P)
        Pp = P[rp]
        im = ax.imshow(Pp, cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{name}\n|W @ A| (row-normalised, permuted)")
        ax.set_xlabel("true source"); ax.set_ylabel("recovered component")
        ax.set_xticks(range(W.shape[0])); ax.set_yticks(range(W.shape[0]))
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    fig.suptitle("recovered demixer aligned with true mixing — "
                 "diagonal = perfect identification", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 5: source-recovery cross correlations
# --------------------------------------------------------------------------

def plot_source_recovery(out: dict, out_path: str):
    S = out["S"]
    methods = [
        ("LOCOCODE", out["H_loco"], "#9467bd"),
        ("PCA", out["H_pca"], "#7f7f7f"),
        ("FastICA", out["H_ica"], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), dpi=120)
    for ax, (name, H, color) in zip(axes, methods):
        C = _abs_corr_matrix(S, H)
        rp = _greedy_permute(C.T).T  # permute columns instead
        # Apply: pick column for each source greedily
        cp = _greedy_permute(C)
        Cp = C[cp]
        im = ax.imshow(Cp, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{name}\n|corr(S, H)| (rows permuted)")
        ax.set_xlabel("hidden unit"); ax.set_ylabel("true source (perm)")
        ax.set_xticks(range(S.shape[1])); ax.set_yticks(range(S.shape[1]))
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    fig.suptitle("cross-correlation between true sources and recovered codes",
                 fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    out = run_one_seed(args.seed, args.k, args.n_samples, args.epochs)
    outdir = _ensure_outdir(args.outdir)

    plot_training_curves(out["history"], os.path.join(outdir, "training_curves.png"))
    plot_amari_comparison(out["metrics"], os.path.join(outdir, "amari_comparison.png"))
    plot_hidden_distributions(out, os.path.join(outdir, "hidden_distributions.png"))
    plot_recovered_demixers(out, os.path.join(outdir, "recovered_demixers.png"))
    plot_source_recovery(out, os.path.join(outdir, "source_recovery.png"))

    print(f"Saved 5 figures under {outdir}/")
    for name in ("lococode", "pca", "fastica"):
        e = out["metrics"][name]
        print(f"  {name:<10} amari={e['amari']:.4f}  "
              f"kurt={e['kurtosis_mean']:.3f}")


if __name__ == "__main__":
    main()
