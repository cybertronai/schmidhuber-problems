"""
Static visualizations for the trained predictability-minimisation network.

Outputs (in `viz/`):
  training_curves.png   -- L_recon, L_pred, pairwise MI, bit accuracy vs step
  pairwise_mi_init_vs_final.png  -- KxK MI matrix between code components
                                    before vs after training
  code_vs_factor_mi.png -- KxK MI matrix between y_i and the latent factors b_j
                          (shows the recovered permutation+sign assignment)
  code_distribution.png -- per-unit histograms of y_i over a fresh batch
                           (sigmoids saturate near 0/1 after PM)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from predictability_min_binary_factors import (
    PMNet, evaluate, sample_batch, train,
)


def plot_training_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    steps = np.asarray(history["step"])

    ax = axes[0, 0]
    ax.plot(steps, history["L_recon"], color="#1f77b4", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_ylabel(r"$L_{recon}$ (MSE on $x$)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Reconstruction loss")

    ax = axes[0, 1]
    ax.plot(steps, history["L_pred"], color="#d62728", linewidth=1.6,
            label=r"$L_{pred}$")
    ax.axhline(0.25, color="black", linestyle=":", linewidth=1.0,
               label="chance = 0.25")
    ax.set_ylabel(r"$L_{pred}$ (predictor MSE)")
    ax.set_xlabel("step")
    ax.set_ylim(-0.02, 0.32)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Predictor MSE  (rises to chance = factorial code)")

    ax = axes[1, 0]
    ax.plot(steps, history["pairwise_mi"], color="#2ca02c", linewidth=1.6)
    ax.set_ylabel("mean pairwise MI (nats)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.set_title("Mutual information between code components")

    ax = axes[1, 1]
    ax.plot(steps, np.array(history["bit_acc"]) * 100,
            color="#9467bd", linewidth=1.6, label="bit accuracy")
    ax.plot(steps, np.array(history["lam"]) * 50,
            color="gray", linewidth=1.0, linestyle="--",
            label=r"$\lambda \times 50$")
    ax.set_ylabel("bit accuracy (%)")
    ax.set_xlabel("step")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Bit recovery (modulo permutation+sign)")

    fig.suptitle("Predictability minimization on synthetic factorial inputs",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pairwise_mi(pmi_init: np.ndarray, pmi_final: np.ndarray,
                     out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=130)
    vmax = max(pmi_init.max(), pmi_final.max(), 1e-6)
    K = pmi_init.shape[0]
    for ax, mat, title in [
        (axes[0], pmi_init, "Initial code (before PM)"),
        (axes[1], pmi_final, "After PM training"),
    ]:
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="magma")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"$y_{i}$" for i in range(K)])
        ax.set_yticklabels([f"$y_{i}$" for i in range(K)])
        ax.set_title(title, fontsize=10)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 * vmax else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise mutual information between code components (nats)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_code_vs_factor(mi_yb: np.ndarray, perm: tuple,
                        out_path: str) -> None:
    K = mi_yb.shape[0]
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=130)
    im = ax.imshow(mi_yb, vmin=0.0, vmax=np.log(2.0), cmap="viridis")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"$b_{j}$" for j in range(K)])
    ax.set_yticklabels([f"$y_{i}$" for i in range(K)])
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{mi_yb[i, j]:.2f}", ha="center", va="center",
                    color="white" if mi_yb[i, j] < 0.4 else "black",
                    fontsize=9)
    # Highlight the recovered assignment.
    for i in range(K):
        ax.add_patch(plt.Rectangle((perm[i] - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="red",
                                   linewidth=2.0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MI (nats)")
    ax.set_title(f"MI between code units and latent factors\n"
                 f"(red = recovered assignment, perfect = ln 2 ≈ 0.693)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_code_distribution(net: PMNet, M: np.ndarray, out_path: str,
                           rng: np.random.Generator, n: int = 4096,
                           noise: float = 0.05) -> None:
    x, _ = sample_batch(n, net.K, M, rng, noise=noise)
    cache = net.forward(x)
    y = cache["y"]
    K = net.K
    fig, axes = plt.subplots(1, K, figsize=(2.4 * K, 2.6), dpi=130, sharey=True)
    if K == 1:
        axes = [axes]
    for i in range(K):
        ax = axes[i]
        ax.hist(y[:, i], bins=40, range=(0.0, 1.0),
                color="#1f77b4", edgecolor="black", linewidth=0.3)
        ax.set_title(f"$y_{i}$", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("activation")
        if i == 0:
            ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle("Code-unit activations on a fresh batch (after PM)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Capture initial pairwise MI before training: train(0 steps) -> just init.
    print(f"Recording initial state (seed={args.seed})...")
    net0, _, M = train(D=args.D, K=args.K, n_steps=1, seed=args.seed,
                       verbose=False)
    init_metrics = evaluate(net0, M, np.random.default_rng(args.seed + 9999),
                            n=4096, noise=0.05)

    print(f"Training {args.steps} steps (seed={args.seed})...")
    net, history, M = train(D=args.D, K=args.K, n_steps=args.steps,
                            seed=args.seed, verbose=False)
    final_metrics = evaluate(net, M, np.random.default_rng(args.seed + 12345),
                             n=4096, noise=0.05)
    print(f"  final: L_recon={final_metrics['L_recon']:.4f}  "
          f"pMI={final_metrics['pairwise_mi']:.4f}  "
          f"bit_acc={final_metrics['bit_acc']*100:.1f}%")

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_pairwise_mi(init_metrics["pairwise_mi_matrix"],
                     final_metrics["pairwise_mi_matrix"],
                     os.path.join(args.outdir, "pairwise_mi_init_vs_final.png"))
    plot_code_vs_factor(final_metrics["mi_yb"], final_metrics["best_perm"],
                        os.path.join(args.outdir, "code_vs_factor_mi.png"))
    plot_code_distribution(net, M,
                           os.path.join(args.outdir, "code_distribution.png"),
                           np.random.default_rng(args.seed + 7777))


if __name__ == "__main__":
    main()
