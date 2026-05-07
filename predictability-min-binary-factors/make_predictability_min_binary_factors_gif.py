"""
Animate the encoder/predictor co-evolution.

Layout per frame:
  Top-left:    pairwise MI matrix between code components (collapses toward 0)
  Top-right:   y_i scatter for two chosen pairs, coloured by ground-truth factor
  Bottom:      training curves: L_recon, L_pred (vs chance line), pairwise MI

Usage:
    python3 make_predictability_min_binary_factors_gif.py
    python3 make_predictability_min_binary_factors_gif.py --steps 1500 \
        --snapshot-every 30 --fps 12
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from predictability_min_binary_factors import (
    PMNet, evaluate, sample_batch, train,
)


PAIR_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def render_frame(step: int,
                 net: PMNet,
                 history: dict,
                 M: np.ndarray,
                 cur_lam: float,
                 total_steps: int,
                 eval_rng: np.random.Generator) -> Image.Image:
    """Render one snapshot frame."""
    metrics = evaluate(net, M, eval_rng, n=1024, noise=0.05)
    pmi = metrics["pairwise_mi_matrix"]

    # Pick the recovered assignment so that y_i shown for pair i corresponds
    # to factor b_{perm[i]} -- consistent colouring across frames.
    perm = metrics["best_perm"]
    K = net.K

    fig = plt.figure(figsize=(11, 6.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.55, wspace=0.40)

    # ---- top-left: pairwise MI matrix ----
    ax_mi = fig.add_subplot(gs[0, 0])
    vmax = max(0.05, 1.1 * pmi.max())
    im = ax_mi.imshow(pmi, vmin=0.0, vmax=vmax, cmap="magma")
    ax_mi.set_xticks(range(K))
    ax_mi.set_yticks(range(K))
    ax_mi.set_xticklabels([f"$y_{i}$" for i in range(K)], fontsize=9)
    ax_mi.set_yticklabels([f"$y_{i}$" for i in range(K)], fontsize=9)
    for i in range(K):
        for j in range(K):
            ax_mi.text(j, i, f"{pmi[i, j]:.2f}",
                       ha="center", va="center",
                       color="white" if pmi[i, j] < 0.5 * vmax else "black",
                       fontsize=8)
    ax_mi.set_title(
        f"Pairwise MI(code, code)  (mean off-diag = {metrics['pairwise_mi']:.3f} nats)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax_mi, fraction=0.046, pad=0.04)

    # ---- top-right: code scatter for one pair ----
    ax_sc = fig.add_subplot(gs[0, 1])
    x_eval, b_eval = sample_batch(512, K, M,
                                  np.random.default_rng(step + 1234),
                                  noise=0.05)
    cache = net.forward(x_eval)
    y_eval = cache["y"]
    # Pair to plot: y_0 vs y_1.
    colours = np.where(b_eval[:, perm[0]] > 0, "#d62728", "#1f77b4")
    ax_sc.scatter(y_eval[:, 0], y_eval[:, 1], c=colours, s=12,
                  alpha=0.7, edgecolors="none")
    ax_sc.set_xlim(-0.05, 1.05)
    ax_sc.set_ylim(-0.05, 1.05)
    ax_sc.set_xlabel(r"$y_0$", fontsize=10)
    ax_sc.set_ylabel(r"$y_1$", fontsize=10)
    ax_sc.set_title(
        f"Code samples (colour = sign of $b_{{{perm[0]}}}$)  "
        f"bit acc = {metrics['bit_acc']*100:.1f}%",
        fontsize=10,
    )
    ax_sc.grid(alpha=0.3)
    ax_sc.set_aspect("equal")

    # ---- bottom: training curves ----
    ax_c = fig.add_subplot(gs[1, :])
    steps = np.asarray(history["step"]) if history["step"] else np.array([0])
    ax_c.plot(steps, history["L_recon"], color="#1f77b4",
              linewidth=1.4, label=r"$L_{recon}$")
    ax_c.plot(steps, history["L_pred"], color="#d62728",
              linewidth=1.4, label=r"$L_{pred}$")
    ax_c.plot(steps, history["pairwise_mi"], color="#2ca02c",
              linewidth=1.4, label="pairwise MI")
    ax_c.axhline(0.25, color="black", linestyle=":", linewidth=0.8,
                 alpha=0.6)
    if step > 0:
        ax_c.axvline(step, color="black", linewidth=0.8, alpha=0.4)
    ax_c.set_xlim(0, max(total_steps - 1, 1))
    ax_c.set_ylim(-0.02, 0.85)
    ax_c.set_xlabel("step", fontsize=9)
    ax_c.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax_c.grid(alpha=0.3)

    fig.suptitle(
        f"Predictability minimization  step {step}/{total_steps - 1}  "
        f"(λ = {cur_lam:.2f})",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--snapshot-every", type=int, default=30)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default="predictability_min_binary_factors.gif")
    p.add_argument("--hold-final", type=int, default=18)
    args = p.parse_args()

    frames: list[Image.Image] = []
    eval_rng = np.random.default_rng(args.seed + 4242)
    lam_max = 1.0
    lam_warmup = 400

    def cb(step, net, history, M):
        cur_lam = lam_max * min(1.0, step / max(lam_warmup, 1))
        frame = render_frame(step, net, history, M, cur_lam,
                             args.steps, eval_rng)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  step {step:5d}  "
              f"L_rec={history['L_recon'][-1]:.4f}  "
              f"L_pred={history['L_pred'][-1]:.4f}  "
              f"pMI={history['pairwise_mi'][-1]:.4f}")

    print(f"Training {args.steps} steps, snapshot every {args.snapshot_every}...")
    net, history, M = train(
        n_steps=args.steps,
        seed=args.seed,
        lam=lam_max,
        lam_warmup=lam_warmup,
        snapshot_callback=cb,
        snapshot_every=args.snapshot_every,
        log_every=args.snapshot_every,
        verbose=False,
    )

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
