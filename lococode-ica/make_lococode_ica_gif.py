"""
Render an animated GIF showing LOCOCODE training dynamics.

Layout per frame:
  Top-left:    |W @ A| heatmap (recovered demixer aligned with true mixing).
               At init this is a Gram-like blob; by the end it is a near
               permutation matrix.
  Top-right:   histogram of one hidden unit, with Laplace and Gaussian
               reference curves overlaid. The unit becomes super-Gaussian
               (kurtotic) over training.
  Bottom:      Amari distance + mean kurtosis curves with a playhead
               at the current epoch.

Usage:
    python3 make_lococode_ica_gif.py
    python3 make_lococode_ica_gif.py --seed 0 --snapshot-every 5 --fps 8
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings(
    "ignore", message=".*not compatible with tight_layout.*"
)

from lococode_ica import (
    generate_dataset, train_lococode, _kurtosis, amari_distance,
)
from visualize_lococode_ica import _greedy_permute


def render_frame(W_xspace, A, X, history, epoch, max_epoch, color="#9467bd"):
    fig = plt.figure(figsize=(10, 5.6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.55, wspace=0.30)

    # ---- top-left: |W @ A| heatmap, row-normalised, permuted ----------
    ax_p = fig.add_subplot(gs[0, 0])
    P = np.abs(W_xspace @ A)
    P = P / (P.max(axis=1, keepdims=True) + 1e-12)
    rp = _greedy_permute(P)
    Pp = P[rp]
    im = ax_p.imshow(Pp, cmap="magma", vmin=0, vmax=1, aspect="equal")
    ax_p.set_title(f"|W @ A|  (epoch {epoch})")
    ax_p.set_xlabel("true source"); ax_p.set_ylabel("recovered (permuted)")
    ax_p.set_xticks(range(P.shape[0])); ax_p.set_yticks(range(P.shape[0]))

    # ---- top-right: hidden distribution -------------------------------
    ax_h = fig.add_subplot(gs[0, 1])
    H = X @ W_xspace.T
    kurt_per = _kurtosis(H)
    idx = int(np.argmax(kurt_per))
    h_unit = H[:, idx]
    h_unit = (h_unit - h_unit.mean()) / (h_unit.std() + 1e-12)
    ax_h.hist(h_unit, bins=np.linspace(-5, 5, 50),
              density=True, color=color, alpha=0.65,
              edgecolor="black", linewidth=0.4)
    xs = np.linspace(-5, 5, 200)
    ax_h.plot(xs, 0.5 * np.exp(-np.abs(xs)),
              color="purple", lw=1.4, ls="--", label="Laplace")
    ax_h.plot(xs, 1.0 / np.sqrt(2 * np.pi) * np.exp(-xs ** 2 / 2),
              color="grey", lw=1.0, ls=":", label="Gaussian")
    ax_h.set_xlim(-5, 5); ax_h.set_ylim(0, 0.65)
    ax_h.set_title(f"hidden unit {idx}  (excess k={kurt_per[idx]:.2f})")
    ax_h.set_xlabel("activation (z-scored)")
    ax_h.legend(fontsize=8, loc="upper right")
    ax_h.grid(alpha=0.3)

    # ---- bottom: training curves --------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    e = np.array(history["epoch"])
    a = np.array(history["amari"])
    k = np.array(history["kurtosis_mean"])

    color_a = "#9467bd"; color_k = "#d62728"
    ax_c.plot(e, a, color=color_a, lw=1.5, label="Amari distance")
    ax_c.set_xlabel("epoch")
    ax_c.set_ylabel("Amari distance", color=color_a)
    ax_c.tick_params(axis="y", labelcolor=color_a)
    ax_c.set_xlim(0, max_epoch)
    ax_c.set_ylim(0, max(0.5, a.max() * 1.05))
    ax_c.axvline(epoch, color="green", lw=1.0, ls="--", alpha=0.7)
    ax_c.grid(alpha=0.3)

    ax_c2 = ax_c.twinx()
    ax_c2.plot(e, k, color=color_k, lw=1.5, label="mean kurtosis")
    ax_c2.set_ylabel("excess kurtosis", color=color_k)
    ax_c2.tick_params(axis="y", labelcolor=color_k)
    ax_c2.set_ylim(min(-0.5, k.min() - 0.5), max(3.5, k.max() + 0.5))
    ax_c2.axhline(3.0, color="purple", lw=0.7, ls="--", alpha=0.6)
    ax_c2.axhline(0.0, color="grey", lw=0.7, ls="--", alpha=0.6)

    fig.suptitle("LOCOCODE-ICA: tied AE on whitened sparse mixtures",
                 fontsize=11)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="lococode_ica.gif")
    args = p.parse_args()

    X, S, A = generate_dataset(seed=args.seed, k=args.k,
                               n_samples=args.n_samples)
    model, hist = train_lococode(
        X, seed=args.seed, epochs=args.epochs, A_true=A, S_true=S,
        snapshot_every=args.snapshot_every,
    )

    K_white = hist["K_white"]

    frames = []
    snaps = hist["snapshots"]
    for snap in snaps:
        W = snap["W"]
        epoch = snap["epoch"]
        W_xspace = W @ K_white
        frame = render_frame(W_xspace, A, X, hist, epoch, args.epochs)
        frames.append(frame)

    # Hold the final frame for a moment.
    duration = int(1000 / args.fps)
    durations = [duration] * len(frames)
    durations[-1] = max(duration * 4, 1500)

    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Saved {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
