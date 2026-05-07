"""
Static visualizations for embedded-reber.

Outputs (under viz/):
    training_curves.png      cross-entropy + outer-T/P accuracy across training
    weight_hinton.png        Hinton diagrams of the LSTM gate weight matrices
    sample_rollout.png       a fresh embedded-Reber string with the model's
                             predicted next-symbol distribution at every step
    grammar.png              the embedded-Reber automaton drawn schematically
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embedded_reber import (
    ALPHABET, N_SYM, SYM2IDX,
    LSTM1997, train, gen_embedded_reber, encode, make_io, legal_next,
)


def hinton(ax, M, title=""):
    """Hinton diagram on `ax` for matrix M (rows × cols)."""
    M = np.asarray(M)
    rows, cols = M.shape
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_facecolor("#dddddd")
    vmax = np.abs(M).max() + 1e-9
    for r in range(rows):
        for c in range(cols):
            v = M[r, c]
            color = "white" if v > 0 else "black"
            size = np.sqrt(abs(v) / vmax) * 0.9
            if size > 0:
                ax.add_patch(plt.Rectangle((c - size / 2, r - size / 2),
                                           size, size, facecolor=color, edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=8000)
    ap.add_argument("--outdir", default="viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("training...")
    out = train(seed=args.seed, max_seqs=args.max_seqs, eval_every=200,
                eval_n=200, verbose=False)
    net: LSTM1997 = out["net"]

    # ------------------------------------------------------------------
    # 1. training curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    losses = np.array(out["losses"])
    # smooth the per-sequence loss trace with a rolling mean
    win = 50
    if len(losses) >= win:
        kern = np.ones(win) / win
        smooth = np.convolve(losses, kern, mode="valid")
        x = np.arange(len(smooth)) + win
        axes[0].plot(x, smooth, color="C0")
    else:
        axes[0].plot(losses, color="C0")
    axes[0].set_xlabel("training sequences")
    axes[0].set_ylabel("loss / step (smoothed)")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    seq_counts = np.array(out["seq_counts"])
    legal = np.array(out["legal_curve"])
    outer = np.array(out["outer_curve"])
    axes[1].plot(seq_counts, legal, label="legal-symbol acc", color="C2")
    axes[1].plot(seq_counts, outer, label="outer T/P acc", color="C3")
    axes[1].set_xlabel("training sequences")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title("Eval accuracy (200 fresh strings)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=130)
    plt.close(fig)
    print("  wrote training_curves.png")

    # ------------------------------------------------------------------
    # 2. weight Hinton diagrams
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    hinton(axes[0], net.W_in, "W_in  (input gate)")
    hinton(axes[1], net.W_out, "W_out (output gate)")
    hinton(axes[2], net.W_c, "W_c   (cell input)")
    hinton(axes[3], net.W_y, "W_y   (logits)")
    fig.suptitle("LSTM weight matrices after training "
                 "(rows=units, cols=[x | h_prev])", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "weight_hinton.png"), dpi=130)
    plt.close(fig)
    print("  wrote weight_hinton.png")

    # ------------------------------------------------------------------
    # 3. sample rollout: a fresh embedded-Reber string + predictions
    # ------------------------------------------------------------------
    rng = np.random.default_rng(2025)
    sample = gen_embedded_reber(rng)
    X, y = make_io(sample)
    probs = net.predict(X)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(sample)), 4.0))
    im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(N_SYM))
    ax.set_yticklabels(ALPHABET)
    ax.set_xticks(range(len(sample) - 1))
    ax.set_xticklabels([f"{i}\n{sample[i]}" for i in range(len(sample) - 1)],
                       fontsize=8)
    ax.set_xlabel("step (input symbol shown below index)")
    ax.set_ylabel("predicted next symbol")
    ax.set_title(f"Next-symbol distribution on a fresh string: {sample}")

    # mark the legal next set with red boxes; the outer-prediction column with
    # a yellow border.
    for t in range(len(sample) - 1):
        for s in legal_next(sample, t):
            r = SYM2IDX[s]
            ax.add_patch(plt.Rectangle((t - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=1.2))
    t_outer = len(sample) - 3
    ax.add_patch(plt.Rectangle((t_outer - 0.5, -0.5), 1, N_SYM,
                               fill=False, edgecolor="yellow", lw=2.2))

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("p(next = sym)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_rollout.png"), dpi=130)
    plt.close(fig)
    print("  wrote sample_rollout.png")

    # ------------------------------------------------------------------
    # 4. grammar diagram (schematic, hand-laid)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # outer skeleton
    outer_layout = [
        (0, 0, "B"),
        (1.5, 0, "T/P"),
        (3.0, 0, "[inner Reber]"),
        (5.0, 0, "T/P"),
        (6.5, 0, "E"),
    ]
    for (x, y, label) in outer_layout:
        ax.add_patch(plt.Circle((x, y), 0.45, fc="#ddeaff", ec="black"))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    for i in range(4):
        x0 = outer_layout[i][0] + 0.45
        x1 = outer_layout[i + 1][0] - 0.45
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(3.5, 1.2,
            "the second T/P MUST match the first --\n"
            "this is the long-range dependency",
            ha="center", fontsize=9, color="#cc3333",
            bbox=dict(boxstyle="round", fc="#fff5e8", ec="#cc3333"))

    # inner schematic, off to the right
    nodes = {
        "i0": (8.0, 0.0, "B"),
        "i1": (9.0, 0.7, "1"),
        "i2": (9.0, -0.7, "2"),
        "i3": (10.5, 0.7, "3"),
        "i4": (10.5, -0.7, "4"),
        "i5": (12.0, 0.0, "5"),
        "i6": (13.5, 0.0, "E"),
    }
    for (x, y, label) in nodes.values():
        ax.add_patch(plt.Circle((x, y), 0.32, fc="#eaffea", ec="black"))
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    arrows = [
        ("i0", "i1", "T"),
        ("i0", "i2", "P"),
        ("i1", "i1", "S"),
        ("i1", "i3", "X"),
        ("i2", "i2", "T"),
        ("i2", "i4", "V"),
        ("i3", "i2", "X"),
        ("i3", "i5", "S"),
        ("i4", "i3", "P"),
        ("i4", "i5", "V"),
        ("i5", "i6", "E"),
    ]
    for a, b, label in arrows:
        x0, y0, _ = nodes[a]
        x1, y1, _ = nodes[b]
        if a == b:
            # self loop, draw a small bump above
            ax.annotate("", xy=(x0 - 0.32, y0 + 0.05), xytext=(x0 - 0.32, y0 - 0.05),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=2.5",
                                        lw=1.0, color="black"))
            ax.text(x0 - 0.85, y0, label, fontsize=7, ha="center", va="center")
        else:
            dx, dy = x1 - x0, y1 - y0
            d = np.hypot(dx, dy)
            ux, uy = dx / d, dy / d
            sx, sy = x0 + 0.34 * ux, y0 + 0.34 * uy
            ex, ey = x1 - 0.34 * ux, y1 - 0.34 * uy
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
            ax.text((sx + ex) / 2, (sy + ey) / 2 + 0.18, label,
                    fontsize=7, ha="center", va="center")
    ax.text(10.7, -1.6, "[inner Reber automaton]", ha="center", fontsize=9, color="#357")

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "grammar.png"), dpi=130)
    plt.close(fig)
    print("  wrote grammar.png")


if __name__ == "__main__":
    main()
