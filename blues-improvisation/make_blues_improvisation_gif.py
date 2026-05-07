"""Generate blues_improvisation.gif — training-time animation.

For each training snapshot, the frame shows:
  (top)    chord track (color cells over 12 bars)
  (middle) pitch piano-roll of the model's free-running generated chorus
  (bottom) training-loss curve so far

Usage:
    python3 make_blues_improvisation_gif.py --seed 0 --epochs 200 \
        --snapshot-every 10
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from blues_improvisation import (
    BLUES_PROGRESSION, BARS_PER_CHORUS, N_CHORDS, N_PITCHES, PITCH_NAMES,
    REST, STEPS_PER_BAR, STEPS_PER_CHORUS, train,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--h1", type=int, default=20)
    ap.add_argument("--h2", type=int, default=24)
    ap.add_argument("--n-pieces", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--lr-decay-every", type=int, default=80)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="blues_improvisation.gif")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    print(f"Training with snapshots every {args.snapshot_every} epochs...")
    params, history, snapshots, _ = train(
        seed=args.seed, h1=args.h1, h2=args.h2,
        n_pieces=args.n_pieces, epochs=args.epochs,
        batch_size=args.batch, lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        eval_every=args.snapshot_every,
        save_snapshots=True, verbose=False,
    )
    print(f"  collected {len(snapshots)} snapshots")
    print(f"  final chord_acc {history.chord_acc[-1]:.3f}  "
          f"pitch_acc {history.pitch_acc[-1]:.3f}")

    fig = plt.figure(figsize=(8.5, 6.0), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.45, 3.0, 1.5],
                          hspace=0.45, left=0.10, right=0.97,
                          top=0.92, bottom=0.10)
    ax_c = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_l = fig.add_subplot(gs[2])

    # global y-range for loss curve
    loss_lo = max(1e-3, min(history.loss) * 0.9)
    loss_hi = max(history.loss) * 1.1
    chord_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # C7, F7, G7

    n = STEPS_PER_CHORUS
    writer = PillowWriter(fps=args.fps)
    print(f"Rendering {args.out}...")
    with writer.saving(fig, args.out, dpi=fig.dpi):
        for snap in snapshots:
            gc = snap["gen_c"]; gp = snap["gen_p"]

            # ---- chord strip ----
            ax_c.clear()
            for t in range(n):
                ax_c.add_patch(plt.Rectangle(
                    (t - 0.5, 0), 1, 1,
                    facecolor=chord_colors[gc[t]], edgecolor="none"))
            ax_c.set_xlim(-0.5, n - 0.5); ax_c.set_ylim(0, 1)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.set_ylabel("chord", fontsize=9)
            for b in range(BARS_PER_CHORUS + 1):
                ax_c.axvline(b * STEPS_PER_BAR - 0.5, color="white", lw=1.2)
            for b in range(BARS_PER_CHORUS):
                cx = b * STEPS_PER_BAR + STEPS_PER_BAR / 2 - 0.5
                ax_c.text(cx, 1.4, BLUES_PROGRESSION[b],
                          ha="center", fontsize=7, color="black")
            ax_c.set_title(
                f"epoch {snap['epoch']:4d}    "
                f"loss {snap['loss']:.3f}    "
                f"chord_acc {snap['chord_acc']:.3f}    "
                f"pitch_acc {snap['pitch_acc']:.3f}",
                fontsize=10, loc="left")

            # ---- piano-roll ----
            ax_p.clear()
            for t in range(n):
                ax_p.add_patch(plt.Rectangle(
                    (t - 0.5, gp[t] - 0.4), 1, 0.8,
                    facecolor="0.25" if gp[t] != REST else "0.85",
                    edgecolor="white", lw=0.4))
            ax_p.set_xlim(-0.5, n - 0.5)
            ax_p.set_ylim(-0.6, N_PITCHES - 0.4)
            ax_p.set_yticks(range(N_PITCHES))
            ax_p.set_yticklabels(PITCH_NAMES, fontsize=7)
            ax_p.set_ylabel("pitch", fontsize=9)
            ax_p.set_xticks([])
            for b in range(BARS_PER_CHORUS + 1):
                ax_p.axvline(b * STEPS_PER_BAR - 0.5,
                             color="0.5", lw=0.5, ls=":")

            # ---- loss curve ----
            ax_l.clear()
            it_idx = history.epochs.index(snap["epoch"])
            xs = history.epochs[: it_idx + 1]
            ys = history.loss[: it_idx + 1]
            ax_l.plot(xs, ys, "C0-", lw=1.5, label="total")
            ax_l.plot(history.epochs[: it_idx + 1],
                      history.loss_c[: it_idx + 1], "C1-", lw=1.0,
                      label="chord head")
            ax_l.plot(history.epochs[: it_idx + 1],
                      history.loss_p[: it_idx + 1], "C2-", lw=1.0,
                      label="pitch head")
            ax_l.scatter([xs[-1]], [ys[-1]], color="C0", s=22, zorder=3)
            ax_l.set_xlim(0, history.epochs[-1])
            ax_l.set_ylim(loss_lo, loss_hi)
            ax_l.set_xlabel("epoch")
            ax_l.set_ylabel("loss")
            ax_l.grid(alpha=0.3)
            ax_l.legend(loc="upper right", fontsize=7)

            writer.grab_frame()

    plt.close(fig)
    size = os.path.getsize(args.out) / 1e6
    print(f"  wrote {args.out}  ({size:.2f} MB, {len(snapshots)} frames)")


if __name__ == "__main__":
    main()
