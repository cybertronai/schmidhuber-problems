"""Generate adding_problem.gif — training-time animation of the LSTM.

For each training snapshot, the frame shows:
  (top)    a fixed sample sequence with the two marked values highlighted,
           and the predicted vs target sum at the current iteration
  (middle) the LSTM cell state c_t over time on that sequence (heatmap)
  (bottom) the test-MSE training curve so far (log scale)

Usage:
    python3 make_adding_problem_gif.py --seed 0 --T 100 --hidden 8
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from adding_problem import lstm_forward, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--snapshot-every", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-decay-every", type=int, default=1500)
    ap.add_argument("--out", type=str, default="adding_problem.gif")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    print(f"Training LSTM with snapshots every {args.snapshot_every} iters...")
    _, history, snapshots = train(
        model="lstm", T=args.T, hidden=args.hidden, seed=args.seed,
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
        eval_every=args.snapshot_every,
        lr_decay_every=args.lr_decay_every, lr_decay_factor=0.5,
        verbose=False, save_snapshots=True,
    )
    print(f"  collected {len(snapshots)} snapshots")
    print(f"  final test MSE = {history.test_mse[-1]:.4f}  "
          f"solve_rate = {history.solve_rate[-1]:.3f}")

    # Build a fixed display sequence (the same one shown across all frames).
    seq_idx = 0  # snapshots stored 4 sequences in Xs; pick the first.

    fig = plt.figure(figsize=(8.5, 5.5), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 2.2, 2.0],
                          hspace=0.65, left=0.10, right=0.97,
                          top=0.93, bottom=0.10)
    ax_seq = fig.add_subplot(gs[0])
    ax_cell = fig.add_subplot(gs[1])
    ax_loss = fig.add_subplot(gs[2])

    # Pre-compute global cell-state colour scale across all snapshots
    cmax = max(float(np.abs(s["c"][1:, seq_idx, :]).max())
               for s in snapshots)
    cmax = max(cmax, 1e-3)

    # Final loss curve range for stable ylim
    loss_lo = max(1e-4, min(history.test_mse) * 0.5)
    loss_hi = max(history.test_mse) * 1.2
    seq_xs = snapshots[0]["Xs"][:, seq_idx, :]
    vals = seq_xs[:, 0]
    marks = seq_xs[:, 1]
    marker_idx = np.where(marks > 0.5)[0]
    target = float(snapshots[0]["ys"][seq_idx])

    writer = PillowWriter(fps=args.fps)
    print(f"Rendering {args.out}...")
    with writer.saving(fig, args.out, dpi=fig.dpi):
        for snap in snapshots:
            # ----- top: input sequence + prediction -----
            ax_seq.clear()
            ax_seq.bar(np.arange(args.T), vals, color="lightgray",
                       edgecolor="lightgray", width=0.9)
            ax_seq.bar(marker_idx, vals[marker_idx], color="C1",
                       edgecolor="C1", width=0.9)
            ax_seq.axhline(0, color="k", lw=0.5)
            ax_seq.set_xlim(-0.5, args.T - 0.5)
            ax_seq.set_ylim(-1.1, 1.1)
            ax_seq.set_ylabel("input value", fontsize=9)
            pred = float(snap["preds"][seq_idx])
            err = pred - target
            ax_seq.set_title(
                f"iter {snap['iter']:5d}    "
                f"sequences seen {snap['sequences']:7d}    "
                f"target {target:+.3f}    "
                f"pred {pred:+.3f}    "
                f"err {err:+.3f}",
                fontsize=10, loc="left")

            # ----- middle: cell state heatmap -----
            ax_cell.clear()
            c = snap["c"][1:, seq_idx, :].T  # (H, T)
            ax_cell.imshow(c, aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-cmax, vmax=cmax,
                           extent=(-0.5, args.T - 0.5,
                                   -0.5, c.shape[0] - 0.5))
            for m in marker_idx:
                ax_cell.axvline(m, color="k", ls=":", lw=0.8, alpha=0.7)
            ax_cell.set_ylabel("LSTM cell\nstate c_t", fontsize=9)
            ax_cell.set_yticks(np.arange(c.shape[0]))

            # ----- bottom: test MSE so far -----
            ax_loss.clear()
            it_idx = history.iters.index(snap["iter"])
            xs = history.sequences_seen[: it_idx + 1]
            ys = history.test_mse[: it_idx + 1]
            ax_loss.plot(xs, ys, "C0-", lw=1.5)
            ax_loss.scatter([xs[-1]], [ys[-1]], color="C0", s=22, zorder=3)
            ax_loss.axhline(0.04, color="k", ls="--", lw=0.8,
                            label="paper threshold (MSE = 0.04)")
            ax_loss.set_yscale("log")
            ax_loss.set_xlim(0, history.sequences_seen[-1])
            ax_loss.set_ylim(loss_lo, loss_hi)
            ax_loss.set_xlabel("sequences seen")
            ax_loss.set_ylabel("test MSE  (log)")
            ax_loss.grid(alpha=0.3, which="both")
            ax_loss.legend(loc="upper right", fontsize=8)

            writer.grab_frame()

    plt.close(fig)
    size = os.path.getsize(args.out) / 1e6
    print(f"  wrote {args.out}  ({size:.2f} MB, {len(snapshots)} frames)")


if __name__ == "__main__":
    main()
