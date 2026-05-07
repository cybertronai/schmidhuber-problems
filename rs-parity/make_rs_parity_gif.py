"""
Animate rs-parity search progression.

Each frame snapshots the random search at trial t:
  Top-left:  best-so-far accuracy curve up to trial t (log-x)
  Top-right: histogram of per-trial accuracies seen so far
  Bottom:    weight matrices of the current best RNN as Hinton diagrams

Usage:
    python3 make_rs_parity_gif.py
    python3 make_rs_parity_gif.py --n 50 --seed 0 --fps 12 --frames 60
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from rs_parity import (
    RNNParams,
    accuracy,
    make_parity_dataset,
    sample_params,
    sample_parity_dataset,
)


# ----------------------------------------------------------------------
# Frame renderer
# ----------------------------------------------------------------------

def hinton_panel(ax, M, title, max_abs):
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            w = M[i, j]
            sz = 0.85 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.3))
    ax.set_xlim(-0.6, nc - 0.4)
    ax.set_ylim(-0.6, nr - 0.4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")


def render_frame(trial: int,
                 best_acc: float,
                 best_trials_x: list[int],
                 best_accs_y: list[float],
                 acc_hist: list[float],
                 best_params: RNNParams,
                 max_trials_xlim: int,
                 N: int, H: int, weight_scale: float,
                 found_at: int | None) -> Image.Image:
    fig = plt.figure(figsize=(8.5, 5.0), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0],
                          width_ratios=[1.4, 1.0, 1.4],
                          hspace=0.55, wspace=0.45)

    # ----- top-left: search curve ---------------------------------------
    ax_curve = fig.add_subplot(gs[0, :2])
    ax_curve.set_xscale("log")
    ax_curve.set_xlim(1, max(max_trials_xlim, 10))
    ax_curve.set_ylim(40, 105)
    ax_curve.set_xlabel("trial #", fontsize=9)
    ax_curve.set_ylabel("accuracy (%)", fontsize=9)
    ax_curve.axhline(50, color="black", linestyle=":",
                     linewidth=0.6, alpha=0.5)
    ax_curve.tick_params(labelsize=8)
    if best_trials_x:
        # extend the step curve out to the current trial
        x = np.asarray(best_trials_x + [trial])
        y = np.asarray(best_accs_y + [best_accs_y[-1]]) * 100
        ax_curve.step(x, y, where="post", color="#cc0000", linewidth=2.0,
                      label="best so far")
    if found_at is not None and trial >= found_at:
        ax_curve.axvline(found_at, color="#cc0000", linestyle="--",
                         linewidth=0.8, alpha=0.6)
    ax_curve.set_title(f"trial {trial:,}   best acc = {best_acc * 100:.2f}%",
                       fontsize=10)
    ax_curve.grid(alpha=0.3)
    if best_trials_x:
        ax_curve.legend(loc="lower right", fontsize=8)

    # ----- top-right: histogram ----------------------------------------
    ax_hist = fig.add_subplot(gs[0, 2])
    if acc_hist:
        ax_hist.hist(np.array(acc_hist) * 100,
                     bins=np.arange(20, 102, 2),
                     color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax_hist.set_xlim(40, 102)
    ax_hist.axvline(50, color="black", linestyle=":",
                    linewidth=0.6, alpha=0.5)
    ax_hist.set_xlabel("acc (%)", fontsize=9)
    ax_hist.set_title("trial scores", fontsize=10)
    ax_hist.tick_params(labelsize=8)
    ax_hist.grid(alpha=0.3)

    # ----- bottom: best-RNN weights ------------------------------------
    max_abs = float(weight_scale)
    ax1 = fig.add_subplot(gs[1, 0])
    hinton_panel(ax1, best_params.W_hh, "$W_{hh}$ (best)", max_abs)
    ax2 = fig.add_subplot(gs[1, 1])
    M_xb = np.concatenate([best_params.W_xh.reshape(1, H).T,
                           best_params.b_h.reshape(H, 1)], axis=1)
    hinton_panel(ax2, M_xb, "input + bias", max_abs)
    ax3 = fig.add_subplot(gs[1, 2])
    M_yb = np.concatenate([best_params.W_hy,
                           best_params.b_y.reshape(1, 1)], axis=0)
    hinton_panel(ax3, M_yb, "readout + bias", max_abs)

    fig.suptitle(f"Random-weight guessing on N={N} parity   "
                 f"(H={H}, weight scale ±{weight_scale:g})",
                 fontsize=11)

    buf = BytesIO()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)
    return img


# ----------------------------------------------------------------------
# Run search and snapshot at log-spaced trial numbers
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--hidden", type=int, default=2)
    p.add_argument("--weight-scale", type=float, default=30.0)
    p.add_argument("--max-trials", type=int, default=50_000)
    p.add_argument("--sample-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=60,
                   help="number of GIF frames (log-spaced over trials)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="rs_parity.gif")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = sample_parity_dataset(args.n, args.sample_size, rng)

    # Decide where to snapshot. Log-spaced trial numbers from 1 -> max_trials.
    snapshot_trials = sorted(set(np.unique(np.round(np.geomspace(
        1, args.max_trials, args.frames)).astype(int)).tolist()))

    # Run RS, capturing snapshots when we hit a snapshot trial number or when
    # best_acc improves.
    best_acc = 0.0
    best_params = None
    best_trials_x: list[int] = []
    best_accs_y: list[float] = []
    acc_hist: list[float] = []
    frames: list[Image.Image] = []
    found_at = None
    snap_idx = 0

    for trial in range(1, args.max_trials + 1):
        params = sample_params(args.hidden, args.weight_scale, rng)
        acc = accuracy(params, X, y)

        # subsample acc_hist so the histogram stays cheap
        if trial % max(1, args.max_trials // 5000) == 0:
            acc_hist.append(acc)

        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_params = params
            best_trials_x.append(trial)
            best_accs_y.append(acc)
            if best_acc >= 1.0 and found_at is None:
                found_at = trial

        # snapshot if we hit the next scheduled trial OR we just improved
        snap_due = (snap_idx < len(snapshot_trials)
                    and trial >= snapshot_trials[snap_idx])
        if snap_due or improved:
            if best_params is not None:
                frames.append(render_frame(
                    trial, best_acc,
                    list(best_trials_x), list(best_accs_y),
                    acc_hist, best_params, args.max_trials,
                    args.n, args.hidden, args.weight_scale, found_at))
            while (snap_idx < len(snapshot_trials)
                   and trial >= snapshot_trials[snap_idx]):
                snap_idx += 1
            print(f"  frame {len(frames):3d}: trial {trial:7d}  "
                  f"best acc {best_acc*100:6.2f}%")

        if found_at is not None and trial > found_at:
            # render a few "victory" frames after solve, then stop
            if trial >= found_at + max(50, found_at // 4):
                break

    # Hold the last frame for ~1.5s at the chosen fps
    hold = max(1, int(args.fps * 1.5))
    frames.extend([frames[-1]] * hold)

    print(f"# {len(frames)} frames")
    frames[0].save(args.out,
                   save_all=True,
                   append_images=frames[1:],
                   duration=int(1000 / args.fps),
                   loop=0,
                   optimize=True)
    sz = os.path.getsize(args.out) / 1024.0
    print(f"# wrote {args.out}  ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
