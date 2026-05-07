"""Build temporal_order_4bit.gif from the snapshots captured during training.

Frames show, for one fixed example sequence per class (8 panels), how the
LSTM's cell state evolves through training. Pure matplotlib + Pillow
(no imageio dependency).
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

CLASS_NAMES = ("XXX", "XXY", "XYX", "XYY", "YXX", "YXY", "YYX", "YYY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=str, default="snapshots.npz")
    ap.add_argument("--out", type=str, default="temporal_order_4bit.gif")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(args.snap):
        raise SystemExit(f"snapshots file not found: {args.snap}")

    snap = np.load(args.snap)
    steps = snap["steps"]
    accs = snap["acc"]
    c_all = snap["c"]    # (n_snapshots, n_examples, T, H)
    record_X = snap["record_X"]
    record_y = snap["record_y"]
    n_snap, n_ex, T, H = c_all.shape

    # subsample frames
    if n_snap > args.max_frames:
        idx = np.linspace(0, n_snap - 1, args.max_frames).astype(int)
    else:
        idx = np.arange(n_snap)

    cmax = float(np.abs(c_all).max()) + 1e-6

    # 2 rows × 4 cols for 8 classes
    n_rows = 2
    n_cols = (n_ex + n_rows - 1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(1.9 * n_cols + 0.5, 2.0 * n_rows + 1.0),
                              sharey=True)
    axes_flat = axes.reshape(-1)

    images = []
    for j, ax in enumerate(axes_flat):
        if j >= n_ex:
            ax.axis("off")
            images.append(None)
            continue
        im = ax.imshow(c_all[0, j].T, aspect="auto", cmap="RdBu_r",
                       vmin=-cmax, vmax=cmax)
        ax.set_xlabel("t", fontsize=8)
        ax.set_title(f"{CLASS_NAMES[int(record_y[j])]}", fontsize=9)
        if j % n_cols == 0:
            ax.set_ylabel("cell", fontsize=8)
        seq = record_X[j]
        for t in range(T):
            tok = int(np.argmax(seq[t]))
            if tok == 4:
                ax.axvline(t, color="C2", alpha=0.6, linewidth=0.8)
            if tok == 5:
                ax.axvline(t, color="C3", alpha=0.6, linewidth=0.8)
        images.append(im)

    title = fig.suptitle(
        f"step {steps[0]:>5d}  val acc {accs[0]:.3f}  "
        "(green ticks = X, red = Y)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def frame(k: int):
        s = idx[k]
        for j, im in enumerate(images):
            if im is None:
                continue
            im.set_data(c_all[s, j].T)
        title.set_text(
            f"step {int(steps[s]):>5d}  val acc {accs[s]:.3f}  "
            "(green ticks = X, red = Y)"
        )
        return [im for im in images if im is not None] + [title]

    anim = FuncAnimation(fig, frame, frames=len(idx), interval=1000 // args.fps,
                         blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
