"""Build temporal_order_3bit.gif from the snapshots captured during training.

Frames show, for one fixed example sequence per class, how the LSTM's cell
state evolves through training. Pure matplotlib + PIL fallback (no imageio
needed if Pillow is available; we use matplotlib's animation -> PillowWriter).
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

CLASS_NAMES = ("XX", "XY", "YX", "YY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=str, default="snapshots.npz")
    ap.add_argument("--out", type=str, default="temporal_order_3bit.gif")
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

    fig, axes = plt.subplots(1, n_ex, figsize=(2.6 * n_ex + 0.5, 3.4), sharey=True)
    if n_ex == 1:
        axes = [axes]

    images = []
    xy_marks = []
    for j, ax in enumerate(axes):
        im = ax.imshow(c_all[0, j].T, aspect="auto", cmap="RdBu_r",
                       vmin=-cmax, vmax=cmax)
        ax.set_xlabel("t")
        ax.set_title(f"class {CLASS_NAMES[int(record_y[j])]}")
        if j == 0:
            ax.set_ylabel("cell index")
        seq = record_X[j]
        marks = []
        for t in range(T):
            tok = int(np.argmax(seq[t]))
            if tok == 4:
                marks.append(ax.axvline(t, color="C2", alpha=0.6, linewidth=1.0))
            if tok == 5:
                marks.append(ax.axvline(t, color="C3", alpha=0.6, linewidth=1.0))
        images.append(im)
        xy_marks.append(marks)

    title = fig.suptitle(
        f"step {steps[0]:>5d}  val acc {accs[0]:.3f}  "
        "(red ticks = X positions, blue = Y)")
    fig.tight_layout()

    def frame(k: int):
        s = idx[k]
        for j, im in enumerate(images):
            im.set_data(c_all[s, j].T)
        title.set_text(
            f"step {int(steps[s]):>5d}  val acc {accs[s]:.3f}  "
            "(green ticks = X, red = Y)"
        )
        return images + [title]

    anim = FuncAnimation(fig, frame, frames=len(idx), interval=1000 // args.fps,
                         blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
