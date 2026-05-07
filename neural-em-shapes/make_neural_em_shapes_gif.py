"""Build neural_em_shapes.gif: animation of slot-assignment evolution.

Two stacked panels per frame:
  TOP    -- per-iteration slot responsibilities for 3 example images at
             the current snapshot (4 columns: input | iter 0 | iter 1 |
             iter T-1).  Same layout as the static viz but for one
             snapshot at a time.
  BOTTOM -- training curves so far: train loss + test NMI.

Frames step through the saved per-epoch snapshots (sub-sampled to ~12).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_run():
    p = os.path.join(HERE, "run.json")
    npz = os.path.join(HERE, "run_viz.npz")
    if not (os.path.exists(p) and os.path.exists(npz)):
        print("run.json / run_viz.npz missing; running neural_em_shapes.py ...", flush=True)
        subprocess.check_call(
            [sys.executable, os.path.join(HERE, "neural_em_shapes.py"),
             "--seed", "0", "--no-grad-check"]
        )
    with open(p) as f:
        return json.load(f), np.load(npz)


def main():
    run, vz = _ensure_run()
    canvas = run["config"]["canvas"]
    K = run["config"]["K"]
    T = run["config"]["T"]

    x_all = vz["viz_x"]
    mask_all = vz["viz_m"]
    snap_gamma = vz["snap_gamma"]   # (n_snaps, T, B, K, D)
    snap_epochs = list(vz["snap_epochs"].tolist())
    n_snaps = snap_gamma.shape[0]

    history = run["history"]
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    test_nmi = history["test_nmi"]

    palette = np.array([
        [0.95, 0.30, 0.30],
        [0.20, 0.80, 0.20],
        [0.30, 0.50, 0.95],
    ])
    if K > 3:
        rng = np.random.default_rng(0)
        extra = rng.uniform(0.2, 0.95, size=(K - 3, 3))
        palette = np.concatenate([palette, extra], axis=0)
    palette = palette[:K]

    n_images_show = 3
    iters_show = [0, max(1, T // 2), T - 1]  # 3 iterations to show

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.35)
    gs_top = gs[0].subgridspec(n_images_show, 1 + len(iters_show), hspace=0.05,
                                wspace=0.05)
    gs_bot = gs[1].subgridspec(1, 1)

    ax_grid = [[fig.add_subplot(gs_top[r, c])
                for c in range(1 + len(iters_show))]
               for r in range(n_images_show)]
    ax_curve = fig.add_subplot(gs_bot[0])
    ax_curve_nmi = ax_curve.twinx()

    # Pre-compute background images
    for r in range(n_images_show):
        ax_grid[r][0].imshow(x_all[r].reshape(canvas, canvas),
                              cmap="gray_r", vmin=0, vmax=1)
        ax_grid[r][0].set_xticks([]); ax_grid[r][0].set_yticks([])
        if r == 0:
            ax_grid[r][0].set_title("input", fontsize=9)
        ax_grid[r][0].set_ylabel(f"img {r}", fontsize=8)

    # Cell containers for the iter columns
    iter_imgs = [[ax_grid[r][1 + c].imshow(np.ones((canvas, canvas, 3)))
                   for c in range(len(iters_show))]
                  for r in range(n_images_show)]
    for c, t in enumerate(iters_show):
        ax_grid[0][1 + c].set_title(f"iter {t}", fontsize=9)
        for r in range(n_images_show):
            ax_grid[r][1 + c].set_xticks([])
            ax_grid[r][1 + c].set_yticks([])

    # Curve panel
    line_loss, = ax_curve.plot([], [], "o-", color="#d62728",
                                label="train loss", markersize=3)
    line_nmi, = ax_curve_nmi.plot([], [], "s-", color="#2ca02c",
                                    label="test NMI", markersize=3)
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("train loss", color="#d62728")
    ax_curve_nmi.set_ylabel("test NMI", color="#2ca02c")
    ax_curve.set_xlim(0, max(epochs) + 0.5)
    ax_curve.set_ylim(0, max(train_loss) * 1.1)
    ax_curve_nmi.set_ylim(0, 1)
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="lower left", fontsize=8)
    ax_curve_nmi.legend(loc="lower right", fontsize=8)

    sup = fig.suptitle("", fontsize=11)

    def render_iter_panel(img_idx, gamma_iter):
        """Build (canvas,canvas,3) RGB hard-assignment image."""
        g = gamma_iter[img_idx]  # (K, D)
        argmax = g.argmax(axis=0)
        rgb = palette[argmax]
        fg_w = x_all[img_idx][:, None]
        rgb_img = 1.0 - fg_w * (1.0 - rgb)
        return rgb_img.reshape(canvas, canvas, 3)

    def update(frame):
        ep = snap_epochs[frame]
        gamma_per_iter = snap_gamma[frame]  # (T, B, K, D)
        for r in range(n_images_show):
            for c, t in enumerate(iters_show):
                img = render_iter_panel(r, gamma_per_iter[t])
                iter_imgs[r][c].set_data(img)
        idx = epochs.index(ep)
        line_loss.set_data(epochs[: idx + 1], train_loss[: idx + 1])
        line_nmi.set_data(epochs[: idx + 1], test_nmi[: idx + 1])
        sup.set_text(
            f"N-EM training | epoch {ep:3d} | "
            f"train_loss {train_loss[idx]:.3f} | "
            f"test_NMI {test_nmi[idx]:.3f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=n_snaps, blit=False)
    out_path = os.path.join(HERE, "neural_em_shapes.gif")
    ani.save(out_path, writer=PillowWriter(fps=2))
    plt.close()
    sz = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({sz / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
