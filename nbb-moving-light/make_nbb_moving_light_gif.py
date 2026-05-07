"""
Render an animated GIF showing the NBB learning dynamics on moving-light.

Layout per frame:
  Top-left:    W_io heatmap (input -> output) — the structure that lets
               cell 0 favour out[0] and cell 4 favour out[1] emerging.
  Top-right:   per-cell output preference (W_io[i, 0] - W_io[i, 1]).
  Bottom:      training curves (frozen-eval accuracy + total weight-substance)
               with a playhead at the current presentation count.

Usage:
    python3 make_nbb_moving_light_gif.py
    python3 make_nbb_moving_light_gif.py --seed 0 --snapshot-every 4 --fps 12
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from nbb_moving_light import NBBMovingLight, train


def render_frame(nbb: NBBMovingLight,
                 history: dict,
                 presentations: int,
                 max_x: int) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.55, wspace=0.30)

    n_cells = nbb.n_cells

    # ---- top-left: W_io heatmap ---------------------------------------
    ax_w = fig.add_subplot(gs[0, 0])
    W = nbb.W_io
    im = ax_w.imshow(W, cmap="magma", aspect="auto",
                     vmin=0.85, vmax=1.15)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax_w.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                      color="white" if W[i, j] < 1.02 else "black",
                      fontsize=8)
    ax_w.set_xticks([0, 1])
    ax_w.set_xticklabels(["out[0]\n(LR)", "out[1]\n(RL)"], fontsize=9)
    yticks = ["bias"] + [f"cell {c}" for c in range(n_cells)]
    ax_w.set_yticks(range(W.shape[0]))
    ax_w.set_yticklabels(yticks, fontsize=8)
    ax_w.set_title("$W_{io}$  (input → output)", fontsize=10)

    # ---- top-right: per-cell output preference --------------------------
    ax_p = fig.add_subplot(gs[0, 1])
    diff = W[:, 0] - W[:, 1]
    labels = ["bias"] + [f"cell {c}" for c in range(n_cells)]
    colors = ["#1f77b4" if d > 0 else "#ff7f0e" for d in diff]
    ax_p.barh(range(len(diff)), diff, color=colors, edgecolor="black",
              linewidth=0.4)
    ax_p.axvline(0, color="black", linewidth=0.6)
    ax_p.set_yticks(range(len(diff)))
    ax_p.set_yticklabels(labels, fontsize=8)
    ax_p.invert_yaxis()
    ax_p.set_title("per-input output preference\n"
                   "(blue = prefers out[0]/LR)", fontsize=9)
    ax_p.set_xlim(-0.16, 0.16)
    ax_p.grid(alpha=0.3, axis="x")

    # ---- bottom: training curves --------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.plot(history["presentations"], history["accuracy"],
              color="#1f77b4", linewidth=1.0, label="# correct (0-2)")
    ax_acc2 = ax_c.twinx()
    ax_acc2.plot(history["presentations"], history["total_substance"],
                 color="#9467bd", linewidth=1.0,
                 label="total weight-substance")
    ax_c.axvline(presentations, color="black", linewidth=1.0, alpha=0.5)
    ax_c.set_xlabel("sequence presentations", fontsize=9)
    ax_c.set_ylabel("# correct", fontsize=9, color="#1f77b4")
    ax_acc2.set_ylabel(r"$\sum w$", fontsize=9, color="#9467bd")
    ax_c.set_xlim(0, max_x)
    ax_c.set_ylim(-0.2, 2.3)
    ax_c.set_yticks([0, 1, 2])
    ax_c.grid(alpha=0.3)

    fig.suptitle(f"NBB moving-light — presentation {presentations}",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-presentations", type=int, default=5000)
    p.add_argument("--n-cells", type=int, default=5)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--out", type=str, default="nbb_moving_light.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames = []

    # Pre-train once to know how long the run is so the playhead has a
    # fixed extent.
    print(f"Probing convergence (seed={args.seed}, n_cells={args.n_cells})...")
    history_probe = {"presentations": [], "accuracy": [],
                     "W_io_norm": [], "W_oo_norm": [], "total_substance": []}
    _, presentations_total, _ = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        n_cells=args.n_cells,
        eta=args.eta, lam=args.lam,
        history=history_probe, log_every=2, verbose=False,
    )
    max_x = presentations_total
    print(f"  total presentations to convergence (or cap): {max_x}")

    def cb(presentations, nbb, history):
        if not history["presentations"]:
            return
        frame = render_frame(nbb, history, presentations, max_x)
        frames.append(frame)
        if len(frames) % 10 == 0:
            print(f"  frame {len(frames):3d}  pres={presentations}  "
                  f"acc={history['accuracy'][-1]}/2")

    print("Re-running with snapshots...")
    history = {"presentations": [], "accuracy": [],
               "W_io_norm": [], "W_oo_norm": [], "total_substance": []}
    nbb, _, _ = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        n_cells=args.n_cells,
        eta=args.eta, lam=args.lam,
        history=history, log_every=2, verbose=False,
        snapshot_callback=cb, snapshot_every=args.snapshot_every,
    )

    # Always include a final frame at the end of training.
    final = render_frame(nbb, history, history["presentations"][-1], max_x)
    frames.append(final)

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
