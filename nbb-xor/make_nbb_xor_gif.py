"""
Render an animated GIF showing the NBB learning dynamics on XOR.

Layout per frame:
  Top-left:    Hinton diagram of W_ih (input -> hidden).
  Top-right:   W_ho output-preference bar (W_ho[:,0] - W_ho[:,1] per hidden).
  Bottom:      training curves (accuracy + total weight-substance) so far.

Usage:
    python3 make_nbb_xor_gif.py
    python3 make_nbb_xor_gif.py --seed 0 --snapshot-every 30 --fps 14
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# tight_layout fights with the explicit gridspec hspace; the layout we set
# is correct, so silence the cosmetic warning.
warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from nbb_xor import NBB, train, make_xor_patterns, evaluate


PATTERN_LABELS = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]


def render_frame(nbb: NBB,
                 history: dict,
                 presentations: int,
                 max_x: int) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.55, wspace=0.30)

    # ---- top-left: W_ih Hinton diagram --------------------------------
    ax_w = fig.add_subplot(gs[0, 0])
    W = nbb.W_ih
    max_abs = max(abs(W).max(), 1e-12)
    for i in range(3):
        for j in range(3):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w >= 0 else "#003366"
            ax_w.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                     facecolor=color, edgecolor="black",
                                     linewidth=0.3))
    ax_w.set_xlim(-0.6, 2.6)
    ax_w.set_ylim(-0.6, 2.6)
    ax_w.invert_yaxis()
    ax_w.set_xticks([0, 1, 2])
    ax_w.set_xticklabels(["h[0]", "h[1]", "h[2]"], fontsize=9)
    ax_w.set_yticks([0, 1, 2])
    ax_w.set_yticklabels(["bias", "x1", "x2"], fontsize=9)
    ax_w.set_aspect("equal")
    ax_w.set_title(f"$W_{{ih}}$  (max = {W.max():.3g})", fontsize=10)

    # ---- top-right: output preference per hidden unit ------------------
    ax_p = fig.add_subplot(gs[0, 1])
    diff = nbb.W_ho[:, 0] - nbb.W_ho[:, 1]
    colors = ["#1f77b4" if d > 0 else "#ff7f0e" for d in diff]
    ax_p.barh(range(3), diff, color=colors, edgecolor="black", linewidth=0.4)
    ax_p.axvline(0, color="black", linewidth=0.6)
    ax_p.set_yticks([0, 1, 2])
    ax_p.set_yticklabels(["h[0]", "h[1]", "h[2]"], fontsize=9)
    ax_p.invert_yaxis()
    ax_p.set_title("$W_{ho}[h,0] - W_{ho}[h,1]$\n"
                   "(blue = prefers out[0] / XOR=0)", fontsize=9)
    ax_p.set_xlim(-0.06, 0.06)
    ax_p.grid(alpha=0.3, axis="x")

    # ---- bottom: training curves --------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.plot(history["presentations"], history["accuracy"],
              color="#1f77b4", linewidth=1.0, label="# correct (0-4)")
    ax_acc2 = ax_c.twinx()
    ax_acc2.plot(history["presentations"], history["total_substance"],
                 color="#9467bd", linewidth=1.0,
                 label="total weight-substance")
    ax_c.axvline(presentations, color="black", linewidth=1.0, alpha=0.5)
    ax_c.set_xlabel("pattern presentations", fontsize=9)
    ax_c.set_ylabel("# correct", fontsize=9, color="#1f77b4")
    ax_acc2.set_ylabel("$\\sum w$", fontsize=9, color="#9467bd")
    ax_c.set_xlim(0, max_x)
    ax_c.set_ylim(-0.2, 4.3)
    ax_c.grid(alpha=0.3)

    fig.suptitle(f"NBB XOR — presentation {presentations}",
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
    p.add_argument("--snapshot-every", type=int, default=40)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--out", type=str, default="nbb_xor.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames = []

    # We pre-train once (cheap) so we know max_x and can put the playhead
    # in a fixed-extent panel.
    print(f"Probing convergence (seed={args.seed})...")
    history_probe = {"presentations": [], "accuracy": [],
                     "W_ih_norm": [], "W_ho_norm": [], "total_substance": []}
    _, presentations_total, _ = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        eta=args.eta, lam=args.lam,
        history=history_probe, log_every=4, verbose=False,
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
                  f"acc={history['accuracy'][-1]}/4")

    print("Re-running with snapshots...")
    history = {"presentations": [], "accuracy": [],
               "W_ih_norm": [], "W_ho_norm": [], "total_substance": []}
    nbb, _, _ = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        eta=args.eta, lam=args.lam,
        history=history, log_every=4, verbose=False,
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
