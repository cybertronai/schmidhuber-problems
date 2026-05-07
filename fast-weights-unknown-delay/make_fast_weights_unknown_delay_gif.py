"""
Render fast_weights_unknown_delay.gif --- the slow programmer net learning to
load and recall a 4-bit pattern across an unknown delay.

Frames are produced from training snapshots; each frame shows, for one fixed
test episode (delay K = 20):
    (a) the input pattern slot bits over time,
    (b) the write gate g_t over time,
    (c) the recall-step output y vs. the true pattern P,
    (d) the recall MSE training curve up to the current frame.
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from fast_weights_unknown_delay import (
    SlowNet, train, make_batch, forward_episode,
)


def render_frame(snap, history, fixed_x, fixed_P, recall_t, p_dim, eta):
    # Reconstruct slow net from snapshot.
    S = SlowNet(p_dim=p_dim, hidden=snap["params"]["W_xh"].shape[0],
                d_k=snap["params"]["W_hk"].shape[0])
    for n, p in zip(S.param_names(), S.params()):
        p[...] = snap["params"][n]
    y, cache = forward_episode(S, fixed_x, recall_t, eta)
    g = cache["g_seq"][:, 0]

    fig = Figure(figsize=(10, 6.5), dpi=110)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.55,
                          wspace=0.30)

    # (a) input pattern slot bits
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(fixed_x[0, :, :p_dim].T, aspect="auto",
                    interpolation="nearest", cmap="bwr", vmin=-1, vmax=1)
    ax1.set_yticks(range(p_dim))
    ax1.set_yticklabels([f"bit {i}" for i in range(p_dim)])
    ax1.set_title(f"Input pattern slot   "
                  f"P = {fixed_P[0].astype(int).tolist()}", fontsize=10)
    ax1.set_xlabel("episode step")

    # (b) write gate g_t
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(range(len(g)), g, "o-", color="#9467bd", linewidth=1.0)
    ax2.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(r"$g_t$")
    ax2.set_xlabel("episode step")
    ax2.set_title("Write gate (sigmoid) over the episode", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # (c) recall output vs true P
    ax3 = fig.add_subplot(gs[2, 0])
    bits = np.arange(p_dim)
    ax3.bar(bits - 0.18, fixed_P[0], width=0.36, color="#2ca02c",
            label=r"true $P$")
    ax3.bar(bits + 0.18, y[0], width=0.36, color="#ff7f0e",
            label=r"output $y$")
    ax3.set_xticks(bits)
    ax3.set_xticklabels([f"b{i}" for i in range(p_dim)])
    ax3.axhline(0, color="#888", linewidth=0.5)
    ax3.set_ylim(-1.6, 1.6)
    ax3.set_title("Recall step", fontsize=10)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # (d) loss curve up to this frame
    ax4 = fig.add_subplot(gs[2, 1])
    upto = snap["step"]
    steps = [s for s in history["step"] if s <= upto]
    losses = [history["loss"][i] for i, s in enumerate(history["step"])
              if s <= upto]
    ax4.plot(steps, losses, color="#d62728", linewidth=1.0)
    ax4.set_yscale("log")
    ax4.set_xlim(0, max(history["step"]))
    if min(history["loss"]) > 0:
        ax4.set_ylim(min(history["loss"]) * 0.5,
                     max(history["loss"]) * 2.0)
    ax4.set_xlabel("training step")
    ax4.set_ylabel("recall MSE")
    ax4.set_title(f"Loss (step {snap['step']})", fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "fast-weights-unknown-delay   "
        f"step {snap['step']}   delay K={fixed_x.shape[1] - 2}",
        fontsize=12)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--snapshot-every", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--delay", type=int, default=20)
    p.add_argument("--out", type=str, default="fast_weights_unknown_delay.gif")
    args = p.parse_args()

    print(f"training with snapshots (seed={args.seed}, iters={args.iters}, "
          f"snapshot_every={args.snapshot_every}) ...")
    S, history, snapshots = train(
        seed=args.seed,
        iters=args.iters,
        log_every=10000,        # quiet
        snapshot_every=args.snapshot_every,
        verbose=False,
    )
    print(f"  {len(snapshots)} snapshots collected.")

    # Trim frames if needed.
    if len(snapshots) > args.max_frames:
        idx = np.linspace(0, len(snapshots) - 1, args.max_frames).round().astype(int)
        snapshots = [snapshots[i] for i in idx]
    print(f"  rendering {len(snapshots)} frames ...")

    # Fixed evaluation episode (same across all frames).
    rng_eval = np.random.default_rng(99999)
    fixed_x, fixed_P, recall_t = make_batch(1, p_dim=4, delay=args.delay,
                                            rng=rng_eval)

    writer = PillowWriter(fps=args.fps)
    fig = plt.figure()
    with writer.saving(fig, args.out, dpi=110):
        for snap in snapshots:
            frame_rgba = render_frame(snap, history, fixed_x, fixed_P,
                                       recall_t, p_dim=4, eta=0.5)
            fig.clear()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(frame_rgba)
            ax.axis("off")
            writer.grab_frame()
    plt.close(fig)
    size = os.path.getsize(args.out)
    print(f"  wrote {args.out}  ({size/1024:.0f} KiB, {len(snapshots)} frames)")


if __name__ == "__main__":
    main()
