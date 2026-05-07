"""Animated training visualization for two-sequence-noise.

Renders ``two_sequence_noise.gif``: a fixed pair of test sequences (one per
class) is replayed at training snapshots so the y_out trace visibly converges
to its targets (0.2 / 0.8) as training progresses.

Usage:
    python3 make_two_sequence_noise_gif.py --seed 0 --steps 8000 --T 100 \
            --max-frames 40 --fps 8
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_sequence_noise import (  # noqa: E402
    LSTM1997,
    forward,
    label_to_target,
    make_sequence,
    train,
)


def fixed_test_sequences(seed: int = 12345):
    """Pick one sequence of each class for the animation."""
    rng = np.random.default_rng(seed)
    out = {}
    while len(out) < 2:
        x_seq, label = make_sequence(rng, T=100, p1=10)
        if label not in out:
            out[label] = x_seq
    return out  # {0: x_seq_class0, 1: x_seq_class1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="two_sequence_noise.gif")
    args = p.parse_args()

    test_seqs = fixed_test_sequences()
    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = []

    snapshot_every = max(args.steps // args.max_frames, 1)

    def cb(step, net: LSTM1997, history, rng_data):
        # evaluate on the two fixed sequences
        traces = []
        for label in (0, 1):
            cache = forward(net, test_seqs[label])
            traces.append(np.asarray(cache["y_out"], dtype=np.float64))
        snapshots.append((step + 1, traces[0], traces[1]))

    print(f"# make_two_sequence_noise_gif  seed={args.seed}  "
          f"steps={args.steps}  T={args.T}  max_frames={args.max_frames}")
    train(
        seed=args.seed,
        n_steps=args.steps,
        T=args.T,
        snapshot_every=snapshot_every,
        snapshot_callback=cb,
        log_every=max(args.steps // 10, 1),
        verbose=False,
    )

    if not snapshots:
        raise RuntimeError("no snapshots captured")

    # ------------------------------------------------------------------
    # Build the animation
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    ts = np.arange(args.T)

    lines = []
    targets = []
    for col, label in enumerate((0, 1)):
        target = label_to_target(label)
        axes[col].plot(
            ts, test_seqs[label][:, 0],
            color="0.65", linewidth=0.7, label="input x"
        )
        axes[col].axvspan(0, 10, color="C0", alpha=0.15)
        axes[col].axhline(target, color="black", linestyle="--",
                          linewidth=0.9, label=f"target {target:.1f}")
        axes[col].axhline(0.5, color="grey", linestyle=":",
                          linewidth=0.7)
        line_y, = axes[col].plot(
            ts, np.zeros(args.T),
            color="C1", linewidth=1.4, label="y_out(t)"
        )
        lines.append(line_y)
        targets.append(target)
        axes[col].set_xlim(0, args.T - 1)
        axes[col].set_ylim(-1.4, 1.4)
        axes[col].set_xlabel("time step t")
        axes[col].set_title(f"class {label}")
        axes[col].grid(True, alpha=0.3)
        axes[col].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("input x  /  y_out")

    title = fig.suptitle("step ?", fontsize=11)

    def render(frame_idx: int):
        step, y0, y1 = snapshots[frame_idx]
        lines[0].set_ydata(y0)
        lines[1].set_ydata(y1)
        title.set_text(
            f"two-sequence-noise (3c)  step {step:6d}  "
            f"y_out[T-1]: cls0={y0[-1]:.3f}  cls1={y1[-1]:.3f}"
        )
        return lines + [title]

    anim = FuncAnimation(
        fig, render,
        frames=len(snapshots),
        interval=1000 / args.fps,
        blit=False,
    )
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    writer = PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(snapshots)} frames)")


if __name__ == "__main__":
    main()
