"""
Render an animated GIF that shows the flip-flop controller learning to latch.

Each frame is one snapshot during training.  At each snapshot we run a
*fixed* test episode (so the events are identical across frames) and plot:

    Top:    A / B / X events as colored vlines
    Middle: target latch state (step) and controller's output y_t
    Bottom: actual pain  (y_t - desired_t)^2  vs M's predicted pain

The accuracy and outer-step counter are annotated on the title.

Usage:
    python3 make_flip_flop_gif.py --seed 0
    python3 make_flip_flop_gif.py --seed 0 --snapshot-every 50 --fps 10
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from flip_flop import (Controller, WorldModel, train, make_episode,
                       rollout_controller, forward_world_model)


def render_frame(C: Controller, M: WorldModel, history: dict,
                 step: int, test_events: np.ndarray, test_desired: np.ndarray,
                 max_step: int) -> Image.Image:
    traj_C = rollout_controller(C, test_events, test_desired)
    traj_M = forward_world_model(M, traj_C["obs"], traj_C["y"])
    T = test_events.shape[0]
    t = np.arange(T)
    acc = float(np.mean((traj_C["y"] > 0.5).astype(float) == test_desired))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 4.5), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [0.5, 1.4, 1.0]})

    # Events
    ax = axes[0]
    a_t = np.flatnonzero(test_events[:, 0] > 0.5)
    b_t = np.flatnonzero(test_events[:, 1] > 0.5)
    x_t = np.flatnonzero(test_events[:, 2] > 0.5)
    for tt in a_t:
        ax.axvline(tt, color="#d62728", linewidth=2.0, alpha=0.85)
    for tt in b_t:
        ax.axvline(tt, color="#1f77b4", linewidth=2.0, alpha=0.85)
    for tt in x_t:
        ax.axvline(tt, color="gray", linewidth=1.0, alpha=0.4)
    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="A"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="B"),
        plt.Line2D([0], [0], color="gray", linewidth=1, label="X"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim(0, T - 1)
    ax.set_title("Events", fontsize=9)

    # Output
    ax = axes[1]
    ax.step(t, test_desired, where="post", color="black", linewidth=1.4,
            label="desired")
    ax.plot(t, traj_C["y"], color="#ff7f0e", linewidth=1.6, label=r"controller $y_t$")
    ax.axhline(0.5, color="gray", linewidth=0.4, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("output")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Latch target vs. controller output", fontsize=9)

    # Pain
    ax = axes[2]
    ax.plot(t, traj_C["pain"], color="#d62728", linewidth=1.2, label="actual pain")
    ax.plot(t, traj_M["pred_pain"], color="#1f77b4", linewidth=1.0,
            linestyle="--", label="M predicts")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time step")
    ax.set_ylabel("pain")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"Pain trajectory  (mean={float(np.mean(traj_C['pain'])):.3f})",
                 fontsize=9)

    fig.suptitle(f"flip-flop  --  outer step {step+1}/{max_step}   "
                 f"accuracy = {acc*100:.0f}%",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr-M", type=float, default=1e-2)
    p.add_argument("--lr-C", type=float, default=5e-3)
    p.add_argument("--M-warmup", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=60)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hold-final", type=int, default=15)
    p.add_argument("--out", type=str, default="flip_flop.gif")
    p.add_argument("--test-T", type=int, default=60)
    p.add_argument("--test-seed", type=int, default=12345)
    p.add_argument("--max-frames", type=int, default=80,
                   help="Hard cap on frames; reduces gif size.")
    args = p.parse_args()

    rng_test = np.random.default_rng(args.test_seed)
    test_events, test_desired = make_episode(args.test_T, rng_test)

    frames = []

    def cb(step: int, C, M, history, _rng_data):
        if step < 0:
            # initial snapshot before training
            frame = render_frame(C, M, history, -1, test_events,
                                 test_desired, args.steps)
            frames.append(frame)
            return
        frame = render_frame(C, M, history, step, test_events,
                             test_desired, args.steps)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  outer step {step+1:5d}  "
              f"({len(frames)} frames so far)")

    # Tune snapshot_every so we don't blow past max_frames
    eff_every = max(args.snapshot_every,
                    max(args.steps // max(args.max_frames - 2, 1), 1))

    print(f"Training {args.steps} steps, snapshot every {eff_every}...")
    C, M, history = train(
        seed=args.seed,
        n_steps=args.steps,
        T=args.T,
        n_hidden=args.hidden,
        lr_M=args.lr_M,
        lr_C=args.lr_C,
        M_warmup=args.M_warmup,
        snapshot_every=eff_every,
        snapshot_callback=cb,
        verbose=False,
    )

    final_acc = history["accuracy"][-1] * 100
    print(f"Final training-episode accuracy: {final_acc:.0f}%")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
