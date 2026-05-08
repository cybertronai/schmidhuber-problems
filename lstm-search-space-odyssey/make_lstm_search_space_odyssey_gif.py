"""
Render the headline animation:
  lstm_search_space_odyssey.gif

Each frame is a snapshot of the 8-variant ablation matrix at one
training-iteration checkpoint. The bar chart shows test MSE per
variant (log scale) with a small panel below tracking the leader's
test MSE over time. Two side-by-side views: test-MSE bars (left) and
solve-rate bars (right).

Trains all 8 variants × N seeds and emits one frame per
`--snapshot-every` iters. With the defaults the full pipeline runs in
roughly 2-3 minutes on a single CPU core.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lstm_search_space_odyssey import (
    VARIANT_NAMES, VARIANT_DESCRIPTIONS, VariantFlags, init_lstm,
    lstm_forward, lstm_backward, make_adding_batch, evaluate, Adam,
    env_info,
)
from visualize_lstm_search_space_odyssey import VARIANT_COLOURS


def train_variant_with_snapshots(name: str, T: int, hidden: int, seed: int,
                                 n_iters: int, batch_size: int, lr: float,
                                 snapshot_every: int):
    """Train one variant and emit (iter, test_mse, solve_rate) snapshots."""
    variant = VariantFlags.from_name(name)
    train_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed + 1_000_003)
    init_rng = np.random.RandomState(seed + 7)
    params = init_lstm(input_dim=2, H=hidden, variant=variant, rng=init_rng)
    opt = Adam(params, lr=lr, clip=1.0)

    snaps = []
    # Iter 0 (untrained) snapshot
    test_mse, solve = evaluate(params, variant, test_rng, T,
                               n_test=256, batch_size=128)
    snaps.append((0, test_mse, solve))
    for it in range(1, n_iters + 1):
        X, y = make_adding_batch(train_rng, T, batch_size)
        pred, cache = lstm_forward(params, X, variant)
        err = pred - y
        dpred = err / batch_size
        grads = lstm_backward(params, cache, dpred, variant)
        opt.step(params, grads)
        if it % snapshot_every == 0 or it == n_iters:
            test_mse, solve = evaluate(params, variant, test_rng, T,
                                       n_test=256, batch_size=128)
            snaps.append((it, test_mse, solve))
    return snaps


def render_frame(snapshots_by_variant: dict, frame_idx: int, T: int,
                 hidden: int, batch_size: int, lr: float,
                 history_so_far: list, total_iters: int):
    """Render one frame of the GIF and return it as a numpy array (H, W, 3)."""
    # All variants share the same snapshot iteration list (same schedule)
    iter_at_frame = snapshots_by_variant[VARIANT_NAMES[0]][frame_idx][0]

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = GridSpec(2, 2, height_ratios=[3.0, 1.2], hspace=0.55, wspace=0.18,
                  left=0.06, right=0.97, top=0.88, bottom=0.10)

    # Test-MSE bars
    ax_mse = fig.add_subplot(gs[0, 0])
    mses = [snapshots_by_variant[n][frame_idx][1] for n in VARIANT_NAMES]
    colours = [VARIANT_COLOURS[n] for n in VARIANT_NAMES]
    ax_mse.bar(VARIANT_NAMES, mses, color=colours)
    ax_mse.set_yscale("log")
    ax_mse.set_ylim(1e-4, 5.0)
    ax_mse.axhline(0.04, color="black", linestyle="--", linewidth=0.8)
    ax_mse.set_ylabel("test MSE (log)")
    ax_mse.set_title("test MSE per variant")
    for xi, m in enumerate(mses):
        ax_mse.text(xi, max(m, 1.5e-4), f"{m:.3f}", ha="center",
                    va="bottom", fontsize=7)

    # Solve-rate bars
    ax_sr = fig.add_subplot(gs[0, 1])
    srs = [snapshots_by_variant[n][frame_idx][2] for n in VARIANT_NAMES]
    ax_sr.bar(VARIANT_NAMES, srs, color=colours)
    ax_sr.set_ylim(0, 1.05)
    ax_sr.set_ylabel("solve rate (|err| < 0.04)")
    ax_sr.set_title("solve rate per variant")
    for xi, s in enumerate(srs):
        ax_sr.text(xi, s, f"{s:.2f}", ha="center", va="bottom", fontsize=7)

    # Test-MSE trajectory (all variants overlaid) at the bottom
    ax_traj = fig.add_subplot(gs[1, :])
    iters_so_far = [h[0] for h in history_so_far]
    for name in VARIANT_NAMES:
        traj = [snapshots_by_variant[name][k][1]
                for k in range(frame_idx + 1)]
        its = [snapshots_by_variant[name][k][0]
               for k in range(frame_idx + 1)]
        ax_traj.plot(its, traj, color=VARIANT_COLOURS[name],
                     label=name, linewidth=1.4)
    ax_traj.set_yscale("log")
    ax_traj.set_xlim(0, total_iters)
    ax_traj.set_ylim(1e-4, 5.0)
    ax_traj.axhline(0.04, color="black", linestyle="--", linewidth=0.7)
    ax_traj.set_xlabel("training iteration")
    ax_traj.set_ylabel("test MSE")
    ax_traj.legend(loc="upper right", ncol=4, fontsize=7,
                   frameon=False)

    fig.suptitle(
        f"LSTM Search Space Odyssey — adding-problem T={T}, "
        f"hidden={hidden}, batch={batch_size}, lr={lr}\n"
        f"iter {iter_at_frame:5d} / {total_iters}",
        fontsize=11)
    # Render to RGB buffer
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb, iter_at_frame


def save_gif(frames: list[np.ndarray], outpath: Path, fps: float):
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio  # type: ignore
    duration = 1.0 / fps
    imageio.mimsave(outpath, frames, duration=duration, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="single seed for the GIF training (snappier)")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--out", type=str,
                    default="lstm_search_space_odyssey.gif")
    args = ap.parse_args()

    print(f"GIF training: T={args.T} hidden={args.hidden} "
          f"iters={args.iters} seed={args.seed}")
    print(f"  env: {env_info()}")

    snapshots_by_variant = {}
    t0 = time.time()
    for name in VARIANT_NAMES:
        ts = time.time()
        snaps = train_variant_with_snapshots(
            name, T=args.T, hidden=args.hidden, seed=args.seed,
            n_iters=args.iters, batch_size=args.batch, lr=args.lr,
            snapshot_every=args.snapshot_every,
        )
        snapshots_by_variant[name] = snaps
        print(f"  [{name:>4s}] trained, {len(snaps)} snapshots "
              f"({time.time() - ts:.1f}s)")
    t_train = time.time() - t0

    # Sanity: every variant must have the same number of snapshots
    n_frames = len(snapshots_by_variant[VARIANT_NAMES[0]])
    for name in VARIANT_NAMES:
        assert len(snapshots_by_variant[name]) == n_frames

    print(f"  training done in {t_train:.1f}s, rendering "
          f"{n_frames} frames...")
    frames = []
    history_so_far = []
    t1 = time.time()
    for idx in range(n_frames):
        rgb, it = render_frame(
            snapshots_by_variant, idx, T=args.T, hidden=args.hidden,
            batch_size=args.batch, lr=args.lr,
            history_so_far=history_so_far, total_iters=args.iters,
        )
        history_so_far.append((it,))
        frames.append(rgb)
    # Hold final frame longer
    for _ in range(int(args.fps * 1.5)):
        frames.append(frames[-1])
    print(f"  rendered in {time.time() - t1:.1f}s, "
          f"writing {args.out}...")
    save_gif(frames, Path(args.out), fps=args.fps)
    print(f"  wrote {args.out} "
          f"({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
