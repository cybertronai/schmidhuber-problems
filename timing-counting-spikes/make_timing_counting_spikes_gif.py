"""Animation for timing-counting-spikes.

Trains a peephole LSTM with snapshots and assembles a GIF showing how
the network learns to fire an output spike at the right time as
training progresses. Each frame shows a single held-out test sequence
plus the test-MSE / solve-rate curve so far.

Output: timing_counting_spikes.gif
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import timing_counting_spikes as tcs


def _build_frame(snapshot, history_iters, history_test_mse,
                 history_solve_rate, total_iters):
    """Render a single frame and return it as an HxWx3 uint8 array."""
    Xs = snapshot["Xs"]      # (T, 4, 1)
    preds = snapshot["preds"]  # (T, 4, 1)
    t1s = snapshot["t1s"]
    t2s = snapshot["t2s"]
    tts = snapshot["tts"]
    it = snapshot["iter"]
    solve = snapshot["solve_rate"]
    test_mse = snapshot["test_mse"]

    T = Xs.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.0),
                             gridspec_kw=dict(height_ratios=[2.4, 1.0]))
    ax_seq, ax_curve = axes
    t_axis = np.arange(T)
    # show first sample
    i = 0
    ax_seq.plot(t_axis, Xs[:, i, 0], color="black", linewidth=1.0,
                label="input spikes", alpha=0.55)
    ax_seq.plot(t_axis, preds[:, i, 0], color="C0", linewidth=2.0,
                label="LSTM output")
    ax_seq.axvline(t1s[i], color="gray", linestyle=":", alpha=0.7)
    ax_seq.axvline(t2s[i], color="gray", linestyle=":", alpha=0.7)
    ax_seq.axvline(tts[i], color="green", linestyle="-", linewidth=1.5,
                   alpha=0.6, label="target spike")
    ax_seq.set_ylim(-0.3, 1.3)
    ax_seq.set_xlim(-1, T)
    ax_seq.set_xlabel("time step")
    ax_seq.set_ylabel("amplitude")
    D = int(t2s[i] - t1s[i])
    ax_seq.set_title(f"Peephole LSTM @ iter {it} / {total_iters}  -  "
                     f"sample D = {D}, target step = {int(tts[i])}",
                     fontsize=11, loc="left")
    ax_seq.legend(loc="upper right", fontsize=8, ncol=3)
    ax_seq.grid(True, alpha=0.25)

    # curves
    ax_curve.plot(history_iters, history_test_mse, color="C0",
                  label="test MSE")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("iteration")
    ax_curve.set_ylabel("test MSE", color="C0")
    ax_curve.tick_params(axis="y", labelcolor="C0")
    ax_curve.grid(True, alpha=0.3)

    ax_solve = ax_curve.twinx()
    ax_solve.plot(history_iters, history_solve_rate, color="C2",
                  label="solve rate")
    ax_solve.set_ylabel("solve rate", color="C2")
    ax_solve.set_ylim(-0.02, 1.02)
    ax_solve.tick_params(axis="y", labelcolor="C2")
    ax_curve.set_title(f"test MSE = {test_mse:.4f}  |  "
                       f"solve rate = {solve:.3f}", fontsize=10, loc="left")

    fig.tight_layout()
    fig.canvas.draw()
    # extract RGB from RGBA buffer (matplotlib >= 3.8)
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[..., :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--T", type=int, default=150)
    parser.add_argument("--D-min", type=int, default=30)
    parser.add_argument("--D-max", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--out", type=str, default="timing_counting_spikes.gif")
    args = parser.parse_args()

    print("[gif] training peephole LSTM with snapshots...")
    params, history, snapshots = tcs.train(
        use_peep=True, T=args.T, D_min=args.D_min, D_max=args.D_max,
        hidden=args.hidden, seed=args.seed, n_iters=args.iters,
        batch_size=args.batch, lr=args.lr,
        eval_every=args.snapshot_every, verbose=False,
        save_snapshots=True,
    )
    print(f"[gif] {len(snapshots)} snapshots collected; "
          f"final test MSE = {history.test_mse[-1]:.5f}, "
          f"solve = {history.solve_rate[-1]:.3f}")

    print("[gif] rendering frames...")
    frames: List[np.ndarray] = []
    cum_iters: List[int] = []
    cum_mse: List[float] = []
    cum_solve: List[float] = []
    for s in snapshots:
        cum_iters.append(s["iter"])
        cum_mse.append(s["test_mse"])
        cum_solve.append(s["solve_rate"])
        frames.append(_build_frame(s, list(cum_iters), list(cum_mse),
                                    list(cum_solve), args.iters))

    # hold last frame
    frames += [frames[-1]] * 4

    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        import imageio  # type: ignore

    print(f"[gif] writing {args.out}  ({len(frames)} frames @ {args.fps} fps)")
    imageio.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    print(f"[gif] done.  size = {os.path.getsize(args.out) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
