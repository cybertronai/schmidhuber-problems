"""Animation of training dynamics for multiplication-problem.

Trains an LSTM and at fixed checkpoints writes a frame showing
  (a) a fixed test sequence with markers,
  (b) the LSTM cell state as it sweeps through that sequence,
  (c) predicted vs ground-truth product.
The frames are stitched into multiplication_problem.gif (matplotlib's
PillowWriter — no external GIF deps).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from multiplication_problem import (
    LSTM,
    Adam,
    TrainConfig,
    make_batch,
    make_sequence,
)


def cell_trajectory(model: LSTM, X):
    """Forward pass that records the cell-state trajectory for one sequence.

    X has shape (T, 2). Returns (c_traj, h_traj, y_pred) where c_traj and
    h_traj have shape (T, H) and y_pred is a scalar in [0, 1].
    """
    H = model.H
    T = X.shape[0]
    h = np.zeros((1, H), dtype=np.float32)
    c = np.zeros((1, H), dtype=np.float32)
    c_traj = np.zeros((T, H))
    h_traj = np.zeros((T, H))
    Xb = X[None]
    for t in range(T):
        x_t = Xb[:, t, :]
        pre = x_t @ model.Wx + h @ model.Wh + model.b
        i_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, 0:H], -50, 50)))
        f_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, H:2 * H], -50, 50)))
        o_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, 2 * H:3 * H], -50, 50)))
        g_g = np.tanh(pre[:, 3 * H:4 * H])
        c = f_g * c + i_g * g_g
        h = o_g * np.tanh(c)
        c_traj[t] = c[0]
        h_traj[t] = h[0]
    y_pre = h @ model.Wy + model.by
    y_pred = 1.0 / (1.0 + np.exp(-np.clip(y_pre, -50, 50)))
    return c_traj, h_traj, float(y_pred[0, 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--n-frames", type=int, default=40)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--out", type=str, default="multiplication_problem.gif")
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 7)

    cfg = TrainConfig(seed=args.seed, max_iters=args.max_iters)
    model = LSTM(D=2, H=cfg.hidden)
    model.reset_with_seed(args.seed)
    opt = Adam(model.params(), lr=cfg.lr)

    # Fixed test sequence used for the animation panel.
    test_X, test_y = make_sequence(args.T, eval_rng)

    # Decide checkpoint iterations (log-ish spacing so early dynamics show).
    ckpt_its = np.unique(
        np.round(np.geomspace(1, args.max_iters, args.n_frames)).astype(int)
    )

    frames = []
    losses = []

    next_ckpt = 0
    for it in range(1, args.max_iters + 1):
        X, y_true, _ = make_batch(cfg.batch_size, cfg.T_min, cfg.T_max, rng)
        y_pred, cache = model.forward(X)
        loss = float(((y_pred - y_true) ** 2).mean())
        losses.append(loss)
        grads = model.backward(y_true, cache)
        opt.step(model.params(), grads, clip=1.0)

        if next_ckpt < len(ckpt_its) and it == int(ckpt_its[next_ckpt]):
            c_traj, h_traj, y_test_pred = cell_trajectory(model, test_X)
            frames.append({
                "iter": it,
                "loss": float(np.mean(losses[-50:])),
                "c_traj": c_traj.copy(),
                "h_traj": h_traj.copy(),
                "y_pred": y_test_pred,
                "loss_history": list(losses),
            })
            next_ckpt += 1

    print(f"Captured {len(frames)} frames over {args.max_iters} iters.")

    # ------------------------------------------------------------------
    # Build the animation.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5),
                             gridspec_kw={"height_ratios": [1, 1.3]})
    ax_seq = axes[0, 0]
    ax_pred = axes[0, 1]
    ax_cell = axes[1, 0]
    ax_loss = axes[1, 1]

    plus_idx = np.where(test_X[:, 1] > 0.5)[0]
    minus_idx = np.where(test_X[:, 1] < -0.5)[0]
    ts = np.arange(args.T)

    # Static panels.
    ax_seq.plot(ts, test_X[:, 0], color="C0", lw=1.0)
    ax_seq.scatter(plus_idx, test_X[plus_idx, 0], color="C3", s=70, marker="^",
                   zorder=5, label=f"x1, x2 → {test_y:.3f}")
    ax_seq.scatter(minus_idx, np.zeros_like(minus_idx, dtype=float), color="k",
                   s=30, marker="v", zorder=5)
    ax_seq.set_title("test sequence (markers in red)", fontsize=10)
    ax_seq.set_xlim(-0.5, args.T - 0.5)
    ax_seq.set_ylim(-0.1, 1.15)
    ax_seq.legend(fontsize=8, loc="upper right")

    cmax = max(np.abs(f["c_traj"]).max() for f in frames)
    im = ax_cell.imshow(frames[0]["c_traj"].T, aspect="auto", cmap="RdBu_r",
                        vmin=-cmax, vmax=cmax,
                        extent=(-0.5, args.T - 0.5, model.H - 0.5, -0.5))
    ax_cell.set_title("cell state c[t] across the sequence", fontsize=10)
    ax_cell.set_xlabel("timestep")
    ax_cell.set_ylabel("LSTM cell")

    bar_pred = ax_pred.bar(["target", "prediction"], [test_y, frames[0]["y_pred"]],
                           color=["k", "C3"])
    ax_pred.set_ylim(0, 1)
    ax_pred.set_title("product prediction vs target", fontsize=10)

    line_loss, = ax_loss.plot([], [], color="C0", lw=1.5)
    ax_loss.set_xlim(0, args.max_iters)
    ax_loss.set_yscale("log")
    chance = 1.0 / 9.0 - 1.0 / 16.0
    ax_loss.axhline(chance, color="k", lw=0.5, ls="--",
                    label=f"chance MSE ≈ {chance:.4f}")
    ax_loss.set_ylim(1e-3, 0.2)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_title("training loss", fontsize=10)
    ax_loss.legend(fontsize=8, loc="upper right")

    title = fig.suptitle("multiplication-problem — iter 0", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(idx):
        f = frames[idx]
        im.set_data(f["c_traj"].T)
        for rect, val in zip(bar_pred, [test_y, f["y_pred"]]):
            rect.set_height(val)
        line_loss.set_data(np.arange(len(f["loss_history"])), f["loss_history"])
        title.set_text(
            f"multiplication-problem — iter {f['iter']}   "
            f"train MSE {f['loss']:.4f}   pred {f['y_pred']:.3f}  target {test_y:.3f}"
        )
        return im, *bar_pred, line_loss, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps,
                         blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=90)
    plt.close(fig)
    sz_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({sz_kb:.0f} KB)")


if __name__ == "__main__":
    main()
