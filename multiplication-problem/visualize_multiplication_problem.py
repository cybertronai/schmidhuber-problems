"""Static training-curve and behavior visualizations for multiplication-problem.

Re-runs training (deterministic with --seed) and writes PNGs into viz/.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiplication_problem import (
    LSTM,
    TrainConfig,
    make_batch,
    make_sequence,
    train,
)


def smooth(xs, k=50):
    if len(xs) < k:
        return np.array(xs)
    kernel = np.ones(k) / k
    return np.convolve(xs, kernel, mode="valid")


def plot_training_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    losses = history["train_losses"]
    ax.plot(losses, color="#bbb", lw=0.5, label="batch MSE")
    sm = smooth(losses, k=50)
    ax.plot(np.arange(len(sm)) + 50, sm, color="C0", lw=2, label="MSE (smoothed, k=50)")
    if history["test_curve"]:
        its = [t for t, _ in history["test_curve"]]
        ms = [m for _, m in history["test_curve"]]
        ax.plot(its, ms, "o-", color="C3", label="test MSE @ T=30")
    chance = 1.0 / 9.0 - 1.0 / 16.0  # Var(XY) for X,Y ~ U[0,1] independent
    ax.axhline(chance, color="k", lw=0.5, ls="--", alpha=0.5,
               label=f"chance MSE ≈ {chance:.4f}")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("MSE")
    ax.set_title("multiplication-problem — training curve")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sample_sequences(model, n_samples, T, rng, out_path):
    """Show the input sequences and predicted vs target product."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 1.6 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    Xs = []
    targets = []
    for _ in range(n_samples):
        X, t = make_sequence(T, rng)
        Xs.append(X)
        targets.append(t)
    X_batch = np.stack(Xs)
    y_pred, _ = model.forward(X_batch)
    for i, ax in enumerate(axes):
        X = Xs[i]
        ts = np.arange(T)
        ax.plot(ts, X[:, 0], color="C0", lw=1.0, label="x_real")
        # Markers: red triangles where mark==1, gray dashed at -1.
        plus_idx = np.where(X[:, 1] > 0.5)[0]
        minus_idx = np.where(X[:, 1] < -0.5)[0]
        ax.scatter(plus_idx, X[plus_idx, 0], color="C3", s=80, marker="^",
                   zorder=5, label="marker=+1")
        ax.scatter(minus_idx, np.zeros_like(minus_idx, dtype=float), color="k",
                   s=40, marker="v", zorder=5, label="marker=-1")
        ax.set_ylabel(f"seq {i}")
        ax.set_ylim(-0.1, 1.15)
        title = f"target = {targets[i]:.3f}   pred = {float(y_pred[i]):.3f}"
        ax.set_title(title, fontsize=9, loc="right")
        if i == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=3)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("multiplication-problem — sample sequences (markers in red, sentinels in black)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cell_state_dynamics(model, T, rng, out_path):
    """Show cell-state and hidden-state of the LSTM across one example sequence.

    Visually demonstrates that the cell line carries information from the
    first marker forward to the second one, without decay.
    """
    X, target = make_sequence(T, rng)
    X_b = X[None]
    # Replicate the forward pass but record cell/hidden trajectories.
    B, _, D = X_b.shape
    H = model.H
    h = np.zeros((B, H), dtype=np.float32)
    c = np.zeros((B, H), dtype=np.float32)
    h_traj = np.zeros((T, H))
    c_traj = np.zeros((T, H))
    f_traj = np.zeros((T, H))
    i_traj = np.zeros((T, H))
    o_traj = np.zeros((T, H))
    for t in range(T):
        x_t = X_b[:, t, :]
        pre = x_t @ model.Wx + h @ model.Wh + model.b
        i_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, 0:H], -50, 50)))
        f_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, H:2 * H], -50, 50)))
        o_g = 1.0 / (1.0 + np.exp(-np.clip(pre[:, 2 * H:3 * H], -50, 50)))
        g_g = np.tanh(pre[:, 3 * H:4 * H])
        c = f_g * c + i_g * g_g
        h = o_g * np.tanh(c)
        c_traj[t] = c[0]
        h_traj[t] = h[0]
        i_traj[t] = i_g[0]
        f_traj[t] = f_g[0]
        o_traj[t] = o_g[0]

    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    ts = np.arange(T)

    ax = axes[0]
    ax.plot(ts, X[:, 0], color="C0", lw=1.0, label="x_real")
    plus_idx = np.where(X[:, 1] > 0.5)[0]
    minus_idx = np.where(X[:, 1] < -0.5)[0]
    ax.scatter(plus_idx, X[plus_idx, 0], color="C3", s=80, marker="^", label="marker=+1")
    ax.scatter(minus_idx, np.zeros_like(minus_idx, dtype=float), color="k", s=40,
               marker="v", label="sentinel=-1")
    ax.set_ylabel("input")
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_title(f"target = {target:.3f}")

    ax = axes[1]
    im = ax.imshow(c_traj.T, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(c_traj).max(), vmax=np.abs(c_traj).max())
    ax.set_ylabel("cell state c (per cell)")
    plt.colorbar(im, ax=ax, fraction=0.025)

    ax = axes[2]
    im = ax.imshow(h_traj.T, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(h_traj).max(), vmax=np.abs(h_traj).max())
    ax.set_ylabel("hidden h (per cell)")
    plt.colorbar(im, ax=ax, fraction=0.025)

    ax = axes[3]
    ax.plot(ts, i_traj.mean(axis=1), label="mean input gate i")
    ax.plot(ts, f_traj.mean(axis=1), label="mean forget gate f")
    ax.plot(ts, o_traj.mean(axis=1), label="mean output gate o")
    ax.set_ylabel("gate (mean over cells)")
    ax.set_xlabel("timestep")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle("multiplication-problem — internal LSTM dynamics on one sequence",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_vs_target(model, n, T, rng, out_path):
    """Scatter of predicted vs ground-truth product on a held-out test batch."""
    X, y_true, _ = make_batch(n, T, T, rng)
    y_pred, _ = model.forward(X)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.6, color="C0")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    ax.set_xlabel("ground truth product")
    ax.set_ylabel("LSTM prediction")
    ax.set_title(f"pred vs target on {n} test sequences (T={T})  —  RMSE = {rmse:.4f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=6000)
    p.add_argument("--out", type=str, default="viz")
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed, max_iters=args.max_iters)
    print(f"training (seed={cfg.seed}, max_iters={cfg.max_iters}) ...")
    model, history = train(cfg, save_dir=None, verbose=False)
    print(f"  done, final test MSE = {history['final_test_mse']:.4f} "
          f"in {history['elapsed_sec']:.1f} s")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 9999)

    plot_training_curve(history, os.path.join(args.out, "training_curve.png"))
    plot_sample_sequences(model, 5, cfg.eval_T, rng,
                          os.path.join(args.out, "sample_sequences.png"))
    plot_cell_state_dynamics(model, cfg.eval_T, rng,
                             os.path.join(args.out, "cell_state.png"))
    plot_pred_vs_target(model, 256, cfg.eval_T, rng,
                        os.path.join(args.out, "pred_vs_target.png"))
    print(f"wrote PNGs to {args.out}/")


if __name__ == "__main__":
    main()
