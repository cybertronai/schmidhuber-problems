"""Static plots for timing-counting-spikes.

Trains a peephole LSTM and a no-peephole baseline on the MSD task, then
saves figures comparing the two:

    viz/training_curves.png  -- test MSE + solve rate vs iteration
    viz/sample_predictions.png -- 4 held-out test sequences with both
                                  models' outputs overlaid
    viz/cell_state.png       -- peephole-LSTM cell-state heatmap on a
                                long-D test sequence; highlights how
                                the cell builds up an interval timer
                                between spikes
    viz/peephole_weights.png -- learned p_i, p_f, p_o vectors per cell
    viz/weights.png          -- input/recurrent gate weight matrices
                                (peephole LSTM)
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np

import timing_counting_spikes as tcs


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def plot_training_curves(hist_peep, hist_nopeep, outpath: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax_mse, ax_solve = axes

    ax_mse.plot(hist_peep.iters, hist_peep.test_mse, label="peephole LSTM",
                color="C0", linewidth=2)
    ax_mse.plot(hist_nopeep.iters, hist_nopeep.test_mse,
                label="vanilla LSTM (no peep)", color="C3", linewidth=2)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("iteration")
    ax_mse.set_ylabel("test MSE (log)")
    ax_mse.set_title("MSD: test MSE per timestep")
    ax_mse.grid(True, which="both", alpha=0.3)
    ax_mse.legend(loc="best", fontsize=9)

    ax_solve.plot(hist_peep.iters, hist_peep.solve_rate,
                  label="peephole LSTM", color="C0", linewidth=2)
    ax_solve.plot(hist_nopeep.iters, hist_nopeep.solve_rate,
                  label="vanilla LSTM (no peep)", color="C3", linewidth=2)
    ax_solve.set_xlabel("iteration")
    ax_solve.set_ylabel("solve rate (exact spike timing)")
    ax_solve.set_title("MSD: fraction of test sequences solved (tol = 0)")
    ax_solve.set_ylim(-0.02, 1.02)
    ax_solve.grid(True, alpha=0.3)
    ax_solve.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_sample_predictions(params_peep, params_nopeep, T, D_min, D_max,
                            seed, outpath: str, n_samples: int = 4) -> None:
    rng = np.random.RandomState(seed + 555)
    X, y, t1, t2, t_target = tcs.make_msd_batch(rng, T, D_min, D_max,
                                                n_samples)
    pred_p, _ = tcs.lstm_forward(params_peep, X)
    pred_n, _ = tcs.lstm_forward(params_nopeep, X)

    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2.0 * n_samples),
                             sharex=True)
    if n_samples == 1:
        axes = [axes]
    t_axis = np.arange(T)
    for i, ax in enumerate(axes):
        ax.plot(t_axis, X[:, i, 0], color="black", linewidth=1.0,
                label="input spikes", alpha=0.6)
        ax.plot(t_axis, pred_p[:, i, 0], color="C0", linewidth=2,
                label="peephole LSTM output")
        ax.plot(t_axis, pred_n[:, i, 0], color="C3", linewidth=1.5,
                label="vanilla LSTM output", alpha=0.85, linestyle="--")
        ax.axvline(t1[i], color="gray", linestyle=":", alpha=0.6)
        ax.axvline(t2[i], color="gray", linestyle=":", alpha=0.6)
        ax.axvline(t_target[i], color="green", linestyle="-", alpha=0.5,
                   linewidth=1.5, label="target spike")
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel(f"D={int(t2[i] - t1[i])}", fontsize=10)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_title(f"sample {i + 1}: t1={int(t1[i])}, t2={int(t2[i])}, "
                     f"target={int(t_target[i])}", fontsize=9, loc="left")
    axes[-1].set_xlabel("time step")
    fig.suptitle("MSD held-out predictions: peephole vs vanilla LSTM",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_cell_state(params, T, D_min, D_max, seed, outpath: str) -> None:
    """Cell-state heatmap on a single test sequence with the largest D."""
    rng = np.random.RandomState(seed + 777)
    # generate a few candidate sequences; pick the one with largest D
    X_b, y_b, t1_b, t2_b, tt_b = tcs.make_msd_batch(rng, T, D_min, D_max, 16)
    D_b = t2_b - t1_b
    idx = int(np.argmax(D_b))
    X = X_b[:, idx:idx + 1]
    pred, cache = tcs.lstm_forward(params, X)
    c = cache["c"][1:, 0]  # (T, H)
    pred = pred[:, 0, 0]
    t1, t2, tt = int(t1_b[idx]), int(t2_b[idx]), int(tt_b[idx])

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5),
                             gridspec_kw=dict(height_ratios=[1, 2.4, 1]),
                             sharex=True)
    t_axis = np.arange(T)

    axes[0].plot(t_axis, X[:, 0, 0], color="black")
    axes[0].axvline(t1, color="gray", linestyle=":", linewidth=1)
    axes[0].axvline(t2, color="gray", linestyle=":", linewidth=1)
    axes[0].axvline(tt, color="green", linestyle="-", alpha=0.5, linewidth=1.5)
    axes[0].set_ylabel("input")
    axes[0].set_title(f"Peephole LSTM cell state on a long-D sample "
                      f"(D = {t2 - t1}, target = {tt})", fontsize=11,
                      loc="left")
    axes[0].set_ylim(-0.2, 1.2)

    H = c.shape[1]
    im = axes[1].imshow(c.T, aspect="auto", origin="lower",
                        cmap="RdBu_r", interpolation="nearest",
                        extent=[0, T, -0.5, H - 0.5],
                        vmin=-np.max(np.abs(c)), vmax=np.max(np.abs(c)))
    axes[1].set_ylabel("cell index")
    axes[1].axvline(t1, color="black", linestyle=":", linewidth=1)
    axes[1].axvline(t2, color="black", linestyle=":", linewidth=1)
    axes[1].axvline(tt, color="green", linestyle="-", alpha=0.6, linewidth=1.5)
    fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.01).set_label("c_t")

    axes[2].plot(t_axis, pred, color="C0")
    axes[2].axvline(t1, color="gray", linestyle=":", linewidth=1)
    axes[2].axvline(t2, color="gray", linestyle=":", linewidth=1)
    axes[2].axvline(tt, color="green", linestyle="-", alpha=0.5, linewidth=1.5)
    axes[2].set_ylabel("output")
    axes[2].set_xlabel("time step")
    axes[2].set_ylim(-0.1, max(1.1, float(np.max(pred)) + 0.1))

    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_peephole_weights(params, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    H = params.p_i.size
    x = np.arange(H)
    width = 0.27
    ax.bar(x - width, params.p_i, width=width, label="p_i (c_{t-1} -> i)",
           color="C0")
    ax.bar(x,         params.p_f, width=width, label="p_f (c_{t-1} -> f)",
           color="C2")
    ax.bar(x + width, params.p_o, width=width, label="p_o (c_t -> o)",
           color="C3")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("cell index")
    ax.set_ylabel("peephole weight")
    ax.set_title("Learned peephole weights (each cell has its own)",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_weight_matrices(params, outpath: str) -> None:
    H = params.Wh.shape[0]
    Wx = params.Wx.reshape(-1, 4, H)  # (D_in, 4, H)
    Wh = params.Wh.reshape(H, 4, H)   # (H, 4, H)
    gate_names = ["i", "f", "g", "o"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.0))
    for j, name in enumerate(gate_names):
        ax = axes[0, j]
        ax.imshow(Wx[:, j, :], aspect="auto", cmap="RdBu_r",
                  vmin=-np.max(np.abs(Wx)), vmax=np.max(np.abs(Wx)))
        ax.set_title(f"Wx -> {name}", fontsize=10)
        ax.set_xlabel("cell")
        if j == 0:
            ax.set_ylabel("input dim")
        ax = axes[1, j]
        ax.imshow(Wh[:, j, :], aspect="auto", cmap="RdBu_r",
                  vmin=-np.max(np.abs(Wh)), vmax=np.max(np.abs(Wh)))
        ax.set_title(f"Wh -> {name}", fontsize=10)
        ax.set_xlabel("cell")
        if j == 0:
            ax.set_ylabel("from cell")
    fig.suptitle("Peephole LSTM weights after training", fontsize=11,
                 y=1.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


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
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    print("[viz] training peephole LSTM...")
    params_peep, hist_peep, _ = tcs.train(
        use_peep=True, T=args.T, D_min=args.D_min, D_max=args.D_max,
        hidden=args.hidden, seed=args.seed, n_iters=args.iters,
        batch_size=args.batch, lr=args.lr,
        eval_every=args.eval_every, verbose=False,
        save_snapshots=False,
    )
    print(f"  final test MSE = {hist_peep.test_mse[-1]:.5f}, "
          f"solve = {hist_peep.solve_rate[-1]:.3f}")

    print("[viz] training vanilla (no-peep) LSTM...")
    params_nopeep, hist_nopeep, _ = tcs.train(
        use_peep=False, T=args.T, D_min=args.D_min, D_max=args.D_max,
        hidden=args.hidden, seed=args.seed, n_iters=args.iters,
        batch_size=args.batch, lr=args.lr,
        eval_every=args.eval_every, verbose=False,
        save_snapshots=False,
    )
    print(f"  final test MSE = {hist_nopeep.test_mse[-1]:.5f}, "
          f"solve = {hist_nopeep.solve_rate[-1]:.3f}")

    print("[viz] plotting...")
    plot_training_curves(hist_peep, hist_nopeep,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_sample_predictions(params_peep, params_nopeep,
                            args.T, args.D_min, args.D_max, args.seed,
                            os.path.join(args.outdir, "sample_predictions.png"))
    plot_cell_state(params_peep, args.T, args.D_min, args.D_max, args.seed,
                    os.path.join(args.outdir, "cell_state.png"))
    plot_peephole_weights(params_peep,
                          os.path.join(args.outdir, "peephole_weights.png"))
    plot_weight_matrices(params_peep,
                         os.path.join(args.outdir, "weights.png"))
    print(f"[viz] wrote {args.outdir}/training_curves.png")
    print(f"[viz] wrote {args.outdir}/sample_predictions.png")
    print(f"[viz] wrote {args.outdir}/cell_state.png")
    print(f"[viz] wrote {args.outdir}/peephole_weights.png")
    print(f"[viz] wrote {args.outdir}/weights.png")


if __name__ == "__main__":
    main()
