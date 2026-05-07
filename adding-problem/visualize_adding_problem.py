"""Static visualizations for the adding-problem experiment.

Trains both the LSTM and the vanilla-RNN baseline and writes:
    viz/training_curves.png    — test MSE + solve rate, LSTM vs RNN
    viz/predictions.png        — scatter pred vs target on a held-out batch
    viz/sample_sequences.png   — 4 example sequences with markers + predictions
    viz/cell_state.png         — LSTM cell state c_t over time on one sequence
    viz/gate_activity.png      — input/forget/output gate activations over time
    viz/weights.png            — final LSTM gate weight matrices

Usage:
    python3 visualize_adding_problem.py --seed 0 --T 100 --hidden 8 --outdir viz
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adding_problem import (
    lstm_forward,
    make_adding_batch,
    rnn_forward,
    train,
)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_training_curves(lstm_h, rnn_h, T, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(lstm_h.sequences_seen, lstm_h.test_mse, "C0-",
                 lw=1.6, label="LSTM")
    axes[0].plot(rnn_h.sequences_seen, rnn_h.test_mse, "C3-",
                 lw=1.6, label="vanilla RNN")
    axes[0].axhline(0.04, color="k", ls="--", lw=0.8,
                    label="paper threshold (MSE = 0.04)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sequences seen")
    axes[0].set_ylabel("test MSE  (log scale)")
    axes[0].set_title(f"adding problem  T={T}")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(lstm_h.sequences_seen, lstm_h.solve_rate, "C0-",
                 lw=1.6, label="LSTM")
    axes[1].plot(rnn_h.sequences_seen, rnn_h.solve_rate, "C3-",
                 lw=1.6, label="vanilla RNN")
    axes[1].set_xlabel("sequences seen")
    axes[1].set_ylabel("fraction of test sequences with |err| < 0.04")
    axes[1].set_title("solve rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _plot_predictions(lstm_p, rnn_p, T, seed, outpath):
    rng = np.random.RandomState(seed + 5_000_000)
    X, y = make_adding_batch(rng, T, 256)
    lp, _ = lstm_forward(lstm_p, X)
    rp, _ = rnn_forward(rnn_p, X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, p, name, mse in [(axes[0], lp, "LSTM", float(((lp - y) ** 2).mean())),
                             (axes[1], rp, "vanilla RNN",
                              float(((rp - y) ** 2).mean()))]:
        ax.scatter(y, p, s=14, alpha=0.65,
                   color="C0" if "LSTM" in name else "C3")
        lo, hi = -2.05, 2.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("target")
        ax.set_title(f"{name}  (test MSE = {mse:.4f})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("prediction")
    fig.suptitle(f"predicted vs target on 256 held-out sequences  (T={T})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _plot_sample_sequences(lstm_p, T, seed, outpath, n_show=4):
    rng = np.random.RandomState(seed + 11_000_007)
    X, y = make_adding_batch(rng, T, n_show)
    pred, cache = lstm_forward(lstm_p, X)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 1.6 * n_show + 0.4),
                             sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        vals = X[:, i, 0]
        marks = X[:, i, 1]
        ax.bar(np.arange(T), vals, width=0.9, color="lightgray",
               edgecolor="lightgray", label="value")
        marker_idx = np.where(marks > 0.5)[0]
        for m in marker_idx:
            ax.bar([m], [vals[m]], width=0.9, color="C1",
                   edgecolor="C1")
        ax.set_xlim(-0.5, T - 0.5)
        ax.axhline(0, color="k", lw=0.5)
        # text annotation
        a, b = marker_idx[0], marker_idx[1]
        ax.set_title(
            f"seq {i}: marked values {vals[a]:+.2f} (t={a}) + "
            f"{vals[b]:+.2f} (t={b}) "
            f"= target {y[i]:+.3f}   |   prediction {pred[i]:+.3f}",
            fontsize=9, loc="left")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("value", fontsize=9)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Sample sequences (orange bars = marked values to be summed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _plot_cell_state(lstm_p, T, seed, outpath):
    rng = np.random.RandomState(seed + 23_456_789)
    X, y = make_adding_batch(rng, T, 1)
    pred, cache = lstm_forward(lstm_p, X)
    c = cache["c"][1:, 0, :]   # (T, H)
    h = cache["h"][1:, 0, :]   # (T, H)
    marks = X[:, 0, 1]
    vals = X[:, 0, 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5),
                             gridspec_kw={"height_ratios": [1, 2, 2]},
                             sharex=True)

    axes[0].bar(np.arange(T), vals, color="lightgray", width=0.9,
                edgecolor="lightgray")
    mi = np.where(marks > 0.5)[0]
    axes[0].bar(mi, vals[mi], color="C1", width=0.9, edgecolor="C1")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("input\nvalue", fontsize=9)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_title(
        f"target = {y[0]:+.3f}   prediction = {pred[0]:+.3f}",
        fontsize=10, loc="left")

    vmax = np.max(np.abs(c))
    im = axes[1].imshow(c.T, aspect="auto", origin="lower",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=(-0.5, T - 0.5, -0.5, c.shape[1] - 0.5))
    for m in mi:
        axes[1].axvline(m, color="k", ls=":", lw=0.8, alpha=0.7)
    axes[1].set_ylabel("cell state c_t  (unit)")
    axes[1].set_title("LSTM cell state (red=positive, blue=negative); "
                      "dotted lines = marker positions",
                      fontsize=10, loc="left")
    fig.colorbar(im, ax=axes[1], pad=0.01)

    vmax_h = np.max(np.abs(h))
    im2 = axes[2].imshow(h.T, aspect="auto", origin="lower",
                         cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h,
                         extent=(-0.5, T - 0.5, -0.5, h.shape[1] - 0.5))
    for m in mi:
        axes[2].axvline(m, color="k", ls=":", lw=0.8, alpha=0.7)
    axes[2].set_ylabel("hidden h_t  (unit)")
    axes[2].set_xlabel("time step")
    fig.colorbar(im2, ax=axes[2], pad=0.01)

    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _plot_gate_activity(lstm_p, T, seed, outpath):
    rng = np.random.RandomState(seed + 33_333_331)
    X, y = make_adding_batch(rng, T, 1)
    pred, cache = lstm_forward(lstm_p, X)
    i_g = cache["i"][:, 0, :]
    f_g = cache["f"][:, 0, :]
    o_g = cache["o"][:, 0, :]
    marks = X[:, 0, 1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.0), sharex=True)
    for ax, arr, name in [(axes[0], i_g, "input gate i_t"),
                          (axes[1], f_g, "forget gate f_t"),
                          (axes[2], o_g, "output gate o_t")]:
        im = ax.imshow(arr.T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.0, vmax=1.0,
                       extent=(-0.5, T - 0.5, -0.5, arr.shape[1] - 0.5))
        mi = np.where(marks > 0.5)[0]
        for m in mi:
            ax.axvline(m, color="w", ls=":", lw=1.0)
        ax.set_ylabel(f"{name}\nunit")
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("time step")
    fig.suptitle(
        "LSTM gate activations over time. "
        "Dotted white = marker positions; "
        "input gate should spike, forget gate should stay near 1.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _plot_weights(lstm_p, outpath):
    H = lstm_p.Wh.shape[0]
    Wx = lstm_p.Wx
    Wh = lstm_p.Wh
    gates = ["i (input)", "f (forget)", "g (cand)", "o (output)"]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    vmax_x = np.max(np.abs(Wx))
    vmax_h = np.max(np.abs(Wh))
    for j in range(4):
        ax = axes[0, j]
        block = Wx[:, j * H:(j + 1) * H]
        im = ax.imshow(block, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax_x, vmax=vmax_x)
        ax.set_title(f"Wx → {gates[j]}", fontsize=9)
        ax.set_xlabel("hidden unit")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["x[0] value", "x[1] marker"])
        if j == 3:
            fig.colorbar(im, ax=ax, pad=0.02)
        ax2 = axes[1, j]
        block = Wh[:, j * H:(j + 1) * H]
        im2 = ax2.imshow(block, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax_h, vmax=vmax_h)
        ax2.set_title(f"Wh → {gates[j]}", fontsize=9)
        ax2.set_xlabel("hidden unit")
        ax2.set_ylabel("from hidden")
        if j == 3:
            fig.colorbar(im2, ax=ax2, pad=0.02)
    fig.suptitle("Final LSTM weights "
                 "(top: input→gate, bottom: hidden→gate)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--rnn-iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-decay-every", type=int, default=1500)
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    print("Training LSTM...")
    lstm_p, lstm_h, _ = train(
        model="lstm", T=args.T, hidden=args.hidden, seed=args.seed,
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
        eval_every=max(50, args.iters // 32),
        lr_decay_every=args.lr_decay_every, lr_decay_factor=0.5,
        verbose=False,
    )
    print(f"  LSTM final test MSE = {lstm_h.test_mse[-1]:.4f}  "
          f"solve_rate = {lstm_h.solve_rate[-1]:.3f}")

    print("Training vanilla RNN baseline...")
    rnn_p, rnn_h, _ = train(
        model="rnn", T=args.T, hidden=args.hidden, seed=args.seed,
        n_iters=args.rnn_iters, batch_size=args.batch, lr=args.lr,
        eval_every=max(50, args.rnn_iters // 32),
        lr_decay_every=args.lr_decay_every, lr_decay_factor=0.5,
        verbose=False,
    )
    print(f"  RNN  final test MSE = {rnn_h.test_mse[-1]:.4f}  "
          f"solve_rate = {rnn_h.solve_rate[-1]:.3f}")

    out = args.outdir
    print(f"Writing plots to {out}/...")
    _plot_training_curves(lstm_h, rnn_h, args.T,
                          os.path.join(out, "training_curves.png"))
    _plot_predictions(lstm_p, rnn_p, args.T, args.seed,
                      os.path.join(out, "predictions.png"))
    _plot_sample_sequences(lstm_p, args.T, args.seed,
                           os.path.join(out, "sample_sequences.png"))
    _plot_cell_state(lstm_p, args.T, args.seed,
                     os.path.join(out, "cell_state.png"))
    _plot_gate_activity(lstm_p, args.T, args.seed,
                        os.path.join(out, "gate_activity.png"))
    _plot_weights(lstm_p, os.path.join(out, "weights.png"))
    print("Done.")


if __name__ == "__main__":
    main()
