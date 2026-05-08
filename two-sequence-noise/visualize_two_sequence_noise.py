"""Static visualizations for two-sequence-noise (variant 3c).

Produces in `viz/`:
    training_curves.png   -- final-step squared error and rolling accuracy
    weights.png           -- Hinton-style diagrams of W_iota, W_omega, W_c, W_out
    test_sequence.png     -- input + cell states + gates + output for two test
                              sequences (one per class).
    output_distribution.png -- histogram of y_out[T-1] split by class on a fresh
                              200-sequence test set.

Usage:
    python3 visualize_two_sequence_noise.py --seed 0 --steps 8000 --T 100 \
            --outdir viz
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# matplotlib already imported with Agg backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_sequence_noise import (  # noqa: E402
    LSTM1997,
    evaluate,
    forward,
    label_to_target,
    make_sequence,
    train,
)


# ----------------------------------------------------------------------
# Hinton diagram
# ----------------------------------------------------------------------

def hinton(ax, W: np.ndarray, max_w: float | None = None, title: str = ""):
    """Plot a single Hinton diagram on the given axis."""
    W = np.asarray(W)
    if max_w is None:
        max_w = float(np.max(np.abs(W))) or 1.0
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("0.4")
    ax.set_xlim(-0.5, W.shape[1] - 0.5)
    ax.set_ylim(-0.5, W.shape[0] - 0.5)
    ax.invert_yaxis()
    for (i, j), w in np.ndenumerate(W):
        color = "white" if w > 0 else "black"
        size = np.sqrt(abs(w) / max_w) * 0.5
        rect = plt.Rectangle(
            (j - size, i - size), 2 * size, 2 * size,
            facecolor=color, edgecolor=color,
        )
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(history["step"], history["loss"], color="C0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("clean final-step squared error (log)")
    axes[0].set_title("Final-step squared error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["step"], np.array(history["acc_train"]) * 100,
                 color="C2")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("rolling accuracy (%)")
    axes[1].set_ylim(40, 102)
    axes[1].axhline(95, color="grey", linestyle=":", linewidth=0.8)
    axes[1].set_title("Rolling training accuracy (per-1000-step window)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_weights(net: LSTM1997, path: str):
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    ax_iota = fig.add_subplot(gs[0, 0])
    hinton(ax_iota, net.W_iota,
           title=f"W_iota  (input gate)  {net.W_iota.shape}")
    ax_omega = fig.add_subplot(gs[0, 1])
    hinton(ax_omega, net.W_omega,
           title=f"W_omega (output gate) {net.W_omega.shape}")
    ax_out = fig.add_subplot(gs[0, 2])
    hinton(ax_out, net.W_out,
           title=f"W_out (output unit) {net.W_out.shape}")

    ax_c = fig.add_subplot(gs[1, :])
    hinton(ax_c, net.W_c,
           title=f"W_c (cell input weights) {net.W_c.shape}  "
                 f"-- columns: [x | y_c_recurrent | bias]")

    fig.suptitle(
        "LSTM-1997 weight matrices after training "
        f"(blocks={net.n_blocks}, cells/block={net.n_cells_per_block}, "
        f"weights={net.n_weights()})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_test_sequence(net: LSTM1997, path: str, seed: int = 12345):
    """Plot one class-0 and one class-1 test sequence side by side."""
    rng = np.random.default_rng(seed)
    # find one of each class
    series = {}
    while len(series) < 2:
        x_seq, label = make_sequence(rng, T=100, p1=10)
        if label in series:
            continue
        cache = forward(net, x_seq)
        series[label] = (x_seq, cache, label_to_target(label))

    fig, axes = plt.subplots(
        4, 2, figsize=(11, 8.8), sharex=True,
    )

    for col, label in enumerate(sorted(series.keys())):
        x_seq, cache, target = series[label]
        T = x_seq.shape[0]
        ts = np.arange(T)

        # row 0: input
        axes[0, col].plot(ts, x_seq[:, 0], color="0.4", linewidth=0.8)
        axes[0, col].axvspan(0, 10, color="C0", alpha=0.15,
                             label="info phase (steps 0..9)")
        axes[0, col].set_ylabel("input x")
        axes[0, col].set_title(f"class {label}  (target = {target:.1f})")
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # row 1: cell states
        for c in range(cache["s"].shape[1]):
            axes[1, col].plot(
                np.arange(1, T + 1),
                cache["s"][1:, c],
                label=f"cell {c}",
                linewidth=0.9,
            )
        axes[1, col].set_ylabel("cell state s_c(t)")
        axes[1, col].legend(loc="upper right", fontsize=7, ncol=2)
        axes[1, col].grid(True, alpha=0.3)

        # row 2: output gate activations (per block)
        for b in range(cache["omega"].shape[1]):
            bias = net.W_omega[b, -1]
            axes[2, col].plot(
                ts, cache["omega"][:, b],
                label=f"block {b} (bias={bias:.0f})",
                linewidth=1.0,
            )
        axes[2, col].set_ylabel("output gate omega_j(t)")
        axes[2, col].set_ylim(-0.05, 1.05)
        axes[2, col].legend(loc="upper right", fontsize=7)
        axes[2, col].grid(True, alpha=0.3)

        # row 3: y_out vs target
        axes[3, col].plot(ts, cache["y_out"], color="C1",
                          label="y_out(t)")
        axes[3, col].axhline(target, color="black", linestyle="--",
                             linewidth=0.8, label=f"target {target:.1f}")
        axes[3, col].axhline(0.5, color="grey", linestyle=":",
                             linewidth=0.7)
        axes[3, col].set_ylabel("y_out")
        axes[3, col].set_xlabel("time step t")
        axes[3, col].set_ylim(-0.05, 1.05)
        axes[3, col].legend(loc="lower right", fontsize=8)
        axes[3, col].grid(True, alpha=0.3)

    fig.suptitle(
        "Test sequences (post-training).  The information phase is the first "
        "10 steps; the rest is N(0,1) noise.\n"
        "Cell states latch on at info phase and are read out by the output "
        "gates only at the end of the sequence.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_output_distribution(net: LSTM1997, path: str, seed: int = 12345):
    res = evaluate(net, n_episodes=500, T=100, p1=10, seed=seed)
    y_outs = np.array(res["y_outs"])
    labels = np.array(res["labels"])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(y_outs[labels == 0], bins=bins, color="C0", alpha=0.6,
            label="class 0 (target 0.2)")
    ax.hist(y_outs[labels == 1], bins=bins, color="C3", alpha=0.6,
            label="class 1 (target 0.8)")
    ax.axvline(0.2, color="C0", linestyle="--", linewidth=0.9)
    ax.axvline(0.8, color="C3", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("y_out at final step")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Output distribution on 500 fresh noiseless test sequences  "
        f"(acc={res['acc'] * 100:.1f}%)"
    )
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"# visualize_two_sequence_noise  seed={args.seed}  "
          f"steps={args.steps}  T={args.T}")
    net, history = train(
        seed=args.seed,
        n_steps=args.steps,
        T=args.T,
        log_every=max(args.steps // 30, 1),
        verbose=False,
    )
    final = evaluate(net, n_episodes=200, T=args.T, p1=10, seed=12345)
    print(f"  final acc={final['acc'] * 100:.1f}%  "
          f"mean|err|={final['abs_err_mean']:.4f}")

    plot_training_curves(history, os.path.join(args.outdir,
                                               "training_curves.png"))
    plot_weights(net, os.path.join(args.outdir, "weights.png"))
    plot_test_sequence(net, os.path.join(args.outdir, "test_sequence.png"))
    plot_output_distribution(net, os.path.join(args.outdir,
                                               "output_distribution.png"))
    print(f"  wrote viz/*.png to {args.outdir}/")


if __name__ == "__main__":
    main()
