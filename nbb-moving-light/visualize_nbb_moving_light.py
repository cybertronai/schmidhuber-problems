"""
Static visualizations for the trained NBB moving-light network.

Outputs (in `viz/`):
  training_curves.png    - frozen-eval accuracy + total substance + weight norms
  weights.png            - final W_io heatmap + W_oo heatmap + per-cell output
                           preference
  sequence_response.png  - per-tick output activations for each direction
                           under frozen-eval at convergence
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nbb_moving_light import (
    NBBMovingLight, train, evaluate,
    make_sequence, make_all_sequences,
)


DIRECTION_LABELS = ["LR (left → right)", "RL (right → left)"]
DIRECTION_COLORS = ["#1f77b4", "#ff7f0e"]


# ----------------------------------------------------------------------
# Plot 1: Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str,
                         converged_at: int | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)

    p = history["presentations"]

    ax = axes[0, 0]
    ax.plot(p, history["accuracy"], color="#1f77b4", linewidth=1.2)
    if converged_at is not None and converged_at > 0:
        ax.axvline(converged_at, color="green", linestyle="--", linewidth=1,
                   label=f"converged @ {converged_at}")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_ylabel("# correct (out of 2)")
    ax.set_xlabel("sequence presentations")
    ax.set_ylim(-0.2, 2.2)
    ax.set_yticks([0, 1, 2])
    ax.grid(alpha=0.3)
    ax.set_title("Frozen-eval accuracy on both directions")

    ax = axes[0, 1]
    ax.plot(p, history["total_substance"], color="#9467bd", linewidth=1.2)
    ax.set_ylabel("sum of all weights")
    ax.set_xlabel("sequence presentations")
    ax.grid(alpha=0.3)
    ax.set_title("Total weight-substance in the network")

    ax = axes[1, 0]
    ax.plot(p, history["W_io_norm"], color="#ff7f0e", linewidth=1.2)
    ax.set_ylabel(r"$\|W_{io}\|_F$")
    ax.set_xlabel("sequence presentations")
    ax.grid(alpha=0.3)
    ax.set_title("input → output norm")

    ax = axes[1, 1]
    ax.plot(p, history["W_oo_norm"], color="#2ca02c", linewidth=1.2)
    ax.set_ylabel(r"$\|W_{oo}\|_F$")
    ax.set_xlabel("sequence presentations")
    ax.grid(alpha=0.3)
    ax.set_title("output → output (recurrent) norm")

    fig.suptitle("NBB moving-light — training dynamics  (Schmidhuber 1989)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Plot 2: Final weights
# ----------------------------------------------------------------------

def plot_weights(nbb: NBBMovingLight, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=130)
    n_cells = nbb.n_cells

    # ---- W_io heatmap (input -> output) ---------------------------------
    ax = axes[0]
    W = nbb.W_io  # shape (n_input, 2)
    im = ax.imshow(W, cmap="magma", aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.3g}", ha="center", va="center",
                    color="white" if W[i, j] < W.max() * 0.6 else "black",
                    fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["out[0]\n(LR target)", "out[1]\n(RL target)"])
    yticks = ["bias"] + [f"cell {c}" for c in range(n_cells)]
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels(yticks)
    ax.set_title(f"$W_{{io}}$  (input → output)")
    plt.colorbar(im, ax=ax, fraction=0.05)

    # ---- W_oo heatmap (recurrent) --------------------------------------
    ax = axes[1]
    W = nbb.W_oo  # shape (2, 2)
    im = ax.imshow(W, cmap="magma", aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.3g}", ha="center", va="center",
                    color="white" if W[i, j] < W.max() * 0.6 else "black",
                    fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["to out[0]", "to out[1]"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["from out[0]", "from out[1]"])
    ax.set_title("$W_{oo}$  (recurrent self-connection)")
    plt.colorbar(im, ax=ax, fraction=0.05)

    # ---- Per-cell output preference -----------------------------------
    # diff[i] = W_io[i, 0] - W_io[i, 1].  Sign = which output that input
    # prefers; magnitude = how strongly. The retina cells should split:
    # left cells favour out[0] (LR target), right cells favour out[1] (RL).
    ax = axes[2]
    W = nbb.W_io
    diff = W[:, 0] - W[:, 1]
    labels = ["bias"] + [f"cell {c}" for c in range(n_cells)]
    colors = ["#1f77b4" if d > 0 else "#ff7f0e" for d in diff]
    ax.barh(range(len(diff)), diff, color=colors, edgecolor="black",
            linewidth=0.5)
    for i, d in enumerate(diff):
        sign = 1 if d > 0 else -1
        pad = 0.06 * max(abs(diff)) * sign + 1e-12
        ax.text(d + pad, i, f"{d:+.2g}", va="center",
                ha="left" if d > 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(range(len(diff)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("$W_{io}$[i, 0] $-$ $W_{io}$[i, 1]")
    ax.set_title("output preference per input\n"
                 "(blue = prefers out[0]/LR, orange = out[1]/RL)")
    pad = max(abs(diff)) * 1.6 + 1e-12
    ax.set_xlim(-pad, pad)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Plot 3: Sequence response (frozen-eval trace per direction)
# ----------------------------------------------------------------------

def plot_sequence_response(nbb: NBBMovingLight, out_path: str):
    """Show input pattern + output activations across all ticks for both directions."""
    n_cells = nbb.n_cells
    n_input = nbb.n_input

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), dpi=130,
                             gridspec_kw={"width_ratios": [n_cells, 2]})
    seqs = make_all_sequences(n_cells)

    for row, (direction, seq) in enumerate(seqs):
        out, trace = nbb.present(seq, direction, learn=False, trace=True)
        x_o_trace = np.array(trace)  # shape (n_cells, 2)

        # left panel: input retina over time. Drop bias for visibility.
        ax = axes[row, 0]
        retina = seq[:, 1:]  # (n_cells, n_cells)
        ax.imshow(retina, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for t in range(n_cells):
            for c in range(n_cells):
                if retina[t, c] > 0:
                    ax.text(c, t, "★", ha="center", va="center",
                            color="#cc6600", fontsize=14)
        ax.set_xticks(range(n_cells))
        ax.set_xticklabels([f"cell {c}" for c in range(n_cells)])
        ax.set_yticks(range(n_cells))
        ax.set_yticklabels([f"t={t}" for t in range(n_cells)])
        ax.set_title(f"{DIRECTION_LABELS[direction]} — retina over 5 ticks")

        # right panel: output activations over time
        ax = axes[row, 1]
        ax.imshow(x_o_trace, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for t in range(n_cells):
            for o in range(2):
                if x_o_trace[t, o] > 0:
                    color = ("#1f7a1f" if o == direction else "#cc0000")
                    ax.text(o, t, "★", ha="center", va="center",
                            color=color, fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["out[0]\n(LR)", "out[1]\n(RL)"])
        ax.set_yticks(range(n_cells))
        ax.set_yticklabels([f"t={t}" for t in range(n_cells)])
        outcome = "✓" if out == direction else "✗"
        ax.set_title(f"output trace  ({outcome} final={out})")

    fig.suptitle("NBB moving-light — frozen-eval per-tick response", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-presentations", type=int, default=5000)
    p.add_argument("--n-cells", type=int, default=5)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training (seed={args.seed}, n_cells={args.n_cells}, "
          f"eta={args.eta}, lam={args.lam})...")
    history = {"presentations": [], "accuracy": [],
               "W_io_norm": [], "W_oo_norm": [], "total_substance": []}
    nbb, presentations, acc = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        n_cells=args.n_cells,
        eta=args.eta, lam=args.lam,
        history=history, log_every=4, verbose=False,
    )
    print(f"  presentations={presentations}  acc={acc}/2")

    converged_at = presentations if acc == 2 else None

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"),
                         converged_at)
    plot_weights(nbb, os.path.join(args.outdir, "weights.png"))
    plot_sequence_response(nbb, os.path.join(args.outdir, "sequence_response.png"))


if __name__ == "__main__":
    main()
