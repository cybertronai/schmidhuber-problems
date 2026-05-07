"""
Static visualizations for the trained flip-flop controller and world-model.

Outputs (in `viz/`):
    training_curves.png       --- mean pain, M-loss, and binary accuracy over outer steps
    test_episode.png          --- one fresh test episode: events, desired vs C output, pain
    controller_weights.png    --- C's W_xh and W_hh as Hinton diagrams
    model_weights.png         --- M's W_xh and W_hh as Hinton diagrams
    pain_landscape.png        --- M's predicted pain as a function of action y at five
                                  representative latch states
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from flip_flop import (Controller, WorldModel, train, make_episode,
                       rollout_controller, forward_world_model)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=120)
    steps = history["step"]

    ax = axes[0]
    ax.plot(steps, history["pain_mean"], color="#d62728", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("outer step")
    ax.set_ylabel("mean pain (per episode)")
    ax.set_title("Mean pain over training (log scale)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, history["M_loss"], color="#1f77b4", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("outer step")
    ax.set_ylabel("M's pain-prediction MSE")
    ax.set_title("World-model loss")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, np.array(history["accuracy"]) * 100,
            color="#2ca02c", linewidth=1.0)
    ax.set_xlabel("outer step")
    ax.set_ylabel("binary accuracy (threshold 0.5)")
    ax.set_ylim(-5, 105)
    ax.set_title("Controller accuracy on training episodes")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"flip-flop training ({history.get('regime','sequential')})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Test-episode rollout
# ----------------------------------------------------------------------

def plot_test_episode(C: Controller, M: WorldModel, out_path: str,
                      seed: int = 12345, T: int = 80):
    rng = np.random.default_rng(seed)
    events, desired = make_episode(T, rng)
    traj_C = rollout_controller(C, events, desired)
    traj_M = forward_world_model(M, traj_C["obs"], traj_C["y"])

    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.5), dpi=120, sharex=True)

    # Top: events as colored vlines
    ax = axes[0]
    a_t = np.flatnonzero(events[:, 0] > 0.5)
    b_t = np.flatnonzero(events[:, 1] > 0.5)
    x_t = np.flatnonzero(events[:, 2] > 0.5)
    for tt in a_t:
        ax.axvline(tt, color="#d62728", linewidth=2.0, alpha=0.8)
    for tt in b_t:
        ax.axvline(tt, color="#1f77b4", linewidth=2.0, alpha=0.8)
    for tt in x_t:
        ax.axvline(tt, color="gray", linewidth=1.0, alpha=0.4)
    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="A (reset)"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="B (set)"),
        plt.Line2D([0], [0], color="gray", linewidth=1, label="X (distractor)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Events")

    # Middle: desired vs controller output
    ax = axes[1]
    ax.step(t, desired, where="post", color="black", linewidth=1.5,
            label="desired (latch state)")
    ax.plot(t, traj_C["y"], color="#ff7f0e", linewidth=1.5,
            label="controller output  $y_t$")
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("output")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Desired latch state vs. controller output")

    # Bottom: actual pain and M's predicted pain
    ax = axes[2]
    ax.plot(t, traj_C["pain"], color="#d62728", linewidth=1.2,
            label="actual pain  $(y_t - desired_t)^2$")
    ax.plot(t, traj_M["pred_pain"], color="#1f77b4", linewidth=1.0,
            linestyle="--", label="M's predicted pain")
    ax.set_ylim(-0.02, max(0.4, traj_C["pain"].max() * 1.05))
    ax.set_xlabel("time step")
    ax.set_ylabel("pain")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pain trajectory  (residual after training)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}  "
          f"(acc={float(np.mean((traj_C['y']>0.5).astype(float)==desired))*100:.1f}%)")


# ----------------------------------------------------------------------
# Hinton-diagram weight matrix
# ----------------------------------------------------------------------

def hinton(ax, W: np.ndarray, row_labels=None, col_labels=None, title=""):
    n_row, n_col = W.shape
    max_abs = max(abs(W).max(), 1e-3)
    ax.set_xlim(-0.6, n_col - 0.4)
    ax.set_ylim(-0.6, n_row - 0.4)
    ax.invert_yaxis()
    for i in range(n_row):
        for j in range(n_col):
            w = W[i, j]
            sz = 0.85 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.3))
    if col_labels is not None:
        ax.set_xticks(range(n_col))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=90)
    if row_labels is not None:
        ax.set_yticks(range(n_row))
        ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def plot_controller_weights(C: Controller, out_path: str):
    n_h = C.n_hidden
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=120,
                             gridspec_kw={"width_ratios": [1.5, n_h / 6.0, 0.8]})

    in_labels = ["A", "B", "X", "bias", "pain", "y_prev"]
    h_labels = [f"h{i}" for i in range(n_h)]
    hinton(axes[0], C.W_xh.T, row_labels=in_labels, col_labels=h_labels,
           title=r"$W_{xh}^T$ (input -> hidden)")
    hinton(axes[1], C.W_hh, row_labels=h_labels, col_labels=h_labels,
           title=r"$W_{hh}$ (hidden -> hidden)")
    hinton(axes[2], C.W_ho.T, row_labels=h_labels, col_labels=["y"],
           title=r"$W_{ho}^T$ (hidden -> output)")

    fig.suptitle("Controller C: weight matrices after training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_model_weights(M: WorldModel, out_path: str):
    n_h = M.n_hidden
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=120,
                             gridspec_kw={"width_ratios": [1.5, n_h / 6.0, 0.8]})

    in_labels = ["A", "B", "X", "bias", "pain", "y"]
    h_labels = [f"h{i}" for i in range(n_h)]
    hinton(axes[0], M.W_xh.T, row_labels=in_labels, col_labels=h_labels,
           title=r"$W_{xh}^T$ (input -> hidden)")
    hinton(axes[1], M.W_hh, row_labels=h_labels, col_labels=h_labels,
           title=r"$W_{hh}$ (hidden -> hidden)")
    hinton(axes[2], M.W_hp.T, row_labels=h_labels, col_labels=["pred_pain"],
           title=r"$W_{hp}^T$ (hidden -> pain)")

    fig.suptitle("World-model M: weight matrices after training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Pain landscape: M's predicted pain across action range
# ----------------------------------------------------------------------

def plot_pain_landscape(M: WorldModel, out_path: str, seed: int = 12345):
    """Drive M with five canonical event sequences (pre-A, just-after-A,
    distractor, just-after-B, well-after-B) and show predicted pain vs y.

    A correctly trained M should learn a bowl with minimum at y=desired.
    """
    rng = np.random.default_rng(seed)
    # Hand-crafted 12-step prefix that lands the latch in known states.
    # The probe is: "what does M predict pain to be if we now output y?"
    contexts = [
        ("just after A (state=0)",
         [(1, 0, 0)] + [(0, 0, 0)] * 1,   # A then rest
         0),
        ("after A + distractor (state=0)",
         [(1, 0, 0)] + [(0, 0, 1)] * 3,
         0),
        ("just after B (state=1)",
         [(1, 0, 0), (0, 0, 0), (0, 1, 0)],
         1),
        ("after B + distractor (state=1)",
         [(1, 0, 0), (0, 0, 0), (0, 1, 0)] + [(0, 0, 1)] * 3,
         1),
        ("long after B (state=1)",
         [(1, 0, 0), (0, 0, 0), (0, 1, 0)] + [(0, 0, 0)] * 6,
         1),
    ]

    y_grid = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    colors = ["#1f77b4", "#9467bd", "#d62728", "#ff7f0e", "#2ca02c"]
    for i, (label, prefix, target) in enumerate(contexts):
        # Build a 12-step prefix padded with no-events.
        prefix_arr = np.array(prefix + [(0, 0, 0)] * (12 - len(prefix)),
                              dtype=np.float64)
        T = prefix_arr.shape[0]
        obs = np.zeros((T, 5))
        obs[:, :3] = prefix_arr
        obs[:, 3] = 1.0
        # Run M with arbitrary y on the prefix to settle hidden state.
        y_prefix = np.full(T, 0.5)
        traj_pre = forward_world_model(M, obs, y_prefix)
        h_settled = traj_pre["h_M"][-1]

        # Now probe a single step with each y in the grid.
        pred = np.zeros_like(y_grid)
        for j, y in enumerate(y_grid):
            obs_step = np.zeros((1, 5))
            obs_step[0, 3] = 1.0  # bias only, no event
            x = np.concatenate([obs_step[0], [y]])
            z = M.W_xh @ x + M.W_hh @ h_settled + M.b_h
            h = np.tanh(z)
            pre = float((M.W_hp @ h + M.b_p)[0])
            pred[j] = 1.0 / (1.0 + np.exp(-np.clip(pre, -50, 50)))
        ax.plot(y_grid, pred, color=colors[i], linewidth=1.4, label=label)
        ax.axvline(target, color=colors[i], linewidth=0.6,
                   linestyle=":", alpha=0.6)

    ax.set_xlabel("controller action  y")
    ax.set_ylabel("M's predicted pain")
    ax.set_title("World-model's pain landscape  (dotted: target = desired)")
    ax.legend(loc="upper center", fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", choices=["sequential", "parallel"],
                   default="sequential")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr-M", type=float, default=1e-2)
    p.add_argument("--lr-C", type=float, default=5e-3)
    p.add_argument("--M-warmup", type=int, default=500)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.steps} outer steps "
          f"(seed={args.seed}, regime={args.regime})...")
    C, M, history = train(
        seed=args.seed,
        regime=args.regime,
        n_steps=args.steps,
        T=args.T,
        n_hidden=args.hidden,
        lr_M=args.lr_M,
        lr_C=args.lr_C,
        M_warmup=args.M_warmup,
        verbose=False,
    )
    print(f"  final accuracy: {history['accuracy'][-1]*100:.0f}%   "
          f"final pain: {history['pain_mean'][-1]:.4f}")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_test_episode(C, M, os.path.join(args.outdir, "test_episode.png"),
                      seed=12345, T=80)
    plot_controller_weights(C, os.path.join(args.outdir, "controller_weights.png"))
    plot_model_weights(M, os.path.join(args.outdir, "model_weights.png"))
    plot_pain_landscape(M, os.path.join(args.outdir, "pain_landscape.png"))


if __name__ == "__main__":
    main()
