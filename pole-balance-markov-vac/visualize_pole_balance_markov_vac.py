"""
Static visualisations for pole-balance-markov-vac.

Outputs (all PNG, in `viz/`):
  learning_curve.png       - balance steps per episode + trailing window
  critic_trajectories.png  - V_pole(t), V_cart(t), and TD residuals on a
                             greedy eval episode after solve
  actor_weight_evolution.png - Hinton-style snapshots of actor W_a1 at
                             init, mid-training, and at solve
  state_phase.png          - phase-space snapshot of the controlled
                             trajectory in (theta, theta_dot)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pole_balance_markov_vac import (
    DT, F_MAG, THETA_THRESHOLD, X_THRESHOLD,
    actor_forward, critic_forward,
    evaluate, run_episode, train_vac,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def hinton(ax, W: np.ndarray, max_w: float | None = None,
           title: str = "") -> None:
    """Hinton-style diagram of a weight matrix."""
    ax.patch.set_facecolor("#f4f4f4")
    if max_w is None:
        max_w = float(np.abs(W).max() + 1e-9)
    nrow, ncol = W.shape
    for i in range(nrow):
        for j in range(ncol):
            w = W[i, j]
            color = "#cc0000" if w >= 0 else "#003c7f"
            size = np.sqrt(min(abs(w) / max_w, 1.0)) * 0.95
            r = plt.Rectangle((j - size / 2, i - size / 2), size, size,
                              facecolor=color, edgecolor="none")
            ax.add_patch(r)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------
# Plot 1: learning curve
# ----------------------------------------------------------------------

def plot_learning_curve(hist, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=140)
    eps = np.arange(1, len(hist.balance_steps) + 1)
    ax.scatter(eps, hist.balance_steps, s=10, color="#888", alpha=0.55,
               label="per-episode balance steps", rasterized=True)
    ax.plot(eps, hist.moving_avg, color="#cc0000", lw=2.0,
            label="trailing-20 mean")
    if hist.solve_episode is not None:
        ax.axvline(hist.solve_episode, color="#cc0000", ls="--", lw=0.9,
                   alpha=0.6)
        ax.text(hist.solve_episode, 50, f" solved @ {hist.solve_episode}",
                color="#cc0000", fontsize=9, va="bottom")
    ax.axhline(950, color="black", ls=":", lw=0.6, alpha=0.5,
               label="solve threshold = 950")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps balanced (max 1000)")
    ax.set_title("VAC learning curve on Markov cart-pole")
    ax.set_ylim(-20, 1080)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 2: critic vector trajectories on a greedy eval episode
# ----------------------------------------------------------------------

def plot_critic_trajectories(p, seed: int, out_path: str,
                             gamma: float = 0.99,
                             max_steps: int = 1000) -> None:
    rng = np.random.default_rng(seed + 200_000)
    info = run_episode(p, rng, gamma=gamma, mix_w=np.array([1.0, 0.3]),
                       actor_lr=0.0, critic_lr=0.0, entropy_coef=0.0,
                       max_steps=max_steps, train=False, greedy=True)
    log = info["log"]
    states = np.array(log["states"])
    v_pole = np.array(log["v_pole"])
    v_cart = np.array(log["v_cart"])
    actions = np.array(log["actions"])
    t = np.arange(len(v_pole)) * DT

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0), dpi=140, sharex=True)

    ax = axes[0]
    ax.plot(t, v_pole, color="#cc0000", lw=1.4, label="V_pole(s_t)")
    ax.plot(t, v_cart, color="#003c7f", lw=1.4, label="V_cart(s_t)")
    ax.set_ylabel("vector critic V")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Vector critic on a greedy 1000-step balance episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, states[:, 0], color="#003c7f", lw=1.0, label="cart x")
    ax.axhline(X_THRESHOLD, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(-X_THRESHOLD, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_ylabel("cart x")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, np.degrees(states[:, 2]), color="#cc0000", lw=1.0,
            label="pole angle (deg)")
    ax.axhline(np.degrees(THETA_THRESHOLD), color="black", ls=":", lw=0.6,
               alpha=0.5)
    ax.axhline(-np.degrees(THETA_THRESHOLD), color="black", ls=":",
               lw=0.6, alpha=0.5)
    # action band at the bottom
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, actions, step="post", alpha=0.15, color="#666",
                     label="action (push right=1)")
    ax2.set_yticks([])
    ax.set_ylabel("pole angle (deg)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 3: actor weight evolution snapshots
# ----------------------------------------------------------------------

def plot_actor_weight_evolution(hist, out_path: str) -> None:
    snaps = hist.snapshot_params
    eps = hist.snapshot_episode
    if len(snaps) > 4:
        # Pick init, two intermediates, final.
        idxs = [0, len(snaps) // 3, 2 * len(snaps) // 3, len(snaps) - 1]
        snaps = [snaps[i] for i in idxs]
        eps = [eps[i] for i in idxs]
    n = len(snaps)
    max_w = max(float(np.abs(s.Wa1).max()) for s in snaps)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.0), dpi=140)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, (snap, ep) in enumerate(zip(snaps, eps)):
        hinton(axes[0, i], snap.Wa1, max_w=max_w,
               title=f"actor Wa1\nepisode {ep}")
        # Critic Wc2 is K x H — also show
        hinton(axes[1, i], snap.Wc2,
               max_w=float(np.abs(snap.Wc2).max() + 1e-9),
               title=f"critic Wc2 (K=2)\nepisode {ep}")
    fig.suptitle("Actor + critic readout weights through training",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 4: phase portrait
# ----------------------------------------------------------------------

def plot_phase_portrait(p, seed: int, out_path: str,
                        gamma: float = 0.99,
                        max_steps: int = 1000) -> None:
    rng = np.random.default_rng(seed + 300_000)
    info = run_episode(p, rng, gamma=gamma, mix_w=np.array([1.0, 0.3]),
                       actor_lr=0.0, critic_lr=0.0, entropy_coef=0.0,
                       max_steps=max_steps, train=False, greedy=True)
    states = np.array(info["log"]["states"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=140)

    ax = axes[0]
    ax.plot(np.degrees(states[:, 2]), states[:, 3], color="#cc0000",
            lw=0.7, alpha=0.8)
    ax.scatter([np.degrees(states[0, 2])], [states[0, 3]], s=30,
               color="#003c7f", label="start", zorder=3)
    ax.axvline(np.degrees(THETA_THRESHOLD), color="black", ls=":",
               lw=0.6, alpha=0.5)
    ax.axvline(-np.degrees(THETA_THRESHOLD), color="black", ls=":",
               lw=0.6, alpha=0.5)
    ax.set_xlabel("pole angle (deg)")
    ax.set_ylabel("pole angular velocity")
    ax.set_title("phase: (theta, theta_dot)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(states[:, 0], states[:, 1], color="#003c7f",
            lw=0.7, alpha=0.8)
    ax.scatter([states[0, 0]], [states[0, 1]], s=30,
               color="#cc0000", label="start", zorder=3)
    ax.axvline(X_THRESHOLD, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(-X_THRESHOLD, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("cart x")
    ax.set_ylabel("cart x_dot")
    ax.set_title("phase: (x, x_dot)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--max-episodes", type=int, default=1000)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    p, hist = train_vac(seed=args.seed, max_episodes=args.max_episodes,
                        verbose=False)

    # Diagnostics
    print(f"[seed={args.seed}] solved={hist.solve_episode}  "
          f"final-trail={hist.moving_avg[-1]:.1f}  "
          f"snapshots={len(hist.snapshot_params)}")

    plot_learning_curve(hist, os.path.join(args.out_dir, "learning_curve.png"))
    plot_critic_trajectories(p, args.seed,
                             os.path.join(args.out_dir,
                                          "critic_trajectories.png"))
    plot_actor_weight_evolution(hist,
                                os.path.join(args.out_dir,
                                             "actor_weight_evolution.png"))
    plot_phase_portrait(p, args.seed,
                        os.path.join(args.out_dir, "state_phase.png"))
    print(f"wrote 4 PNGs to {args.out_dir}/")


if __name__ == "__main__":
    main()
