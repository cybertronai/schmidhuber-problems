"""Static visualizations for double-pole-no-velocity (ESP co-evolution).

Outputs (in `viz/`):
  training_curves.png  - per-generation best-assembly fitness, mean fitness,
                         #solved trials per generation; burst markers
  rollout.png          - 1000-step rollout under the trained net showing
                         observed positions and (diagnostic-only) hidden
                         velocities, plus action trace
  weights.png          - heatmap of W_x, W_h, b, V for the assembled net

Usage:
    python3 visualize_double_pole_no_velocity.py --seed 0 --outdir viz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from double_pole_no_velocity import (
    RunConfig, run, run_episode, assemble_network,
    init_state, init_state_random, normalize_obs, double_pole_step,
    is_failed, F_MAX, X_LIMIT, THETA_LIMIT,
)


def plot_training_curves(history: dict, out_path: str, solve_threshold: int):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=120)

    gens = history["gen"]

    ax = axes[0]
    ax.plot(gens, history["best_fitness"], "o-",
            color="#2ca02c", markersize=3, linewidth=0.9,
            label="best-assembly balance")
    ax.axhline(solve_threshold, color="red", linestyle="--",
               linewidth=0.7, alpha=0.6,
               label=f"{solve_threshold}-step target")
    for g, b in zip(gens, history["burst"]):
        if b:
            ax.axvline(g, color="purple", linestyle=":", linewidth=0.6,
                       alpha=0.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("balance time (steps)")
    ax.set_title("Best assembly balance time per generation")
    ax.set_ylim(0, max(solve_threshold, max(history["best_fitness"])) * 1.1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(gens, history["mean_fitness"], color="#1f77b4", linewidth=0.9)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean per-individual fitness")
    ax.set_title("Population mean fitness")
    ax.grid(alpha=0.3)

    ax = axes[2]
    trials = np.array(history["trials_per_gen"], dtype=float)
    solved = np.array(history["n_solved_in_gen"], dtype=float)
    ax.plot(gens, solved / np.maximum(trials, 1) * 100.0,
            color="#ff7f0e", linewidth=0.9)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"% trials reaching {solve_threshold} steps")
    ax.set_title("Fraction of trial assemblies that solved")
    ax.set_ylim(0, max(5.0,
                       float((solved / np.maximum(trials, 1)).max() * 100.0
                             * 1.1) if len(solved) else 5.0))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def collect_rollout(net: dict, hidden: int, T_max: int,
                    state0: np.ndarray):
    """Roll out the trained net, returning state and action trajectories."""
    state = state0.copy()
    h = np.zeros(hidden)
    states = np.zeros((T_max + 1, 6))
    actions = np.zeros(T_max)
    states[0] = state
    last = T_max
    for t in range(T_max):
        x_obs = normalize_obs(state)
        pre = net["W_x"] @ x_obs + net["W_h"] @ h + net["b"]
        h = np.tanh(pre)
        u_pre = (net["V"] @ h + net["c"]).item()
        u = float(np.tanh(u_pre))
        actions[t] = u
        state = double_pole_step(state, u * F_MAX)
        states[t + 1] = state
        if is_failed(state):
            last = t + 1
            break
    return states[:last + 1], actions[:last]


def plot_rollout(net: dict, hidden: int, out_path: str, T_max: int = 1000):
    states, actions = collect_rollout(net, hidden, T_max,
                                      init_state(4.5 * np.pi / 180.0))
    T = len(actions)
    t_axis = np.arange(T + 1) * 0.01    # seconds (dt = 0.01)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=120, sharex=True)

    ax = axes[0]
    ax.plot(t_axis, states[:, 0], color="#1f77b4", label="x (m)")
    ax.axhline(X_LIMIT, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axhline(-X_LIMIT, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.plot(t_axis, np.degrees(states[:, 2]),
            color="#d62728", label=r"$\theta_1$ (deg)")
    ax.plot(t_axis, np.degrees(states[:, 4]),
            color="#9467bd", label=r"$\theta_2$ (deg)")
    ax.set_ylabel("position")
    ax.set_title("Observed positions (the only inputs to the net)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, states[:, 1], color="#1f77b4",
            label=r"$\dot x$ (m/s)")
    ax.plot(t_axis, states[:, 3], color="#d62728",
            label=r"$\dot{\theta}_1$ (rad/s)")
    ax.plot(t_axis, states[:, 5], color="#9467bd",
            label=r"$\dot{\theta}_2$ (rad/s)")
    ax.set_ylabel("velocity (HIDDEN)")
    ax.set_title("Hidden velocities — diagnostic only, the net never sees these")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_axis[:-1], actions, color="#2ca02c", linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("action u")
    ax.set_title(f"Action trace ({T}-step rollout from $\\theta_1$ = 4.5 deg)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(net: dict, out_path: str):
    H = net["W_h"].shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), dpi=120,
                             gridspec_kw={"width_ratios": [1.0, 1.4, 0.4, 0.4]})

    vmin = min(net["W_x"].min(), net["W_h"].min(), net["b"].min(),
               net["V"].min())
    vmax = max(net["W_x"].max(), net["W_h"].max(), net["b"].max(),
               net["V"].max())
    bound = max(abs(vmin), abs(vmax))

    ax = axes[0]
    im = ax.imshow(net["W_x"], aspect="auto", cmap="RdBu_r",
                   vmin=-bound, vmax=bound)
    ax.set_title(r"$W_x$ (H $\times$ 3)")
    ax.set_xticks(range(3), [r"$x$", r"$\theta_1$", r"$\theta_2$"])
    ax.set_yticks(range(H), [f"h{i}" for i in range(H)])

    ax = axes[1]
    ax.imshow(net["W_h"], aspect="auto", cmap="RdBu_r",
              vmin=-bound, vmax=bound)
    ax.set_title(r"$W_h$ (H $\times$ H)")
    ax.set_xticks(range(H), [f"h{i}" for i in range(H)])
    ax.set_yticks(range(H), [f"h{i}" for i in range(H)])

    ax = axes[2]
    ax.imshow(net["b"].reshape(-1, 1), aspect="auto", cmap="RdBu_r",
              vmin=-bound, vmax=bound)
    ax.set_title("b")
    ax.set_xticks([])
    ax.set_yticks(range(H), [f"h{i}" for i in range(H)])

    ax = axes[3]
    ax.imshow(net["V"].reshape(-1, 1), aspect="auto", cmap="RdBu_r",
              vmin=-bound, vmax=bound)
    ax.set_title(r"$V^\top$")
    ax.set_xticks([])
    ax.set_yticks(range(H), [f"h{i}" for i in range(H)])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.02)
    cbar.set_label("weight value")
    fig.suptitle("ESP-assembled recurrent net — weights", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gen", type=int, default=200)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed, max_generations=args.max_gen)
    print(f"Running evolution (seed={cfg.seed}, max_gen={cfg.max_generations})")
    res = run(cfg, verbose=True)

    os.makedirs(args.outdir, exist_ok=True)
    plot_training_curves(res["evolve"]["history"],
                         os.path.join(args.outdir, "training_curves.png"),
                         cfg.solve_threshold)
    net = assemble_network(res["evolve"]["best_genomes"])
    plot_rollout(net, cfg.hidden,
                 os.path.join(args.outdir, "rollout.png"),
                 T_max=cfg.eval_T_max)
    plot_weights(net, os.path.join(args.outdir, "weights.png"))


if __name__ == "__main__":
    main()
