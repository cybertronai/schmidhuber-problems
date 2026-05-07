"""
Static visualizations for the HQ-learning POM run.

Outputs (in viz/):
    maze.png                  --- the maze layout with start, goal, obs labels.
    learning_curves.png       --- running-mean steps and solve rate, HQ vs flat-Q.
    hq_tables.png             --- HQ-table heatmap per sub-agent.
    q_tables.png              --- Q-table heatmap per sub-agent + flat-Q's table.
    subagent_trajectory.png   --- one trial's path coloured by active sub-agent.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from hq_learning_pomdp import (
    POMMaze, HQAgent, FlatQAgent,
    train_hq, train_flat,
    ACTION_NAMES, ACTIONS,
)


SUBAGENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def maze_to_rgb(env: POMMaze) -> np.ndarray:
    """White=free, black=wall, green=start, red=goal."""
    img = np.ones((env.rows, env.cols, 3), dtype=np.float32)
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] == 1:
                img[r, c] = [0.15, 0.15, 0.15]
    img[env.start] = [0.2, 0.7, 0.2]
    img[env.goal] = [0.85, 0.2, 0.2]
    return img


def plot_maze(env: POMMaze, out_path: str):
    fig, ax = plt.subplots(figsize=(4.0, 1.0 + 0.6 * env.rows), dpi=140)
    ax.imshow(maze_to_rgb(env), origin="upper")
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] == 1:
                continue
            obs = env._wall_mask(r, c)
            ax.text(c, r, str(obs), ha="center", va="center",
                    fontsize=7, color="black")
    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr - 0.45, "S", ha="center", va="center",
            color="darkgreen", fontsize=10, fontweight="bold")
    ax.text(gc, gr - 0.45, "G", ha="center", va="center",
            color="darkred", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"POM maze ({env.rows}x{env.cols})  free={int((env.grid==0).sum())}  "
                 f"obs={len(env._observed_obs)}  optimal={env._optimal_steps}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_learning_curves(hq_run, flat_run, out_path: str, optimal: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130)
    hq_h = hq_run["history"]
    flat_h = flat_run["history"]

    ax = axes[0]
    ax.plot(hq_h["trial"], hq_h["running_steps"],
            color="#1f77b4", linewidth=1.0, label="HQ-learning")
    ax.plot(flat_h["trial"], flat_h["running_steps"],
            color="#d62728", linewidth=1.0, label="Flat Q(lambda)")
    ax.axhline(optimal, color="black", linestyle="--", linewidth=0.8,
               label=f"BFS optimal ({optimal})")
    ax.set_xlabel("trial")
    ax.set_ylabel("running mean steps (window=200)")
    ax.set_title("Episodic step count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hq_h["trial"], np.array(hq_h["running_solved"]) * 100,
            color="#1f77b4", linewidth=1.0, label="HQ-learning")
    ax.plot(flat_h["trial"], np.array(flat_h["running_solved"]) * 100,
            color="#d62728", linewidth=1.0, label="Flat Q(lambda)")
    ax.set_xlabel("trial")
    ax.set_ylabel("solve rate (%, window=200)")
    ax.set_title("Goal-reaching rate during training")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("HQ-learning vs flat Q(lambda) on the POM maze", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_hq_tables(hq: HQAgent, valid_obs, out_path: str):
    if hq.HQ is None:
        return
    fig, axes = plt.subplots(1, hq.M - 1, figsize=(2.0 * (hq.M - 1) + 1, 3.5),
                              dpi=130, squeeze=False)
    axes = axes[0]
    vals = hq.HQ[:, valid_obs]
    vmax = max(1.0, np.abs(vals).max())
    for i in range(hq.M - 1):
        ax = axes[i]
        im = ax.imshow(hq.HQ[i, valid_obs].reshape(-1, 1),
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for j, o in enumerate(valid_obs):
            v = hq.HQ[i, o]
            ax.text(0, j, f"{v:.1f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > vmax * 0.5 else "black")
        ax.set_yticks(range(len(valid_obs)))
        ax.set_yticklabels([f"o={o}" for o in valid_obs], fontsize=8)
        ax.set_xticks([])
        best_idx = int(np.argmax(hq.HQ[i, valid_obs]))
        best_o = valid_obs[best_idx]
        ax.set_title(f"sub-agent {i}\nbest sub-goal: o={best_o}",
                     fontsize=9)
    fig.suptitle("HQ-table values (sub-goal scores)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_q_tables(hq: HQAgent, flat: FlatQAgent, valid_obs, out_path: str):
    cols = hq.M + 1
    fig, axes = plt.subplots(1, cols, figsize=(2.0 * cols + 1, 3.5),
                              dpi=130, squeeze=False)
    axes = axes[0]

    all_q = np.concatenate([
        hq.Q[:, valid_obs, :].ravel(),
        flat.Q[valid_obs, :].ravel(),
    ])
    vmax = max(1.0, np.abs(all_q).max())

    for i in range(hq.M):
        ax = axes[i]
        Q = hq.Q[i, valid_obs, :]
        im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        for j, o in enumerate(valid_obs):
            for a in range(4):
                v = Q[j, a]
                ax.text(a, j, f"{v:.1f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > vmax * 0.5 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ACTION_NAMES, fontsize=8)
        ax.set_yticks(range(len(valid_obs)))
        ax.set_yticklabels([f"o={o}" for o in valid_obs], fontsize=8)
        ax.set_title(f"HQ sub-agent {i}", fontsize=9)

    ax = axes[-1]
    Q = flat.Q[valid_obs, :]
    im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for j, o in enumerate(valid_obs):
        for a in range(4):
            v = Q[j, a]
            ax.text(a, j, f"{v:.1f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(ACTION_NAMES, fontsize=8)
    ax.set_yticks(range(len(valid_obs)))
    ax.set_yticklabels([f"o={o}" for o in valid_obs], fontsize=8)
    ax.set_title("Flat Q(lambda)", fontsize=9)

    fig.suptitle("Action-value tables Q(o, a)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_subagent_trajectory(env: POMMaze, hq: HQAgent, out_path: str,
                              seed: int = 1234):
    rng = np.random.default_rng(seed)
    saved = hq.rng
    hq.rng = rng
    res = hq.run_episode(env, p_max_a=0.95, p_max_sg=1.0, learn=False)
    hq.rng = saved

    fig, ax = plt.subplots(figsize=(4.0, 1.0 + 0.6 * env.rows), dpi=140)
    ax.imshow(maze_to_rgb(env), origin="upper")

    traj = res["trajectory"]
    positions = traj["pos"]
    agent_ids = traj["agent_id"]

    # Draw path with sub-agent colors.
    for i in range(len(positions) - 1):
        r1, c1 = positions[i]
        r2, c2 = positions[i + 1]
        a_id = agent_ids[i]
        col = SUBAGENT_COLORS[a_id % len(SUBAGENT_COLORS)]
        ax.plot([c1, c2], [r1, r2], "-", color=col, alpha=0.7, linewidth=2.0)
        ax.plot(c1, r1, "o", color=col, markersize=4)
    if positions:
        r, c = positions[-1]
        ax.plot(c, r, "*", color="gold", markersize=12)

    # Legend.
    handles = [plt.Line2D([0], [0], color=SUBAGENT_COLORS[i % len(SUBAGENT_COLORS)],
                          linewidth=2, label=f"sub-agent {i}")
               for i in range(hq.M)]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"trajectory ({res['steps']} steps, "
                 f"reached goal={res['reached_goal']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=5000)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--M", type=int, default=5)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    env = POMMaze(max_steps=200)
    valid_obs = sorted(env._observed_obs)

    plot_maze(env, os.path.join(outdir, "maze.png"))
    print(f"  wrote {outdir}/maze.png")

    print(f"Training HQ-learning (M={args.M}, seed={args.seed}) ...")
    rng_h = np.random.default_rng(args.seed)
    hq = HQAgent(n_obs=env.n_obs, n_actions=env.n_actions, M=args.M,
                 alpha_q=0.1, alpha_hq=0.2, gamma=0.95, lam=0.9,
                 temperature=0.5, valid_subgoals=env._observed_obs,
                 min_subagent_steps=2, rng=rng_h)
    hq_run = train_hq(env, hq, args.n_trials,
                      p_max_start=0.0, p_max_end=1.0, verbose=False)

    print(f"Training flat Q(lambda) (seed={args.seed+1}) ...")
    rng_f = np.random.default_rng(args.seed + 1)
    flat = FlatQAgent(n_obs=env.n_obs, n_actions=env.n_actions,
                      alpha=0.1, gamma=0.95, lam=0.9, temperature=0.5,
                      rng=rng_f)
    flat_run = train_flat(env, flat, args.n_trials,
                          p_max_start=0.0, p_max_end=1.0, verbose=False)

    plot_learning_curves(hq_run, flat_run,
                          os.path.join(outdir, "learning_curves.png"),
                          env._optimal_steps)
    print(f"  wrote {outdir}/learning_curves.png")

    plot_hq_tables(hq, valid_obs, os.path.join(outdir, "hq_tables.png"))
    print(f"  wrote {outdir}/hq_tables.png")

    plot_q_tables(hq, flat, valid_obs, os.path.join(outdir, "q_tables.png"))
    print(f"  wrote {outdir}/q_tables.png")

    plot_subagent_trajectory(env, hq,
                              os.path.join(outdir, "subagent_trajectory.png"))
    print(f"  wrote {outdir}/subagent_trajectory.png")


if __name__ == "__main__":
    main()
