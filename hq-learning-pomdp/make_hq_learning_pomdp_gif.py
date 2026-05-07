"""
Render an animated GIF of HQ-learning training on the POM maze.

Each frame is a snapshot during training, showing:
    Top:   the maze with the latest test trajectory, coloured by the
           sub-agent in control at each step.
    Bottom: running step count and solve rate so far.

Usage:
    python3 make_hq_learning_pomdp_gif.py --seed 0
    python3 make_hq_learning_pomdp_gif.py --seed 0 --max-frames 60 --fps 8
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from hq_learning_pomdp import POMMaze, HQAgent, train_hq
from visualize_hq_learning_pomdp import maze_to_rgb, SUBAGENT_COLORS


def render_frame(env: POMMaze, hq: HQAgent, history: dict,
                 max_trial: int, test_seed: int = 1234) -> Image.Image:
    rng = np.random.default_rng(test_seed)
    saved = hq.rng
    hq.rng = rng
    res = hq.run_episode(env, p_max_a=0.95, p_max_sg=1.0, learn=False)
    hq.rng = saved

    fig = plt.figure(figsize=(7.5, 5.0), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1, 1.4])

    # Maze panel (top-left).
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(maze_to_rgb(env), origin="upper")
    traj = res["trajectory"]
    positions = traj["pos"]
    agent_ids = traj["agent_id"]
    for i in range(len(positions) - 1):
        r1, c1 = positions[i]
        r2, c2 = positions[i + 1]
        a_id = agent_ids[i]
        col = SUBAGENT_COLORS[a_id % len(SUBAGENT_COLORS)]
        ax.plot([c1, c2], [r1, r2], "-", color=col, alpha=0.7, linewidth=2.0)
        ax.plot(c1, r1, "o", color=col, markersize=3)
    if positions:
        r, c = positions[-1]
        ax.plot(c, r, "*", color="gold", markersize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"trial {max_trial}: {res['steps']} steps  "
                 f"reached={res['reached_goal']}",
                 fontsize=9)

    # HQ-table panel (top-right).
    ax = fig.add_subplot(gs[0, 1])
    if hq.HQ is not None:
        valid = sorted(env._observed_obs)
        Z = hq.HQ[:, valid]
        vmax = max(1.0, np.abs(Z).max())
        im = ax.imshow(Z.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(hq.M - 1))
        ax.set_xticklabels([f"sub-{i}" for i in range(hq.M - 1)], fontsize=8)
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels([f"o={o}" for o in valid], fontsize=8)
        for i in range(hq.M - 1):
            best = int(np.argmax(hq.HQ[i, valid]))
            ax.add_patch(plt.Rectangle((i - 0.4, best - 0.4), 0.8, 0.8,
                                       fill=False, edgecolor="lime",
                                       linewidth=1.8))
        ax.set_title("HQ-table (greedy sub-goal in green)", fontsize=9)
    else:
        ax.set_title("no HQ-table (M=1)", fontsize=9)

    # Curves (bottom).
    ax = fig.add_subplot(gs[1, :])
    trials = history["trial"][:max_trial]
    steps = history["running_steps"][:max_trial]
    solve = history["running_solved"][:max_trial]
    ax.plot(trials, steps, color="#1f77b4", linewidth=1.0,
            label="running steps")
    ax.set_xlabel("trial")
    ax.set_ylabel("running steps", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(trials, np.array(solve) * 100, color="#2ca02c", linewidth=1.0,
             label="solve rate %")
    ax2.set_ylabel("solve rate (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(-5, 105)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=5000)
    p.add_argument("--snapshot-every", type=int, default=100,
                   help="trials between training snapshots")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="hq_learning_pomdp.gif")
    p.add_argument("--M", type=int, default=5)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    env = POMMaze(max_steps=200)
    rng = np.random.default_rng(args.seed)
    hq = HQAgent(n_obs=env.n_obs, n_actions=env.n_actions, M=args.M,
                 alpha_q=0.1, alpha_hq=0.2, gamma=0.95, lam=0.9,
                 temperature=0.5, valid_subgoals=env._observed_obs,
                 min_subagent_steps=2, rng=rng)

    print(f"Training {args.n_trials} trials, snapshots every "
          f"{args.snapshot_every} ...")
    out = train_hq(env, hq, args.n_trials,
                   p_max_start=0.0, p_max_end=1.0,
                   snapshot_every=args.snapshot_every, verbose=False)

    snapshots = out["snapshots"]
    history = out["history"]
    if len(snapshots) > args.max_frames:
        idx = np.linspace(0, len(snapshots) - 1, args.max_frames).astype(int)
        snapshots = [snapshots[i] for i in idx]

    print(f"Rendering {len(snapshots)} frames ...")
    frames = []
    for snap in snapshots:
        # Restore training-state at the snapshot for rendering.
        hq.Q = snap["Q"].copy()
        if snap["HQ"] is not None:
            hq.HQ = snap["HQ"].copy()
        frame = render_frame(env, hq, history, snap["trial"])
        frames.append(frame)

    out_path = os.path.join(here, args.out)
    duration = max(int(1000 / args.fps), 1)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    optimize=True, duration=duration, loop=0)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"Wrote {out_path}  ({size_kb} KB, {len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
