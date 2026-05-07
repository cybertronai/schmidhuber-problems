"""Animate the trained pomdp-flag-maze controller in the real env.

The animation alternates between an indicator=+1 episode (top-flag is the
correct goal) and an indicator=-1 episode (bottom-flag), showing how the
recurrent controller picks the correct branch at the T-junction even though
the indicator is no longer visible after t=0.

Usage:
    python3 make_pomdp_flag_maze_gif.py --seed 0
    python3 make_pomdp_flag_maze_gif.py --seed 0 --fps 6
"""
from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

from pomdp_flag_maze import (
    RunConfig, run, TMazeEnv, softmax,
    ROWS, COLS, START_RC, TJUNC_RC, TOP_FLAG_RC, BOT_FLAG_RC,
    is_walkable,
)


ARROW_NAMES = ["N", "E", "S", "W"]
ARROW_DXY = [(0.0, 0.4), (0.4, 0.0), (0.0, -0.4), (-0.4, 0.0)]


def render_frame(env: TMazeEnv, agent_rc, indicator: float,
                 last_action: int, step_idx: int, ep_label: str,
                 path_so_far: list, h_C_now: np.ndarray) -> Image.Image:
    fig = plt.figure(figsize=(7.5, 4.5), dpi=85)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    # paint cells
    for r in range(ROWS):
        for c in range(COLS):
            if is_walkable(r, c):
                color = "#f5f5f5"
            else:
                color = "#444"
            ax.add_patch(Rectangle((c, ROWS - 1 - r), 1, 1,
                                   facecolor=color, edgecolor="#aaa",
                                   linewidth=0.5))
    # flags
    for (r, c), color, label in [
        (TOP_FLAG_RC, "#d62728", "F+"),
        (BOT_FLAG_RC, "#2ca02c", "F-"),
    ]:
        ax.text(c + 0.5, ROWS - 1 - r + 0.5, label, ha="center",
                va="center", fontsize=14, color=color, fontweight="bold")
    # start label
    ax.text(START_RC[1] + 0.18, ROWS - 1 - START_RC[0] + 0.18, "S",
            ha="left", va="bottom", fontsize=9, color="#1f77b4")

    # path so far
    if path_so_far:
        xs = [c + 0.5 for (_, c) in path_so_far]
        ys = [ROWS - 1 - r + 0.5 for (r, _) in path_so_far]
        ax.plot(xs, ys, "-", color="#888", linewidth=1.0, alpha=0.6)

    # current agent
    r, c = agent_rc
    agent_color = "#d62728" if indicator > 0 else "#2ca02c"
    ax.add_patch(Circle((c + 0.5, ROWS - 1 - r + 0.5), 0.28,
                        facecolor=agent_color, edgecolor="black", linewidth=1.0,
                        zorder=4))
    # indicator hint at start cell only at t=0
    if step_idx == 0:
        ax.text(c + 0.5, ROWS - 1 - r + 0.85,
                f"indicator = {int(indicator):+d}",
                ha="center", va="bottom", fontsize=8,
                color=agent_color, fontweight="bold")

    # last action arrow
    if last_action is not None and step_idx > 0:
        dx, dy = ARROW_DXY[last_action]
        ax.annotate("", xy=(c + 0.5 + dx, ROWS - 1 - r + 0.5 + dy),
                    xytext=(c + 0.5, ROWS - 1 - r + 0.5),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.6, alpha=0.8), zorder=5)

    ax.set_xlim(-0.1, COLS + 0.1)
    ax.set_ylim(-0.1, ROWS + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{ep_label}    step {step_idx}    "
                 f"a = {ARROW_NAMES[last_action] if last_action is not None else '-'}    "
                 f"agent at ({r},{c})",
                 fontsize=10)

    # bottom: hidden-state bars
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(np.arange(h_C_now.shape[0]), h_C_now,
            color=["#1f77b4" if h >= 0 else "#ff7f0e" for h in h_C_now],
            edgecolor="black", linewidth=0.4)
    ax2.set_ylim(-1.0, 1.0)
    ax2.axhline(0.0, color="black", linewidth=0.5)
    ax2.set_ylabel(r"$h_C[i]$", fontsize=9)
    ax2.set_xlabel(f"hidden unit (the indicator latch lives here)", fontsize=9)
    ax2.set_xticks([])
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def collect_episode(C, env, indicator: float, T_max: int):
    obs = env.reset_to(indicator)
    h_C = np.zeros(C.hid_dim)
    states = [(env.r, env.c)]
    actions = []
    h_seq = [h_C.copy()]
    for _ in range(T_max):
        h_C, a_logit, _ = C.step_(obs, h_C)
        a = int(np.argmax(softmax(a_logit)))
        obs, r, done = env.step(a)
        states.append((env.r, env.c))
        actions.append(a)
        h_seq.append(h_C.copy())
        if done:
            break
    return states, actions, np.array(h_seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T-max", type=int, default=15)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", type=str, default="pomdp_flag_maze.gif")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed, run_baselines=False)
    print(f"Training (seed={args.seed}) ...")
    res = run(cfg, verbose=False)
    print(f"  recurrent C: success = {res['final_success']:.3f}  "
          f"({res['final_mean_steps']:.1f} mean steps)")
    C = res["C"]
    env = TMazeEnv(t_max=args.T_max)

    # one episode each indicator
    eps_data = []
    for ind, label in [(+1.0, "indicator = +1   target = top flag (F+)"),
                       (-1.0, "indicator = -1   target = bottom flag (F-)")]:
        states, actions, h_seq = collect_episode(C, env, ind, args.T_max)
        eps_data.append((ind, label, states, actions, h_seq))

    # render
    frames = []
    for (ind, label, states, actions, h_seq) in eps_data:
        path_so_far = [states[0]]
        # opening frame at t=0 (no action yet)
        frames.append(render_frame(env, states[0], ind, None, 0, label,
                                   [], h_seq[0]))
        for i, a in enumerate(actions):
            path_so_far.append(states[i + 1])
            frames.append(render_frame(env, states[i + 1], ind, a, i + 1,
                                       label, path_so_far[:-1], h_seq[i + 1]))
        # hold final frame
        frames.extend([frames[-1]] * 5)

    duration = max(1000 // max(args.fps, 1), 60)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
