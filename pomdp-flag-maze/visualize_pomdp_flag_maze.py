"""Static visualizations for the trained pomdp-flag-maze system.

Outputs (in `viz/`):
  maze_layout.png           - the T-maze with cell colors annotated
  agent_paths.png           - greedy real-env paths under trained C, both
                              indicator settings, side by side
  hidden_state.png          - h_C activations across the indicator=+/-1
                              trajectories, showing the latching mechanism
  training_curves.png       - phase-1 M MSE; phase-2 imagined R; cycle eval
                              success; FF baseline curve
  reward_table.png          - small table of the recurrent vs FF vs random
                              eval success / mean-steps numbers

Usage:
    python3 visualize_pomdp_flag_maze.py --seed 0 --outdir viz
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

from pomdp_flag_maze import (
    RunConfig, run, TMazeEnv, softmax,
    ROWS, COLS, START_RC, TJUNC_RC, TOP_FLAG_RC, BOT_FLAG_RC,
    is_walkable,
)


# ----------------------------------------------------------------------
# 1. Maze layout
# ----------------------------------------------------------------------

def plot_maze_layout(out_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=120)
    cell_size = 1.0
    for r in range(ROWS):
        for c in range(COLS):
            if is_walkable(r, c):
                color = "#e9f5e9"
            else:
                color = "#444"
            ax.add_patch(Rectangle((c, ROWS - 1 - r), cell_size, cell_size,
                                   facecolor=color, edgecolor="#888",
                                   linewidth=0.6))
    # annotations
    for (r, c), label, color in [
        (START_RC, "S", "#1f77b4"),
        (TJUNC_RC, "T", "#666"),
        (TOP_FLAG_RC, "F+", "#d62728"),
        (BOT_FLAG_RC, "F-", "#2ca02c"),
    ]:
        ax.text(c + 0.5, ROWS - 1 - r + 0.5, label, ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
    ax.set_xlim(-0.1, COLS + 0.1)
    ax.set_ylim(-0.1, ROWS + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(COLS) + 0.5)
    ax.set_xticklabels(range(COLS), fontsize=8)
    ax.set_yticks(np.arange(ROWS) + 0.5)
    ax.set_yticklabels(range(ROWS - 1, -1, -1), fontsize=8)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title("T-maze layout. S=start (indicator visible at t=0). "
                 "F+=top flag (indicator=+1).\n"
                 "F-=bottom flag (indicator=-1). T=T-junction (no indicator).",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 2. Agent paths (recurrent C, both indicators)
# ----------------------------------------------------------------------

def rollout_path(C, env: TMazeEnv, indicator: float):
    """Return list of (r, c) along greedy real-env rollout, plus h_C array."""
    obs = env.reset_to(indicator)
    h_C = np.zeros(C.hid_dim)
    path = [(env.r, env.c)]
    h_seq = [h_C.copy()]
    for _ in range(env.t_max):
        h_C, a_logit, _ = C.step_(obs, h_C)
        a = int(np.argmax(softmax(a_logit)))
        obs, r, done = env.step(a)
        path.append((env.r, env.c))
        h_seq.append(h_C.copy())
        if done:
            break
    return path, np.array(h_seq)


def plot_agent_paths(C, out_path: str):
    env = TMazeEnv(t_max=20)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
    for ax, ind, title_color, flag_rc, flag_label in [
        (axes[0], +1.0, "#d62728", TOP_FLAG_RC, "F+"),
        (axes[1], -1.0, "#2ca02c", BOT_FLAG_RC, "F-"),
    ]:
        path, _ = rollout_path(C, env, ind)
        for r in range(ROWS):
            for c in range(COLS):
                if is_walkable(r, c):
                    color = "#f5f5f5"
                else:
                    color = "#444"
                ax.add_patch(Rectangle((c, ROWS - 1 - r), 1, 1,
                                       facecolor=color, edgecolor="#aaa",
                                       linewidth=0.5))
        # path
        xs = [c + 0.5 for (_, c) in path]
        ys = [ROWS - 1 - r + 0.5 for (r, _) in path]
        ax.plot(xs, ys, "o-", color=title_color, linewidth=2.0, markersize=6,
                alpha=0.9)
        # arrows
        for i in range(len(path) - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            if abs(dx) + abs(dy) > 0.01:
                ax.annotate("", xy=(xs[i + 1], ys[i + 1]),
                            xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="->", color=title_color,
                                            alpha=0.6, lw=1.5))
        # labels
        ax.text(START_RC[1] + 0.5, ROWS - 1 - START_RC[0] + 0.5, "S",
                ha="center", va="center", fontsize=11, color="#1f77b4")
        ax.text(flag_rc[1] + 0.5, ROWS - 1 - flag_rc[0] + 0.5, flag_label,
                ha="center", va="center", fontsize=11, color=title_color)
        ax.set_xlim(-0.1, COLS + 0.1)
        ax.set_ylim(-0.1, ROWS + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # check success
        on_target = path[-1] == flag_rc
        success_str = "SUCCESS" if on_target else "FAIL"
        ax.set_title(f"indicator = {int(ind):+d}    {len(path) - 1} steps    "
                     f"[{success_str}]", fontsize=10)
    fig.suptitle("Recurrent C in real env (greedy rollouts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 3. Hidden-state trajectory (the latch)
# ----------------------------------------------------------------------

def plot_hidden_state(C, out_path: str):
    """Show C's hidden-state activations across steps for indicator=+1 vs -1.
    The visible difference is C's learned indicator latch.
    """
    env = TMazeEnv(t_max=20)
    _, h_pos = rollout_path(C, env, +1.0)
    _, h_neg = rollout_path(C, env, -1.0)
    T = min(h_pos.shape[0], h_neg.shape[0])
    h_pos = h_pos[:T]
    h_neg = h_neg[:T]
    diff = h_pos - h_neg

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=120, sharex=True)
    vmax = max(abs(h_pos).max(), abs(h_neg).max())
    for ax, mat, title in [
        (axes[0], h_pos.T, r"$h_C$ along trajectory  (indicator=+1)"),
        (axes[1], h_neg.T, r"$h_C$ along trajectory  (indicator=$-$1)"),
        (axes[2], diff.T,  r"$h_C^{+1} - h_C^{-1}$  (the indicator latch)"),
    ]:
        v = max(abs(mat).max(), 0.01)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                       origin="lower")
        ax.set_ylabel("hidden unit", fontsize=9)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[2].set_xlabel("time-step")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 4. Training curves
# ----------------------------------------------------------------------

def plot_training_curves(res: dict, out_path: str):
    p1 = res["phase1_losses"]
    p2 = res["phase2_history"]
    refresh = res.get("refresh_losses", [])
    cycle_eval = res.get("cycle_eval", [])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=120)

    # left: M's training MSE (phase 1 + refreshes)
    ax = axes[0]
    ax.plot(np.arange(1, len(p1) + 1), p1, color="#1f77b4", linewidth=0.6,
            label="phase 1 (random + scripted)")
    offset = len(p1)
    for i, rl in enumerate(refresh):
        xs = np.arange(offset, offset + len(rl)) + 1
        ax.plot(xs, rl, color="#9467bd", linewidth=0.6,
                label="refresh on C-rollouts" if i == 0 else None)
        offset += len(rl)
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("weighted MSE  (per output)")
    ax.set_title("World-model M training")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # middle: imagined return per controller iteration
    ax = axes[1]
    ax.plot(p2["iter"], p2["R"], color="#ff7f0e", linewidth=0.5,
            label="imagined R")
    ax.plot(p2["iter"], p2["objective"], color="#888", linewidth=0.3,
            label="objective (R + ent)")
    ax.set_xlabel("phase-2 iteration  (across cycles)")
    ax.set_ylabel("R / step  (sum gamma^t r_pred)")
    ax.set_title("Controller C imagined return")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # right: cycle-end real-env success
    ax = axes[2]
    cycles = [ce["cycle"] for ce in cycle_eval]
    successes = [ce["success"] for ce in cycle_eval]
    if cycles:
        ax.plot(cycles, successes, "o-", color="#2ca02c", linewidth=1.5,
                markersize=8, label="recurrent C (real env)")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.6,
               label="feedforward ceiling (50%)")
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=0.6,
               label="solved (100%)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("real-env success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Cycle-end eval success")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 5. Results summary table image
# ----------------------------------------------------------------------

def plot_results_table(res: dict, out_path: str):
    rows = [
        ("Recurrent C  (BPTT through M)", res["final_success"],
         res["final_mean_steps"]),
        ("Feed-forward C  (W_h = 0)",
         res.get("ff_success") if res.get("ff_success") is not None else float("nan"),
         res.get("ff_mean_steps") if res.get("ff_mean_steps") is not None else float("nan")),
        ("Random walk", res["random_success"], res["random_mean_steps"]),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.4), dpi=120)
    ax.axis("off")
    table_rows = [["agent", "success", "mean steps"]]
    for name, succ, steps in rows:
        if isinstance(succ, float) and np.isnan(succ):
            succ_str = "n/a"
            steps_str = "n/a"
        else:
            succ_str = f"{succ:.3f}"
            steps_str = f"{steps:.1f}"
        table_rows.append([name, succ_str, steps_str])
    tab = ax.table(cellText=table_rows, loc="center",
                   cellLoc="center", colWidths=[0.55, 0.2, 0.25])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.6)
    # header style
    for c in range(3):
        tab[0, c].set_facecolor("#dde")
        tab[0, c].set_text_props(weight="bold")
    fig.suptitle(f"pomdp-flag-maze eval (seed {res.get('_seed', '?')}, "
                 f"{res.get('_eval_eps', '?')} episodes)", fontsize=10)
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
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = RunConfig(seed=args.seed, run_baselines=not args.no_baseline)
    print(f"Training (seed={args.seed}) ...")
    res = run(cfg, verbose=False)
    res["_seed"] = args.seed
    res["_eval_eps"] = cfg.final_eval_eps
    print(f"  recurrent C: success = {res['final_success']:.3f}  "
          f"({res['final_mean_steps']:.1f} mean steps)")

    plot_maze_layout(os.path.join(args.outdir, "maze_layout.png"))
    plot_agent_paths(res["C"], os.path.join(args.outdir, "agent_paths.png"))
    plot_hidden_state(res["C"], os.path.join(args.outdir, "hidden_state.png"))
    plot_training_curves(res, os.path.join(args.outdir, "training_curves.png"))
    plot_results_table(res, os.path.join(args.outdir, "results_table.png"))


if __name__ == "__main__":
    main()
