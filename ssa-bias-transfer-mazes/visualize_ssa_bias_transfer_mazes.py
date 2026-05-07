"""Static visualisations for ssa-bias-transfer-mazes.

Outputs (in `viz/`):
  per_task_steps.png       — bar chart: tail mean steps per task across the
                             three regimes (ssa, no_ssa, restart).
  per_task_solve.png       — bar chart: tail solve rate per task across the
                             three regimes.
  learning_curves.png      — episode-by-episode steps-to-goal for each
                             regime, with task boundaries marked. Smoothed
                             over a 20-episode window.
  stack_evolution.png      — number of retained modifications on the SSA
                             stack over time, with task boundaries.
  pop_timeline.png         — every push and pop event coloured by the task
                             that owned the modification. Lets you see which
                             task's modifications survive after later tasks.
  multi_seed_solve.png     — solve-rate summary across 10 seeds (the
                             headline plot for the bias-transfer claim).

Pure numpy + matplotlib. Run after a clean training pass.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from ssa_bias_transfer_mazes import (
    run_all, TrainConfig, TASKS, MAZE,
)


REGIME_COLOURS = {"ssa": "#0072B2", "no_ssa": "#D55E00", "restart": "#009E73"}
REGIME_LABEL = {"ssa": "SSA (filtered)",
                "no_ssa": "no-SSA (continual, raw)",
                "restart": "random restart per task"}


# ----------------------------------------------------------------------
# Per-task summaries
# ----------------------------------------------------------------------

def plot_per_task_bars(summary: dict, out_path: str, key: str, title: str):
    n_tasks = summary["config"]["n_tasks"]
    regimes = ("ssa", "no_ssa", "restart")
    width = 0.27
    x = np.arange(n_tasks)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=120)
    for i, r in enumerate(regimes):
        vals = [summary["results"][r]["per_task"][t][key] for t in range(n_tasks)]
        ax.bar(x + (i - 1) * width, vals, width=width,
               color=REGIME_COLOURS[r], label=REGIME_LABEL[r])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{TASKS[i].name}" for i in range(n_tasks)])
    ax.set_xlabel("task index (executed in order)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if key == "tail_mean_steps":
        ax.set_ylabel("steps to goal\n(mean over last 20% of each task's episodes)")
    elif key == "tail_solve_rate":
        ax.set_ylabel("solve rate\n(fraction reaching goal in last 20% of eps)")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Learning curves
# ----------------------------------------------------------------------

def smooth(arr, w=20):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    pad = np.concatenate([np.full(w - 1, arr[0]), arr])
    out = np.convolve(pad, np.ones(w) / w, mode="valid")
    return out


def plot_learning_curves(traces: dict, summary: dict, out_path: str):
    n_tasks = summary["config"]["n_tasks"]
    eps_per_task = summary["config"]["episodes_per_task"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=120)
    for r, trace in traces.items():
        steps = np.array(trace.ep_steps, dtype=float)
        ax.plot(np.arange(len(steps)), smooth(steps, 20),
                color=REGIME_COLOURS[r], label=REGIME_LABEL[r], lw=1.4)
    for t in range(1, n_tasks):
        ax.axvline(t * eps_per_task, color="grey",
                   linestyle="--", alpha=0.5)
    for t in range(n_tasks):
        ax.text((t + 0.5) * eps_per_task, ax.get_ylim()[1] * 0.95,
                TASKS[t].name, ha="center", va="top", fontsize=8,
                color="grey")
    ax.set_xlabel("episode (across all tasks, in order)")
    ax.set_ylabel("steps to goal (smoothed, w=20)")
    ax.set_title("Learning curves across the 4-task sequence")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Stack evolution + pop timeline (SSA only)
# ----------------------------------------------------------------------

def plot_stack_evolution(ssa_trace, summary: dict, out_path: str):
    """Stack size over time, with push/pop marks coloured by task."""
    eps_per_task = summary["config"]["episodes_per_task"]
    n_tasks = summary["config"]["n_tasks"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=120)
    if not ssa_trace.mod_events:
        ax.text(0.5, 0.5, "no modification events recorded", ha="center")
        fig.savefig(out_path); plt.close(fig); return

    times = [ev[0] for ev in ssa_trace.mod_events]
    sizes = []
    sz = 0
    for _, _, _, kind in ssa_trace.mod_events:
        sz += (1 if kind == "push" else -1)
        sizes.append(sz)
    ax.step(times, sizes, where="post", color="#0072B2", lw=1.5,
            label="retained modifications")

    task_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for time, _, task_idx, kind in ssa_trace.mod_events:
        if kind == "pop":
            ax.scatter([time], [0], marker="x", s=40,
                       color=task_colours[task_idx % len(task_colours)],
                       alpha=0.5)

    # mark task boundaries: cumulative time at the END of task i = sum of
    # ep_steps for episodes whose task <= i
    ep_task = np.array(ssa_trace.ep_task)
    cum_time_arr = np.array(ssa_trace.cum_time)
    for t in range(1, n_tasks):
        idx = np.where(ep_task == t)[0]
        if len(idx):
            tt = cum_time_arr[idx[0] - 1]
            ax.axvline(tt, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("environment step (cumulative across tasks)")
    ax.set_ylabel("# modifications on the success-story stack")
    ax.set_title("SSA stack evolution — retained modifications over time")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pop_timeline(ssa_trace, summary: dict, out_path: str):
    """Each push/pop event as a coloured tick at its timestamp."""
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=120)
    task_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    n_tasks = summary["config"]["n_tasks"]

    pushes_x, pushes_c = [], []
    pops_x, pops_c = [], []
    for time, _, task_idx, kind in ssa_trace.mod_events:
        c = task_colours[task_idx % len(task_colours)]
        if kind == "push":
            pushes_x.append(time); pushes_c.append(c)
        else:
            pops_x.append(time); pops_c.append(c)
    if pushes_x:
        ax.scatter(pushes_x, np.full(len(pushes_x), 1), c=pushes_c, marker="^",
                   s=22, alpha=0.7, label="push")
    if pops_x:
        ax.scatter(pops_x, np.full(len(pops_x), 0), c=pops_c, marker="v",
                   s=22, alpha=0.7, label="pop")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["pop (rolled back)", "push (proposed)"])
    ax.set_xlabel("environment step")
    cum_time_arr = np.array(ssa_trace.cum_time)
    ep_task = np.array(ssa_trace.ep_task)
    for t in range(1, n_tasks):
        idx = np.where(ep_task == t)[0]
        if len(idx):
            tt = cum_time_arr[idx[0] - 1]
            ax.axvline(tt, color="grey", linestyle="--", alpha=0.5)
    handles = [plt.Line2D([], [], marker="o", color=task_colours[t], lw=0,
                          label=f"task {t} ({TASKS[t].name})")
               for t in range(n_tasks)]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_title("Modification events coloured by the task that proposed them")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Maze layout reference
# ----------------------------------------------------------------------

def plot_maze_layout(out_path: str):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.0 * len(TASKS), 3.2),
                             dpi=120)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, task in zip(axes, TASKS):
        ax.imshow(MAZE, cmap="binary", origin="upper", vmin=0, vmax=1)
        sr, sc = task.start
        gr, gc = task.goal
        ax.scatter([sc], [sr], marker="s", s=240, c="#009E73",
                   edgecolors="black", lw=1.2, label="start")
        ax.scatter([gc], [gr], marker="*", s=300, c="#D55E00",
                   edgecolors="black", lw=1.2, label="goal")
        ax.set_title(task.name)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(MAZE.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(MAZE.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.4)
    fig.suptitle("Same maze, four tasks (different goal cell). Start = centre.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Multi-seed solve rate summary
# ----------------------------------------------------------------------

def plot_multi_seed_solve(out_path: str, n_seeds: int = 10):
    print(f"  running multi-seed sweep across {n_seeds} seeds...")
    cfg = TrainConfig()
    n_tasks = cfg.n_tasks
    agg = {r: [[] for _ in range(n_tasks)]
           for r in ("ssa", "no_ssa", "restart")}
    for seed in range(n_seeds):
        summary, *_ = run_all(seed, cfg, quiet=True)
        for r in ("ssa", "no_ssa", "restart"):
            for t in range(n_tasks):
                agg[r][t].append(
                    summary["results"][r]["per_task"][t]["tail_solve_rate"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)
    width = 0.27
    x = np.arange(n_tasks)
    for i, r in enumerate(("ssa", "no_ssa", "restart")):
        means = [np.mean(agg[r][t]) for t in range(n_tasks)]
        sems = [np.std(agg[r][t]) / np.sqrt(n_seeds) for t in range(n_tasks)]
        ax0.bar(x + (i - 1) * width, means, width=width, yerr=sems,
                color=REGIME_COLOURS[r], label=REGIME_LABEL[r],
                error_kw={"linewidth": 0.8})
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{i}\n{TASKS[i].name}" for i in range(n_tasks)])
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("tail solve rate (mean over 10 seeds, error = SEM)")
    ax0.set_title(f"Tail solve rate per task across {n_seeds} seeds")
    ax0.legend(loc="lower left", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    # cumulative solve count
    cumulative = {r: 0.0 for r in ("ssa", "no_ssa", "restart")}
    cum_curves = {r: [] for r in ("ssa", "no_ssa", "restart")}
    for r in ("ssa", "no_ssa", "restart"):
        c = 0.0
        for t in range(n_tasks):
            c += np.mean(agg[r][t])
            cum_curves[r].append(c)
        ax1.plot(np.arange(1, n_tasks + 1), cum_curves[r],
                 marker="o", color=REGIME_COLOURS[r], label=REGIME_LABEL[r])
    ax1.set_xlabel("number of tasks completed")
    ax1.set_ylabel("cumulative tail solve rate (sum over tasks)")
    ax1.set_title("Cumulative solve rate over the task sequence")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

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
    p.add_argument("--multi-seed", type=int, default=10,
                   help="Number of seeds for the aggregate solve-rate plot.")
    p.add_argument("--no-multi-seed", action="store_true",
                   help="Skip the multi-seed sweep (faster).")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = TrainConfig()
    summary, ssa_trace, nossa_trace, restart_trace = run_all(
        args.seed, cfg, quiet=True)
    print(f"trained in {summary['elapsed_sec']:.2f} s "
          f"(seed {args.seed}, {cfg.n_tasks} tasks x "
          f"{cfg.episodes_per_task} eps each)")
    traces = {"ssa": ssa_trace, "no_ssa": nossa_trace, "restart": restart_trace}

    plot_maze_layout(os.path.join(args.outdir, "maze_layout.png"))
    plot_per_task_bars(summary,
                       os.path.join(args.outdir, "per_task_steps.png"),
                       key="tail_mean_steps",
                       title=f"Per-task mean steps to goal (seed {args.seed})")
    plot_per_task_bars(summary,
                       os.path.join(args.outdir, "per_task_solve.png"),
                       key="tail_solve_rate",
                       title=f"Per-task solve rate (seed {args.seed})")
    plot_learning_curves(traces, summary,
                         os.path.join(args.outdir, "learning_curves.png"))
    plot_stack_evolution(ssa_trace, summary,
                         os.path.join(args.outdir, "stack_evolution.png"))
    plot_pop_timeline(ssa_trace, summary,
                      os.path.join(args.outdir, "pop_timeline.png"))

    if not args.no_multi_seed:
        plot_multi_seed_solve(
            os.path.join(args.outdir, "multi_seed_solve.png"),
            n_seeds=args.multi_seed,
        )


if __name__ == "__main__":
    main()
