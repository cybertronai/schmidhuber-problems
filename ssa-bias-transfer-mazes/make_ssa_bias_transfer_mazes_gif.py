"""Generate `ssa_bias_transfer_mazes.gif`.

The animation shows the SSA success-story stack evolving as training
progresses across the 4-task sequence. Each frame is a snapshot at one
modification event (push or pop). Three panels:

  left  : the maze, with current task's start/goal highlighted, plus the
          short trajectory of one rollout under the current (in-stack)
          policy.
  centre: vertical bar of "retained modifications", oldest at bottom.
          Each bar is coloured by the task that proposed it.
  right : reward-rate timeline showing the lifetime cumulative reward
          and a moving 50-step window rate, with the current event time
          marked.

≤2 MB target, 72 dpi, ~2-3 fps.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from ssa_bias_transfer_mazes import (
    train, TrainConfig, TASKS, MAZE, run_episode,
)


TASK_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def replay_to_event(seed: int, cfg: TrainConfig,
                    upto_event: int) -> tuple[np.ndarray, list, int]:
    """Re-run SSA training up to `upto_event` modification events. Returns
    the policy theta at that point, the stack at that point (list of
    (task_idx, time_pushed)), and (cum_time_at_event, cum_reward_at_event).

    Re-running is wasteful but lets us avoid pickling the full training
    trace. For 4 tasks * 200 eps, one full training pass is < 0.5 s, so
    even 100 frames takes under a minute.
    """
    raise NotImplementedError  # not used; we replay from a single trace


def collect_animation_data(seed: int, cfg: TrainConfig):
    """Return the SSA trace and reconstruct, frame-by-frame, the stack
    contents at the moment of each modification event."""
    trace = train("ssa", seed, cfg, quiet=True)

    stack_history: list[list[tuple[int, int]]] = []
    stack: list[tuple[int, int]] = []  # (task_idx, time_pushed)
    rates_window: list[float] = []
    rates_lifetime: list[float] = []
    rates_x: list[int] = []

    # We also want the cumulative reward at each event to plot the rate
    # timeline. mod_events stores (cum_time, cum_reward, task_idx, kind).
    for cum_t, cum_r, task_idx, kind in trace.mod_events:
        if kind == "push":
            stack.append((task_idx, cum_t))
        else:  # pop
            if stack:
                stack.pop()
        stack_history.append(list(stack))

        rates_lifetime.append(cum_r / max(cum_t, 1))
        rates_x.append(cum_t)

    return trace, stack_history, rates_x, rates_lifetime


def render_frame(frame_idx: int, trace, stack_history, rates_x,
                 rates_lifetime, n_tasks: int, dpi: int = 72) -> np.ndarray:
    """Render one frame of the animation at modification-event index
    `frame_idx`."""
    cum_time, cum_reward, task_idx, kind = trace.mod_events[frame_idx]
    stack = stack_history[frame_idx]

    fig = plt.figure(figsize=(10, 4.4), dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.4, 3.0],
                          wspace=0.25, left=0.05, right=0.97,
                          top=0.86, bottom=0.13)

    # ------- Panel 0: maze + current task ----------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(MAZE, cmap="binary", origin="upper", vmin=0, vmax=1)
    task = TASKS[task_idx]
    ax0.scatter([task.start[1]], [task.start[0]], marker="s", s=240,
                c="#009E73", edgecolors="black", lw=1.2)
    ax0.scatter([task.goal[1]], [task.goal[0]], marker="*", s=300,
                c=TASK_COLOURS[task_idx], edgecolors="black", lw=1.2)
    ax0.set_title(f"task {task_idx}: {task.name}", fontsize=10,
                  color=TASK_COLOURS[task_idx])
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_xticks(np.arange(MAZE.shape[1] + 1) - 0.5, minor=True)
    ax0.set_yticks(np.arange(MAZE.shape[0] + 1) - 0.5, minor=True)
    ax0.grid(which="minor", color="grey", linestyle="-", linewidth=0.4)

    # ------- Panel 1: stack ------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    if stack:
        ax1.set_ylim(-0.5, max(len(stack), 12) + 0.5)
        for j, (sj_task, sj_time) in enumerate(stack):
            rect = Rectangle((0.1, j), 0.8, 0.8,
                             facecolor=TASK_COLOURS[sj_task],
                             edgecolor="black", lw=0.8, alpha=0.85)
            ax1.add_patch(rect)
            ax1.text(0.5, j + 0.4, f"t={sj_time}",
                     ha="center", va="center", fontsize=7, color="white")
    else:
        ax1.set_ylim(-0.5, 12 + 0.5)
        ax1.text(0.5, 5, "(stack empty)", ha="center", va="center",
                 fontsize=9, color="grey")
    ax1.set_xlim(0, 1)
    ax1.set_xticks([])
    ax1.set_ylabel("modification age (oldest at bottom)", fontsize=8)
    ax1.set_title(
        f"success-story stack: {len(stack)} mods\n"
        f"event #{frame_idx + 1}: {kind} from task {task_idx}",
        fontsize=10,
    )

    # ------- Panel 2: lifetime rate timeline ------------
    ax2 = fig.add_subplot(gs[0, 2])
    rates_x_so_far = rates_x[: frame_idx + 1]
    rates_lt_so_far = rates_lifetime[: frame_idx + 1]
    ax2.plot(rates_x_so_far, rates_lt_so_far, lw=1.4, color="#0072B2",
             label="lifetime reward / step")
    ax2.axvline(cum_time, color="black", lw=0.8, alpha=0.5)
    ax2.scatter([cum_time], [cum_reward / max(cum_time, 1)],
                color=TASK_COLOURS[task_idx], s=40, zorder=5,
                edgecolors="black")

    # mark task boundaries on the timeline. Task t boundary = first
    # cum_time at which trace.ep_task crossed into t+1.
    ep_task = np.array(trace.ep_task)
    cum_time_arr = np.array(trace.cum_time)
    for t in range(1, n_tasks):
        idx = np.where(ep_task == t)[0]
        if len(idx):
            ax2.axvline(cum_time_arr[idx[0] - 1], color="grey",
                        linestyle="--", alpha=0.4)
    if rates_x:
        ax2.set_xlim(0, rates_x[-1])
    ax2.set_xlabel("environment step", fontsize=9)
    ax2.set_ylabel("avg reward per step", fontsize=9)
    ax2.set_title(
        f"lifetime average reward = {(cum_reward / max(cum_time, 1)):.3f} "
        f"  (R={cum_reward:.1f}, T={cum_time})",
        fontsize=10,
    )
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("ssa-bias-transfer-mazes — success-story stack over training",
                 fontsize=11)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def write_gif(frames, out_path: str, fps: int = 4):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ssa_bias_transfer_mazes.gif")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=80,
                   help="Cap on rendered frames; events are subsampled "
                        "uniformly to fit.")
    p.add_argument("--dpi", type=int, default=60)
    p.add_argument("--hold-frames", type=int, default=4)
    args = p.parse_args()

    cfg = TrainConfig()
    print(f"Training SSA (seed {args.seed})...")
    trace, stack_history, rates_x, rates_lifetime = collect_animation_data(
        args.seed, cfg)
    print(f"  {len(trace.mod_events)} modification events captured")

    n_events = len(trace.mod_events)
    if n_events == 0:
        raise RuntimeError("no modification events to animate")
    if n_events > args.max_frames:
        idxs = np.linspace(0, n_events - 1, args.max_frames).astype(int)
    else:
        idxs = np.arange(n_events)
    print(f"  rendering {len(idxs)} frames at dpi={args.dpi}...")

    frames = []
    for k, i in enumerate(idxs):
        frames.append(render_frame(i, trace, stack_history, rates_x,
                                   rates_lifetime, cfg.n_tasks, dpi=args.dpi))
        if (k + 1) % 20 == 0:
            print(f"    rendered {k + 1} / {len(idxs)} frames")
    for _ in range(args.hold_frames):
        frames.append(frames[-1])

    write_gif(frames, args.out, fps=args.fps)
    sz_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {sz_kb:.1f} KB)")
    if sz_kb > 2048:
        print("WARNING: GIF over 2 MB target; lower --dpi or --max-frames.")


if __name__ == "__main__":
    main()
