"""Generate curiosity_three_regions.gif.

Two-panel animation showing, frame-by-frame across the run:
  Left panel:  cumulative visit counts per region as a bar chart.
  Right panel: per-region curiosity signal (max(0, error reduction)) as
               three lines, drawn up to the current step.

The frames are sampled along a log-spaced index so the early dynamics
(when curiosity is shifting fastest) get more frames than the late steady
state. Output is ~30-50 frames at 12 fps -- well under 2 MB for the
default settings.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from curiosity_three_regions import run_experiment


COLORS = ["#3b82f6", "#f59e0b", "#10b981"]
SHORT = ["A: det", "B: rand", "C: learn"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--out", type=str, default="curiosity_three_regions.gif")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    res = run_experiment(seed=args.seed, steps=args.steps)
    cfg = res["config"]
    chosen = np.asarray(res["chosen"])
    cur_log = np.asarray(res["cur_log"])

    cum = np.zeros((args.steps, 3), dtype=np.int64)
    for t in range(args.steps):
        if t > 0:
            cum[t] = cum[t - 1]
        cum[t, chosen[t]] += 1

    # log-spaced + a few linear at the end so the steady state shows up too
    early = np.unique(np.geomspace(20, args.steps // 2, args.frames // 2).astype(int))
    late = np.linspace(args.steps // 2, args.steps - 1, args.frames - len(early)).astype(int)
    frame_t = np.unique(np.concatenate([early, late]))

    fig, (ax_visits, ax_cur) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: bar chart of visits
    bars = ax_visits.bar(SHORT, [0, 0, 0], color=COLORS)
    ax_visits.set_ylabel("cumulative visits")
    ax_visits.set_ylim(0, args.steps)
    title_visits = ax_visits.set_title("")

    # Right: curiosity over time
    cur_lines = []
    for i in range(3):
        line, = ax_cur.plot([], [], color=COLORS[i], label=SHORT[i], linewidth=1.2)
        cur_lines.append(line)
    ax_cur.set_xlim(0, args.steps)
    cmax = max(cur_log.max() * 1.1, 1e-3)
    ax_cur.set_ylim(0, cmax)
    ax_cur.set_xlabel("step")
    ax_cur.set_ylabel("curiosity")
    ax_cur.set_title("Per-region curiosity")
    ax_cur.axvline(cfg["burn_in"], color="red", linestyle="--",
                   alpha=0.5, label="burn-in end")
    ax_cur.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "curiosity-three-regions  (Schmidhuber 1991)",
        y=1.02, fontsize=11,
    )

    def update(frame_idx):
        t = int(frame_t[frame_idx])
        for i, b in enumerate(bars):
            b.set_height(int(cum[t - 1, i]))
        a, br, cr = (int(cum[t - 1, i]) for i in range(3))
        title_visits.set_text(
            f"step {t}/{args.steps}\nvisits  A={a}  B={br}  C={cr}"
        )
        for i, line in enumerate(cur_lines):
            line.set_data(np.arange(t), cur_log[:t, i])
        return list(bars) + cur_lines + [title_visits]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_t),
        interval=1000 // args.fps, blit=False,
    )
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {args.out}  ({len(frame_t)} frames)")


if __name__ == "__main__":
    main()
