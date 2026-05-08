"""Build self_referential_weight_matrix.gif.

For each of the 4 tasks, run one episode with the trained model and animate:
  * left   : the demo and query inputs as text
  * middle : W_fast at every time step (heatmap)
  * right  : write_gate * write_value bar across time + the prediction at
             each query step

The GIF is concatenated across the 4 tasks so the viewer can see the same
slow weights producing different W_fast trajectories per task.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from self_referential_weight_matrix import (
    SRWM, TASK_NAMES, make_episode, train,
)


def make_gif(seed: int, out_path: str, n_episodes: int = 3000, n_h: int = 6,
             eta: float = 0.5, lr: float = 0.01, quick: bool = False, fps: int = 2):
    print("training for GIF ...")
    model, history, summary = train(
        seed=seed, n_episodes=n_episodes, n_h=n_h, eta=eta, lr=lr,
        quick=quick, verbose=False,
    )
    print(f"  final acc {summary['final_overall_acc']:.3f}")

    # Run one episode per task and collect frames.
    episodes = []
    rng = np.random.default_rng(seed + 12_345)
    for task_id in range(4):
        inputs, targets, is_query = make_episode(rng, task_id)
        ys = model.episode(inputs)
        episodes.append({
            "task_id": task_id,
            "inputs": inputs,
            "targets": targets,
            "is_query": is_query,
            "ys": ys,
            "fast_history": [m.copy() for m in model.fast_history],
            "rows": np.stack([tp["row"] for tp in model.tape]),
            "cols": np.stack([tp["col"] for tp in model.tape]),
            "vals": np.array([float(tp["val"][0]) for tp in model.tape]),
            "gates": np.array([float(tp["gate"][0]) for tp in model.tape]),
        })

    # Determine global colour scale.
    all_w = np.concatenate([np.stack(ep["fast_history"]) for ep in episodes], axis=0)
    vmax = float(np.max(np.abs(all_w))) + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [0.9, 1.1, 1.4]})
    plt.subplots_adjust(top=0.85)

    # Each task contributes T+1 = 9 frames; 4 tasks -> 36 frames.
    T = 8
    frames = []
    for k, ep in enumerate(episodes):
        for t in range(T + 1):
            frames.append((k, t))

    # Pre-create stable artists.
    ax_text, ax_w, ax_bars = axes
    im = ax_w.imshow(np.zeros((n_h, n_h)), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_w.set_xticks([]); ax_w.set_yticks([])
    cbar = plt.colorbar(im, ax=ax_w, fraction=0.046)
    cbar.set_label("W_fast entry", fontsize=8)

    bars = ax_bars.bar(range(T), np.zeros(T), color="C0")
    pred_dots = ax_bars.scatter([], [], s=40, color="black", zorder=5)
    target_dots = ax_bars.scatter([], [], s=40, marker="x", color="C3", zorder=5)
    ax_bars.set_xlim(-0.5, T - 0.5)
    ax_bars.set_ylim(-1.1, 1.4)
    ax_bars.axhline(0, color="black", lw=0.5)
    ax_bars.axvspan(-0.5, 3.5, color="grey", alpha=0.1)
    ax_bars.set_xlabel("t"); ax_bars.grid(alpha=0.3)

    text_artists = ax_text.text(
        0.02, 0.98, "", transform=ax_text.transAxes,
        fontfamily="monospace", fontsize=9, va="top",
    )
    ax_text.axis("off")

    def fmt_episode_text(ep, t_now):
        lines = [f"Task: {TASK_NAMES[ep['task_id']]}", "", "demo:"]
        for t in range(4):
            x0, x1 = ep["inputs"][t][:2]
            yl = ep["inputs"][t][2]
            arrow = "  <-- now" if t == t_now - 1 else ""
            lines.append(f"  t={t}  x=({x0:+.0f},{x1:+.0f})  y={yl:+.0f}{arrow}")
        lines.append("")
        lines.append("query:")
        for t in range(4, 8):
            x0, x1 = ep["inputs"][t][:2]
            tgt = int(ep["targets"][t])
            pr = ep["ys"][t]
            shown = ""
            if t < t_now:
                shown = f"  pred={pr:.2f} target={tgt}"
                if (pr > 0.5) == bool(tgt):
                    shown += " OK"
                else:
                    shown += " ERR"
            arrow = "  <-- now" if t == t_now - 1 else ""
            lines.append(f"  t={t}  x=({x0:+.0f},{x1:+.0f}){shown}{arrow}")
        return "\n".join(lines)

    def update(frame_idx):
        k, t = frames[frame_idx]
        ep = episodes[k]
        im.set_data(ep["fast_history"][t])
        ax_w.set_title(
            f"W_fast (post-step {t-1})" if t > 0 else "W_fast (pre-episode)",
            fontsize=10,
        )
        # Bar plot: cumulative gate*val up to current time, others greyed.
        write_strength = ep["gates"] * ep["vals"]
        for i, b in enumerate(bars):
            if i < t:
                v = write_strength[i]
                b.set_height(v)
                b.set_color("C0" if i < 4 else "C2")
                b.set_alpha(1.0)
            else:
                b.set_height(0.0)
        # Predictions at past query steps.
        pred_x, pred_y, tgt_x, tgt_y = [], [], [], []
        for tt in range(min(t, 8)):
            if ep["is_query"][tt]:
                pred_x.append(tt); pred_y.append(2.0 * ep["ys"][tt] - 1.0)
                tgt_x.append(tt); tgt_y.append(2.0 * ep["targets"][tt] - 1.0)
        pred_dots.set_offsets(np.column_stack([pred_x, pred_y]) if pred_x else np.empty((0, 2)))
        target_dots.set_offsets(np.column_stack([tgt_x, tgt_y]) if tgt_x else np.empty((0, 2)))
        ax_bars.set_title(
            "writes (blue=demo, green=query)\n"
            "dots=pred (•) / target (x), rescaled to [-1,1]",
            fontsize=9,
        )
        text_artists.set_text(fmt_episode_text(ep, t))
        fig.suptitle(
            f"Self-referential weight matrix — {TASK_NAMES[ep['task_id']]} "
            f"(episode {k+1}/4)",
            y=0.97,
        )
        return [im, text_artists, pred_dots, target_dots] + list(bars)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps,
                         blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"writing {out_path} ...")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=3000)
    parser.add_argument("--n-h", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default="self_referential_weight_matrix.gif")
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    make_gif(seed=args.seed, out_path=out, n_episodes=args.n_episodes,
             n_h=args.n_h, eta=args.eta, lr=args.lr, quick=args.quick,
             fps=args.fps)


if __name__ == "__main__":
    main()
