"""Make world_models_carracing.gif: animate one greedy rollout of the trained
V+M+C policy.

Two-panel layout:
  left   - top-down 2-D track + centerline + the car (triangle) and trail
           (color-graded over time)
  right  - the 16x16 birds-eye observation the V network is currently seeing,
           plus the latent z bars and cumulative reward.

If run.json does not exist, runs world_models_carracing.py once with seed=0.
Output: world_models_carracing.gif (target ≤ 2 MB).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        print("[gif] run.json not found, running world_models_carracing.py ...")
        subprocess.run(
            ["python3", os.path.join(HERE, "world_models_carracing.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )
    with open(json_path) as f:
        out = json.load(f)

    track = out["track"]
    mask = np.array(track["mask"], dtype=np.float32)
    cx = np.array(track["centerline_x"])
    cy = np.array(track["centerline_y"])
    E = track["grid_extent"]

    demo = out["demo"]
    states = np.array(demo["demo_states"])  # (T+1, 4)
    obs_seq = np.array(demo["demo_obs"])     # (T+1, 256)
    z_seq = np.array(demo["demo_z"])         # (T, z_dim)
    actions = np.array(demo["demo_actions"])  # (T, 2)
    rand = out["random_baseline"]
    final = out["final_eval"]

    # subsample frames if too long for ≤2 MB target
    T = states.shape[0]
    stride = max(1, T // 80)  # cap at ~80 frames
    frame_idx = list(range(0, T, stride))
    if frame_idx[-1] != T - 1:
        frame_idx.append(T - 1)
    n_frames = len(frame_idx)

    fig = plt.figure(figsize=(11.0, 5.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.4],
                          width_ratios=[2.4, 1.0, 1.0, 1.0],
                          hspace=0.4, wspace=0.35)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_obs = fig.add_subplot(gs[0:2, 1:3])
    ax_z = fig.add_subplot(gs[2, 1:3])
    ax_rew = fig.add_subplot(gs[:, 3])

    # static: track + centerline
    ax_world.imshow(mask, extent=[-E, E, -E, E], origin="lower",
                    cmap="Greys", alpha=0.55)
    ax_world.plot(cx, cy, color="#bbbbbb", linewidth=0.7)
    ax_world.set_xlim(-E, E); ax_world.set_ylim(-E, E)
    ax_world.set_aspect("equal")
    ax_world.set_title("V+M+C controller on numpy 2-D track", fontsize=11)
    ax_world.set_xticks([]); ax_world.set_yticks([])
    car_dot, = ax_world.plot([], [], "o", color="#d4694e", markersize=10,
                             zorder=5, markeredgecolor="black",
                             markeredgewidth=1.0)
    trail_line, = ax_world.plot([], [], "-", color="#d4694e", linewidth=1.8,
                                alpha=0.85)

    # arrow for heading
    heading_arrow = ax_world.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#5a9bd4", lw=1.4),
    )

    # cumulative reward over time (right panel)
    cum_rewards = np.zeros(T)
    # demo info doesn't carry per-step reward; recompute from final cum + states
    # Approximate per-step progress along centerline:
    cur_s = []
    for k in range(T):
        x, y = states[k, 0], states[k, 1]
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        i = int(np.argmin(d2))
        cur_s.append(i / cx.shape[0])
    cur_s = np.array(cur_s)
    # signed deltas
    progress = np.zeros(T)
    for k in range(1, T):
        d = (cur_s[k] - cur_s[k - 1]) % 1.0
        if d > 0.5:
            d -= 1.0
        progress[k] = progress[k - 1] + 30.0 * d

    ax_rew.set_xlim(0, T)
    ax_rew.set_ylim(progress.min() - 5, max(10, progress.max()) + 5)
    ax_rew.set_xlabel("step")
    ax_rew.set_ylabel("cum reward (progress)")
    ax_rew.set_title("cumulative reward", fontsize=10)
    ax_rew.axhline(rand["mean_return"], color="#999",
                   linestyle="--", linewidth=1,
                   label=f"random ({rand['mean_return']:+.1f})")
    ax_rew.grid(alpha=0.3)
    rew_line, = ax_rew.plot([], [], color="#d4694e", linewidth=1.8,
                            label="V+M+C")
    ax_rew.legend(fontsize=8, loc="upper left")

    # observation
    obs_im = ax_obs.imshow(obs_seq[0].reshape(16, 16),
                           cmap="Greys", vmin=0, vmax=1, origin="lower")
    ax_obs.set_xticks([]); ax_obs.set_yticks([])
    ax_obs.set_title("16×16 obs (car-rotated, forward = up)", fontsize=10)
    # add a "car" marker at the patch center
    ax_obs.plot([7.5], [7.5], marker="^", markersize=14,
                markerfacecolor="#d4694e", markeredgecolor="black",
                markeredgewidth=1.0)

    # latent z bars
    z_dim = z_seq.shape[1] if z_seq.shape[0] > 0 else 16
    z_bars = ax_z.bar(np.arange(z_dim), np.zeros(z_dim),
                      color="#5a9bd4", width=0.85)
    ax_z.set_xlim(-0.6, z_dim - 0.4)
    z_lo, z_hi = z_seq.min() - 0.5 if z_seq.size else -1, z_seq.max() + 0.5 if z_seq.size else 1
    ax_z.set_ylim(z_lo, z_hi)
    ax_z.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax_z.set_title(f"latent z (R^{z_dim})", fontsize=10)
    ax_z.set_xticks([])

    text_box = ax_world.text(
        0.02, 0.98, "", transform=ax_world.transAxes,
        ha="left", va="top", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85)
    )

    def update(i):
        k = frame_idx[i]
        x, y, theta, v = states[k, 0], states[k, 1], states[k, 2], states[k, 3]
        car_dot.set_data([x], [y])
        trail_line.set_data(states[:k + 1, 0], states[:k + 1, 1])
        # heading arrow
        ax_len = 1.0
        heading_arrow.xy = (x + ax_len * np.cos(theta),
                            y + ax_len * np.sin(theta))
        heading_arrow.set_position((x, y))
        # obs: there are T+1 obs total
        if k < obs_seq.shape[0]:
            obs_im.set_data(obs_seq[k].reshape(16, 16))
        # z bars: only T entries (one per action step)
        if k < z_seq.shape[0]:
            for j, b in enumerate(z_bars):
                b.set_height(z_seq[k, j])
        rew_line.set_data(np.arange(k + 1), progress[:k + 1])
        a_str = (f"steer={actions[min(k, actions.shape[0]-1), 0]:+.2f} "
                 f"throttle={actions[min(k, actions.shape[0]-1), 1]:+.2f}"
                 if actions.shape[0] > 0 else "")
        text_box.set_text(
            f"t = {k:3d} / {T - 1}\n"
            f"v = {v:.2f}\n"
            f"cum R = {progress[k]:+.1f}\n"
            f"{a_str}"
        )
        return [car_dot, trail_line, obs_im, rew_line, text_box, *z_bars]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False,
                         interval=80)
    out_gif = os.path.join(HERE, "world_models_carracing.gif")
    anim.save(out_gif, writer=PillowWriter(fps=12))
    plt.close(fig)
    size = os.path.getsize(out_gif) / (1024 * 1024)
    print(f"Wrote {out_gif} ({size:.2f} MB, {n_frames} frames)")


if __name__ == "__main__":
    main()
