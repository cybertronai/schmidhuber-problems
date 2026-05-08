"""Render an animated GIF of the trained ESP-evolved net balancing the
double cart-pole.

Frame layout:
    Top:    cart on track + long pole + short pole + action arrow,
            failure markers at +/- X_LIMIT
    Bottom: x and theta_1, theta_2 traces up to current frame

Usage:
    python3 make_double_pole_no_velocity_gif.py --seed 0
    python3 make_double_pole_no_velocity_gif.py --seed 0 --T-max 600 --fps 30
"""
from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from PIL import Image

from double_pole_no_velocity import (
    RunConfig, run, assemble_network,
    init_state, normalize_obs, double_pole_step, is_failed,
    F_MAX, X_LIMIT, THETA_LIMIT, L_HALF_1, L_HALF_2,
)


def render_frame(state: np.ndarray, action: float,
                 history_t: np.ndarray, history_x: np.ndarray,
                 history_t1: np.ndarray, history_t2: np.ndarray,
                 t_idx: int, T_max: int) -> Image.Image:
    fig = plt.figure(figsize=(8, 5.6), dpi=85)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.35)

    # ---- top: cart + two poles ----
    ax = fig.add_subplot(gs[0, 0])
    x, _, t1, _, t2, _ = state
    cart_w, cart_h = 0.4, 0.22

    # track
    ax.axhline(0.0, color="#444", linewidth=2)
    for xl in (-X_LIMIT, X_LIMIT):
        ax.plot([xl, xl], [-0.1, 0.1], color="red", linewidth=2)
    # cart
    ax.add_patch(Rectangle((x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                           facecolor="#1f77b4", edgecolor="black",
                           linewidth=1.0))
    # long pole
    p1_top_x = x + 2 * L_HALF_1 * np.sin(t1)
    p1_top_y = 2 * L_HALF_1 * np.cos(t1)
    ax.plot([x, p1_top_x], [0.0, p1_top_y], color="#d62728",
            linewidth=4, solid_capstyle="round")
    ax.plot(p1_top_x, p1_top_y, "o", color="#d62728", markersize=8)
    # short pole
    p2_top_x = x + 2 * L_HALF_2 * np.sin(t2)
    p2_top_y = 2 * L_HALF_2 * np.cos(t2)
    ax.plot([x, p2_top_x], [0.0, p2_top_y], color="#9467bd",
            linewidth=3, solid_capstyle="round")
    ax.plot(p2_top_x, p2_top_y, "o", color="#9467bd", markersize=6)
    # pivot
    ax.plot(x, 0.0, "o", color="black", markersize=4)
    # action arrow
    if abs(action) > 0.01:
        arrow_len = 0.5 * action
        ax.add_patch(FancyArrow(
            x, -cart_h / 2 - 0.15, arrow_len, 0.0,
            head_width=0.07, head_length=0.06, length_includes_head=True,
            color="#2ca02c", linewidth=1.2))
    # info
    ax.text(0.02, 0.96,
            f"step {t_idx:4d} / {T_max}    "
            f"x = {x:+5.2f}    "
            r"$\theta_1$ = " + f"{np.degrees(t1):+6.2f}" + r"$^\circ$    "
            r"$\theta_2$ = " + f"{np.degrees(t2):+6.2f}" + r"$^\circ$    "
            f"u = {action:+.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            family="monospace")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.55, 1.35)
    ax.set_aspect("equal")
    ax.set_xticks([-2.4, -1.2, 0.0, 1.2, 2.4])
    ax.set_yticks([])
    ax.set_title("double-pole-no-velocity: ESP-evolved recurrent net "
                 "(positions only)", fontsize=10)

    # ---- bottom: traces so far ----
    ax = fig.add_subplot(gs[1, 0])
    if len(history_t) > 1:
        ax.plot(history_t, history_x, color="#1f77b4", linewidth=0.7,
                label="x (m)")
        ax.plot(history_t, np.degrees(history_t1), color="#d62728",
                linewidth=0.7, label=r"$\theta_1$ (deg)")
        ax.plot(history_t, np.degrees(history_t2), color="#9467bd",
                linewidth=0.7, label=r"$\theta_2$ (deg)")
    ax.axhline(np.degrees(THETA_LIMIT), color="red", linestyle="--",
               linewidth=0.5, alpha=0.5)
    ax.axhline(-np.degrees(THETA_LIMIT), color="red", linestyle="--",
               linewidth=0.5, alpha=0.5)
    ax.set_xlim(0.0, T_max * 0.01)
    ax.set_ylim(-np.degrees(THETA_LIMIT) - 5, np.degrees(THETA_LIMIT) + 5)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gen", type=int, default=200)
    p.add_argument("--T-max", type=int, default=400,
                   help="number of frames in the GIF (= simulation steps)")
    p.add_argument("--frame-stride", type=int, default=4,
                   help="render every N-th simulation step")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out", type=str, default="double_pole_no_velocity.gif")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed, max_generations=args.max_gen)
    print(f"Running evolution (seed={cfg.seed})")
    res = run(cfg, verbose=True)
    net = assemble_network(res["evolve"]["best_genomes"])

    state = init_state(4.5 * np.pi / 180.0)
    h = np.zeros(cfg.hidden)
    history_t, history_x, history_t1, history_t2 = [], [], [], []
    frames = []
    last_action = 0.0

    print(f"Rendering up to {args.T_max} steps "
          f"(every {args.frame_stride}-th step) at {args.fps} fps...")
    for t in range(args.T_max):
        x_obs = normalize_obs(state)
        pre = net["W_x"] @ x_obs + net["W_h"] @ h + net["b"]
        h = np.tanh(pre)
        u_pre = (net["V"] @ h + net["c"]).item()
        u = float(np.tanh(u_pre))
        last_action = u
        state = double_pole_step(state, u * F_MAX)
        history_t.append((t + 1) * 0.01)
        history_x.append(state[0])
        history_t1.append(state[2])
        history_t2.append(state[4])
        if t % args.frame_stride == 0:
            frames.append(render_frame(
                state, u, np.array(history_t), np.array(history_x),
                np.array(history_t1), np.array(history_t2),
                t + 1, args.T_max))
        if is_failed(state):
            print(f"  failed at step {t + 1}")
            break

    if not frames:
        raise RuntimeError("no frames rendered")
    print(f"  saving {len(frames)} frames -> {args.out}")
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"  GIF size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
