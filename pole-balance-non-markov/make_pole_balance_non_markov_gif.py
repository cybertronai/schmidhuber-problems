"""Render an animated GIF of cart-pole rollout under the trained controller.

Frame layout:
    Top:    cart-pole rendering (cart on track, pole at angle theta, action u
            shown as horizontal arrow)
    Bottom: time series of x and theta up to current step

Usage:
    python3 make_pole_balance_non_markov_gif.py --seed 0
    python3 make_pole_balance_non_markov_gif.py --seed 0 --T-max 600 --fps 30
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

from pole_balance_non_markov import (
    RunConfig, run, cart_pole_step, init_state, normalize_pos, is_failed,
    FORCE, X_LIMIT, THETA_LIMIT, L_HALF,
)


def render_frame(state: np.ndarray, action: float,
                 history: dict, t_idx: int) -> Image.Image:
    fig = plt.figure(figsize=(8, 5), dpi=85)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.35)

    # ---- top: cart-pole rendering ----
    ax = fig.add_subplot(gs[0, 0])
    x, _, theta, _ = state
    cart_w, cart_h = 0.4, 0.25
    pole_len = 2 * L_HALF      # full visible pole

    # track
    ax.axhline(0.0, color="#444", linewidth=2)
    # x-limit markers
    for xl in (-X_LIMIT, X_LIMIT):
        ax.plot([xl, xl], [-0.1, 0.1], color="red", linewidth=2)
    # cart
    ax.add_patch(Rectangle((x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                           facecolor="#1f77b4", edgecolor="black", linewidth=1.0))
    # pole
    pole_x_top = x + pole_len * np.sin(theta)
    pole_y_top = pole_len * np.cos(theta)
    ax.plot([x, pole_x_top], [0.0, pole_y_top], color="#d62728", linewidth=4,
            solid_capstyle="round")
    # tip
    ax.plot(pole_x_top, pole_y_top, "o", color="#d62728", markersize=8)
    # pivot
    ax.plot(x, 0.0, "o", color="black", markersize=4)
    # action arrow
    if abs(action) > 0.01:
        arrow_len = 0.5 * action
        ax.add_patch(FancyArrow(
            x, -cart_h / 2 - 0.15, arrow_len, 0.0,
            head_width=0.08, head_length=0.07, length_includes_head=True,
            color="#2ca02c", linewidth=1.2))
    # info text
    ax.text(0.02, 0.95, f"step {t_idx}    "
            f"x = {x:+5.2f}    $\\theta$ = {np.degrees(theta):+6.2f}$^\\circ$    "
            f"u = {action:+.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            family="monospace")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.6, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([-2.4, -1.2, 0.0, 1.2, 2.4])
    ax.set_yticks([])
    ax.set_title("pole-balance-non-markov: trained controller in real env",
                 fontsize=10)

    # ---- bottom: time series so far ----
    ax2 = fig.add_subplot(gs[1, 0])
    times = np.arange(len(history["x"])) * 0.02
    if len(times) > 0:
        ax2.plot(times, history["x"], color="#1f77b4", linewidth=1.0,
                 label="x (m)")
        ax2.plot(times, np.degrees(history["theta"]), color="#d62728",
                 linewidth=1.0, label=r"$\theta$ (deg)")
    ax2.axhline(np.degrees(THETA_LIMIT), color="#d62728", linestyle=":",
                linewidth=0.6, alpha=0.6)
    ax2.axhline(-np.degrees(THETA_LIMIT), color="#d62728", linestyle=":",
                linewidth=0.6, alpha=0.6)
    ax2.axhline(X_LIMIT, color="#1f77b4", linestyle=":", linewidth=0.6,
                alpha=0.6)
    ax2.axhline(-X_LIMIT, color="#1f77b4", linestyle=":", linewidth=0.6,
                alpha=0.6)
    ax2.set_xlabel("time (s)", fontsize=9)
    ax2.set_ylim(-15, 15)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def collect_real_rollout(C, rng, T_max: int):
    state = init_state(rng)
    h_C = np.zeros(C.hid_dim)
    states, actions = [state.copy()], []
    for _ in range(T_max):
        pos_n = normalize_pos(np.array([state[0], state[2]]))
        pre_C = C.W_h @ h_C + C.W_x @ pos_n + C.b
        h_C = np.tanh(pre_C)
        u_pre = C.V @ h_C + C.c
        u = float(np.tanh(u_pre[0]))
        state = cart_pole_step(state, u * FORCE)
        states.append(state.copy())
        actions.append(u)
        if is_failed(state):
            break
    return np.array(states), np.array(actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--T-max", type=int, default=400,
                   help="length of the rendered rollout")
    p.add_argument("--frame-stride", type=int, default=4,
                   help="render every Nth step (controls GIF length / size)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", type=str, default="pole_balance_non_markov.gif")
    p.add_argument("--rollout-seed", type=int, default=11)
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed, n_cycles=args.cycles)
    print(f"Training (seed={args.seed}, cycles={args.cycles}) ...")
    res = run(cfg, verbose=False)
    print(f"  final solved: {res['final_solved']}/30  "
          f"(mean balance {np.mean(res['final_times']):.1f})")

    # Real-env rollout
    rng = np.random.default_rng(args.rollout_seed)
    states, actions = collect_real_rollout(res["C"], rng, T_max=args.T_max)
    n = len(actions)
    print(f"  rollout: {n} steps before fall (cap {args.T_max})")

    frames = []
    history = {"x": [], "theta": []}
    for t in range(n):
        history["x"].append(states[t, 0])
        history["theta"].append(states[t, 2])
        if t % args.frame_stride == 0 or t == n - 1:
            frames.append(render_frame(states[t], actions[t], history, t))
    print(f"  {len(frames)} frames")

    # hold final frame for emphasis
    if frames:
        frames.extend([frames[-1]] * 12)

    duration = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
