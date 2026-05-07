"""
Animation of the trained VAC controller balancing the cart-pole.

Renders a 2-panel matplotlib animation:
  left  : the cart-pole physical scene (cart, pole, force arrow)
  right : on-line vector critic V_pole(t), V_cart(t), advancing by frame

Output: pole_balance_markov_vac.gif (target <= 2 MB).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pole_balance_markov_vac import (
    DT, F_MAG, L_HALF, THETA_THRESHOLD, X_THRESHOLD,
    actor_forward, critic_forward, normalize, reset_env,
    run_episode, step_env, train_vac,
)


def collect_frames(p, *, seed: int, max_steps: int = 400):
    """Run a greedy episode capturing per-step state, action, V."""
    rng = np.random.default_rng(seed + 400_000)
    state = reset_env(rng)
    states, actions, v_pole, v_cart = [], [], [], []
    for t in range(max_steps):
        s_norm = normalize(state)
        pr, _, _ = actor_forward(p, s_norm)
        action = 1 if pr >= 0.5 else 0
        v, _ = critic_forward(p, s_norm)
        states.append(state.copy())
        actions.append(action)
        v_pole.append(float(v[0]))
        v_cart.append(float(v[1]))
        state, terminated = step_env(state, action)
        if terminated:
            break
    return (np.array(states), np.array(actions),
            np.array(v_pole), np.array(v_cart))


def render(states, actions, v_pole, v_cart, *, frames: int,
           out_path: str, stride: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=110,
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    # --- Left: the cart-pole scene
    ax_scene = axes[0]
    ax_scene.set_xlim(-X_THRESHOLD - 0.3, X_THRESHOLD + 0.3)
    ax_scene.set_ylim(-0.4, 1.3)
    ax_scene.set_aspect("equal")
    ax_scene.axhline(0, color="#999", lw=0.8)
    ax_scene.axvline(X_THRESHOLD, color="#cc0000", ls=":", lw=0.8)
    ax_scene.axvline(-X_THRESHOLD, color="#cc0000", ls=":", lw=0.8)
    ax_scene.set_xticks(np.linspace(-2, 2, 5))
    ax_scene.set_yticks([])
    ax_scene.set_title("VAC controller on Markov cart-pole")

    cart_w, cart_h = 0.5, 0.25
    cart_patch = plt.Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h,
                               facecolor="#003c7f", edgecolor="black",
                               lw=1.0)
    ax_scene.add_patch(cart_patch)
    pole_line, = ax_scene.plot([0, 0], [0, 2 * L_HALF], color="#cc0000",
                               lw=4.0, solid_capstyle="round")
    force_arrow = ax_scene.annotate("", xy=(0, -0.1), xytext=(0, -0.1),
                                    arrowprops=dict(arrowstyle="->",
                                                    color="#cc0000",
                                                    lw=1.6))
    step_text = ax_scene.text(0.02, 0.95, "", transform=ax_scene.transAxes,
                              fontsize=10, va="top",
                              bbox=dict(facecolor="white", alpha=0.8,
                                        edgecolor="none"))

    # --- Right: vector critic trace
    ax_v = axes[1]
    ax_v.set_xlim(0, len(v_pole) * DT)
    y_lo = float(min(v_pole.min(), v_cart.min()) - 0.2)
    y_hi = float(max(v_pole.max(), v_cart.max()) + 0.2)
    ax_v.set_ylim(y_lo, y_hi)
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("V")
    ax_v.set_title("vector critic V(s_t)")
    ax_v.grid(True, alpha=0.3)
    line_pole, = ax_v.plot([], [], color="#cc0000", lw=1.6,
                           label="V_pole")
    line_cart, = ax_v.plot([], [], color="#003c7f", lw=1.6,
                           label="V_cart")
    ax_v.legend(loc="lower right", fontsize=9)

    t_axis = np.arange(len(v_pole)) * DT

    def draw_frame(i: int):
        idx = min(i * stride, len(states) - 1)
        x, _, theta, _ = states[idx]
        cart_patch.set_xy((x - cart_w / 2, -cart_h / 2))
        tip_x = x + 2 * L_HALF * np.sin(theta)
        tip_y = 2 * L_HALF * np.cos(theta)
        pole_line.set_data([x, tip_x], [0, tip_y])
        force_dir = +1 if actions[idx] == 1 else -1
        force_arrow.xy = (x + 0.5 * force_dir, -0.1)
        force_arrow.set_position((x, -0.1))
        step_text.set_text(f"t={idx * DT:5.2f}s   "
                           f"step={idx:4d}/{len(states)-1}\n"
                           f"x={x:+.2f}   theta={np.degrees(theta):+.1f}deg")
        line_pole.set_data(t_axis[: idx + 1], v_pole[: idx + 1])
        line_cart.set_data(t_axis[: idx + 1], v_cart[: idx + 1])
        return cart_patch, pole_line, force_arrow, step_text, line_pole, line_cart

    n_frames = max(1, len(states) // stride)
    if frames is not None and frames < n_frames:
        n_frames = frames

    anim = FuncAnimation(fig, draw_frame, frames=n_frames,
                         interval=50, blit=False)
    writer = PillowWriter(fps=20)
    anim.save(out_path, writer=writer, dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default="pole_balance_markov_vac.gif")
    parser.add_argument("--frames", type=int, default=120,
                        help="number of GIF frames (default 120, 6s @ 20fps)")
    parser.add_argument("--episode-steps", type=int, default=400,
                        help="how many simulation steps to render at most")
    parser.add_argument("--stride", type=int, default=2,
                        help="simulation steps between rendered frames")
    parser.add_argument("--max-episodes", type=int, default=1000)
    args = parser.parse_args(argv)

    p, hist = train_vac(seed=args.seed, max_episodes=args.max_episodes,
                        verbose=False)
    print(f"[seed={args.seed}] solved={hist.solve_episode}  "
          f"trail-avg={hist.moving_avg[-1]:.1f}")
    states, actions, v_pole, v_cart = collect_frames(
        p, seed=args.seed, max_steps=args.episode_steps,
    )
    print(f"  rendered episode length: {len(states)} steps")
    render(states, actions, v_pole, v_cart,
           frames=args.frames, out_path=args.out, stride=args.stride)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size/1024:.1f} KB)")
    if size > 2 * 1024 * 1024:
        print("WARNING: GIF exceeds 2 MB target", file=sys.stderr)


if __name__ == "__main__":
    main()
