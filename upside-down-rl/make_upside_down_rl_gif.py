"""Make upside_down_rl.gif: animate four greedy rollouts of the trained UDRL
policy under different commanded returns, side by side.

The agent starts in the same state for all four panels. Each panel is
conditioned on a different desired_return R^*. The animation shows the
agent's position over time + the cumulative reward, illustrating that
*the same network* produces opposite trajectories depending purely on the
return command -- the headline UDRL claim.

If run.json does not exist, runs upside_down_rl.py once with seed=0 first.

Output: upside_down_rl.gif (target <= 2 MB).
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
sys.path.insert(0, HERE)

from upside_down_rl import (  # noqa: E402
    ChainMDP, MLP, RunConfig, Buffer, encode, policy_rollout,
    random_rollout,
)


def _load_or_train(seed: int = 0):
    """Train fresh and return the trained policy + env + config (so we can
    rerun rollouts at GIF time without persisting full network state to JSON).
    """
    cfg = RunConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    env = ChainMDP(N=cfg.N, t_max=cfg.t_max)
    return_scale = float(max(abs(env.right_reward), abs(env.left_reward)))
    horizon_scale = float(env.t_max)
    pol = MLP(
        in_dim=env.N + 2, hidden=cfg.hidden, out_dim=2,
        rng=np.random.default_rng(cfg.seed + 1),
    )
    buf = Buffer(capacity=cfg.buffer_size)
    for _ in range(cfg.n_warmup_random):
        buf.add(random_rollout(env, rng))
    for it in range(cfg.n_iters):
        cmd_R, cmd_H = buf.top_k_command(cfg.top_k)
        for _ in range(cfg.episodes_per_iter):
            ep = policy_rollout(
                env, pol, cmd_R, cmd_H, rng,
                return_scale=return_scale,
                horizon_scale=horizon_scale,
                greedy=False, explore_sigma=cfg.explore_sigma,
            )
            buf.add(ep)
        for _ in range(cfg.grad_steps_per_iter):
            s_b, a_b, dR_b, dH_b = buf.sample_transitions(rng, cfg.batch_size)
            x_b = encode(s_b, dR_b, dH_b, env.N, return_scale, horizon_scale)
            _, cache = pol.forward(x_b)
            _, grads = pol.cross_entropy_grad(cache, a_b)
            pol.adam_step(grads, lr=cfg.lr)
    eval_H, _ = (lambda r, h: (h, r))(*buf.top_k_command(cfg.top_k))
    # eval_H is the second tuple element actually -- correct by recomputing
    _, eval_H = buf.top_k_command(cfg.top_k)
    return pol, env, cfg, return_scale, horizon_scale, max(eval_H, 1.0)


def collect_rollout(pol, env, cfg, return_scale, horizon_scale,
                    desired_return, eval_H):
    rng = np.random.default_rng(cfg.seed + 100 + int(desired_return * 10))
    ep = policy_rollout(
        env, pol, desired_return, eval_H, rng,
        return_scale=return_scale,
        horizon_scale=horizon_scale,
        greedy=True, explore_sigma=0.0,
    )
    states = ep.states + [ep.states[-1]]  # pad final position
    rewards = list(ep.rewards)
    cum = np.cumsum([0.0] + rewards)
    return np.array(states), cum


def main():
    seed = 0
    pol, env, cfg, return_scale, horizon_scale, eval_H = _load_or_train(seed)
    desired_returns = [-1.0, 1.0, 3.5, 5.0]
    rollouts = [
        collect_rollout(pol, env, cfg, return_scale, horizon_scale, d, eval_H)
        for d in desired_returns
    ]
    max_T = max(s.shape[0] for s, _ in rollouts)

    fig, axs = plt.subplots(len(desired_returns), 1, figsize=(7.0, 6.5))
    if len(desired_returns) == 1:
        axs = [axs]

    # Pre-draw chain layout and persistent text once
    artists = []
    for i, (ax, d, (states, cum)) in enumerate(zip(axs, desired_returns,
                                                   rollouts)):
        ax.set_xlim(-0.7, env.N - 0.3)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        for s in range(env.N):
            if s == 0:
                color = "#5a9bd4"
                label = "+1"
            elif s == env.N - 1:
                color = "#d4694e"
                label = "+5"
            else:
                color = "#eeeeee"
                label = ""
            circ = plt.Circle((s, 0), 0.32, facecolor=color,
                              edgecolor="black", linewidth=0.8)
            ax.add_patch(circ)
            if label:
                ax.text(s, 0, label, ha="center", va="center", fontsize=8,
                        fontweight="bold")
        for s in range(env.N - 1):
            ax.plot([s + 0.32, s + 1 - 0.32], [0, 0], color="black",
                    linewidth=0.8)
        # agent marker
        agent = plt.Circle((states[0], 0), 0.18, facecolor="#222",
                           edgecolor="white", linewidth=1.0, zorder=5)
        ax.add_patch(agent)
        title = ax.text(env.N / 2 - 0.5, 0.85,
                        f"$R^*$ = {d:.1f}    t = 0    cum_R = 0.00",
                        ha="center", fontsize=9)
        artists.append({"agent": agent, "title": title,
                        "states": states, "cum": cum, "d": d})

    fig.suptitle("Upside-Down RL: same policy, different return commands",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def update(frame):
        out = []
        for a in artists:
            t = min(frame, a["states"].shape[0] - 1)
            a["agent"].center = (a["states"][t], 0.0)
            a["title"].set_text(
                f"$R^*$ = {a['d']:.1f}    t = {t:>2d}    "
                f"cum_R = {a['cum'][t]:>+.2f}"
            )
            out.append(a["agent"])
            out.append(a["title"])
        return out

    n_frames = max_T + 4  # tail-pause
    anim = FuncAnimation(fig, update, frames=n_frames, interval=400, blit=False)
    out_path = os.path.join(HERE, "upside_down_rl.gif")
    writer = PillowWriter(fps=2)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    sz = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({sz/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
