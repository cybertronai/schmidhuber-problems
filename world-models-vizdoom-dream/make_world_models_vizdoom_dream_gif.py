"""Make world_models_vizdoom_dream.gif: animate the dream-trained controller
running in the REAL env, side-by-side with what M was hallucinating during
its dream rollouts.

LEFT panel  : the real DodgingEnv. Agent and fireballs as they actually exist.
RIGHT panel : the dream. C_dream is run inside M starting from the same z_0.
              We decode each predicted z_t back into an obs via V to render
              what M *thinks* the next step looks like. Note that M's dream
              is not pixel-faithful -- it is a learned compression -- but it
              is good enough to train a controller that transfers.

If run.json or the trained weights are missing, retrains once with seed=0
(adds ~9s).

Output: world_models_vizdoom_dream.gif (target <= 2 MB).
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

from world_models_vizdoom_dream import (  # noqa: E402
    DodgingEnv, V_Autoencoder, M_LSTM, C_Controller,
    RunConfig, collect_random_data, train_V, encode_seqs, train_M,
    train_C_in_dreams, real_rollout,
)


def _train_artifacts(seed: int = 0):
    """Train V, M, and C_dream and return everything we need for the GIF."""
    cfg = RunConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    env = DodgingEnv(W=cfg.W, H=cfg.H,
                     max_fireballs=cfg.max_fireballs,
                     spawn_prob=cfg.spawn_prob,
                     max_steps=cfg.max_steps)
    obs_seqs, act_seqs, rew_seqs, done_seqs = collect_random_data(env, cfg, rng)
    V = V_Autoencoder(in_dim=env.obs_dim, z_dim=cfg.z_dim,
                      hidden=cfg.v_hidden,
                      rng=np.random.default_rng(cfg.seed + 11))
    train_V(V, obs_seqs, cfg, rng)
    z_seqs = encode_seqs(V, obs_seqs)
    M = M_LSTM(in_dim=cfg.z_dim + cfg.n_actions, z_dim=cfg.z_dim,
               hidden=cfg.m_hidden,
               rng=np.random.default_rng(cfg.seed + 21))
    train_M(M, z_seqs, act_seqs, rew_seqs, done_seqs, cfg, rng)
    z0_pool = np.stack([z[0] for z in z_seqs])
    C_dream, _, _ = train_C_in_dreams(V, M, env, cfg, z0_pool, rng)
    return V, M, C_dream, env, cfg


def real_frames(env, V, M, C, rng, n_actions=3):
    """Collect frames of an actual real-env rollout."""
    obs = env.reset()
    z = V.encode(obs[None])
    h, c = M.init_state(batch=1)
    frames = []
    while not env.done:
        a = C.act(z[0], h[0], greedy=True, rng=rng)
        frames.append({
            "agent_x": env.agent_x,
            "fireballs": list(env.fireballs),
            "t": env.t,
            "alive": True,
        })
        o2, r, done = env.step(a)
        a_oh = np.zeros(n_actions); a_oh[a] = 1.0
        x_t = np.concatenate([z[0], a_oh])[None]
        h, c, _ = M.lstm_step(x_t, h, c)
        z = V.encode(o2[None])
    frames.append({
        "agent_x": env.agent_x,
        "fireballs": list(env.fireballs),
        "t": env.t,
        "alive": False,
    })
    return frames


def dream_frames(M, V, C, z0, env, max_steps, n_actions=3, rng=None):
    """Collect frames of a *dream* rollout: M hallucinates z_{t+1}, we decode
    back through V to a grid for rendering. We threshold the agent and
    fireball channels at >0.4 to render binary cells.
    """
    h, c = M.init_state(batch=1)
    z = z0.reshape(1, -1).copy()
    frames = []
    W, H = env.W, env.H
    obs_size = 3 * H * W
    for t in range(max_steps):
        # Decode current z to grid
        x_hat = V.decode(z)[0]   # (3*H*W,)
        ch = x_hat.reshape(3, H, W)
        agent_ch = ch[0]
        fire_ch = ch[1]
        ay, ax = np.unravel_index(int(np.argmax(agent_ch)), agent_ch.shape)
        # Show all fire cells with channel value above threshold
        fire_cells = []
        thresh = max(0.3, float(fire_ch.max()) * 0.5) if fire_ch.max() > 0.1 else 1e9
        for yy in range(H):
            for xx in range(W):
                if fire_ch[yy, xx] > thresh:
                    fire_cells.append((xx, yy))
        d_logit = float((h @ M.W_d + M.b_d).reshape(-1)[0])
        d_prob = float(M._sigmoid(np.array([d_logit]))[0])
        frames.append({
            "agent_xy": (ax, ay),
            "fire_cells": fire_cells,
            "t": t,
            "alive": d_prob < 0.5,
            "d_prob": d_prob,
        })
        # Step C and M forward
        a = C.act(z[0], h[0], greedy=True, rng=rng)
        a_oh = np.zeros(n_actions); a_oh[a] = 1.0
        x_in = np.concatenate([z[0], a_oh])[None]
        h, c, _ = M.lstm_step(x_in, h, c)
        z_pred, r_pred, d_logit2 = M.predict_step(h)
        z = z_pred
        if d_prob > 0.5:
            break
    return frames


def _draw_grid(ax, W, H, color_bg="#fafafa"):
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # rectangle background (no grid lines for clarity)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, facecolor=color_bg,
                               edgecolor="black", linewidth=0.8, zorder=0))


def main():
    # Match the README's headline run (--seed 1) so the GIF illustrates the
    # numbers reported in the table.
    seed = 1
    V, M, C, env, cfg = _train_artifacts(seed)

    rng_real = np.random.default_rng(seed + 5000)
    env.seed(rng_real)
    real = real_frames(env, V, M, C, rng_real, n_actions=cfg.n_actions)

    # Use the same initial z that the real rollout started from for the dream
    # rollout. The first real frame's grid encodes the initial obs; just
    # encode the env's reset obs.
    rng_dream = np.random.default_rng(seed + 6000)
    env.seed(rng_dream)
    obs0 = env.reset()
    z0 = V.encode(obs0[None])[0]
    dream = dream_frames(M, V, C, z0, env, cfg.dream_max_steps,
                         n_actions=cfg.n_actions, rng=rng_dream)

    n_frames = max(len(real), len(dream))
    # add a tail pause
    n_frames += 4

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 5.0))
    real_ax, dream_ax = axs

    real_artists = {}
    dream_artists = {}

    def init_panel(ax, title):
        _draw_grid(ax, env.W, env.H)
        ax.set_title(title, fontsize=10)

    init_panel(real_ax, "REAL env (zero-shot transfer)")
    init_panel(dream_ax, "M's DREAM (controller trained here)")

    real_ax_text = real_ax.text(env.W / 2 - 0.5, env.H + 0.05,
                                "t=0", ha="center", va="bottom", fontsize=9)
    dream_ax_text = dream_ax.text(env.W / 2 - 0.5, env.H + 0.05,
                                  "t=0  d=0.00", ha="center", va="bottom",
                                  fontsize=9)

    real_agent = plt.Circle((env.W // 2, 0), 0.32, facecolor="#5a9bd4",
                            edgecolor="black", linewidth=1.0, zorder=3)
    real_ax.add_patch(real_agent)
    real_fire_artists = []

    dream_agent = plt.Circle((env.W // 2, 0), 0.32, facecolor="#7faa53",
                             edgecolor="black", linewidth=1.0, zorder=3)
    dream_ax.add_patch(dream_agent)
    dream_fire_artists = []

    def clear_artists(lst):
        for a in lst:
            a.remove()
        lst.clear()

    def update(frame):
        # Real
        i_real = min(frame, len(real) - 1)
        f = real[i_real]
        real_agent.center = (f["agent_x"], 0)
        real_agent.set_facecolor("#5a9bd4" if f["alive"] else "#999999")
        clear_artists(real_fire_artists)
        for fx, fy in f["fireballs"]:
            # Convert env's y (where 0 = top) to plot y (0 = bottom)
            py = env.H - 1 - fy
            circ = plt.Circle((fx, py), 0.27, facecolor="#d4694e",
                              edgecolor="black", linewidth=0.5, zorder=2)
            real_ax.add_patch(circ)
            real_fire_artists.append(circ)
        real_ax_text.set_text(f"t={f['t']}  "
                              + ("alive" if f["alive"] else "DEAD"))

        # Dream
        i_dream = min(frame, len(dream) - 1)
        d = dream[i_dream]
        ax_, ay_ = d["agent_xy"]
        # Convert dream's y (where 0 = top) to plot y (0 = bottom)
        py = env.H - 1 - ay_
        dream_agent.center = (ax_, py)
        dream_agent.set_facecolor("#7faa53" if d["alive"] else "#999999")
        clear_artists(dream_fire_artists)
        for fx, fy in d["fire_cells"]:
            py = env.H - 1 - fy
            circ = plt.Circle((fx, py), 0.27, facecolor="#d4694e",
                              edgecolor="black", linewidth=0.5, alpha=0.85,
                              zorder=2)
            dream_ax.add_patch(circ)
            dream_fire_artists.append(circ)
        dream_ax_text.set_text(
            f"t={d['t']}  d={d['d_prob']:.2f}  "
            + ("alive" if d["alive"] else "DEAD")
        )
        return [real_agent, dream_agent, real_ax_text, dream_ax_text]

    fig.suptitle(
        "world-models-vizdoom-dream: same controller, two envs (real vs dream)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=150,
                         blit=False)
    out_path = os.path.join(HERE, "world_models_vizdoom_dream.gif")
    writer = PillowWriter(fps=6)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    sz = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({sz/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
