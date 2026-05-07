"""Static visualizations for the trained pole-balance-non-markov system.

Outputs (in `viz/`):
  training_curves.png  - phase 1 M MSE; phase 2 C cost/step; balance over iters
  rollout.png          - state trajectories under trained C in real env
  model_error.png      - M's predicted vs true (x, theta) on a held-out trajectory

Usage:
    python3 visualize_pole_balance_non_markov.py --seed 0 --outdir viz
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pole_balance_non_markov import (
    RunConfig, run, cart_pole_step, init_state, normalize_pos,
    is_failed, eval_controller, FORCE, X_LIMIT, THETA_LIMIT,
    collect_random_rollout,
)


def plot_training_curves(res: dict, out_path: str):
    p1 = res["phase1_losses"]
    p2 = res["phase2_history"]
    cycle_eval = res["cycle_eval"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=120)

    # phase 1 M loss
    ax = axes[0]
    ax.plot(np.arange(1, len(p1) + 1), p1, color="#1f77b4", linewidth=0.7)
    # add refresh phases (cumulative)
    if res.get("refresh_losses"):
        offset = len(p1)
        for i, rl in enumerate(res["refresh_losses"]):
            xs = np.arange(offset, offset + len(rl)) + 1
            ax.plot(xs, rl, color="#9467bd", linewidth=0.7,
                    label="M refresh" if i == 0 else None)
            offset += len(rl)
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("MSE (normalized positions)")
    ax.set_title("Phase 1 + refresh: world-model M loss")
    ax.grid(alpha=0.3)
    if res.get("refresh_losses"):
        ax.legend(loc="upper right", fontsize=8)

    # phase 2 controller cost / step
    ax = axes[1]
    ax.plot(p2["iter"], p2["cost"], color="#ff7f0e", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"cost / step  ($\theta_n^2 + \lambda\,x_n^2$)")
    ax.set_title("Phase 2: controller C imagined cost")
    ax.grid(alpha=0.3)

    # phase 2 real-env balance time
    ax = axes[2]
    if p2["eval_iter"]:
        ax.plot(p2["eval_iter"], p2["balance"], "o-", color="#2ca02c",
                linewidth=1.0, markersize=3, label="mean (5 eps)")
        ax.plot(p2["eval_iter"], p2["balance_max"], "x", color="#666",
                markersize=4, label="max (5 eps)")
    ax.axhline(1000, color="red", linestyle="--", linewidth=0.7, alpha=0.6,
               label="1000-step target")
    for ce in cycle_eval[:-1]:
        ax.axvline(ce["cycle"] * (len(p2["iter"]) // len(cycle_eval)),
                   color="purple", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("balance time (steps, real env)")
    ax.set_title("Phase 2: real-env balance time")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1100)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rollout(res: dict, out_path: str, T_max: int = 1000, seed: int = 99):
    """One real-env rollout under the trained C; plot state trajectories."""
    rng = np.random.default_rng(seed)
    C = res["C"]
    state = init_state(rng)
    h_C = np.zeros(C.hid_dim)
    states = [state.copy()]
    actions = []
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
    states = np.array(states)
    actions = np.array(actions)
    times = np.arange(states.shape[0]) * 0.02

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=120, sharex=True)

    ax = axes[0]
    ax.plot(times, states[:, 0], color="#1f77b4", label="x  (cart position)")
    ax.plot(times, states[:, 2], color="#d62728", label=r"$\theta$  (pole angle)")
    ax.axhline(X_LIMIT, color="#1f77b4", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.axhline(-X_LIMIT, color="#1f77b4", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.axhline(THETA_LIMIT, color="#d62728", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.axhline(-THETA_LIMIT, color="#d62728", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.set_ylabel("position (m, rad)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"Real-env rollout under trained C ({states.shape[0] - 1} steps "
                 f"= {(states.shape[0] - 1) * 0.02:.1f}s)")

    ax = axes[1]
    ax.plot(times, states[:, 1], color="#1f77b4",
            label=r"$\dot{x}$  (hidden from C)", linewidth=0.8)
    ax.plot(times, states[:, 3], color="#d62728",
            label=r"$\dot{\theta}$  (hidden from C)", linewidth=0.8)
    ax.set_ylabel("velocity (m/s, rad/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(times[:-1], actions, color="#2ca02c", linewidth=0.7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("action u  (force / F)")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_model_error(res: dict, out_path: str, seed: int = 7):
    """Compare M's predicted next position to ground truth on a held-out
    random rollout (open-loop M unrolled with the same actions)."""
    rng = np.random.default_rng(seed)
    M = res["M"]
    in_seq, target_seq = collect_random_rollout(rng, T_max=120)
    T = in_seq.shape[0]
    if T < 2:
        return
    _, y_seq = M.forward(in_seq)
    times = np.arange(T) * 0.02

    # Open-loop unroll: M predicts next-pos using its OWN previous prediction.
    h_M = np.zeros(M.hid_dim)
    open_loop_pred = np.zeros((T, 2))
    pos_n = in_seq[0, :2]
    for t in range(T):
        in_M = np.concatenate([pos_n, in_seq[t, 2:3]])
        pre_M = M.W_h @ h_M + M.W_x @ in_M + M.b
        h_M = np.tanh(pre_M)
        pos_n = M.V @ h_M + M.c
        open_loop_pred[t] = pos_n

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=120, sharex=True)
    ax = axes[0]
    ax.plot(times, target_seq[:, 0], color="black", label="ground truth $x$",
            linewidth=1.2)
    ax.plot(times, y_seq[:, 0], color="#1f77b4", label="M (teacher-forced)",
            linewidth=0.8)
    ax.plot(times, open_loop_pred[:, 0], color="#ff7f0e", linestyle="--",
            label="M (open-loop)", linewidth=0.8)
    ax.set_ylabel(r"$x / X_{\rm limit}$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"World-model accuracy (random rollout, T={T} steps)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(times, target_seq[:, 1], color="black",
            label=r"ground truth $\theta$", linewidth=1.2)
    ax.plot(times, y_seq[:, 1], color="#1f77b4", label="M (teacher-forced)",
            linewidth=0.8)
    ax.plot(times, open_loop_pred[:, 1], color="#ff7f0e", linestyle="--",
            label="M (open-loop)", linewidth=0.8)
    ax.set_ylabel(r"$\theta / \theta_{\rm limit}$")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--C-iters", type=int, default=400)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = RunConfig(seed=args.seed, n_cycles=args.cycles, C_iters=args.C_iters)
    print(f"Training (seed={args.seed}, cycles={args.cycles}) ...")
    res = run(cfg, verbose=False)
    print(f"  final solved: {res['final_solved']}/30  "
          f"(mean balance {np.mean(res['final_times']):.1f})")

    plot_training_curves(res, os.path.join(args.outdir, "training_curves.png"))
    plot_rollout(res, os.path.join(args.outdir, "rollout.png"))
    plot_model_error(res, os.path.join(args.outdir, "model_error.png"))


if __name__ == "__main__":
    main()
