"""Static visualisations for upside-down-rl.

Reads run.json produced by `upside_down_rl.py --save-json run.json` and writes
PNGs to viz/:

  1. env_layout.png             chain-MDP layout, terminals, step costs
  2. training_curves.png        loss + buffer mean return + rollout return
  3. command_sweep.png          achieved return vs desired return + random
                                baseline (THE headline figure)
  4. action_heatmap.png         P(right) over (state, desired_return) at the
                                buffer's eval horizon -- the conditioned
                                policy in one picture
  5. eval_per_command.png       achieved return per eval-command over training
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def env_layout(out):
    env = out["env"]
    N = env["N"]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    for s in range(N):
        if s == 0:
            color = "#5a9bd4"
            label = "+1"
        elif s == N - 1:
            color = "#d4694e"
            label = "+5"
        elif s == env["start"]:
            color = "#bbbbbb"
            label = "S"
        else:
            color = "#eeeeee"
            label = ""
        circle = plt.Circle((s, 0), 0.34, facecolor=color, edgecolor="black",
                            linewidth=1.0)
        ax.add_patch(circle)
        ax.text(s, 0, label, ha="center", va="center", fontsize=10,
                fontweight="bold")
        ax.text(s, -0.55, str(s), ha="center", va="center", fontsize=8,
                color="#666")
    # connecting lines
    for s in range(N - 1):
        ax.plot([s + 0.34, s + 1 - 0.34], [0, 0], color="black", linewidth=1)
    ax.text(N / 2 - 0.5, 0.85, "step cost = -0.1   t_max = 30",
            ha="center", fontsize=9, color="#444")
    ax.text(0, 0.7, "left\nterminal", ha="center", fontsize=8, color="#5a9bd4")
    ax.text(N - 1, 0.7, "right\nterminal", ha="center", fontsize=8, color="#d4694e")
    ax.text(env["start"], -0.85, "start (S)", ha="center", fontsize=8,
            color="#666")
    ax.set_axis_off()
    ax.set_title("chain-MDP environment", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "env_layout.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def training_curves(out):
    h = out["history"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    axs[0].plot(h["iter"], h["loss"], color="#444")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("training loss")
    axs[0].set_title("UDRL behavior-cloning loss")
    axs[0].set_yscale("log")
    axs[0].grid(alpha=0.3)

    axs[1].plot(h["iter"], h["buffer_mean_return"], color="#5a9bd4",
                label="buffer mean R", linewidth=1.6)
    axs[1].plot(h["iter"], h["rollout_mean_return"], color="#d4694e",
                label="rollout mean R", linewidth=1.6, alpha=0.8)
    axs[1].axhline(out["random_baseline"]["mean_return"], color="#999",
                   linestyle="--", linewidth=1, label="random baseline")
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("mean episode return")
    axs[1].set_title("buffer + on-policy rollouts")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    axs[2].plot(h["iter"], h["buffer_top_command_R"], color="#5a9bd4",
                label="cmd_R (top-K mean R)", linewidth=1.6)
    axs[2].plot(h["iter"], h["buffer_top_command_H"], color="#7faa53",
                label="cmd_H (top-K mean len)", linewidth=1.6)
    axs[2].set_xlabel("iteration")
    axs[2].set_title("exploration command (from buffer top-K)")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "training_curves.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def command_sweep(out):
    sweep = out["sweep_results"]
    desired = np.array([s["desired_return"] for s in sweep])
    achieved = np.array([s["achieved_return_mean"] for s in sweep])
    achieved_std = np.array([s["achieved_return_std"] for s in sweep])
    rand = out["random_baseline"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([min(desired), max(desired)], [min(desired), max(desired)],
            color="#bbbbbb", linestyle="--", linewidth=1, label="ideal (achieved = desired)")
    ax.errorbar(desired, achieved, yerr=achieved_std, fmt="o-",
                color="#d4694e", capsize=3, linewidth=1.8,
                label="UDRL greedy rollout")
    ax.axhline(rand["mean_return"], color="#999", linestyle=":",
               linewidth=1, label=f"random policy ({rand['mean_return']:.2f})")
    ax.set_xlabel("commanded return $R^*$")
    ax.set_ylabel("achieved return (mean over rollouts)")
    ax.set_title("UDRL: achieved return tracks commanded return")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "command_sweep.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def action_heatmap(out):
    grid = np.array(out["p_right_grid"])  # shape (N, len(R_axis))
    R_axis = np.array(out["p_right_grid_R_axis"])
    state_axis = np.array(out["p_right_grid_state_axis"])
    horizon = out.get("p_right_grid_horizon", 4.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=0.0, vmax=1.0,
        extent=[R_axis[0], R_axis[-1], state_axis[0] - 0.5,
                state_axis[-1] + 0.5],
    )
    ax.set_xlabel("commanded return $R^*$")
    ax.set_ylabel("state")
    ax.set_title(
        f"P(action = right) given (state, $R^*$) at horizon = {horizon:.0f}"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(right)")
    # mark start state
    env = out["env"]
    ax.axhline(env["start"], color="black", linestyle=":", linewidth=1,
               alpha=0.5)
    ax.text(R_axis[0] + 0.05, env["start"] + 0.15, "start",
            color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "action_heatmap.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def eval_per_command(out):
    h = out["history"]
    eval_iter = np.array(h["eval_iter"])
    eval_achieved = np.array(h["eval_achieved"])  # (n_eval, n_commands)
    cmds = h["eval_commands"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = plt.get_cmap("viridis")
    for j, c in enumerate(cmds):
        ax.plot(eval_iter, eval_achieved[:, j],
                color=cmap(j / max(1, len(cmds) - 1)),
                marker="o", markersize=3, linewidth=1.4,
                label=f"$R^*$ = {c:.1f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("achieved return (greedy)")
    ax.set_title("achieved return per commanded $R^*$ over training")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "eval_per_command.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        # Run with defaults if no JSON yet.
        import subprocess
        subprocess.run(
            ["python3", os.path.join(HERE, "upside_down_rl.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )
    out = _load(json_path)
    env_layout(out)
    training_curves(out)
    command_sweep(out)
    action_heatmap(out)
    eval_per_command(out)
    print(f"Wrote 5 PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
