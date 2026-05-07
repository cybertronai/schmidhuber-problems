"""Static visualizations for relational-nem-bouncing-balls.

Reads run.json (produced by `python3 relational_nem_bouncing_balls.py`) and
writes PNGs into viz/. Run AFTER training the models.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(run_path: str):
    with open(run_path) as f:
        return json.load(f)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_training_curves(run, out_dir: str):
    """Train/val loss curves for both models."""
    nr = run["training"]["non_relational"]["history"]
    re = run["training"]["relational"]["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # train loss
    ax = axes[0]
    ax.plot(nr["epoch"], nr["train_loss"], label="non-relational", color="#d62728")
    ax.plot(re["epoch"], re["train_loss"], label="relational",     color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BPTT MSE (train)")
    ax.set_yscale("log")
    ax.set_title("Training loss (BPTT roll, mean over t_bptt steps)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # val 1-step MSE
    ax = axes[1]
    ax.plot(nr["epoch"], nr["val_loss_1step"], label="non-relational",
            color="#d62728")
    ax.plot(re["epoch"], re["val_loss_1step"], label="relational",
            color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("1-step MSE (val)")
    ax.set_yscale("log")
    ax.set_title("Val 1-step prediction MSE")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # val multi-step
    ax = axes[2]
    ax.plot(nr["epoch"], nr["val_loss_bptt"], label="non-relational",
            color="#d62728")
    ax.plot(re["epoch"], re["val_loss_bptt"], label="relational",
            color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{run['training']['relational']['t_bptt']}-step MSE (val)")
    ax.set_yscale("log")
    ax.set_title(
        f"Val {run['training']['relational']['t_bptt']}-step rollout MSE"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_rollout_errors(run, out_dir: str):
    """Per-step rollout pos/vel error for K=K_train and extrapolation."""
    K_train = run["config"]["K_train"]
    K_list = [K_train] + run["config"]["extrapolate_K"]
    n = len(K_list)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.4), sharex=True)
    if n == 1:
        axes = axes[:, None]
    for i, K in enumerate(K_list):
        rk = run["rollout"][f"K{K}"]
        T = rk["T"]
        x = np.arange(T)
        # position
        ax = axes[0, i]
        ax.plot(x, rk["pos_err_non_relational"], color="#d62728",
                label="non-rel")
        ax.plot(x, rk["pos_err_relational"],     color="#2ca02c",
                label="rel")
        title = f"K={K}" + (" (train)" if K == K_train else " (extrap)")
        ax.set_title(f"{title}: position-MSE")
        if i == 0:
            ax.set_ylabel("RMSE (box units)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(rk["pos_err_non_relational"]),
                           max(rk["pos_err_relational"])) * 1.1)

        # velocity
        ax = axes[1, i]
        ax.plot(x, rk["vel_err_non_relational"], color="#d62728",
                label="non-rel")
        ax.plot(x, rk["vel_err_relational"],     color="#2ca02c",
                label="rel")
        ax.set_title(f"K={K}: velocity-MSE  "
                     f"(rel/non-rel mean={rk['mean_vel_err_relational']/(rk['mean_vel_err_non_relational']+1e-12):.2f})")
        if i == 0:
            ax.set_ylabel("RMSE (vel units)")
        ax.set_xlabel("rollout step")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(rk["vel_err_non_relational"]),
                           max(rk["vel_err_relational"])) * 1.1)

    fig.suptitle(
        "Closed-loop rollout error vs step. Relational dynamics consistently "
        "lower velocity error\n(velocity changes happen at collisions, where "
        "the pairwise message is informative)."
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "rollout_errors.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_extrapolation_summary(run, out_dir: str):
    """Bar chart: mean velocity-MSE for non-rel vs rel across all K."""
    K_train = run["config"]["K_train"]
    K_list = [K_train] + run["config"]["extrapolate_K"]
    nr_v = [run["rollout"][f"K{K}"]["mean_vel_err_non_relational"] for K in K_list]
    re_v = [run["rollout"][f"K{K}"]["mean_vel_err_relational"]      for K in K_list]
    nr_p = [run["rollout"][f"K{K}"]["mean_pos_err_non_relational"] for K in K_list]
    re_p = [run["rollout"][f"K{K}"]["mean_pos_err_relational"]      for K in K_list]

    x = np.arange(len(K_list))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # velocity
    ax = axes[0]
    ax.bar(x - w/2, nr_v, w, label="non-relational", color="#d62728")
    ax.bar(x + w/2, re_v, w, label="relational",      color="#2ca02c")
    for i, K in enumerate(K_list):
        ax.text(i, max(nr_v[i], re_v[i]) + 0.005,
                f"{re_v[i]/(nr_v[i]+1e-12):.2f}",
                ha="center", fontsize=9, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={K}" + ("\n(train)" if K == K_train else "\n(extrap)")
                        for K in K_list])
    ax.set_ylabel("Mean velocity-MSE (rollout T)")
    ax.set_title("Velocity prediction (collision-sensitive)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # position
    ax = axes[1]
    ax.bar(x - w/2, nr_p, w, label="non-relational", color="#d62728")
    ax.bar(x + w/2, re_p, w, label="relational",      color="#2ca02c")
    for i, K in enumerate(K_list):
        ax.text(i, max(nr_p[i], re_p[i]) + 0.002,
                f"{re_p[i]/(nr_p[i]+1e-12):.2f}",
                ha="center", fontsize=9, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={K}" + ("\n(train)" if K == K_train else "\n(extrap)")
                        for K in K_list])
    ax.set_ylabel("Mean position-MSE (rollout T)")
    ax.set_title("Position prediction (ballistic-dominated)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Relational vs non-relational dynamics, mean rollout error "
                 "by K (numbers above bars = rel/non-rel ratio)")
    fig.tight_layout()
    out = os.path.join(out_dir, "extrapolation_summary.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_sample_trajectory(run, out_dir: str):
    """One sequence: ground truth vs both rollouts as overlaid 2D trajectories."""
    samples = run["samples"]
    true = np.array(samples["true"])              # (T, B, K, 4)
    nr   = np.array(samples["non_relational"])
    re   = np.array(samples["relational"])
    K_train = run["config"]["K_train"]

    n_show = min(3, true.shape[1])
    fig, axes = plt.subplots(1, n_show, figsize=(4.0 * n_show, 4.0),
                             sharex=True, sharey=True)
    if n_show == 1:
        axes = [axes]
    radius = run["config"]["radius"]
    for i in range(n_show):
        ax = axes[i]
        for k in range(K_train):
            ax.plot(true[:, i, k, 0], true[:, i, k, 1],
                    color="black", lw=2.0, alpha=0.7,
                    label="truth" if (i == 0 and k == 0) else None)
            ax.plot(nr[:, i, k, 0], nr[:, i, k, 1],
                    color="#d62728", lw=1.0, alpha=0.7,
                    label="non-rel" if (i == 0 and k == 0) else None)
            ax.plot(re[:, i, k, 0], re[:, i, k, 1],
                    color="#2ca02c", lw=1.0, alpha=0.7,
                    label="rel" if (i == 0 and k == 0) else None)
            # mark start position
            ax.plot(true[0, i, k, 0], true[0, i, k, 1], "ko",
                    markersize=4)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"sample {i+1}, K={K_train}, T={true.shape[0]}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("y")
        ax.set_xlabel("x")
    fig.suptitle("Closed-loop rollout trajectories: ground truth vs predictions")
    fig.tight_layout()
    out = os.path.join(out_dir, "sample_trajectories.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_rendered_frames(run, out_dir: str):
    """Render true vs predicted frames at a few timesteps as a grid."""
    samples = run["samples"]
    true = np.array(samples["true"])
    nr   = np.array(samples["non_relational"])
    re   = np.array(samples["relational"])
    K_train = run["config"]["K_train"]
    radius = run["config"]["radius"]
    H = W = 48
    sigma = max(0.6 * radius, 0.04)

    # pick timesteps at 0, T/3, 2T/3, T-1
    T = true.shape[0]
    steps = [0, T // 3, 2 * T // 3, T - 1]
    fig, axes = plt.subplots(3, len(steps), figsize=(2.4 * len(steps), 7.4))
    rows = [("ground truth", true, "Greys"),
            ("non-relational", nr, "Reds"),
            ("relational", re, "Greens")]
    for r, (label, traj, cmap) in enumerate(rows):
        for c, t in enumerate(steps):
            img = _render(traj[t, 0, :, :], H, W, sigma)  # use first sample
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, origin="lower",
                      extent=(0, 1, 0, 1), vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t}")
            if c == 0:
                ax.set_ylabel(label)
    fig.suptitle("Rendered frames: ground truth vs rolled-out predictions "
                 f"(K={K_train})")
    fig.tight_layout()
    out = os.path.join(out_dir, "rendered_frames.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def _render(state_k: np.ndarray, H: int, W: int, sigma: float) -> np.ndarray:
    img = np.zeros((H, W))
    ys = np.linspace(0.0, 1.0, H)
    xs = np.linspace(0.0, 1.0, W)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    for k in range(state_k.shape[0]):
        cx = float(state_k[k, 0]); cy = float(state_k[k, 1])
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        img += np.exp(-d2 / (2.0 * sigma ** 2))
    return np.clip(img, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(HERE, "run.json"))
    p.add_argument("--out", default=os.path.join(HERE, "viz"))
    args = p.parse_args()

    if not os.path.exists(args.run):
        raise SystemExit(
            f"{args.run} not found. Run "
            f"`python3 relational_nem_bouncing_balls.py --seed 0` first."
        )
    run = _load(args.run)
    _ensure_dir(args.out)

    plot_training_curves(run, args.out)
    plot_rollout_errors(run, args.out)
    plot_extrapolation_summary(run, args.out)
    plot_sample_trajectory(run, args.out)
    plot_rendered_frames(run, args.out)
    print(f"all visualizations saved under {args.out}/")


if __name__ == "__main__":
    main()
