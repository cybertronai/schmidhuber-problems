"""Static visualizations for the trained saccadic-target-detection controller.

Outputs (in `viz/`):
  scene_examples.png        — 6 scenes with target halo + sample fovea trajectory
  saccade_trajectories.png  — overlay of 16 trajectories (start to target)
  training_curves.png       — M loss + C predicted-score, find-rate, median-saccades
  fovea_strip.png           — fovea contents along one trajectory, frame by frame
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from saccadic_target_detection import (
    train_full, make_scene, extract_fovea, halo_intensity,
    rollout_controller_step, Controller, WorldModel,
    SCENE_SIZE, FOVEA_SIZE, HALF, T_MAX, DETECT_RADIUS, target_indicator, clip_pos,
)


def _restore(state, hidden, depth=None):
    """Reconstruct an MLP-style object from saved weights."""
    return state["W"], state["b"]


def _rebuild(C_state, M_state, c_hidden, m_hidden, m_depth):
    rng = np.random.default_rng(0)  # init only (will be overwritten)
    C = Controller.make(rng, hidden=c_hidden)
    C.W = [w.copy() for w in C_state["W"]]
    C.b = [bb.copy() for bb in C_state["b"]]
    M = WorldModel.make(rng, hidden=m_hidden, depth=m_depth)
    M.W = [w.copy() for w in M_state["W"]]
    M.b = [bb.copy() for bb in M_state["b"]]
    return C, M


def rollout_one(C, M, scene, target):
    """Roll out the policy on a single scene, return list of positions visited."""
    pos = np.array([SCENE_SIZE / 2.0 - 0.5, SCENE_SIZE / 2.0 - 0.5], dtype=np.float32)
    positions = [pos.copy()]
    fovea_history = [extract_fovea(scene, pos)]
    for step in range(T_MAX):
        sub_pos = pos[None]
        new_pos, ind, _ = rollout_controller_step(
            C, M, [scene], target[None], sub_pos, lr=0.0, train=False
        )
        pos = new_pos[0]
        positions.append(pos.copy())
        fovea_history.append(extract_fovea(scene, pos))
        if ind[0] > 0.5:
            break
    return positions, fovea_history


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_scene_examples(C, M, rng, out_path: str, n: int = 6):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=120)
    for ax in axes.flat:
        scene, target = make_scene(rng)
        positions, _ = rollout_one(C, M, scene, target)
        positions = np.stack(positions)
        ax.imshow(scene, cmap="magma", origin="upper", vmin=0, vmax=1.0)
        # plot trajectory: positions are (x, y); imshow x-axis is column, y-axis is row
        ax.plot(positions[:, 0], positions[:, 1], "-o",
                color="#7fff7f", markersize=4, linewidth=1.2, alpha=0.9)
        ax.plot(positions[0, 0], positions[0, 1], "s",
                color="white", markersize=8, markeredgecolor="black", label="start")
        ax.plot(target[0], target[1], "*",
                color="cyan", markersize=14, markeredgecolor="black", label="target")
        # Draw last fovea box
        fx, fy = positions[-1]
        rect = Rectangle((fx - HALF - 0.5, fy - HALF - 0.5),
                         FOVEA_SIZE, FOVEA_SIZE,
                         linewidth=1.2, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
        # Detection radius circle
        circ = Circle((target[0], target[1]), DETECT_RADIUS,
                      linewidth=1.0, edgecolor="cyan", facecolor="none", linestyle="--")
        ax.add_patch(circ)
        n_sacc = positions.shape[0] - 1
        found = target_indicator(positions[-1], target) > 0.5
        ax.set_title(f"saccades={n_sacc} {'(found)' if found else '(timeout)'}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Saccadic target detection — fovea trajectory on test scenes",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_trajectories(C, M, rng, out_path: str, n: int = 32):
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=120)
    bg = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
    ax.imshow(bg, cmap="gray", origin="upper", vmin=0, vmax=1)
    for k in range(n):
        scene, target = make_scene(rng)
        positions, _ = rollout_one(C, M, scene, target)
        positions = np.stack(positions)
        # translate trajectory so target sits at center, to overlay all trajectories
        offset = np.array([SCENE_SIZE / 2.0, SCENE_SIZE / 2.0]) - target
        adj = positions + offset
        color = plt.cm.viridis(k / max(n - 1, 1))
        ax.plot(adj[:, 0], adj[:, 1], "-", color=color, alpha=0.55, linewidth=1.0)
        ax.plot(adj[0, 0], adj[0, 1], "o", color=color, markersize=3)
    # mark the recentered target
    ax.plot(SCENE_SIZE / 2.0, SCENE_SIZE / 2.0, "*",
            color="red", markersize=18, markeredgecolor="black", label="target")
    circ = Circle((SCENE_SIZE / 2.0, SCENE_SIZE / 2.0), DETECT_RADIUS,
                  linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--")
    ax.add_patch(circ)
    ax.set_title(f"{n} trajectories, recentered on target")
    ax.set_xlim(-0.5, SCENE_SIZE - 0.5)
    ax.set_ylim(SCENE_SIZE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(result: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=120)
    losses = result["m_losses"]
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#1f77b4")
    ax.set_xlabel("M epoch")
    ax.set_ylabel("M MSE on delta")
    ax.set_title("Phase 1: world-model M training")
    ax.grid(alpha=0.3)

    h = result["c_history"]
    ax = axes[0, 1]
    ax.plot(h["epoch"], h["mean_score"], color="#2ca02c")
    ax.set_xlabel("C epoch")
    ax.set_ylabel("mean predicted halo score")
    ax.set_title("Phase 2: M's predicted score during C training")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(h["epoch"], h["find_rate"], color="#ff7f0e")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("C epoch")
    ax.set_ylabel("find rate")
    ax.set_title("Find rate over C training")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(h["epoch"], h["median_saccades"], color="#9467bd")
    ax.axhline(T_MAX, color="gray", linestyle="--", linewidth=0.8, label=f"T_max={T_MAX}")
    ax.set_xlabel("C epoch")
    ax.set_ylabel("median saccades to find target")
    ax.set_title("Saccades to target")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    eval_m = result["eval"]
    rand_m = result["random_baseline"]
    fig.suptitle(
        f"saccadic-target-detection — eval find_rate {eval_m['find_rate']:.2f}, "
        f"median saccades {eval_m['median_saccades']:.0f} "
        f"(random: {rand_m['find_rate']:.2f}, {rand_m['median_saccades']:.0f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_fovea_strip(C, M, rng, out_path: str):
    """Show the fovea content frame-by-frame for a single trajectory."""
    # find a scene that needed several saccades to look interesting
    chosen = None
    for _ in range(30):
        scene, target = make_scene(rng)
        positions, fhist = rollout_one(C, M, scene, target)
        if 3 <= len(positions) <= 8:
            chosen = (scene, target, positions, fhist)
            break
    if chosen is None:
        scene, target = make_scene(rng)
        positions, fhist = rollout_one(C, M, scene, target)
        chosen = (scene, target, positions, fhist)
    scene, target, positions, fhist = chosen
    n = len(fhist)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4), dpi=120,
                             gridspec_kw={"height_ratios": [3, 2]})
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, (pos, fov) in enumerate(zip(positions, fhist)):
        ax = axes[0, i]
        ax.imshow(scene, cmap="magma", origin="upper", vmin=0, vmax=1)
        ax.plot(target[0], target[1], "*", color="cyan", markersize=10,
                markeredgecolor="black")
        rect = Rectangle((pos[0] - HALF - 0.5, pos[1] - HALF - 0.5),
                         FOVEA_SIZE, FOVEA_SIZE,
                         linewidth=1.0, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
        if i > 0:
            ax.plot([positions[i - 1][0], pos[0]], [positions[i - 1][1], pos[1]],
                    "-", color="#7fff7f", linewidth=1.0)
        ax.set_title(f"t={i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax2 = axes[1, i]
        ax2.imshow(fov, cmap="magma", origin="upper", vmin=0, vmax=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(f"max={fov.max():.2f}", fontsize=8)
    axes[0, 0].set_ylabel("scene + fovea")
    axes[1, 0].set_ylabel("fovea content")
    fig.suptitle(f"Single-trajectory fovea strip — found in {n - 1} saccades",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-epochs", type=int, default=150)
    p.add_argument("--c-epochs", type=int, default=150)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Training (this is a fresh full pipeline)...")
    result = train_full(seed=args.seed, m_epochs=args.m_epochs,
                        c_epochs=args.c_epochs, quiet=True)
    print(f"  M held-out R^2: {result['m_metrics']['explained_var']:.3f}")
    print(f"  Eval find_rate: {result['eval']['find_rate']:.3f}, "
          f"mean saccades: {result['eval']['mean_saccades']:.2f}")

    cfg = result["config"]
    C, M = _rebuild(result["C_state"], result["M_state"],
                    cfg["c_hidden"], cfg["m_hidden"], cfg["m_depth"])

    rng = np.random.default_rng(args.seed + 1000)
    plot_training_curves(result, os.path.join(args.outdir, "training_curves.png"))
    plot_scene_examples(C, M, rng, os.path.join(args.outdir, "scene_examples.png"))
    plot_trajectories(C, M, np.random.default_rng(args.seed + 2000),
                      os.path.join(args.outdir, "saccade_trajectories.png"))
    plot_fovea_strip(C, M, np.random.default_rng(args.seed + 3000),
                     os.path.join(args.outdir, "fovea_strip.png"))


if __name__ == "__main__":
    main()
