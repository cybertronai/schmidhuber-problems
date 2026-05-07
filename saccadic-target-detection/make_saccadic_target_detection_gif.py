"""Generate `saccadic_target_detection.gif`: fovea trajectory over a scene.

The animation shows three test scenes laid side by side; on each, the trained
fovea moves toward the target. After all three episodes finish, the GIF holds
on the final frame for a moment, then loops.
"""
from __future__ import annotations
import argparse
import io
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from saccadic_target_detection import (
    train_full, make_scene, extract_fovea, halo_intensity,
    rollout_controller_step, Controller, WorldModel,
    SCENE_SIZE, FOVEA_SIZE, HALF, T_MAX, DETECT_RADIUS, target_indicator,
)


def _rebuild(C_state, M_state, c_hidden, m_hidden, m_depth):
    rng = np.random.default_rng(0)
    C = Controller.make(rng, hidden=c_hidden)
    C.W = [w.copy() for w in C_state["W"]]
    C.b = [bb.copy() for bb in C_state["b"]]
    M = WorldModel.make(rng, hidden=m_hidden, depth=m_depth)
    M.W = [w.copy() for w in M_state["W"]]
    M.b = [bb.copy() for bb in M_state["b"]]
    return C, M


def rollout_collect(C, M, scene, target, max_steps=T_MAX):
    pos = np.array([SCENE_SIZE / 2.0 - 0.5, SCENE_SIZE / 2.0 - 0.5], dtype=np.float32)
    positions = [pos.copy()]
    foveas = [extract_fovea(scene, pos)]
    for _ in range(max_steps):
        new_pos, ind, _ = rollout_controller_step(
            C, M, [scene], target[None], pos[None], lr=0.0, train=False
        )
        pos = new_pos[0]
        positions.append(pos.copy())
        foveas.append(extract_fovea(scene, pos))
        if ind[0] > 0.5:
            break
    return positions, foveas


def render_frame(scenes, targets, traj_lists, fovea_lists, t_idx_list, found_list,
                 dpi: int = 72) -> np.ndarray:
    """Render the three-panel view at frame index t (per panel)."""
    n_panels = len(scenes)
    fig, axes = plt.subplots(2, n_panels, figsize=(2.4 * n_panels, 4.0), dpi=dpi,
                             gridspec_kw={"height_ratios": [3, 2]})
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    for k in range(n_panels):
        scene = scenes[k]
        target = targets[k]
        traj = traj_lists[k]
        foveas = fovea_lists[k]
        ti = min(t_idx_list[k], len(traj) - 1)

        ax = axes[0, k]
        ax.imshow(scene, cmap="magma", origin="upper", vmin=0, vmax=1.0)
        ax.plot(target[0], target[1], "*", color="cyan", markersize=12,
                markeredgecolor="black")
        circ = Circle((target[0], target[1]), DETECT_RADIUS,
                      linewidth=1.0, edgecolor="cyan", facecolor="none", linestyle="--")
        ax.add_patch(circ)
        # path so far
        path = np.stack(traj[:ti + 1])
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], "-", color="#7fff7f",
                    linewidth=1.2, alpha=0.9)
        ax.plot(path[:, 0], path[:, 1], "o", color="#7fff7f", markersize=3)
        # current fovea box
        cur = traj[ti]
        rect = Rectangle((cur[0] - HALF - 0.5, cur[1] - HALF - 0.5),
                         FOVEA_SIZE, FOVEA_SIZE,
                         linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"scene {k + 1}: t={ti}"
        if found_list[k] and ti == len(traj) - 1:
            title += "  ✓ found"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, SCENE_SIZE - 0.5)
        ax.set_ylim(SCENE_SIZE - 0.5, -0.5)

        ax2 = axes[1, k]
        ax2.imshow(foveas[ti], cmap="magma", origin="upper", vmin=0, vmax=1.0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(f"fovea max={foveas[ti].max():.2f}", fontsize=9)

    fig.suptitle("Saccadic target detection — controller + world-model fovea trajectory",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def write_gif(frames, out_path: str, fps: int = 4):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-epochs", type=int, default=150)
    p.add_argument("--c-epochs", type=int, default=150)
    p.add_argument("--n-panels", type=int, default=3)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--dpi", type=int, default=72)
    p.add_argument("--hold-frames", type=int, default=4,
                   help="Number of frames to hold the final view before looping.")
    p.add_argument("--out", type=str, default="saccadic_target_detection.gif")
    args = p.parse_args()

    print("Training pipeline (this regenerates C and M)...")
    result = train_full(seed=args.seed, m_epochs=args.m_epochs,
                        c_epochs=args.c_epochs, quiet=True)
    print(f"  eval find_rate: {result['eval']['find_rate']:.3f}, "
          f"mean saccades {result['eval']['mean_saccades']:.2f}")

    cfg = result["config"]
    C, M = _rebuild(result["C_state"], result["M_state"],
                    cfg["c_hidden"], cfg["m_hidden"], cfg["m_depth"])

    rng = np.random.default_rng(args.seed + 9000)
    n_panels = args.n_panels
    # Pick non-trivial trajectories (longer is more visually interesting). The
    # trained controller is fast (median ~2 saccades) so we sample many scenes
    # and keep the longest few for the GIF.
    candidates = []
    for _ in range(60):
        scene, target = make_scene(rng)
        traj, foveas = rollout_collect(C, M, scene, target)
        candidates.append((len(traj), scene, target, traj, foveas))
    # Sort by trajectory length descending, take the top few that found target.
    candidates.sort(key=lambda c: -c[0])
    scenes, targets, traj_lists, fovea_lists, found_list = [], [], [], [], []
    for length, scene, target, traj, foveas in candidates:
        if length < 2:
            continue
        scenes.append(scene)
        targets.append(target)
        traj_lists.append(traj)
        fovea_lists.append(foveas)
        found_list.append(target_indicator(traj[-1], target) > 0.5)
        if len(scenes) == n_panels:
            break

    max_len = max(len(t) for t in traj_lists)
    print(f"max trajectory length: {max_len}")
    frames = []
    for t in range(max_len):
        t_idx_list = [min(t, len(tj) - 1) for tj in traj_lists]
        frames.append(render_frame(scenes, targets, traj_lists, fovea_lists,
                                   t_idx_list, found_list, dpi=args.dpi))
    # hold final
    for _ in range(args.hold_frames):
        frames.append(frames[-1])

    write_gif(frames, args.out, fps=args.fps)
    sz = os.path.getsize(args.out)
    print(f"wrote {args.out} ({len(frames)} frames, {sz / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
