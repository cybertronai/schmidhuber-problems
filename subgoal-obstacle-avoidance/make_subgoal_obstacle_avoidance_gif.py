"""Generate `subgoal_obstacle_avoidance.gif`.

Three side-by-side arenas. On each, the agent steps from start through the
two SGG-emitted sub-goals to the goal. The doomed direct-line trajectory is
drawn in faint red as the no-sub-goal counterfactual.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from subgoal_obstacle_avoidance import (
    train_full, sample_arena, featurize_state, rollout,
    SubgoalGenerator, LowLevelPolicy,
    START, GOAL, ARENA_SIZE, OBS_RADIUS, GOAL_RADIUS, N_SUBGOALS,
)


def _rebuild(result):
    rng = np.random.default_rng(0)
    cfg = result["config"]
    SGG = SubgoalGenerator.make(rng, hidden=cfg["sgg_hidden"])
    SGG.W = [w.copy() for w in result["_SGG_W"]]
    SGG.b = [bb.copy() for bb in result["_SGG_b"]]
    LL = LowLevelPolicy.make(rng, hidden=cfg["ll_hidden"])
    LL.W = [w.copy() for w in result["_LL_W"]]
    LL.b = [bb.copy() for bb in result["_LL_b"]]
    return SGG, LL


def collect_episode(SGG, LL, obs):
    state = featurize_state(START, GOAL, obs)
    sgs, _, _ = SGG.forward(state[None])
    sgs = sgs[0]
    wps = [sgs[k] for k in range(N_SUBGOALS)]
    sg_run = rollout(START, GOAL, obs, wps, ll_policy=LL)
    direct_run = rollout(START, GOAL, obs, [], ll_policy=LL)
    return {
        "obstacles": obs,
        "subgoals": sgs,
        "traj_sgg": sg_run["trajectory"],
        "success_sgg": sg_run["success"],
        "traj_direct": direct_run["trajectory"],
        "collided_direct": direct_run["collided"],
    }


def render_frame(panels, t_idx_list, dpi: int = 80) -> np.ndarray:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0), dpi=dpi)
    if n == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        p = panels[k]
        obs = p["obstacles"]
        sgs = p["subgoals"]
        traj = p["traj_sgg"]
        traj_d = p["traj_direct"]
        ti_sgg = min(t_idx_list[k], len(traj) - 1)
        ti_d = min(t_idx_list[k], len(traj_d) - 1)
        # arena
        ax.set_xlim(0, ARENA_SIZE)
        ax.set_ylim(0, ARENA_SIZE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for o in obs:
            ax.add_patch(Circle((o[0], o[1]), o[2],
                                facecolor="#444", edgecolor="black",
                                linewidth=0.6, alpha=0.9))
        # direct trajectory (counterfactual, ghost-red)
        if ti_d > 0:
            d = traj_d[:ti_d + 1]
            ax.plot(d[:, 0], d[:, 1], "-", color="#d62728", linewidth=1.0,
                    alpha=0.45)
            if p["collided_direct"] and ti_d == len(traj_d) - 1:
                ax.plot(d[-1, 0], d[-1, 1], "x", color="#d62728",
                        markersize=10, mew=2)
        # sub-goal markers (only show ones not yet visited)
        for k_sg in range(N_SUBGOALS):
            ax.plot(sgs[k_sg, 0], sgs[k_sg, 1], "D", color="#1f77b4",
                    markersize=10, markeredgecolor="black", alpha=0.85)
            ax.annotate(f"SG{k_sg + 1}", (sgs[k_sg, 0], sgs[k_sg, 1]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=9, color="#1f77b4")
        # SGG trajectory
        s = traj[:ti_sgg + 1]
        if len(s) > 1:
            ax.plot(s[:, 0], s[:, 1], "-", color="#2ca02c", linewidth=2.0,
                    alpha=0.95)
        ax.plot(s[-1, 0], s[-1, 1], "o", color="#2ca02c", markersize=8,
                markeredgecolor="black")
        # start, goal
        ax.plot(START[0], START[1], "s", color="#2ca02c",
                markersize=10, markeredgecolor="black")
        ax.plot(GOAL[0], GOAL[1], "*", color="#d62728",
                markersize=14, markeredgecolor="black")
        ax.add_patch(Circle((GOAL[0], GOAL[1]), GOAL_RADIUS,
                            edgecolor="#d62728", facecolor="none",
                            linewidth=1.0, linestyle="--"))
        title = f"arena {k + 1}: t={ti_sgg}"
        if p["success_sgg"] and ti_sgg == len(traj) - 1:
            title += "  ✓"
        ax.set_title(title, fontsize=10)

    fig.suptitle("subgoal-obstacle-avoidance — green: SG-guided  red: direct (counterfactual)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def write_gif(frames, out_path: str, fps: int = 6):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sgg-epochs", type=int, default=400)
    p.add_argument("--ll-epochs", type=int, default=20)
    p.add_argument("--n-panels", type=int, default=3)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=72)
    p.add_argument("--hold-frames", type=int, default=8)
    p.add_argument("--out", type=str, default="subgoal_obstacle_avoidance.gif")
    args = p.parse_args()

    print("Training pipeline...")
    result = train_full(seed=args.seed,
                        sgg_epochs=args.sgg_epochs,
                        ll_epochs=args.ll_epochs,
                        quiet=True)
    sgg = result["eval"]["sgg"]
    direct = result["eval"]["direct"]
    print(f"  eval: SGG success {sgg['success_rate']*100:.1f}%, "
          f"direct {direct['success_rate']*100:.1f}%")

    SGG, LL = _rebuild(result)

    # Pick non-trivial arenas: SGG must succeed AND direct must crash, AND the
    # SGG trajectory must be long enough to be visually interesting.
    rng = np.random.default_rng(args.seed + 9000)
    candidates = []
    while len(candidates) < args.n_panels and len(candidates) < 60:
        for _ in range(80):
            obs = sample_arena(rng)
            ep = collect_episode(SGG, LL, obs)
            ok = ep["success_sgg"] and ep["collided_direct"] and len(ep["traj_sgg"]) >= 30
            if ok:
                candidates.append(ep)
                if len(candidates) == args.n_panels:
                    break
        if len(candidates) < args.n_panels:
            # relax the trajectory-length constraint
            for _ in range(80):
                obs = sample_arena(rng)
                ep = collect_episode(SGG, LL, obs)
                if ep["success_sgg"] and ep["collided_direct"]:
                    candidates.append(ep)
                    if len(candidates) == args.n_panels:
                        break
            break

    panels = candidates[:args.n_panels]
    max_len = max(len(p["traj_sgg"]) for p in panels)
    print(f"  selected {len(panels)} arenas, max trajectory length {max_len}")

    frames = []
    # Subsample frames to keep file size moderate
    step = max(1, max_len // 60)
    for t in range(0, max_len, step):
        t_idx_list = [t for _ in panels]
        frames.append(render_frame(panels, t_idx_list, dpi=args.dpi))
    # hold the final frame
    final_t = [len(p["traj_sgg"]) - 1 for p in panels]
    frames.append(render_frame(panels, final_t, dpi=args.dpi))
    for _ in range(args.hold_frames):
        frames.append(frames[-1])

    write_gif(frames, args.out, fps=args.fps)
    sz = os.path.getsize(args.out)
    print(f"  wrote {args.out} ({len(frames)} frames, {sz / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
