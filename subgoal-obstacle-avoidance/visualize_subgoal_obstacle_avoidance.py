"""Static visualizations for the trained subgoal-obstacle-avoidance system.

Outputs (in `viz/`):
  sample_paths.png        — 6 example arenas with sub-goal sequence + rollout
  obstacle_layouts.png    — 12 random arenas (the dataset look)
  subgoal_distribution.png — heatmap of where SG1 / SG2 land across 500 arenas
  training_curves.png     — LL imitation MSE + SGG cost / length / penalty / grad
  cost_landscape.png      — for one fixed arena, cost as a function of SG1 with
                            SG2 fixed at the SGG output (sanity check that the
                            generator sits at a low-cost point)
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from subgoal_obstacle_avoidance import (
    train_full, sample_arena, featurize_state, rollout,
    SubgoalGenerator, LowLevelPolicy, total_cost_with_grad,
    START, GOAL, ARENA_SIZE, OBS_RADIUS, N_SUBGOALS, N_OBSTACLES,
    GOAL_RADIUS,
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


def _draw_arena(ax, obstacles, *, draw_start_goal: bool = True):
    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for obs in obstacles:
        ax.add_patch(Circle((obs[0], obs[1]), obs[2],
                            facecolor="#444", edgecolor="black",
                            linewidth=0.6, alpha=0.85))
    if draw_start_goal:
        ax.plot(START[0], START[1], "s", color="#2ca02c",
                markersize=11, markeredgecolor="black", label="start")
        ax.plot(GOAL[0], GOAL[1], "*", color="#d62728",
                markersize=16, markeredgecolor="black", label="goal")
        ax.add_patch(Circle((GOAL[0], GOAL[1]), GOAL_RADIUS,
                            edgecolor="#d62728", facecolor="none",
                            linewidth=1.2, linestyle="--"))


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_obstacle_layouts(rng, out_path: str, n: int = 12):
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5), dpi=120)
    for ax in axes.flat:
        obs = sample_arena(rng)
        _draw_arena(ax, obs)
        # also draw the unobstructed straight line for reference
        ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
                "--", color="#888", linewidth=0.8, alpha=0.6)
    fig.suptitle("Random arena layouts (N=3 obstacles, one anchored on the diagonal)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sample_paths(SGG, LL, rng, out_path: str, n: int = 6):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), dpi=120)
    for ax in axes.flat:
        obs = sample_arena(rng)
        state = featurize_state(START, GOAL, obs)
        sgs, _, _ = SGG.forward(state[None])
        sgs = sgs[0]                                          # (K, 2)
        wps = [sgs[k] for k in range(N_SUBGOALS)]
        traj = rollout(START, GOAL, obs, wps, ll_policy=LL)
        direct = rollout(START, GOAL, obs, [], ll_policy=LL)

        _draw_arena(ax, obs)
        # direct (failed) trajectory in red
        traj_direct = direct["trajectory"]
        ax.plot(traj_direct[:, 0], traj_direct[:, 1], "-", color="#d62728",
                linewidth=1.2, alpha=0.6, label=f"direct ({'crashed' if direct['collided'] else 'ok'})")
        # SGG trajectory in green
        traj_sgg = traj["trajectory"]
        ax.plot(traj_sgg[:, 0], traj_sgg[:, 1], "-o", color="#2ca02c",
                markersize=2.5, linewidth=1.4,
                label=f"SGG ({'reached' if traj['success'] else 'crashed' if traj['collided'] else 'timeout'})")
        # sub-goals
        for k in range(N_SUBGOALS):
            ax.plot(sgs[k, 0], sgs[k, 1], "D", color="#1f77b4",
                    markersize=8, markeredgecolor="black",
                    label=f"sub-goal" if k == 0 else None)
            ax.annotate(f"SG{k+1}", (sgs[k, 0], sgs[k, 1]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#1f77b4")
        ax.set_title(f"steps={traj['n_steps']}  "
                     f"len={traj['path_length']:.1f}", fontsize=9)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Sample paths: direct (red) vs sub-goal-guided (green)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_subgoal_distribution(SGG, rng, out_path: str, n_arenas: int = 500):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=120)
    sg1_pts = []
    sg2_pts = []
    obstacle_pts = []
    for _ in range(n_arenas):
        obs = sample_arena(rng)
        state = featurize_state(START, GOAL, obs)
        sgs, _, _ = SGG.forward(state[None])
        sg1_pts.append(sgs[0, 0])
        sg2_pts.append(sgs[0, 1])
        obstacle_pts.append(obs[:, :2].copy())
    sg1 = np.stack(sg1_pts)
    sg2 = np.stack(sg2_pts)
    all_obs = np.concatenate(obstacle_pts, axis=0)

    for ax, pts, title, color in [
        (axes[0], all_obs, "Obstacle centers (over all arenas)", "#777"),
        (axes[1], sg1, "Sub-goal 1 placement", "#1f77b4"),
        (axes[2], sg2, "Sub-goal 2 placement", "#9467bd"),
    ]:
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=30,
                                   range=[[0, ARENA_SIZE], [0, ARENA_SIZE]])
        ax.imshow(H.T, origin="lower",
                  extent=(0, ARENA_SIZE, 0, ARENA_SIZE),
                  cmap="magma", aspect="equal")
        ax.plot(START[0], START[1], "s", color="#2ca02c", markersize=8,
                markeredgecolor="black")
        ax.plot(GOAL[0], GOAL[1], "*", color="#d62728", markersize=12,
                markeredgecolor="black")
        ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
                "--", color="white", linewidth=0.8, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sub-goal placement heatmap over {n_arenas} arenas "
                 f"(start green, goal red, diagonal dashed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(result: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120)
    ll = result["ll_losses"]
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(ll) + 1), ll, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("LL epoch")
    ax.set_ylabel("LL imitation MSE")
    ax.set_title("Phase 1: LL learns to head toward target (log y)")
    ax.grid(alpha=0.3)

    h = result["sgg_history"]
    ax = axes[0, 1]
    ax.plot(h["epoch"], h["total_cost"], color="#2ca02c", label="total")
    ax.plot(h["epoch"], h["path_length"], color="#ff7f0e", label="path length")
    ax.set_xlabel("SGG epoch")
    ax.set_ylabel("cost")
    ax.set_title("Phase 2: SGG cost via differentiable model")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(h["epoch"], h["obstacle_penalty"], color="#d62728")
    ax.set_xlabel("SGG epoch")
    ax.set_ylabel("mean obstacle penalty")
    ax.set_title("Obstacle-line-integral penalty")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(h["epoch"], h["grad_norm"], color="#9467bd")
    ax.set_yscale("log")
    ax.set_xlabel("SGG epoch")
    ax.set_ylabel("gradient norm")
    ax.set_title("|grad| (clipped at 5.0)")
    ax.grid(alpha=0.3)

    sgg = result["eval"]["sgg"]
    direct = result["eval"]["direct"]
    fig.suptitle(
        f"subgoal-obstacle-avoidance — eval success {sgg['success_rate']*100:.1f}%  "
        f"(direct baseline {direct['success_rate']*100:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cost_landscape(SGG, rng, out_path: str, grid: int = 60):
    """For a single fixed arena, sweep SG1 over a grid; SG2 fixed at SGG output.
    Show cost surface and mark where the SGG net actually places SG1."""
    obs = sample_arena(rng)
    state = featurize_state(START, GOAL, obs)
    sgs, _, _ = SGG.forward(state[None])
    sgs = sgs[0]

    xs = np.linspace(0.0, ARENA_SIZE, grid)
    ys = np.linspace(0.0, ARENA_SIZE, grid)
    Z = np.zeros((grid, grid), dtype=np.float32)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            sg_try = sgs.copy()
            sg_try[0] = np.array([x, y], dtype=np.float32)
            c, _, _, _ = total_cost_with_grad(sg_try, START, GOAL, obs)
            # row index is y, col index is x for imshow
            Z[iy, ix] = c

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=120)
    im = ax.imshow(Z, origin="lower",
                   extent=(0, ARENA_SIZE, 0, ARENA_SIZE),
                   cmap="magma", aspect="equal")
    fig.colorbar(im, ax=ax, label="total cost (sweeping SG1)")
    # overlay obstacles, start, goal
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2],
                            facecolor="black", edgecolor="white",
                            linewidth=0.7, alpha=0.7))
    ax.plot(START[0], START[1], "s", color="#2ca02c", markersize=10,
            markeredgecolor="black", label="start")
    ax.plot(GOAL[0], GOAL[1], "*", color="#d62728", markersize=14,
            markeredgecolor="black", label="goal")
    ax.plot(sgs[1, 0], sgs[1, 1], "D", color="#9467bd", markersize=10,
            markeredgecolor="white", label="SG2 (fixed)")
    ax.plot(sgs[0, 0], sgs[0, 1], "o", color="cyan", markersize=11,
            markeredgecolor="black", label="SGG output (SG1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, ARENA_SIZE); ax.set_ylim(0, ARENA_SIZE)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Cost landscape over SG1 (SG2 fixed). "
                 f"SGG output cost = {Z.min():.2f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sgg-epochs", type=int, default=400)
    p.add_argument("--ll-epochs", type=int, default=20)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Training fresh pipeline...")
    result = train_full(seed=args.seed,
                        sgg_epochs=args.sgg_epochs,
                        ll_epochs=args.ll_epochs,
                        quiet=True)
    sgg_e = result["eval"]["sgg"]
    direct_e = result["eval"]["direct"]
    print(f"  Eval: SGG success {sgg_e['success_rate']*100:.1f}%, "
          f"direct {direct_e['success_rate']*100:.1f}%")

    SGG, LL = _rebuild(result)

    plot_training_curves(result, os.path.join(args.outdir, "training_curves.png"))
    plot_obstacle_layouts(np.random.default_rng(args.seed + 100),
                          os.path.join(args.outdir, "obstacle_layouts.png"))
    plot_sample_paths(SGG, LL, np.random.default_rng(args.seed + 200),
                      os.path.join(args.outdir, "sample_paths.png"))
    plot_subgoal_distribution(SGG, np.random.default_rng(args.seed + 300),
                              os.path.join(args.outdir, "subgoal_distribution.png"))
    plot_cost_landscape(SGG, np.random.default_rng(args.seed + 400),
                        os.path.join(args.outdir, "cost_landscape.png"))


if __name__ == "__main__":
    main()
