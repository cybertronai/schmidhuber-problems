"""Build relational_nem_bouncing_balls.gif

Side-by-side animation: ground truth | non-relational rollout | relational
rollout. Uses sample trajectories saved in run.json (rolled out closed-loop
on the trained models for the K=K_train evaluation set).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))


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
    p.add_argument("--out", default=os.path.join(HERE,
                                                 "relational_nem_bouncing_balls.gif"))
    p.add_argument("--sample", type=int, default=0,
                   help="Which sample index from run.json to animate.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--frame-size", type=int, default=64)
    args = p.parse_args()

    if not os.path.exists(args.run):
        raise SystemExit(
            f"{args.run} not found. Run "
            f"`python3 relational_nem_bouncing_balls.py --seed 0` first."
        )
    with open(args.run) as f:
        run = json.load(f)

    samples = run["samples"]
    true = np.array(samples["true"])              # (T, B, K, 4)
    nr_  = np.array(samples["non_relational"])
    re_  = np.array(samples["relational"])
    T = true.shape[0]
    s = min(args.sample, true.shape[1] - 1)
    radius = run["config"]["radius"]
    sigma = max(0.6 * radius, 0.04)
    H = W = args.frame_size

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.6))
    titles = ["ground truth", "non-relational rollout", "relational rollout"]
    cmaps = ["Greys", "Reds", "Greens"]
    trajs = [true, nr_, re_]
    ims = []
    for ax, t, cm in zip(axes, titles, cmaps):
        img = np.zeros((H, W))
        im = ax.imshow(img, cmap=cm, origin="lower", extent=(0, 1, 0, 1),
                       vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=11)
        ims.append(im)

    fig.suptitle(
        f"Bouncing balls (K={true.shape[2]}, T={T}); closed-loop rollout from "
        f"frame 0\n"
        f"non-rel mean vel-MSE = {run['rollout']['K' + str(run['config']['K_train'])]['mean_vel_err_non_relational']:.3f}, "
        f"rel = {run['rollout']['K' + str(run['config']['K_train'])]['mean_vel_err_relational']:.3f}",
        fontsize=10
    )
    step_text = fig.text(0.5, 0.04, "", ha="center", fontsize=9)
    fig.tight_layout()

    def update(t):
        for im, traj in zip(ims, trajs):
            im.set_data(_render(traj[t, s], H, W, sigma))
        step_text.set_text(f"step {t+1}/{T}")
        return ims + [step_text]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // args.fps,
                         blit=False)
    print(f"writing {args.out} ({T} frames at {args.fps} fps) ...")
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
