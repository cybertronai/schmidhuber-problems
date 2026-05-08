"""Generate torcs_vision_evolution.gif — the best DCT-compressed controller
driving around the synthetic track.

Each frame shows two panels:
  left  : top-down view of the full track with the car's trajectory so far
  right : the 16x16 grayscale observation the controller currently sees
The frame title prints the current step, action, and lap fraction.

Reads viz/run_dct_seed{seed}.npz produced by torcs_vision_evolution.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

import torcs_vision_evolution as tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="viz")
    ap.add_argument("--gif", type=str, default="torcs_vision_evolution.gif")
    ap.add_argument("--theta-offset", type=float, default=0.0)
    ap.add_argument("--T-max", type=int, default=420)
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="render every N-th env step")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    base = Path(args.outdir)
    npz_path = base / f"run_dct_seed{args.seed}.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"missing {npz_path}. Run torcs_vision_evolution.py "
            f"--seed {args.seed} --save-npz {npz_path} first."
        )
    npz = np.load(npz_path)
    theta = npz["theta_best"]

    nc = tv.NetConfig()
    env = tv.EnvConfig(max_steps=args.T_max)
    track = tv.build_track(env.track)
    M = tv.build_idct_matrix(nc.img_size, nc.dct_k)

    r = tv.rollout(theta, nc, M, track, env,
                   return_traj=True, theta_offset=args.theta_offset)
    traj = r["traj_car"]
    obs = r["traj_obs"]
    actions = r["traj_act"]
    perim = track["perimeter"]

    frames = []
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(2.4, 1.0))
    ax_track = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])

    n_steps = obs.shape[0]
    for step in range(0, n_steps, args.frame_stride):
        ax_track.clear(); ax_obs.clear()
        ax_track.imshow(track["mask"], cmap="Greys", origin="lower",
                        extent=(-env.track.x_range, env.track.x_range,
                                -env.track.y_range, env.track.y_range),
                        alpha=0.5)
        ax_track.plot(track["cl"][:, 0], track["cl"][:, 1], "g--", alpha=0.5, lw=0.7)
        ax_track.plot(traj[:step + 1, 0], traj[:step + 1, 1], "r-", lw=1.3)
        car = traj[step]
        # car as a small triangle pointing along heading
        L = 0.25
        cos_t, sin_t = np.cos(car[2]), np.sin(car[2])
        nose = (car[0] + L * cos_t, car[1] + L * sin_t)
        l = (car[0] - 0.5 * L * cos_t + 0.3 * L * sin_t,
             car[1] - 0.5 * L * sin_t - 0.3 * L * cos_t)
        rgt = (car[0] - 0.5 * L * cos_t - 0.3 * L * sin_t,
               car[1] - 0.5 * L * sin_t + 0.3 * L * cos_t)
        tri = plt.Polygon([nose, l, rgt], color="red", ec="black", lw=0.5)
        ax_track.add_patch(tri)
        ax_track.set_aspect("equal")
        ax_track.set_xlim(-env.track.x_range, env.track.x_range)
        ax_track.set_ylim(-env.track.y_range, env.track.y_range)
        ax_track.set_title(
            f"step {step:3d}/{n_steps - 1}   action {actions[step]:+.2f}   "
            f"lap_frac {step / max(n_steps - 1, 1) * r['lap_frac']:.2f}",
            fontsize=10)
        ax_track.set_xticks([]); ax_track.set_yticks([])

        ax_obs.imshow(obs[step], cmap="Greys", interpolation="nearest",
                      vmin=0, vmax=1)
        ax_obs.set_title("16x16 observation\n(DCT controller input)", fontsize=9)
        ax_obs.set_xticks([]); ax_obs.set_yticks([])

        fig.tight_layout()
        fig.canvas.draw()
        # NB: buffer_rgba returns (H, W, 4) uint8; drop alpha for GIF.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())

    plt.close(fig)
    out_path = Path(args.gif)
    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
    size_kb = out_path.stat().st_size / 1024.0
    print(f"wrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
