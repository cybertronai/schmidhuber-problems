"""Static visualizations for torcs-vision-evolution.

Reads viz/run_dct_seed{seed}.json + .npz (and optionally viz/run_raw_seed{seed}.json)
produced by torcs_vision_evolution.py and writes:

  viz/headline_compression.png   - bar chart: raw vs DCT-compressed search-space size
  viz/training_curves.png        - lap-fraction per generation, best + mean
  viz/decoded_filters.png        - the 16 hidden-unit weight images, decoded from DCT
  viz/track_and_rollout.png      - track mask + best-controller trajectory
  viz/observation_strip.png      - sample 16x16 observations along the rollout
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torcs_vision_evolution as tv


def _load(seed: int, base: Path):
    dct_json = base / f"run_dct_seed{seed}.json"
    dct_npz = base / f"run_dct_seed{seed}.npz"
    if not dct_json.exists() or not dct_npz.exists():
        raise SystemExit(
            f"missing {dct_json} or {dct_npz}. Run torcs_vision_evolution.py "
            f"--seed {seed} --save-json {dct_json} --save-npz {dct_npz} first."
        )
    with open(dct_json) as f:
        summary = json.load(f)
    npz = np.load(dct_npz)
    raw_json = base / f"run_raw_seed{seed}.json"
    raw_summary = None
    if raw_json.exists():
        with open(raw_json) as f:
            raw_summary = json.load(f)
    return summary, npz, raw_summary


def plot_headline_compression(summary, raw_summary, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    n_raw = summary["n_raw"]
    n_dct = summary["n_compressed"]
    bars = ax.bar(["Raw weights\n(K = N = 16)", f"DCT-compressed\n(K = {summary['config']['dct_k']})"],
                  [n_raw, n_dct],
                  color=["#888", "#1f77b4"], edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax.set_ylabel("evolved-parameter count")
    ax.set_title(
        f"DCT-compressed search space: {n_raw} raw weights -> {n_dct} coefficients "
        f"({n_raw / n_dct:.1f}x compression)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def plot_training_curves(summary, raw_summary, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    h = summary["history"]
    ax.plot(h["gen"], h["best_lap"], color="#1f77b4", label=f"DCT K={summary['config']['dct_k']} best")
    ax.plot(h["gen"], h["mean_lap"], color="#1f77b4", alpha=0.4,
            label=f"DCT K={summary['config']['dct_k']} mean")
    if raw_summary is not None:
        rh = raw_summary["history"]
        ax.plot(rh["gen"], rh["best_lap"], color="#888", label=f"Raw K=16 best")
        ax.plot(rh["gen"], rh["mean_lap"], color="#888", alpha=0.4, label=f"Raw K=16 mean")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.7, label="one full lap")
    ax.set_xlabel("generation")
    ax.set_ylabel("lap fraction (mean over 3 trials)")
    ax.set_title(f"Evolution progress, seed {summary['seed']}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def plot_decoded_filters(summary, npz, out: Path):
    nc = tv.NetConfig(hidden=summary["config"]["hidden"], dct_k=summary["config"]["dct_k"])
    M = tv.build_idct_matrix(nc.img_size, nc.dct_k)
    theta = npz["theta_best"]
    p = tv.split_params(theta, nc)
    W1 = tv.decode_W1(p["coefs"], M, nc.img_size)        # (256, H)
    imgs = W1.T.reshape(nc.hidden, nc.img_size, nc.img_size)

    cols = 4
    rows = (nc.hidden + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
    vmax = float(np.max(np.abs(imgs)))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < nc.hidden:
            ax.imshow(imgs[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(f"h{i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Decoded W1 filters (16x16) — each is reconstructed from {nc.dct_k}x{nc.dct_k} = "
        f"{nc.dct_k**2} DCT coefficients",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def plot_track_and_rollout(summary, npz, out: Path):
    nc = tv.NetConfig(hidden=summary["config"]["hidden"], dct_k=summary["config"]["dct_k"])
    env = tv.EnvConfig(max_steps=summary["config"]["max_steps"])
    track = tv.build_track(env.track)
    M = tv.build_idct_matrix(nc.img_size, nc.dct_k)
    theta = npz["theta_best"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, off in zip(axes, env.init_theta_offsets):
        r = tv.rollout(theta, nc, M, track, env, return_traj=True, theta_offset=off)
        # plot mask
        ax.imshow(track["mask"], cmap="Greys", origin="lower",
                  extent=(-env.track.x_range, env.track.x_range,
                          -env.track.y_range, env.track.y_range),
                  alpha=0.4)
        ax.plot(track["cl"][:, 0], track["cl"][:, 1], "g--", alpha=0.6, lw=0.8, label="centre line")
        traj = r["traj_car"]
        ax.plot(traj[:, 0], traj[:, 1], "r-", lw=1.4, label=f"trajectory ({r['steps']} steps)")
        ax.plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="start")
        ax.plot(traj[-1, 0], traj[-1, 1], "rs", markersize=7, label="end")
        ax.set_aspect("equal")
        ax.set_xlim(-env.track.x_range, env.track.x_range)
        ax.set_ylim(-env.track.y_range, env.track.y_range)
        ax.set_title(f"heading offset = {off:+.2f} rad   lap_frac = {r['lap_frac']:.2f}",
                     fontsize=10)
        ax.grid(linestyle=":", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Best DCT-compressed controller, three initial-heading trials", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def plot_observation_strip(summary, npz, out: Path, n_show: int = 8):
    nc = tv.NetConfig(hidden=summary["config"]["hidden"], dct_k=summary["config"]["dct_k"])
    env = tv.EnvConfig(max_steps=summary["config"]["max_steps"])
    track = tv.build_track(env.track)
    M = tv.build_idct_matrix(nc.img_size, nc.dct_k)
    r = tv.rollout(npz["theta_best"], nc, M, track, env, return_traj=True, theta_offset=0.0)
    obs = r["traj_obs"]
    if obs.shape[0] < n_show:
        n_show = obs.shape[0]
    idxs = np.linspace(0, obs.shape[0] - 1, n_show).astype(int)

    fig, axes = plt.subplots(1, n_show, figsize=(2.0 * n_show, 2.4))
    for ax, idx in zip(axes, idxs):
        ax.imshow(obs[idx], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"step {idx}\naction {r['traj_act'][idx]:+.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("16x16 grayscale observations along the best controller's trajectory",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    base = Path(args.outdir)
    base.mkdir(exist_ok=True, parents=True)
    summary, npz, raw_summary = _load(args.seed, base)

    plot_headline_compression(summary, raw_summary, base / "headline_compression.png")
    plot_training_curves(summary, raw_summary, base / "training_curves.png")
    plot_decoded_filters(summary, npz, base / "decoded_filters.png")
    plot_track_and_rollout(summary, npz, base / "track_and_rollout.png")
    plot_observation_strip(summary, npz, base / "observation_strip.png")


if __name__ == "__main__":
    main()
