"""Static visualisations for world-models-carracing.

Reads run.json produced by `world_models_carracing.py --seed 0 --save-json
run.json` and writes PNGs to viz/:

  1. track_layout.png             rendered track + centerline + spawn point
  2. training_curves.png          V loss, M loss, CMA-ES best/mean fitness
  3. cma_es_curve.png             CMA-ES generation vs best return + sigma
                                  (THE headline figure)
  4. vae_reconstruction.png       8 input patches vs VAE-reconstructed patches
  5. policy_trajectory.png        car path on track for the trained controller
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


def track_layout(out):
    track = out["track"]
    mask = np.array(track["mask"], dtype=np.float32)
    cx = np.array(track["centerline_x"])
    cy = np.array(track["centerline_y"])
    E = track["grid_extent"]
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(mask, extent=[-E, E, -E, E], origin="lower",
              cmap="Greys", alpha=0.85)
    ax.plot(cx, cy, color="#d4694e", linewidth=1.2,
            label="centerline")
    # spawn point: centerline sample 0
    ax.scatter([cx[0]], [cy[0]], color="#5a9bd4", s=80, zorder=5,
               edgecolor="black", linewidth=1.0, label="spawn")
    # tangent arrow at spawn
    dx = cx[1] - cx[0]
    dy = cy[1] - cy[0]
    n = np.hypot(dx, dy) + 1e-9
    ax.arrow(cx[0], cy[0], dx / n * 1.5, dy / n * 1.5,
             head_width=0.3, head_length=0.3,
             fc="#5a9bd4", ec="#5a9bd4", linewidth=1.5)
    ax.set_xlim(-E, E)
    ax.set_ylim(-E, E)
    ax.set_aspect("equal")
    ax.set_title(
        f"numpy 2-D racing track  (half-width = {track['half_width']:.2f}, "
        f"on-track ratio = {float(mask.mean()):.2f})"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "track_layout.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def training_curves(out):
    v = np.array(out["v_loss"])
    m = np.array(out["m_loss"])
    h = out["cma_history"]
    cma_best = np.array([d["best"] for d in h])
    cma_mean = np.array([d["mean"] for d in h])
    cma_med = np.array([d["median"] for d in h])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].plot(v, color="#444", linewidth=0.9)
    axs[0].set_xlabel("batch")
    axs[0].set_ylabel("BCE loss")
    axs[0].set_title("V (autoencoder) — reconstruction loss")
    axs[0].set_yscale("log")
    axs[0].grid(alpha=0.3)

    axs[1].plot(m, color="#5a9bd4", linewidth=0.9)
    axs[1].set_xlabel("batch")
    axs[1].set_ylabel("MSE on next-z")
    axs[1].set_title("M (LSTM) — world-model loss")
    axs[1].set_yscale("log")
    axs[1].grid(alpha=0.3)

    gens = np.arange(len(cma_best))
    axs[2].plot(gens, cma_best, color="#d4694e", label="best", linewidth=1.6)
    axs[2].plot(gens, cma_mean, color="#7faa53", label="mean", linewidth=1.2)
    axs[2].plot(gens, cma_med, color="#7faa53", linestyle=":",
                label="median", linewidth=1.2, alpha=0.6)
    axs[2].axhline(out["random_baseline"]["mean_return"], color="#999",
                   linestyle="--", linewidth=1, label="random baseline")
    axs[2].set_xlabel("generation")
    axs[2].set_ylabel("episode return")
    axs[2].set_title("C (CMA-ES) — controller fitness")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "training_curves.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def cma_es_curve(out):
    h = out["cma_history"]
    gens = np.arange(len(h))
    best = np.array([d["best"] for d in h])
    mean = np.array([d["mean"] for d in h])
    sigma = np.array([d["sigma"] for d in h])
    rand = out["random_baseline"]
    final = out["final_eval"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(gens, best, color="#d4694e", marker="o", markersize=3,
            linewidth=1.8, label="best in pop")
    ax.plot(gens, mean, color="#7faa53", linewidth=1.4,
            label="population mean")
    ax.fill_between(gens, mean, best, color="#d4694e", alpha=0.08)
    ax.axhline(rand["mean_return"], color="#999", linestyle="--",
               linewidth=1, label=f"random ({rand['mean_return']:+.1f})")
    ax.axhline(final["mean_return"], color="#5a9bd4", linestyle=":",
               linewidth=1.2,
               label=f"final greedy ({final['mean_return']:+.1f})")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel("episode return")
    ax.set_title(
        "CMA-ES: linear controller (z, h_M)→(steer, throttle), "
        f"{out['config']['cma_popsize']} indiv × "
        f"{out['config']['cma_gens']} gens"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # right axis: sigma
    ax2 = ax.twinx()
    ax2.plot(gens, sigma, color="#888", linestyle=":", linewidth=1.2,
             label="σ")
    ax2.set_ylabel("CMA-ES step σ", color="#888")
    ax2.tick_params(axis="y", colors="#888")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "cma_es_curve.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def vae_reconstruction(out):
    sr = out["sample_recon"]
    inp = np.array(sr["input"])     # (8, 256)
    rec = np.array(sr["output"])    # (8, 256)
    z = np.array(sr["z"])           # (8, z_dim)
    n = inp.shape[0]
    fig, axs = plt.subplots(3, n, figsize=(1.6 * n, 5.0))
    for i in range(n):
        axs[0, i].imshow(inp[i].reshape(16, 16), cmap="Greys",
                         vmin=0, vmax=1)
        axs[0, i].set_xticks([]); axs[0, i].set_yticks([])
        axs[1, i].imshow(rec[i].reshape(16, 16), cmap="Greys",
                         vmin=0, vmax=1)
        axs[1, i].set_xticks([]); axs[1, i].set_yticks([])
        # latent code as a bar
        axs[2, i].bar(np.arange(z.shape[1]), z[i],
                      color="#5a9bd4", width=0.8)
        axs[2, i].set_xticks([])
        axs[2, i].set_ylim(z.min() - 0.2, z.max() + 0.2)
        axs[2, i].axhline(0, color="#aaaaaa", linewidth=0.5)
    axs[0, 0].set_ylabel("obs (16×16)")
    axs[1, 0].set_ylabel("V recon")
    axs[2, 0].set_ylabel(f"z (R^{z.shape[1]})")
    fig.suptitle("V (autoencoder): observation → latent → reconstruction",
                 y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "vae_reconstruction.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def policy_trajectory(out):
    track = out["track"]
    mask = np.array(track["mask"], dtype=np.float32)
    cx = np.array(track["centerline_x"])
    cy = np.array(track["centerline_y"])
    E = track["grid_extent"]
    states = np.array(out["demo"]["demo_states"])  # (T+1, 4)
    actions = np.array(out["demo"]["demo_actions"])  # (T, 2)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 6.0))
    ax = axs[0]
    ax.imshow(mask, extent=[-E, E, -E, E], origin="lower",
              cmap="Greys", alpha=0.4)
    ax.plot(cx, cy, color="#bbbbbb", linewidth=0.8)
    # color path by step to show progress
    cmap = plt.get_cmap("viridis")
    n_steps = states.shape[0]
    for i in range(n_steps - 1):
        ax.plot(states[i:i + 2, 0], states[i:i + 2, 1],
                color=cmap(i / max(1, n_steps - 1)), linewidth=1.6)
    ax.scatter([states[0, 0]], [states[0, 1]], color="#5a9bd4", s=70,
               edgecolor="black", linewidth=1.0, zorder=5, label="start")
    ax.scatter([states[-1, 0]], [states[-1, 1]], color="#d4694e", s=70,
               edgecolor="black", linewidth=1.0, zorder=5, label="end")
    ax.set_xlim(-E, E); ax.set_ylim(-E, E)
    ax.set_aspect("equal")
    ax.set_title(
        f"trained controller trajectory  (return = "
        f"{out['demo']['demo_cum_reward']:+.1f}, T = {n_steps - 1})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axs[1]
    t = np.arange(actions.shape[0])
    ax2.plot(t, actions[:, 0], color="#5a9bd4", label="steer", linewidth=1.4)
    ax2.plot(t, actions[:, 1], color="#d4694e", label="throttle",
             linewidth=1.4)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax2.set_xlabel("step")
    ax2.set_ylabel("action ∈ [-1, 1]")
    ax2.set_title("controller actions over the rollout")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "policy_trajectory.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        import subprocess
        print("[viz] run.json not found, running world_models_carracing.py first ...")
        subprocess.run(
            ["python3", os.path.join(HERE, "world_models_carracing.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )
    out = _load(json_path)
    track_layout(out)
    training_curves(out)
    cma_es_curve(out)
    vae_reconstruction(out)
    policy_trajectory(out)
    print(f"Wrote 5 PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
