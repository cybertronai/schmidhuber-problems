"""Static visualisations for world-models-vizdoom-dream.

Reads run.json produced by `world_models_vizdoom_dream.py --save-json run.json`
and writes PNGs to viz/:

  1. env_layout.png            -- the dodging-fireballs gridworld
  2. v_m_curves.png            -- V autoencoder loss + M (LSTM) losses
  3. survival_real_vs_dream.png -- THE headline figure: C survival in
                                   real env vs in M's dream over ES
                                   iterations, plus the direct-trained
                                   baseline
  4. final_survival_dist.png   -- histogram of survival times (random,
                                   C_dream, C_real) at final eval
  5. weight_matrix_C.png       -- C's tiny linear policy as a heatmap
                                   ([z|h] -> action)
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
    cfg = out["config"]
    W, H = cfg["W"], cfg["H"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    # Grid lines
    for x in range(W + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, H - 0.5], color="#dddddd",
                linewidth=0.6)
    for y in range(H + 1):
        ax.plot([-0.5, W - 0.5], [y - 0.5, y - 0.5], color="#dddddd",
                linewidth=0.6)
    # Top row "monster spawner"
    ax.add_patch(plt.Rectangle((-0.5, H - 1 - 0.5), W, 1.0,
                               facecolor="#ffe6e6", edgecolor="none",
                               alpha=0.8, zorder=0))
    ax.text(W / 2 - 0.5, H - 1, "monsters spawn fireballs",
            ha="center", va="center", fontsize=9, color="#a04040")
    # Example fireballs falling
    for fx, fy in [(2, H - 3), (5, H - 4), (1, H - 2)]:
        ax.add_patch(plt.Circle((fx, fy), 0.25, facecolor="#d4694e",
                                edgecolor="black", linewidth=0.6))
    # Agent at bottom
    agent_x = W // 2
    ax.add_patch(plt.Circle((agent_x, 0), 0.3, facecolor="#5a9bd4",
                            edgecolor="black", linewidth=1.0))
    ax.text(agent_x, -0.85, "agent", ha="center", va="top", fontsize=9,
            color="#3a6db4")
    # Action arrows
    ax.annotate("", xy=(agent_x - 1, -0.2), xytext=(agent_x - 0.4, -0.2),
                arrowprops=dict(arrowstyle="->", color="#666"))
    ax.annotate("", xy=(agent_x + 1, -0.2), xytext=(agent_x + 0.4, -0.2),
                arrowprops=dict(arrowstyle="->", color="#666"))
    ax.text(agent_x, -1.25, "left / stay / right", ha="center",
            fontsize=8, color="#666")
    ax.set_axis_off()
    ax.set_title(
        f"Dodging-fireballs gridworld ({W}x{H}, "
        f"max_fireballs={cfg['max_fireballs']}, "
        f"spawn_prob={cfg['spawn_prob']}, t_max={cfg['max_steps']})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "env_layout.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def v_m_curves(out):
    v = out["v_losses"]
    m = out["m_losses"]
    m_total = [d["total"] for d in m]
    m_z = [d["z"] for d in m]
    m_r = [d["r"] for d in m]
    m_d = [d["done"] for d in m]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].plot(v, color="#5a9bd4", linewidth=1.4)
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("MSE")
    axs[0].set_title("V (autoencoder) reconstruction loss")
    axs[0].set_yscale("log")
    axs[0].grid(alpha=0.3)
    axs[1].plot(m_total, color="#444", label="total", linewidth=1.4)
    axs[1].plot(m_z, color="#5a9bd4", label="z (MSE)", linewidth=1.0)
    axs[1].plot(m_r, color="#7faa53", label="reward (MSE)", linewidth=1.0)
    axs[1].plot(m_d, color="#d4694e", label="done (BCE)", linewidth=1.0)
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("loss")
    axs[1].set_title("M (LSTM world model) training losses")
    axs[1].set_yscale("log")
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "v_m_curves.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def survival_real_vs_dream(out):
    """THE headline figure -- side-by-side real vs dream survival curves."""
    dh = out["dream_history"]
    rh = out["real_history"]
    rand_mu = out["data"]["rand_baseline_len_mean"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)

    # LEFT: dream-trained C, both inside M (dream curve) and zero-shot in
    # the real env (real curve).
    ax = axs[0]
    ax.plot(dh["iter"], dh["dream_len"], color="#7faa53",
            linewidth=1.6, label="C_dream eval IN M's dream")
    if dh["real_iter"]:
        ax.plot(dh["real_iter"], dh["real_len"], color="#d4694e",
                marker="o", markersize=4, linewidth=1.6,
                label="C_dream eval IN REAL env (zero-shot transfer)")
    ax.axhline(rand_mu, color="#999", linestyle="--", linewidth=1,
               label=f"random policy ({rand_mu:.1f})")
    ax.set_xlabel("ES iteration")
    ax.set_ylabel("mean survival steps")
    ax.set_title("C trained INSIDE M's dream\n"
                 "(no real-env interaction during training)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # RIGHT: baseline C trained directly in real env (reference)
    ax = axs[1]
    if rh is not None:
        ax.plot(rh["iter"], rh["real_len"], color="#5a9bd4",
                marker="o", markersize=4, linewidth=1.6,
                label="C_real eval IN REAL env")
    ax.axhline(rand_mu, color="#999", linestyle="--", linewidth=1,
               label=f"random policy ({rand_mu:.1f})")
    final = out["final_eval"]
    if final["C_dream_mean_len"] is not None:
        ax.axhline(final["C_dream_mean_len"], color="#d4694e",
                   linestyle=":", linewidth=1.4,
                   label=f"C_dream final ({final['C_dream_mean_len']:.1f})")
    ax.set_xlabel("ES iteration")
    ax.set_title("Baseline: C trained DIRECTLY in real env\n"
                 "(reference -- standard ES, no world model)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Headline: dream-trained controller transfers to the real env",
        fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "survival_real_vs_dream.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def final_survival_dist(out):
    fe = out["final_eval"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0, max(fe["C_dream_lens"]
                              + fe["random_lens"]
                              + (fe["C_real_lens"] or [0]) + [1]) + 4, 20)
    ax.hist(fe["random_lens"], bins=bins, alpha=0.55, color="#999999",
            label=f"random ({fe['random_mean_len']:.1f} +/- "
                  f"{fe['random_std_len']:.1f})")
    if fe["C_real_lens"]:
        ax.hist(fe["C_real_lens"], bins=bins, alpha=0.55, color="#5a9bd4",
                label=f"C_real ({fe['C_real_mean_len']:.1f} +/- "
                      f"{fe['C_real_std_len']:.1f})")
    ax.hist(fe["C_dream_lens"], bins=bins, alpha=0.65, color="#d4694e",
            label=f"C_dream ({fe['C_dream_mean_len']:.1f} +/- "
                  f"{fe['C_dream_std_len']:.1f})")
    ax.set_xlabel("survival steps in real env")
    ax.set_ylabel("count")
    ax.set_title(f"Final survival distribution "
                 f"({fe['n_episodes']} eps each, real env)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "final_survival_dist.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def weight_matrix_C(out):
    cfg = out["config"]
    z_dim, h_dim, n_act = cfg["z_dim"], cfg["m_hidden"], cfg["n_actions"]
    c_hid = cfg.get("c_hidden", 0)
    theta_d = np.array(out["weights"]["theta_C_dream"])
    in_dim = z_dim + h_dim
    if c_hid > 0:
        # Slice the flat ES vector back into (W1, b1, W2, b2)
        sizes = [in_dim * c_hid, c_hid, c_hid * n_act, n_act]
        offs = np.cumsum([0] + sizes)
        W1 = theta_d[offs[0]:offs[1]].reshape(in_dim, c_hid)
        W2 = theta_d[offs[2]:offs[3]].reshape(c_hid, n_act)
        b2 = theta_d[offs[3]:offs[4]]
        # Effective input -> action linearisation: W1 @ W2 (ignoring tanh
        # nonlinearity around 0; this is a useful low-bias summary of how
        # each input dimension net-maps to each action).
        W_eff = W1 @ W2
        title = (f"C_dream: effective input -> action map (W1 @ W2)\n"
                 f"MLP arch: in={in_dim} -> tanh({c_hid}) -> {n_act};  "
                 f"b_out = [{b2[0]:+.2f}, {b2[1]:+.2f}, {b2[2]:+.2f}]")
    else:
        W_eff = theta_d[:in_dim * n_act].reshape(in_dim, n_act)
        b = theta_d[in_dim * n_act:]
        title = (f"C_dream linear policy: [z | h] -> action\n"
                 f"(b = [{b[0]:+.2f}, {b[1]:+.2f}, {b[2]:+.2f}])")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmax = float(np.abs(W_eff).max() + 1e-9)
    im = ax.imshow(W_eff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axhline(z_dim - 0.5, color="black", linewidth=1.2)
    ax.text(-0.6, z_dim / 2 - 0.5, "z", ha="right", va="center",
            fontsize=10, fontweight="bold")
    ax.text(-0.6, z_dim + h_dim / 2 - 0.5, "h", ha="right", va="center",
            fontsize=10, fontweight="bold")
    ax.set_xticks(range(n_act))
    ax.set_xticklabels(["left", "stay", "right"])
    ax.set_yticks(range(in_dim))
    labels = [f"z[{i}]" for i in range(z_dim)] + [f"h[{i}]" for i in range(h_dim)]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "weight_matrix_C.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        import subprocess
        subprocess.run(
            ["python3", os.path.join(HERE, "world_models_vizdoom_dream.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )
    out = _load(json_path)
    env_layout(out)
    v_m_curves(out)
    survival_real_vs_dream(out)
    final_survival_dist(out)
    weight_matrix_C(out)
    print(f"Wrote 5 PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
