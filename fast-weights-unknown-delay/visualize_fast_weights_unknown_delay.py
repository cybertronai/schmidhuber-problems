"""
Static visualizations for fast-weights-unknown-delay.

Outputs (in `viz/`):
    training_curves.png       --- loss + bit-accuracy + delay scatter over training
    delay_generalization.png  --- per-delay bit-accuracy on a held-out eval set
    test_episode.png          --- one fresh test episode: events, gates, output
    fast_weight_evolution.png --- |W_fast| over the steps of one episode
    head_activations.png      --- per-step k_t, v_t, q_t, g_t for one episode
    slow_weights.png          --- Hinton diagrams of the slow-net weight matrices
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from fast_weights_unknown_delay import (
    SlowNet, train, evaluate, make_batch, forward_episode,
)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=120)
    steps = history["step"]

    ax = axes[0]
    ax.plot(steps, history["loss"], color="#d62728", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("recall MSE")
    ax.set_title("Recall-step MSE (log)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, np.array(history["bit_acc"]) * 100,
            color="#2ca02c", linewidth=1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("bit-accuracy at recall (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Recall-bit accuracy on training batch")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(steps, history["delay"], s=4, color="#1f77b4", alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("delay K (steps)")
    ax.set_title("Per-batch delay (sampled uniformly)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"fast-weights-unknown-delay  seed={history['seed']}  "
        f"p_dim={history['p_dim']}  hidden={history['hidden']}  "
        f"d_k={history['d_k']}  eta={history['eta']}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-delay generalization
# ----------------------------------------------------------------------

def plot_delay_generalization(eval_in: dict, eval_extrap: dict,
                              d_min_train: int, d_max_train: int,
                              out_path: str):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    delays = eval_extrap["delays"]
    accs = np.array(eval_extrap["bit_acc"]) * 100
    ax.plot(delays, accs, "o-", linewidth=1.2, markersize=3, color="#1f77b4")
    ax.axvspan(d_min_train, d_max_train, color="#e8e8e8", alpha=0.6,
               label=f"trained range [{d_min_train}, {d_max_train}]")
    ax.set_xlabel("delay K between store and recall")
    ax.set_ylabel("bit-accuracy at recall (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Pattern recall accuracy vs. delay K (50 episodes per K)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Test episode rollout
# ----------------------------------------------------------------------

def plot_test_episode(S: SlowNet, eta: float, p_dim: int,
                      delay: int, out_path: str, seed: int = 12345):
    rng = np.random.default_rng(seed)
    x, P, recall_t = make_batch(1, p_dim, delay, rng)
    y, cache = forward_episode(S, x, recall_t, eta)
    T = x.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), dpi=120, sharex=True)

    # 1) input pattern slot bits over time
    ax = axes[0]
    im = ax.imshow(x[0, :, :p_dim].T, aspect="auto", interpolation="nearest",
                   cmap="bwr", vmin=-1, vmax=1)
    ax.set_yticks(range(p_dim))
    ax.set_yticklabels([f"bit {i}" for i in range(p_dim)])
    ax.set_ylabel("input pattern slot")
    ax.set_title(f"Episode (delay K={delay}). "
                 f"True P = {P[0].astype(int).tolist()}.  "
                 f"Recovered y = {y[0].round(2).tolist()}")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # 2) store / recall flags
    ax = axes[1]
    store = x[0, :, p_dim]
    recall = x[0, :, p_dim + 1]
    ax.bar(np.arange(T) - 0.18, store, width=0.36, color="#1f77b4",
           label="store flag")
    ax.bar(np.arange(T) + 0.18, recall, width=0.36, color="#d62728",
           label="recall flag")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("control flag")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) write gate g_t over time
    ax = axes[2]
    g = cache["g_seq"][:, 0]                 # (T,)
    ax.plot(range(T), g, "o-", linewidth=1.0, color="#9467bd",
            label=r"write gate $g_t$")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$g_t$ (sigmoid)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) recall output (compared against P) at the recall step
    ax = axes[3]
    bits = np.arange(p_dim)
    ax.bar(bits - 0.18, P[0], width=0.36, color="#2ca02c",
           label=r"true pattern $P$")
    ax.bar(bits + 0.18, y[0], width=0.36, color="#ff7f0e",
           label=r"recall output $y$")
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xticks(bits)
    ax.set_xticklabels([f"bit {i}" for i in range(p_dim)])
    ax.set_ylabel("value")
    ax.set_title("Recall step: true P vs. network output y")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("episode step")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Fast-weight Frobenius-norm evolution within one episode
# ----------------------------------------------------------------------

def plot_fast_weight_evolution(S: SlowNet, eta: float, p_dim: int,
                               delay: int, out_path: str, seed: int = 12345):
    rng = np.random.default_rng(seed)
    x, P, recall_t = make_batch(1, p_dim, delay, rng)
    y, cache = forward_episode(S, x, recall_t, eta)
    Wf = cache["Wfast_history"]               # list of (1, p_dim, d_k)
    norms = [float(np.linalg.norm(W[0])) for W in Wf]
    g_seq = cache["g_seq"][:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=120,
                             gridspec_kw={"width_ratios": [3, 4]})

    # Norm trace
    ax = axes[0]
    ax.plot(range(len(norms)), norms, "o-", linewidth=1.0, color="#d62728")
    ax.fill_between(range(len(norms)), 0, norms, alpha=0.15, color="#d62728")
    ax.set_xlabel("episode step")
    ax.set_ylabel(r"$\|W_{\mathrm{fast}}\|_F$")
    ax.set_title("Fast-weight norm grows at store, holds across delay")
    # Mark store and recall steps.
    ax.axvspan(-0.5, 0.5, color="#1f77b4", alpha=0.15, label="store")
    ax.axvspan(recall_t - 0.5, recall_t + 0.5, color="#2ca02c", alpha=0.15,
               label="recall")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Heatmap of W_fast at recall (final).
    ax = axes[1]
    Wfinal = Wf[-1][0]                        # (p_dim, d_k)
    vmax = np.abs(Wfinal).max() + 1e-9
    im = ax.imshow(Wfinal, aspect="auto", interpolation="nearest",
                   cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("key dimension")
    ax.set_ylabel("value (pattern) dimension")
    ax.set_title("$W_{\\mathrm{fast}}$ at recall step")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-step k / v / q / g activations
# ----------------------------------------------------------------------

def plot_head_activations(S: SlowNet, eta: float, p_dim: int,
                          delay: int, out_path: str, seed: int = 12345):
    rng = np.random.default_rng(seed)
    x, P, recall_t = make_batch(1, p_dim, delay, rng)
    y, cache = forward_episode(S, x, recall_t, eta)

    k = cache["k_seq"][:, 0, :]               # (T, d_k)
    v = cache["v_seq"][:, 0, :]
    q = cache["q_seq"][:, 0, :]
    g = cache["g_seq"][:, 0]

    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), dpi=120, sharex=True)
    for ax, mat, name in zip(axes[:3],
                             [k, v, q],
                             ["key  $k_t$", "value $v_t$", "query $q_t$"]):
        im = ax.imshow(mat.T, aspect="auto", interpolation="nearest",
                       cmap="bwr", vmin=-1, vmax=1)
        ax.set_ylabel(name)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax = axes[3]
    ax.plot(range(len(g)), g, "o-", color="#9467bd")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"gate $g_t$")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("episode step")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Slow-net head activations  delay K={delay}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Slow-net weight Hinton diagrams
# ----------------------------------------------------------------------

def plot_slow_weights(S: SlowNet, out_path: str):
    mats = [S.W_xh, S.W_hk, S.W_hv, S.W_hq, S.W_hg]
    names = ["W_xh", "W_hk", "W_hv", "W_hq", "W_hg"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=120)
    for ax, M, n in zip(axes, mats, names):
        vmax = np.abs(M).max() + 1e-9
        im = ax.imshow(M, aspect="auto", interpolation="nearest",
                       cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{n}  shape={tuple(M.shape)}")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Slow-net weights after training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--delay", type=int, default=20,
                   help="Delay used for per-episode visualizations.")
    p.add_argument("--extrap-max", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"training (seed={args.seed}, iters={args.iters})...")
    S, history, _ = train(seed=args.seed, iters=args.iters,
                          log_every=10, verbose=False)
    print("  done.")

    print("plotting...")
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))

    eval_in = evaluate(S, p_dim=4, eta=0.5, d_min=5, d_max=30)
    eval_extrap = evaluate(S, p_dim=4, eta=0.5, d_min=1, d_max=args.extrap_max)
    plot_delay_generalization(
        eval_in, eval_extrap,
        d_min_train=5, d_max_train=30,
        out_path=os.path.join(args.outdir, "delay_generalization.png"))

    plot_test_episode(S, eta=0.5, p_dim=4, delay=args.delay,
                      out_path=os.path.join(args.outdir, "test_episode.png"))
    plot_fast_weight_evolution(
        S, eta=0.5, p_dim=4, delay=args.delay,
        out_path=os.path.join(args.outdir, "fast_weight_evolution.png"))
    plot_head_activations(
        S, eta=0.5, p_dim=4, delay=args.delay,
        out_path=os.path.join(args.outdir, "head_activations.png"))
    plot_slow_weights(S, os.path.join(args.outdir, "slow_weights.png"))


if __name__ == "__main__":
    main()
