"""
Static visualizations for the trained NBB XOR network.

Outputs (in `viz/`):
  training_curves.png     - accuracy + total substance + weight norms
  weights.png             - final W_ih (Hinton diagram) + final W_ho heatmap
  hidden_response.png     - which hidden unit fires for each XOR pattern
  per_pattern_history.png - per-pattern correctness as training progresses
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nbb_xor import NBB, train, evaluate, make_xor_patterns


PATTERN_LABELS = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]
PATTERN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_training_curves(history: dict, out_path: str, converged_at: int | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)

    p = history["presentations"]

    ax = axes[0, 0]
    ax.plot(p, history["accuracy"], color="#1f77b4", linewidth=1.2)
    if converged_at is not None and converged_at > 0:
        ax.axvline(converged_at, color="green", linestyle="--", linewidth=1,
                   label=f"converged @ {converged_at}")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_ylabel("# correct (out of 4)")
    ax.set_xlabel("pattern presentations")
    ax.set_ylim(-0.2, 4.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.set_title("Frozen-eval accuracy")

    ax = axes[0, 1]
    ax.plot(p, history["total_substance"], color="#9467bd", linewidth=1.2)
    ax.set_ylabel("sum of all weights")
    ax.set_xlabel("pattern presentations")
    ax.grid(alpha=0.3)
    ax.set_title("Total weight-substance in the network")

    ax = axes[1, 0]
    ax.plot(p, history["W_ih_norm"], color="#ff7f0e", linewidth=1.2)
    ax.set_ylabel(r"$\|W_{ih}\|_F$")
    ax.set_xlabel("pattern presentations")
    ax.grid(alpha=0.3)
    ax.set_title("input → hidden norm")

    ax = axes[1, 1]
    ax.plot(p, history["W_ho_norm"], color="#2ca02c", linewidth=1.2)
    ax.set_ylabel(r"$\|W_{ho}\|_F$")
    ax.set_xlabel("pattern presentations")
    ax.grid(alpha=0.3)
    ax.set_title("hidden → output norm")

    fig.suptitle("NBB XOR — training dynamics (Schmidhuber 1989)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(nbb: NBB, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), dpi=130)

    # ---- input -> hidden (Hinton diagram) ----
    ax = axes[0]
    W = nbb.W_ih
    max_abs = max(abs(W).max(), 1e-12)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w >= 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.4))
    ax.set_xlim(-0.7, W.shape[1] - 0.3)
    ax.set_ylim(-0.7, W.shape[0] - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"h[{j}]" for j in range(W.shape[1])])
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels(["bias", "x1", "x2"])
    ax.set_aspect("equal")
    ax.set_title(f"$W_{{ih}}$  (max={W.max():.3g})")

    # ---- hidden -> output (heatmap, with text) ----
    ax = axes[1]
    W = nbb.W_ho
    im = ax.imshow(W, cmap="magma", aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i,j]:.4g}",
                    ha="center", va="center",
                    color="white" if W[i, j] < W.max() * 0.6 else "black",
                    fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["out[0]\n(XOR=0)", "out[1]\n(XOR=1)"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["h[0]", "h[1]", "h[2]"])
    ax.set_title("$W_{ho}$  (raw weights)")
    plt.colorbar(im, ax=ax, fraction=0.05)

    # ---- W_ho asymmetry per row -----------------------------------------
    # The decision per hidden unit is W_ho[h, 0] - W_ho[h, 1].
    # Sign tells which output that hidden unit prefers; magnitude is small
    # but consistent (this is what the bucket brigade asymmetry actually
    # encodes).
    ax = axes[2]
    diff = W[:, 0] - W[:, 1]
    colors = ["#1f77b4" if d > 0 else "#ff7f0e" for d in diff]
    bars = ax.barh(range(len(diff)), diff, color=colors, edgecolor="black",
                   linewidth=0.5)
    for i, d in enumerate(diff):
        ax.text(d + (0.05 * max(abs(diff)) * (1 if d > 0 else -1)),
                i, f"{d:+.2e}", va="center",
                ha="left" if d > 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(range(len(diff)))
    ax.set_yticklabels(["h[0]", "h[1]", "h[2]"])
    ax.invert_yaxis()
    ax.set_xlabel("$W_{ho}$[h, 0] $-$ $W_{ho}$[h, 1]")
    ax.set_title("output preference per hidden unit\n"
                 "(blue → out[0]=XOR=0, orange → out[1]=XOR=1)")
    pad = max(abs(diff)) * 1.6 + 1e-12
    ax.set_xlim(-pad, pad)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_response(nbb: NBB, out_path: str):
    """Show which hidden unit fires (and which output) for each XOR pattern."""
    patterns = make_xor_patterns()
    n_hidden = nbb.n_hidden
    n_output = nbb.n_output

    h_fires = np.zeros((4, n_hidden))
    o_fires = np.zeros((4, n_output))
    correct = []
    for r, (x1, x2, target) in enumerate(patterns):
        out = nbb.present(int(x1), int(x2), int(target), learn=False)
        # snapshot final activations
        h_fires[r] = nbb.x_h
        o_fires[r] = nbb.x_o
        correct.append(out == int(target))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=130)

    ax = axes[0]
    im = ax.imshow(h_fires, cmap="Greys", vmin=0, vmax=1)
    for r in range(4):
        for h in range(n_hidden):
            if h_fires[r, h] > 0:
                ax.text(h, r, "★", ha="center", va="center",
                        color="#cc0000", fontsize=16)
    ax.set_xticks(range(n_hidden))
    ax.set_xticklabels([f"h[{h}]" for h in range(n_hidden)])
    ax.set_yticks(range(4))
    ax.set_yticklabels(PATTERN_LABELS)
    ax.set_title("Which hidden unit fires per pattern")

    ax = axes[1]
    im = ax.imshow(o_fires, cmap="Greys", vmin=0, vmax=1)
    for r in range(4):
        mark = "✓" if correct[r] else "✗"
        for o in range(n_output):
            if o_fires[r, o] > 0:
                ax.text(o, r, "★", ha="center", va="center",
                        color="#1f7a1f" if correct[r] else "#cc0000",
                        fontsize=16)
        ax.text(n_output - 0.4, r, mark,
                ha="left", va="center", fontsize=14,
                color="#1f7a1f" if correct[r] else "#cc0000")
    ax.set_xticks(range(n_output))
    ax.set_xticklabels(["out[0]\n(XOR=0)", "out[1]\n(XOR=1)"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(PATTERN_LABELS)
    ax.set_title("Which output unit fires per pattern")

    fig.suptitle(f"NBB XOR — frozen-eval responses  ({sum(correct)}/4 correct)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_pattern_history(seed: int, max_presentations: int,
                             eta: float, lam: float, out_path: str):
    """Track per-pattern correctness through training (re-runs training)."""
    patterns = make_xor_patterns()
    nbb = NBB(eta=eta, lam=lam, seed=seed)
    pres_rng = np.random.default_rng(seed + 12345)

    pres_log = []
    pat_log = [[] for _ in range(4)]  # bool per pattern

    presentations = 0
    log_every = 4
    while presentations < max_presentations:
        order = pres_rng.permutation(4)
        for p_idx in order:
            x1, x2, target = patterns[p_idx]
            nbb.present(int(x1), int(x2), int(target), learn=True)
            presentations += 1

            if presentations % log_every == 0:
                pres_log.append(presentations)
                # frozen eval per pattern
                for r in range(4):
                    out = nbb.present(int(patterns[r, 0]),
                                      int(patterns[r, 1]),
                                      int(patterns[r, 2]),
                                      learn=False)
                    pat_log[r].append(int(out == int(patterns[r, 2])))

            if presentations >= max_presentations:
                break
        if all(p[-1] for p in pat_log if p):
            break

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=130)
    for r in range(4):
        # offset each pattern vertically for legibility
        ax.fill_between(pres_log,
                        np.array(pat_log[r]) * 0.85 + r * 1.05,
                        r * 1.05,
                        color=PATTERN_COLORS[r], alpha=0.45,
                        step="post")
        ax.plot(pres_log,
                np.array(pat_log[r]) * 0.85 + r * 1.05,
                color=PATTERN_COLORS[r], linewidth=0.8, drawstyle="steps-post")
    ax.set_yticks([r * 1.05 + 0.42 for r in range(4)])
    ax.set_yticklabels(PATTERN_LABELS)
    ax.set_xlabel("pattern presentations")
    ax.set_title("Per-pattern correctness (filled = correct, gap = wrong)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-presentations", type=int, default=5000)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--ticks", type=int, default=6)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training (seed={args.seed}, eta={args.eta}, lam={args.lam})...")
    history = {"presentations": [], "accuracy": [],
               "W_ih_norm": [], "W_ho_norm": [], "total_substance": []}
    nbb, presentations, acc = train(
        seed=args.seed,
        max_presentations=args.max_presentations,
        n_ticks=args.ticks, eta=args.eta, lam=args.lam,
        history=history, log_every=4, verbose=False,
    )
    print(f"  presentations={presentations}  acc={acc}/4")

    converged_at = presentations if acc == 4 else None

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"),
                         converged_at)
    plot_weights(nbb, os.path.join(args.outdir, "weights.png"))
    plot_hidden_response(nbb, os.path.join(args.outdir, "hidden_response.png"))
    plot_per_pattern_history(args.seed, args.max_presentations,
                             args.eta, args.lam,
                             os.path.join(args.outdir, "per_pattern_history.png"))


if __name__ == "__main__":
    main()
