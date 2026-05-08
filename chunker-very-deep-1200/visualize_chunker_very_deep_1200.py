"""
Static PNG visualizations for chunker-very-deep-1200.

Reads `results.json` produced by `chunker_very_deep_1200.py` and writes:

    viz/training_curves.png        --- automatizer / chunker / baseline losses
    viz/surprise_pattern.png       --- per-step automatizer loss on one fresh
                                       sequence; surprise mask highlighted
    viz/grad_vs_depth.png          --- ||d L_terminal / d h_t|| versus
                                       reverse-time-step for the baseline RNN
                                       (the canonical vanishing-gradient
                                       picture); chunker comparison overlaid
    viz/depth_ratio_bar.png        --- bar chart of effective BPTT depth:
                                       baseline (vanishes at ~5 steps) vs
                                       chunker (~2 steps), and the cited
                                       T = 1200 reference line

Usage
-----

    python3 visualize_chunker_very_deep_1200.py --seed 0
    python3 visualize_chunker_very_deep_1200.py --seed 0 --T 500 --outdir viz
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from chunker_very_deep_1200 import (
    NUM_SYMBOLS, RNN, make_sequence, train_automatizer, train_baseline,
    train_chunker, detect_surprises, effective_depth, SYMBOL_NAMES,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def ensure_results(seed: int, T: int):
    """Load results.json if it matches (seed, T); otherwise re-run."""
    here = os.path.dirname(os.path.abspath(__file__))
    rpath = os.path.join(here, "results.json")
    if os.path.exists(rpath):
        try:
            with open(rpath) as f:
                r = json.load(f)
            if r.get("seed") == seed and r.get("T") == T:
                return r
        except Exception:
            pass
    raise SystemExit(
        f"results.json missing or stale; please run: "
        f"python3 chunker_very_deep_1200.py --seed {seed} --T {T}"
    )


# ----------------------------------------------------------------------
# Plot 1: training curves
# ----------------------------------------------------------------------

def plot_training_curves(results: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=120)

    ax = axes[0]
    ax.plot(results["automatizer_loss"], color="#1f77b4", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Automatizer (level 0)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results["chunker_loss"], color="#2ca02c", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(results["chunker_target_acc"], color="#d62728",
             linewidth=1.0, linestyle="--", label="target acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss", color="#2ca02c")
    ax2.set_ylabel("recall-target accuracy", color="#d62728")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("Chunker (level 1) on compressed surprises")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(results["baseline_loss"], color="#7f7f7f", linewidth=1.0,
            label="loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax2 = ax.twinx()
    ax2.plot(results["baseline_target_acc"], color="#d62728",
             linewidth=1.0, linestyle="--")
    ax2.set_ylabel("recall-target accuracy", color="#d62728")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("Single-net baseline (full BPTT, T={})".format(results["T"]))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 2: surprise pattern on one fresh sequence
# ----------------------------------------------------------------------

def plot_surprise_pattern(seed: int, T: int, results: dict, out_path: str):
    """Re-train automatizer on the same seed, then probe one fresh sequence."""
    rng = np.random.default_rng(seed)
    hp = results["hyperparameters"]
    A, _ = train_automatizer(T, rng,
                             hidden=hp["auto_hidden"],
                             epochs=hp["auto_epochs"],
                             lr=hp["auto_lr"],
                             truncate=hp["auto_truncate"],
                             verbose=False)
    # Use a fresh draw for visualization.
    x, y, trig = make_sequence(T, rng)
    threshold = hp["threshold"]
    mask, losses = detect_surprises(A, x, y, threshold)

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), dpi=120, sharex=True)
    ax = axes[0]
    ax.plot(np.arange(len(losses)), losses, color="#1f77b4", linewidth=0.8)
    ax.axhline(threshold, color="#d62728", linewidth=0.8, linestyle="--",
               label=f"surprise threshold = {threshold:.2f}")
    surp_idx = np.where(mask)[0]
    ax.scatter(surp_idx, losses[surp_idx], color="#d62728", s=22, zorder=3,
               label=f"surprise events ({len(surp_idx)})")
    ax.set_ylabel("automatizer\nper-step CE loss")
    ax.set_title(f"Automatizer surprise pattern on a fresh T={T} sequence "
                 f"(trigger={SYMBOL_NAMES[trig]})")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(len(x)), x, marker="|", color="#666666", linewidth=0.0,
            markersize=8)
    # Mark surprise positions
    for i in surp_idx:
        ax.axvline(i, color="#d62728", linewidth=0.6, alpha=0.6)
    ax.set_yticks(range(NUM_SYMBOLS))
    ax.set_yticklabels(SYMBOL_NAMES)
    ax.set_xlabel("sequence position t")
    ax.set_ylabel("input symbol")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 3: gradient norms vs reverse-time step
# ----------------------------------------------------------------------

def plot_grad_vs_depth(results: dict, out_path: str):
    grads = np.array(results["baseline_grad_norms"])
    T_total = len(grads)
    # x-axis: reverse-time distance from terminal target
    rev = np.arange(T_total)[::-1]
    rev_dist = T_total - 1 - rev   # 0 at the terminal step; T_total-1 at start

    # Plot ||dh|| vs reverse distance
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    base = grads[-1]
    norm_grads = grads[::-1] / max(base, 1e-12)
    ax.semilogy(np.arange(T_total), norm_grads,
                color="#1f77b4", linewidth=1.2,
                label=f"baseline RNN (full BPTT, T={results['T']})")

    # 1% threshold
    ax.axhline(0.01, color="#d62728", linewidth=0.8, linestyle="--",
               label="1% of terminal gradient (effective-depth cutoff)")

    # Effective depths
    d_baseline = results["effective_depth_baseline"]
    d_chunker = results["effective_depth_chunker"]
    ax.axvline(d_baseline, color="#1f77b4", linewidth=0.6, linestyle=":")
    ax.text(d_baseline + 1, 1.5e-2,
            f"baseline\neffective depth = {d_baseline}",
            color="#1f77b4", fontsize=9, va="bottom")

    # The chunker only ever sees k_compressed steps; its 'curve' is just a
    # flat segment of length k_chunker at norm = 1 (no propagation needed
    # through the filler).
    ax.plot([0, d_chunker], [1.0, 1.0], color="#2ca02c", linewidth=2.5,
            label=f"chunker on compressed surprises (k = {d_chunker} steps)")
    ax.scatter([d_chunker], [1.0], color="#2ca02c", zorder=3)

    ax.set_xlabel("reverse-time distance from recall-target loss "
                  "(0 = output step)")
    ax.set_ylabel(r"$\|\partial L_{\mathrm{terminal}} / \partial h_t\|$  "
                  "(normalised)")
    ax.set_title("Gradient flow backward through time -- "
                 "credit-assignment trace")
    ax.set_ylim(1e-25, 5)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 4: depth ratio bar
# ----------------------------------------------------------------------

def plot_depth_ratio(results: dict, out_path: str):
    T = results["T"]
    d_b = results["effective_depth_baseline"]
    d_c = results["effective_depth_chunker"]
    ratio = (T - 1) / max(1, d_c)

    labels = [f"raw\nsequence\n(T - 1 = {T - 1})",
              f"single-net\nbaseline\n(eff. depth = {d_b})",
              f"chunker on\ncompressed\nsurprises\n(k = {d_c})"]
    values = [T - 1, d_b, d_c]
    colors = ["#999999", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=120)
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("BPTT depth (steps the gradient must traverse)")
    ax.set_title(
        f"Effective BPTT depth: history compression reduces "
        f"{T - 1} virtual layers to {d_c}\n"
        f"(depth-reduction ratio: {ratio:.1f}x)"
    )
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.1, f"{v}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, args.outdir)
    os.makedirs(out, exist_ok=True)

    results = ensure_results(args.seed, args.T)

    plot_training_curves(results, os.path.join(out, "training_curves.png"))
    plot_surprise_pattern(args.seed, args.T, results,
                          os.path.join(out, "surprise_pattern.png"))
    plot_grad_vs_depth(results, os.path.join(out, "grad_vs_depth.png"))
    plot_depth_ratio(results, os.path.join(out, "depth_ratio_bar.png"))

    print(f"[viz] wrote 4 PNGs to {out}/")


if __name__ == "__main__":
    main()
