"""Static PNG visualisations for curiosity-three-regions.

Outputs to <outdir>/:
  region_targets.png      — the three regions' target functions
  visit_distribution.png  — total visit counts per region
  cumulative_visits.png   — cumulative visits per region across the run
  curiosity_signal.png    — windowed curiosity per region across the run
  per_region_error.png    — smoothed per-region squared error vs visit index
  model_vs_target.png     — final M[c] for each region next to the target
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from curiosity_three_regions import run_experiment


COLORS = ["#3b82f6", "#f59e0b", "#10b981"]  # A=blue, B=amber, C=green
SHORT = ["A: deterministic", "B: random", "C: learnable-but-unlearned"]


def smooth(x, w):
    if len(x) < w:
        return np.array(x, dtype=np.float64)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    res = run_experiment(seed=args.seed, steps=args.steps)
    cfg = res["config"]
    names = [r["name"] for r in res["regions"]]
    chosen = np.asarray(res["chosen"])
    cur_log = np.asarray(res["cur_log"])  # (T, 3)

    # 1) region_targets.png
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, ax in enumerate(axes):
        targets = res["targets"][i]
        if targets is None:
            ax.text(0.5, 0.5,
                    "(noise resampled\neach visit\n"
                    f"sigma={cfg['sigma_rand']})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            ax.set_title(SHORT[i] + f" (K={cfg['K_rand']})")
            ax.set_xlim(0, cfg["K_rand"])
            ax.set_ylim(-2.5, 2.5)
        else:
            t = np.asarray(targets)
            ax.bar(np.arange(len(t)), t, color=COLORS[i])
            ax.set_title(SHORT[i] + f" (K={len(t)})")
        ax.set_xlabel("context")
        ax.set_ylabel("target")
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("Region target functions", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "region_targets.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) visit_distribution.png
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = res["visit_counts"]
    bars = ax.bar(SHORT, counts, color=COLORS)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100.0*v/sum(counts):.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("visit count")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_title(f"Visits over {args.steps} steps  (seed {args.seed})")
    fig.tight_layout()
    fig.savefig(out / "visit_distribution.png", dpi=120)
    plt.close(fig)

    # 3) cumulative_visits.png
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cum = np.zeros((args.steps, 3), dtype=np.int64)
    for t in range(args.steps):
        if t > 0:
            cum[t] = cum[t - 1]
        cum[t, chosen[t]] += 1
    for i in range(3):
        ax.plot(cum[:, i], color=COLORS[i], label=SHORT[i], linewidth=1.6)
    ax.axvline(cfg["burn_in"], color="red", linestyle="--", alpha=0.5,
               label=f"burn-in end (t={cfg['burn_in']})")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative visits")
    ax.set_title("Cumulative visits per region")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "cumulative_visits.png", dpi=120)
    plt.close(fig)

    # 4) curiosity_signal.png
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i in range(3):
        ax.plot(cur_log[:, i], color=COLORS[i], label=SHORT[i],
                linewidth=1.0, alpha=0.85)
    ax.axvline(cfg["burn_in"], color="red", linestyle="--", alpha=0.5,
               label=f"burn-in end (t={cfg['burn_in']})")
    ax.set_xlabel("step")
    ax.set_ylabel("curiosity = max(0, mean_old_err - mean_new_err)")
    ax.set_title(f"Per-region curiosity (window W={cfg['window']})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "curiosity_signal.png", dpi=120)
    plt.close(fig)

    # 5) per_region_error.png
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sw = 50
    for i, h in enumerate(res["err_hist"]):
        if len(h) < sw:
            continue
        s = smooth(np.asarray(h), sw)
        ax.plot(s, color=COLORS[i], label=f"{SHORT[i]} ({len(h)} visits)",
                linewidth=1.6)
    ax.set_xlabel("visit index within region (independent x per series)")
    ax.set_ylabel(f"squared error (smoothed, window={sw})")
    ax.set_title("Per-region prediction error vs visit count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "per_region_error.png", dpi=120)
    plt.close(fig)

    # 6) model_vs_target.png
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, ax in enumerate(axes):
        M = np.asarray(res["final_M"][i])
        targets = res["targets"][i]
        K = len(M)
        x = np.arange(K)
        if targets is not None:
            ax.bar(x - 0.2, np.asarray(targets), 0.4,
                   label="target", color="#888", alpha=0.6)
            ax.bar(x + 0.2, M, 0.4, label="learned M", color=COLORS[i])
        else:
            # random region: show only learned M (which should be ~0)
            ax.bar(x, M, color=COLORS[i], label="learned M (target = noise)")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--",
                       label="true mean = 0")
        ax.set_title(SHORT[i] + f" (K={K})")
        ax.set_xlabel("context")
        ax.set_ylabel("value")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Final per-context predictions vs targets", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "model_vs_target.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"saved 6 PNGs under {out}/")


if __name__ == "__main__":
    main()
