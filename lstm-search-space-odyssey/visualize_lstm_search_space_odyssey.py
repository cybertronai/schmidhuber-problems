"""
Static visualisations for lstm-search-space-odyssey.

Reads (or generates) the ablation-matrix JSON and writes:
  viz/ablation_matrix.png       headline bar chart over all 8 variants
  viz/learning_curves.png       per-variant test-MSE curve
  viz/solve_rate_curves.png     per-variant solve-rate curve
  viz/wallclock.png             per-variant wallclock (seconds)
  viz/summary_table.png         numerical results rendered as a table

If `viz/ablation_results.json` exists, we reuse it. Otherwise we run
`run_ablation_matrix` with the same defaults the GIF script uses.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lstm_search_space_odyssey import (
    VARIANT_NAMES, VARIANT_DESCRIPTIONS, run_ablation_matrix, env_info,
)


# Colour per variant — colour-blind-friendly tab10 ordering
VARIANT_COLOURS = {
    "V":    "#1f77b4",
    "NIG":  "#ff7f0e",
    "NFG":  "#2ca02c",
    "NOG":  "#d62728",
    "NIAF": "#9467bd",
    "NOAF": "#8c564b",
    "CIFG": "#e377c2",
    "NP":   "#7f7f7f",
}


def median_history(histories: list[dict], key: str):
    """Return iters, median array over seeds for the given key."""
    iters = histories[0]["iters"]
    arr = np.array([h[key] for h in histories])  # (n_seeds, n_evals)
    med = np.median(arr, axis=0)
    return iters, med, arr


def load_or_run(results_path: Path, T: int, hidden: int, n_iters: int,
                batch_size: int, lr: float, eval_every: int,
                seeds: list[int]):
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return data["results"], data.get("args"), data.get("env")
    print(f"  no cached results at {results_path}; running ablation...")
    results = run_ablation_matrix(
        T=T, hidden=hidden, n_iters=n_iters, batch_size=batch_size,
        lr=lr, eval_every=eval_every, seeds=seeds, verbose=True,
    )
    out = {
        "args": dict(T=T, hidden=hidden, iters=n_iters, batch=batch_size,
                     lr=lr, seeds=seeds, eval_every=eval_every),
        "env": env_info(),
        "results": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    return results, out["args"], out["env"]


def plot_ablation_matrix(results: dict, outpath: Path, title: str):
    """Headline bar chart: median test MSE per variant + error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = VARIANT_NAMES

    # Test MSE bars (log scale)
    med_mse = [float(np.median(results[n]["final_test_mse_per_seed"]))
               for n in names]
    lo_mse = [float(np.min(results[n]["final_test_mse_per_seed"]))
              for n in names]
    hi_mse = [float(np.max(results[n]["final_test_mse_per_seed"]))
              for n in names]
    err_lo = [m - lo for m, lo in zip(med_mse, lo_mse)]
    err_hi = [hi - m for m, hi in zip(med_mse, hi_mse)]
    colours = [VARIANT_COLOURS[n] for n in names]
    ax = axes[0]
    ax.bar(names, med_mse, color=colours,
           yerr=[err_lo, err_hi], capsize=4)
    ax.set_yscale("log")
    ax.set_ylabel("final test MSE (log)")
    ax.set_title("Final test MSE per variant\n(median over seeds, "
                 "min/max whiskers)")
    ax.axhline(0.04, color="black", linestyle="--", linewidth=0.8,
               label="paper threshold 0.04")
    ax.legend(loc="upper right", fontsize=8)
    for xi, m in enumerate(med_mse):
        ax.text(xi, m, f"{m:.4f}", ha="center", va="bottom", fontsize=7)

    # Solve-rate bars
    med_sr = [float(np.median(results[n]["final_solve_rate_per_seed"]))
              for n in names]
    lo_sr = [float(np.min(results[n]["final_solve_rate_per_seed"]))
             for n in names]
    hi_sr = [float(np.max(results[n]["final_solve_rate_per_seed"]))
             for n in names]
    err_lo = [m - lo for m, lo in zip(med_sr, lo_sr)]
    err_hi = [hi - m for m, hi in zip(med_sr, hi_sr)]
    ax = axes[1]
    ax.bar(names, med_sr, color=colours,
           yerr=[err_lo, err_hi], capsize=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("solve rate (|err| < 0.04)")
    ax.set_title("Solve rate per variant\n(median over seeds, "
                 "min/max whiskers)")
    for xi, m in enumerate(med_sr):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_learning_curves(results: dict, outpath: Path, key: str,
                         ylabel: str, title: str, log: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in VARIANT_NAMES:
        histories = results[name]["history_per_seed"]
        iters, med, arr = median_history(histories, key)
        ax.plot(iters, med, label=name, color=VARIANT_COLOURS[name],
                linewidth=1.6)
        # min/max envelope across seeds
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        ax.fill_between(iters, lo, hi, color=VARIANT_COLOURS[name],
                        alpha=0.12, linewidth=0)
    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
        ax.axhline(0.04, color="black", linestyle="--", linewidth=0.7,
                   label="paper threshold")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_wallclock(results: dict, outpath: Path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = VARIANT_NAMES
    med = [float(np.median(results[n]["wallclock_per_seed_sec"]))
           for n in names]
    colours = [VARIANT_COLOURS[n] for n in names]
    ax.bar(names, med, color=colours)
    ax.set_ylabel("wallclock per run (seconds)")
    ax.set_title("Per-variant training wallclock "
                 "(median over seeds, single CPU core)")
    for xi, m in enumerate(med):
        ax.text(xi, m, f"{m:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_summary_table(results: dict, outpath: Path, args_used):
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off")
    headers = ["Variant", "Description", "test MSE", "solve",
               "wall (s)"]
    rows = []
    for name in VARIANT_NAMES:
        r = results[name]
        med_mse = float(np.median(r["final_test_mse_per_seed"]))
        med_sr = float(np.median(r["final_solve_rate_per_seed"]))
        med_wall = float(np.median(r["wallclock_per_seed_sec"]))
        rows.append([name, VARIANT_DESCRIPTIONS[name],
                     f"{med_mse:.4f}", f"{med_sr:.3f}",
                     f"{med_wall:.2f}"])
    col_widths = [0.07, 0.55, 0.13, 0.10, 0.10]
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="left", colLoc="left",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    cfg_str = (f"T={args_used.get('T')} hidden={args_used.get('hidden')} "
               f"iters={args_used.get('iters')} "
               f"batch={args_used.get('batch')} "
               f"lr={args_used.get('lr')} "
               f"seeds={args_used.get('seeds')}")
    ax.set_title("LSTM Search Space Odyssey — small-scale ablation\n"
                 + cfg_str, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--outdir", type=str, default="viz")
    ap.add_argument("--results", type=str,
                    default="viz/ablation_results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results)

    results, args_used, env = load_or_run(
        results_path, T=args.T, hidden=args.hidden,
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
        eval_every=args.eval_every, seeds=seeds)
    if args_used is None:
        args_used = vars(args)

    plot_ablation_matrix(
        results, outdir / "ablation_matrix.png",
        title=(f"Ablation matrix on adding-problem T={args.T} "
               f"(median over {len(seeds)} seeds)"))
    plot_learning_curves(
        results, outdir / "learning_curves.png", key="test_mse",
        ylabel="test MSE (log)",
        title=("Test-MSE learning curves per variant "
               f"(adding-problem T={args.T})"),
        log=True)
    plot_learning_curves(
        results, outdir / "solve_rate_curves.png", key="solve_rate",
        ylabel="solve rate (|err| < 0.04)",
        title=("Solve-rate learning curves per variant "
               f"(adding-problem T={args.T})"),
        log=False)
    plot_wallclock(results, outdir / "wallclock.png")
    plot_summary_table(results, outdir / "summary_table.png", args_used)

    print(f"  wrote {outdir}/ablation_matrix.png")
    print(f"  wrote {outdir}/learning_curves.png")
    print(f"  wrote {outdir}/solve_rate_curves.png")
    print(f"  wrote {outdir}/wallclock.png")
    print(f"  wrote {outdir}/summary_table.png")


if __name__ == "__main__":
    main()
