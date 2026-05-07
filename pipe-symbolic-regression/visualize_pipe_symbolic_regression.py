"""Static visualizations for pipe-symbolic-regression.

Writes four PNGs to viz/:

* fitness_curve.png      — best-of-gen and elite fitness over generations.
* sse_curve.png          — log-scale SSE.
* hits_curve.png         — Koza-hits over generations.
* fit_curve_overlay.png  — target curve vs elite at three checkpoints
                           (early / mid / final).
* program_size.png       — elite program size + depth over generations.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipe_symbolic_regression as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3,
                    help="Seed 3 is the headline reproducer (solves at gen 60).")
    ap.add_argument("--max-gen", type=int, default=200)
    ap.add_argument("--funcs", choices=("arith", "full"), default="arith")
    args = ap.parse_args()

    P.set_function_set(P.ARITH_KEYS if args.funcs == "arith" else P.FULL_KEYS)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "viz")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Training PIPE seed={args.seed} max_gen={args.max_gen} funcs={args.funcs} ...")
    out = P.train(P.Hyper(pop_size=100, max_gen=args.max_gen, max_depth=6),
                  seed=args.seed, verbose=False)
    hist = out["history"]
    gens = np.array([h["gen"] for h in hist])
    fit_best = np.array([h["fit_best"] for h in hist])
    fit_elite = np.array([h["fit_elite"] for h in hist])
    sse_best = np.array([h["sse_best"] for h in hist])
    sse_elite = np.array([h["sse_elite"] for h in hist])
    hits_elite = np.array([h["hits_elite"] for h in hist])
    elite_size = np.array([h["elite_size"] for h in hist])
    elite_depth = np.array([h["elite_depth"] for h in hist])

    X = np.array(out["X"])
    Y = np.array(out["Y"])
    yhat_per_gen = [np.array(y) for y in out["best_yhat_per_gen"]]
    elite_str_per_gen = out["elite_str_per_gen"]

    print(f"  elite SSE={out['elite_sse']:.4e}  hits={out['elite_hits']}/20  "
          f"solved_at={out['solved_at']}  hits_solved_at={out['hits_solved_at']}")

    # ----- fitness curve
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(gens, fit_best, color="#888", lw=1, label="best of gen")
    ax.plot(gens, fit_elite, color="C0", lw=2, label="elite (best ever)")
    if out["hits_solved_at"] is not None:
        ax.axvline(out["hits_solved_at"], color="C2", ls="--", alpha=0.6,
                   label=f"hits 20/20 (gen {out['hits_solved_at']})")
    if out["solved_at"] is not None:
        ax.axvline(out["solved_at"], color="C1", ls="--", alpha=0.6,
                   label=f"SSE < 1e-6 (gen {out['solved_at']})")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness = 1/(1+SSE)")
    ax.set_title(f"PIPE on Koza f(x)=x^4+x^3+x^2+x — seed {args.seed}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fitness_curve.png"), dpi=120)
    plt.close(fig)

    # ----- SSE curve (log)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(gens, np.maximum(sse_best, 1e-32), color="#888", lw=1, label="best of gen")
    ax.semilogy(gens, np.maximum(sse_elite, 1e-32), color="C0", lw=2, label="elite")
    ax.axhline(1e-6, color="C1", ls="--", alpha=0.6, label="SSE = 1e-6 (fit-solved)")
    ax.set_xlabel("generation")
    ax.set_ylabel("sum of squared error (log)")
    ax.set_title("Elite SSE shrinks by orders of magnitude across generations")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sse_curve.png"), dpi=120)
    plt.close(fig)

    # ----- hits curve
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(gens, hits_elite, color="C2", lw=2)
    ax.axhline(20, color="C1", ls="--", alpha=0.6, label="Koza hits = 20/20")
    ax.set_xlabel("generation")
    ax.set_ylabel("Koza hits  (|err| < 0.01)")
    ax.set_title("Elite Koza-hits (20 = problem solved per Koza criterion)")
    ax.set_ylim(-1, 21)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hits_curve.png"), dpi=120)
    plt.close(fig)

    # ----- fit-curve overlay (early / mid / final)
    n_gen = len(yhat_per_gen)
    cps = [0, n_gen // 4, n_gen // 2, n_gen - 1]
    cps = sorted(set(c for c in cps if 0 <= c < n_gen))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X, Y, color="black", lw=2.5, label="target  x^4+x^3+x^2+x")
    cmap = plt.get_cmap("viridis")
    for i, gen_idx in enumerate(cps):
        col = cmap(i / max(1, len(cps) - 1))
        label = f"elite at gen {gen_idx}"
        ax.plot(X, yhat_per_gen[gen_idx], color=col, lw=1.4, alpha=0.85,
                marker="o", markersize=3, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Elite fit improves across generations")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fit_curve_overlay.png"), dpi=120)
    plt.close(fig)

    # ----- program size and depth
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(gens, elite_size, color="C3", lw=1.6, label="elite size (nodes)")
    ax.plot(gens, elite_depth, color="C4", lw=1.6, label="elite depth")
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Elite program size and depth over generations")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "program_size.png"), dpi=120)
    plt.close(fig)

    # ----- final fit, scatter view
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(X, Y, "k-", lw=2, label="target")
    elite_y = np.array(out["elite_yhat"])
    ax.plot(X, elite_y, "o-", color="C0", lw=1.5, label="elite")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Final elite fit  (SSE = {out['elite_sse']:.2e}, "
                 f"hits = {out['elite_hits']}/20)\n"
                 f"{out['elite_prog_str'][:60]}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_fit.png"), dpi=120)
    plt.close(fig)

    print(f"Wrote 6 PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
