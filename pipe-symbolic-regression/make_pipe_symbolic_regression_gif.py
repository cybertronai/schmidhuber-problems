"""Generate pipe_symbolic_regression.gif — elite fit over generations.

Each frame shows: black target curve x^4+x^3+x^2+x, blue elite-curve at
that generation, and a text panel with the (truncated) elite expression
plus its SSE / Koza-hits.

Implemented with matplotlib's ImageMagick-free PillowWriter (no extra
deps beyond matplotlib + Pillow which matplotlib already pulls in)."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import pipe_symbolic_regression as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3,
                    help="Default 3 — solves at gen 60, makes a satisfying frame loop.")
    ap.add_argument("--max-gen", type=int, default=120)
    ap.add_argument("--funcs", choices=("arith", "full"), default="arith")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="pipe_symbolic_regression.gif")
    args = ap.parse_args()

    P.set_function_set(P.ARITH_KEYS if args.funcs == "arith" else P.FULL_KEYS)

    print(f"Training PIPE seed={args.seed} max_gen={args.max_gen} ...")
    out = P.train(P.Hyper(pop_size=100, max_gen=args.max_gen, max_depth=6),
                  seed=args.seed, verbose=False)
    X = np.array(out["X"])
    Y = np.array(out["Y"])
    yhat_per_gen = [np.array(y) for y in out["best_yhat_per_gen"]]
    str_per_gen = out["elite_str_per_gen"]
    hist = out["history"]
    n_gen = len(yhat_per_gen)
    print(f"  ran {n_gen} generations  elite SSE={out['elite_sse']:.2e}  "
          f"hits={out['elite_hits']}/20  hits_solved_at={out['hits_solved_at']}")

    # Pick frames evenly across the run, plus pin a few late frames to
    # show the converged fit.
    idxs = list(np.linspace(0, n_gen - 1, args.frames - 5).astype(int))
    idxs += [n_gen - 1] * 5  # hold final frame
    idxs = list(dict.fromkeys(idxs))  # de-dup early
    while len(idxs) < args.frames:
        idxs.append(n_gen - 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(X.min() - 0.05, X.max() + 0.05)
    y_lo = float(min(Y.min(), -0.5))
    y_hi = float(max(Y.max(), 4.5))
    ax.set_ylim(y_lo - 0.3, y_hi + 0.3)
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    target_line, = ax.plot(X, Y, "k-", lw=2, label="target  x^4+x^3+x^2+x")
    elite_line, = ax.plot(X, yhat_per_gen[0], "o-", color="C0", lw=1.6,
                          markersize=4, label="PIPE elite")
    ax.legend(loc="upper left")
    title = ax.set_title("PIPE on Koza  —  generation 0")
    info = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                   fontsize=8, family="monospace",
                   verticalalignment="bottom",
                   bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    def render(frame_i: int):
        gen_idx = idxs[frame_i]
        elite_line.set_ydata(yhat_per_gen[gen_idx])
        h = hist[gen_idx]
        sse = h["sse_elite"]
        hits = h["hits_elite"]
        prog = str_per_gen[gen_idx]
        if len(prog) > 56:
            prog = prog[:53] + "..."
        title.set_text(f"PIPE on Koza  —  generation {gen_idx}")
        info.set_text(f"elite : {prog}\n"
                      f"SSE   = {sse:.4e}\n"
                      f"hits  = {hits}/20  (Koza solve = 20)")
        return elite_line, title, info

    ani = animation.FuncAnimation(fig, render, frames=len(idxs),
                                  interval=1000 // args.fps, blit=False)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)
    print(f"Saving {out_path} ...")
    ani.save(out_path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    size = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
