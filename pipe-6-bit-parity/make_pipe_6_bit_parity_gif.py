"""Build ``pipe_6_bit_parity.gif`` of PIPE training.

The animation has two panels:

* Left: best-so-far fitness over generations (line) + current generation best
  (dot), with the chance and target lines marked.
* Right: a square grid of which inputs the *current best-so-far* program
  classifies correctly (green) vs incorrectly (red). The grid evolves from
  ~50/50 at chance toward all-green as PIPE solves the task.

The GIF runs PIPE on 4-bit even parity by default (it solves cleanly in a few
hundred generations and the GIF stays small). Use ``--n-bits 6`` to instead
animate the 6-bit run, which only reaches partial fitness in the budget.

Frames come from a snapshot callback hooked into the *same* ``train()`` used
elsewhere, so the dynamics shown match the headline run exactly.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import pipe_6_bit_parity as P


HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6,
                        help="Default seed=6: solves 4-bit parity in ~258 gens.")
    parser.add_argument("--n-bits", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-gens", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--p-mut", type=float, default=0.4)
    parser.add_argument("--mut-rate", type=float, default=0.4)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--elitist-prob", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--stagnation-window", type=int, default=80)
    parser.add_argument("--reset-alpha", type=float, default=1.0)
    parser.add_argument("--max-time-s", type=float, default=30.0)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--out", default="pipe_6_bit_parity.gif")
    args = parser.parse_args()

    P.configure_n_bits(args.n_bits)
    n_cases = 1 << args.n_bits

    snapshots: List[Dict[str, Any]] = []

    def cb(gen, gen_best_fit, overall_best_fit, overall_best_tree, n_restarts):
        snapshots.append({
            "gen": gen,
            "gen_best_fit": int(gen_best_fit),
            "overall_best_fit": int(overall_best_fit),
            "overall_best_tree": overall_best_tree,
            "n_restarts": int(n_restarts),
        })

    P.train(
        seed=args.seed,
        n_bits=args.n_bits,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        lr=args.lr,
        p_mut=args.p_mut,
        mut_rate=args.mut_rate,
        max_depth=args.max_depth,
        elitist_prob=args.elitist_prob,
        eps=args.eps,
        stagnation_window=args.stagnation_window,
        reset_alpha=args.reset_alpha,
        max_time_s=args.max_time_s,
        verbose=False,
        early_stop=True,
        snapshot_callback=cb,
        snapshot_every=args.snapshot_every,
    )

    print(f"  trained {len(snapshots)} snapshots, "
          f"final fitness={snapshots[-1]['overall_best_fit']}/{n_cases}")

    target = np.array(
        [int(bin(j).count("1") % 2 == 0) for j in range(n_cases)], dtype=int
    )
    grid_h = max(1, int(np.sqrt(n_cases)))
    grid_w = (n_cases + grid_h - 1) // grid_h
    pad = grid_h * grid_w - n_cases

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.2))
    line_so_far, = ax_left.plot([], [], color="C3", lw=2, label="best so far")
    dot_gen = ax_left.scatter([], [], color="C0", s=24, label="gen best")
    ax_left.axhline(n_cases, color="grey", ls="--", lw=0.7, label="target")
    ax_left.axhline(n_cases / 2, color="grey", ls=":", lw=0.7, label="chance")
    ax_left.set_xlim(0, max(snap["gen"] for snap in snapshots) + 1)
    ax_left.set_ylim(0, n_cases + 1)
    ax_left.set_xlabel("generation")
    ax_left.set_ylabel("fitness (correct / N)")
    ax_left.legend(loc="lower right", fontsize=8)
    ax_left.grid(alpha=0.3)

    im = ax_right.imshow(
        np.zeros((grid_h, grid_w)),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_title("correct (green) / wrong (red)")

    title = fig.suptitle("")
    gens_so_far: List[int] = []
    fits_so_far: List[int] = []

    def update(frame_idx: int):
        snap = snapshots[frame_idx]
        gens_so_far.append(snap["gen"])
        fits_so_far.append(snap["overall_best_fit"])
        line_so_far.set_data(gens_so_far, fits_so_far)
        dot_gen.set_offsets([[snap["gen"], snap["gen_best_fit"]]])

        tree = snap["overall_best_tree"]
        if tree is None:
            grid = np.zeros((grid_h, grid_w))
        else:
            out = P.evaluate_tree_bitmask(tree)
            pred = np.array([(out >> j) & 1 for j in range(n_cases)], dtype=int)
            correct = (pred == target).astype(float)
            if pad > 0:
                correct = np.concatenate([correct, np.full(pad, np.nan)])
            grid = correct.reshape(grid_h, grid_w)
        im.set_data(grid)
        title.set_text(
            f"PIPE on {args.n_bits}-bit even parity   "
            f"gen {snap['gen']:4d}   "
            f"best {snap['overall_best_fit']:3d}/{n_cases}   "
            f"restarts {snap['n_restarts']}"
        )
        return line_so_far, dot_gen, im, title

    fps = 12
    interval_ms = 1000 // fps
    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=interval_ms, blit=False
    )

    out_path = os.path.join(HERE, args.out)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
