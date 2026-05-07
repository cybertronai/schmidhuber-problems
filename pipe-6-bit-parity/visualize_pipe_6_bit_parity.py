"""Static visualisations for PIPE on n-bit even parity.

Runs PIPE inline (no JSON dependency) for the two reproducible configurations
documented in the README and writes seven PNGs to ``viz/``:

* ``training_curves_6bit.png`` — fitness trajectories for the 6-bit run
* ``training_curves_4bit.png`` — same for the 4-bit run
* ``ppt_max_prob.png`` — PPT-sharpness over time for both widths
* ``best_program_size.png`` — elite tree size over time
* ``solution_truth_table_4bit.png`` — the 4-bit solution checked vs. ground
  truth on all 16 inputs
* ``error_pattern_6bit.png`` — for the 6-bit best program, which of the 64
  inputs are correctly classified
* ``ppt_heatmap.png`` — final PPT distributions on the elite path of the
  4-bit run as a (path-position x instruction) heatmap

The two PIPE runs are short — together they take about 4–5 minutes — and
match the headline configurations from ``pipe_6_bit_parity.py`` exactly.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipe_6_bit_parity as P


HERE = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(HERE, "viz")


def _ensure_viz_dir() -> None:
    os.makedirs(VIZ_DIR, exist_ok=True)


def _run_4bit() -> Dict[str, Any]:
    """Run the headline 4-bit configuration (seed 6, ~2.4 s, solves)."""
    return P.train(
        seed=6,
        n_bits=4,
        pop_size=30,
        max_gens=5000,
        lr=0.3,
        p_mut=0.4,
        mut_rate=0.4,
        max_depth=12,
        elitist_prob=0.5,
        eps=0.05,
        stagnation_window=80,
        reset_alpha=1.0,
        max_time_s=30.0,
        verbose=False,
        early_stop=True,
    )


def _run_6bit() -> Dict[str, Any]:
    """Run the headline 6-bit configuration (seed 0, 240 s cap, ~46/64)."""
    return P.train(
        seed=0,
        n_bits=6,
        pop_size=30,
        max_gens=200000,
        lr=0.3,
        p_mut=0.4,
        mut_rate=0.4,
        max_depth=14,
        elitist_prob=0.5,
        eps=0.05,
        stagnation_window=80,
        reset_alpha=1.0,
        max_time_s=240.0,
        verbose=False,
        early_stop=True,
    )


def plot_training_curves(result: Dict[str, Any], n_bits: int, out_path: str,
                         label: str) -> None:
    h = result["history"]
    n_cases = 1 << n_bits
    gens = h["gen"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, h["gen_best_fit"], lw=0.7, alpha=0.5, label="generation best")
    ax.plot(gens, h["gen_mean_fit"], lw=0.7, alpha=0.5, label="generation mean")
    ax.plot(gens, h["best_fit_so_far"], lw=2.0, color="C3", label="overall best")
    ax.axhline(n_cases, color="grey", ls="--", lw=0.8, label=f"target = {n_cases}")
    ax.axhline(n_cases / 2, color="grey", ls=":", lw=0.8, label="chance")
    for r in h.get("restarts", []):
        ax.axvline(r, color="C2", lw=0.4, alpha=0.3)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (correct / N)")
    ax.set_title(f"PIPE training on {label}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ppt_max_prob(res4: Optional[Dict[str, Any]],
                      res6: Optional[Dict[str, Any]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for res, n_bits, color in [(res4, 4, "C0"), (res6, 6, "C1")]:
        if res is None:
            continue
        h = res["history"]
        ax.plot(h["gen"], h["ppt_max_prob"],
                label=f"{n_bits}-bit", color=color, lw=1.0)
    ax.axhline(0.1, color="grey", ls=":", lw=0.7, label="uniform (≈ 1/10)")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean(max(P(I|d))) over PPT nodes")
    ax.set_title("PPT sharpness over generations")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_best_size(res4: Optional[Dict[str, Any]],
                   res6: Optional[Dict[str, Any]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for res, n_bits, color in [(res4, 4, "C0"), (res6, 6, "C1")]:
        if res is None:
            continue
        h = res["history"]
        ax.plot(h["gen"], h["best_size"],
                label=f"{n_bits}-bit", color=color, lw=1.0)
    ax.set_xlabel("generation")
    ax.set_ylabel("best-so-far program size (# nodes)")
    ax.set_title("Elite program size over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_truth_table_4bit(result: Dict[str, Any], out_path: str) -> None:
    n_bits = 4
    n_cases = 16
    P.configure_n_bits(n_bits)
    tree = result["best_tree"]
    out = P.evaluate_tree_bitmask(tree)
    pred = np.array([(out >> j) & 1 for j in range(n_cases)], dtype=int)
    target = np.array(
        [int(bin(j).count("1") % 2 == 0) for j in range(n_cases)], dtype=int
    )
    inputs = np.array(
        [[(j >> b) & 1 for b in range(n_bits)] for j in range(n_cases)],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.column_stack([inputs, target.reshape(-1, 1), pred.reshape(-1, 1)])
    im = ax.imshow(grid.T, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_yticks(list(range(n_bits + 2)))
    ax.set_yticklabels([f"x{i}" for i in range(n_bits)] + ["target", "PIPE"])
    ax.set_xlabel("input index (j = binary to decimal)")
    correct = int((pred == target).sum())
    ax.set_title(
        f"4-bit even parity truth table — PIPE matches {correct}/{n_cases}"
    )
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_error_pattern_6bit(result: Dict[str, Any], out_path: str) -> None:
    n_bits = 6
    n_cases = 64
    P.configure_n_bits(n_bits)
    tree = result["best_tree"]
    out = P.evaluate_tree_bitmask(tree)
    pred = np.array([(out >> j) & 1 for j in range(n_cases)], dtype=int)
    target = np.array(
        [int(bin(j).count("1") % 2 == 0) for j in range(n_cases)], dtype=int
    )
    correct_mask = (pred == target).astype(int)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(
        correct_mask.reshape(8, 8),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    correct = int(correct_mask.sum())
    n_wrong = n_cases - correct
    ax.set_title(
        f"6-bit parity: which of 64 inputs the elite classifies correctly.  "
        f"green = correct ({correct}/64),  red = wrong ({n_wrong}/64)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ppt_path_heatmap(result: Dict[str, Any], out_path: str) -> None:
    """Visualise the final PPT distributions along an elite-path sample."""
    P.configure_n_bits(4)
    ppt = result["ppt"]
    rng = np.random.default_rng(6)
    _, paths = P.sample_tree(ppt, rng, max_depth=12)
    paths = paths[: 60]
    matrix = np.array([n.probs for n, _ in paths])
    fig, ax = plt.subplots(figsize=(8, max(3, len(paths) * 0.18)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(list(range(P.N_INSTR)))
    ax.set_xticklabels(P.INSTR_NAMES, rotation=45, ha="right")
    ax.set_ylabel("position along elite path")
    ax.set_title(
        "Final PPT distributions on the 4-bit elite path "
        "(rows = path positions, cols = instruction)"
    )
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-6bit", action="store_true",
                        help="Skip the slow 240-s 6-bit run.")
    args = parser.parse_args()

    _ensure_viz_dir()

    print("Running 4-bit PIPE (~2.4 s) ...")
    res4 = _run_4bit()
    print(f"  4-bit: best={res4['best_fitness']}/16  size={res4['best_size']}"
          f"  solved_at={res4['solved_at']}")

    res6: Optional[Dict[str, Any]] = None
    if not args.skip_6bit:
        print("Running 6-bit PIPE (~240 s) ...")
        res6 = _run_6bit()
        print(f"  6-bit: best={res6['best_fitness']}/64  size={res6['best_size']}"
              f"  restarts={res6['n_restarts']}")

    plot_training_curves(res4, 4,
                         os.path.join(VIZ_DIR, "training_curves_4bit.png"),
                         label="4-bit even parity")
    plot_truth_table_4bit(res4,
                          os.path.join(VIZ_DIR, "solution_truth_table_4bit.png"))
    plot_ppt_path_heatmap(res4, os.path.join(VIZ_DIR, "ppt_heatmap.png"))

    if res6 is not None:
        plot_training_curves(res6, 6,
                             os.path.join(VIZ_DIR, "training_curves_6bit.png"),
                             label="6-bit even parity")
        plot_error_pattern_6bit(res6,
                                os.path.join(VIZ_DIR, "error_pattern_6bit.png"))

    plot_ppt_max_prob(res4, res6, os.path.join(VIZ_DIR, "ppt_max_prob.png"))
    plot_best_size(res4, res6, os.path.join(VIZ_DIR, "best_program_size.png"))

    print(f"Wrote PNGs to {VIZ_DIR}")


if __name__ == "__main__":
    main()
