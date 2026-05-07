"""
Static visualizations for Levin search on add-positions.

Outputs (in `viz/`):
  dsl.png               -- DSL alphabet table
  search_progress.png   -- programs evaluated per phase / per length
  program_trace.png     -- execution trace of the found program on example 0
  generalization.png    -- induced weight vector vs ground-truth ramp,
                           plus held-out accuracy bar

Usage:
  python3 visualize_levin_add_positions.py --seed 0 --outdir viz
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from levin_add_positions import (
    ALPHABET,
    induced_weight_vector,
    levin_search,
    make_examples,
    prog_str,
    run_body_traced,
    test_generalization,
)


def plot_dsl(out_path: str) -> None:
    rows = [
        ("+", "A := A + T", "accumulate temp into output"),
        ("*", "A := A * T", "multiply output by temp"),
        ("m", "T := T * B", "gate temp by current bit"),
        ("i", "T := I",     "load current index into temp"),
        ("b", "T := B",     "load current bit into temp"),
        ("1", "T := 1",     "load constant 1 into temp"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=130)
    table = ax.table(
        cellText=[[op, eff, expl] for op, eff, expl in rows],
        colLabels=["op", "effect", "what it does"],
        loc="center", cellLoc="left", colLoc="left",
        colWidths=[0.08, 0.32, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for c in range(3):
        table[(0, c)].set_facecolor("#dddddd")
    ax.axis("off")
    ax.set_title(
        "DSL: 6 ops. Body executed once per (bit B, index I) over the 100-bit input.\n"
        "A persists across iterations and is the final output. T resets to 0 each iteration.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_search_progress(log: dict, found_prog: str, out_path: str) -> None:
    """Show which programs were evaluated, broken down by length and phase."""
    evals = log.get("evals", [])
    if not evals:
        return

    # Cumulative count by (phase, length)
    lengths_seen = sorted({e["length"] for e in evals})
    phases_seen = sorted({e["phase"] for e in evals})

    cum = {L: 0 for L in lengths_seen}
    series = {L: {} for L in lengths_seen}
    last_seen = {L: 0 for L in lengths_seen}
    for ph in phases_seen:
        for ev in evals:
            if ev["phase"] != ph:
                continue
            cum[ev["length"]] += 1
        for L in lengths_seen:
            series[L][ph] = cum[L]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(lengths_seen)))
    for color, L in zip(colors, lengths_seen):
        xs = phases_seen
        ys = [series[L][ph] for ph in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.6,
                label=f"length {L}", color=color)
    ax.set_xlabel("phase  (Kt-cost = len(p) + log2(time(p)))")
    ax.set_ylabel("cumulative programs evaluated")
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    n_total = len(evals)
    ax.set_title(f"Search progress -- found {found_prog!r}  "
                 f"({n_total} total evaluations)")

    # Per-length histogram of pass/fail
    ax = axes[1]
    by_len_pass = {L: 0 for L in lengths_seen}
    by_len_fail = {L: 0 for L in lengths_seen}
    for ev in evals:
        if ev["ok"]:
            by_len_pass[ev["length"]] += 1
        else:
            by_len_fail[ev["length"]] += 1
    xs = lengths_seen
    fail = [by_len_fail[L] for L in xs]
    pas = [by_len_pass[L] for L in xs]
    ax.bar(xs, fail, color="#d62728", label=f"failed ({sum(fail)})",
           edgecolor="black", linewidth=0.6)
    ax.bar(xs, pas, bottom=fail, color="#2ca02c",
           label=f"passed ({sum(pas)})", edgecolor="black", linewidth=0.6)
    for L, f, p in zip(xs, fail, pas):
        if p > 0:
            ax.text(L, f + p + 0.5, f"{found_prog}",
                    ha="center", va="bottom", fontsize=9, color="#1f7a1f")
    ax.set_xlabel("program length")
    ax.set_ylabel("# programs evaluated")
    ax.set_xticks(xs)
    ax.set_title(f"Programs evaluated per length\n"
                 f"(alphabet={len(ALPHABET)} ops, search stopped on first match)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_program_trace(prog, examples, n_bits: int, out_path: str) -> None:
    """Visualize the execution of the found program on a single example."""
    x, target = examples[0]
    A_final, trace = run_body_traced(prog, x)

    A_per_iter = [tick[-1][0] for _, _, tick in trace]
    bits = [B for _, B, _ in trace]

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), dpi=130, sharex=True)

    ax = axes[0]
    ax.plot(range(n_bits), A_per_iter, color="#1f77b4", linewidth=1.4,
            label="A after each iteration")
    bit_idxs = [i for i, b in enumerate(bits) if b]
    ax.scatter(bit_idxs, [A_per_iter[i] for i in bit_idxs],
               s=30, color="#d62728", zorder=3,
               label="iterations where bit = 1 (A jumps by I)")
    ax.set_ylabel("accumulator A")
    popcount = sum(bits)
    ax.set_title(f"Execution of {prog_str(prog)!r} on training example 0  "
                 f"(popcount={popcount}, target={target}, A_final={A_final})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.bar(range(n_bits), bits, color="#7f7f7f", width=1.0)
    ax.set_xlabel("input bit index I")
    ax.set_ylabel("input bit B")
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.5, n_bits - 0.5)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generalization(prog, n_bits: int, seed: int, out_path: str) -> None:
    """Show induced weight vector vs ramp + held-out accuracy."""
    weights = induced_weight_vector(prog, n_bits)
    correct, total = test_generalization(prog, 200, n_bits, seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), dpi=130,
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    ax = axes[0]
    ax.bar(range(n_bits), weights, color="#1f77b4", width=1.0,
           label="induced weight w_i = output(e_i)")
    ax.plot(range(n_bits), range(n_bits), color="#d62728",
            linewidth=1.2, linestyle="--",
            label="ground-truth ramp w_i = i")
    ax.set_xlabel("bit index i")
    ax.set_ylabel("induced weight w_i")
    ax.set_title(f"Implicit linear weight vector of {prog_str(prog)!r}  "
                 f"(matches ramp: {weights == list(range(n_bits))})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.barh([0], [correct], color="#2ca02c", edgecolor="black",
            label=f"correct: {correct}")
    if total - correct > 0:
        ax.barh([0], [total - correct], left=[correct], color="#d62728",
                edgecolor="black", label=f"wrong: {total-correct}")
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"held-out trials")
    ax.set_title(f"Generalization on {total} fresh\nrandom 100-bit inputs:\n"
                 f"{correct}/{total} = {100*correct/total:.1f}%")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bits", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--max-length", type=int, default=6)
    p.add_argument("--max-phase", type=int, default=25)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    examples = make_examples(args.n_examples, args.n_bits, args.seed)
    log: dict = {}
    print(f"Running Levin search (seed={args.seed})...")
    prog, info = levin_search(examples, max_length=args.max_length,
                               max_phase=args.max_phase, log=log)
    if prog is None:
        print("Search failed; nothing to visualize.")
        return
    print(f"  found {prog_str(prog)!r}, length {info['length_found']}, "
          f"{info['n_visited']} evaluations.")

    plot_dsl(os.path.join(args.outdir, "dsl.png"))
    plot_search_progress(log, prog_str(prog),
                         os.path.join(args.outdir, "search_progress.png"))
    plot_program_trace(prog, examples, args.n_bits,
                       os.path.join(args.outdir, "program_trace.png"))
    plot_generalization(prog, args.n_bits, args.seed,
                        os.path.join(args.outdir, "generalization.png"))


if __name__ == "__main__":
    main()
