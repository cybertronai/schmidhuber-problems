"""
Static visualizations for OOPS solving Towers of Hanoi.

Outputs (in `viz/`):
  search_cost_vs_n.png      - per-task wallclock + nodes expanded; the drop
                              to ~0 at n >= 4 is the OOPS reuse signature.
  found_programs.png        - the frozen subroutine library, disassembled.
  subroutine_reuse_graph.png- chain graph s_1 <- s_2 <- ... <- s_N showing
                              the recursive call structure.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from oops_towers_of_hanoi import (oops_solve, tokens_to_str, TOKENS, ALPHABET)


# ----------------------------------------------------------------------
# Plot 1: search cost vs n
# ----------------------------------------------------------------------

def plot_search_cost(history: list[dict], out_path: str) -> None:
    ns = [h["n"] for h in history]
    times_ms = [h["elapsed_s"] * 1000 for h in history]
    nodes = [max(h["nodes"], 0) for h in history]
    modes = [h["mode"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)

    # --- wallclock per task ---
    ax = axes[0]
    colors = ["#1f77b4" if m == "found" else "#2ca02c" for m in modes]
    bars = ax.bar(ns, times_ms, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("n (disks)")
    ax.set_ylabel("search wallclock (ms)")
    ax.set_title("Per-task search time")
    ax.grid(alpha=0.3, axis="y")
    for bar, t in zip(bars, times_ms):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15 if h > 0.05 else 0.06,
                f"{t:.2f}", ha="center", fontsize=7)

    # legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#1f77b4"), Patch(facecolor="#2ca02c")],
              ["search (Levin enumeration)", "reused frozen program"],
              loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xticks(ns)

    # --- nodes expanded ---
    ax = axes[1]
    bars = ax.bar(ns, nodes, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("n (disks)")
    ax.set_ylabel("programs enumerated")
    ax.set_title("Per-task search nodes")
    ax.grid(alpha=0.3, axis="y")
    for bar, nd in zip(bars, nodes):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h * 1.4 if h > 1 else 1.5,
                f"{nd}", ha="center", fontsize=7)
    ax.set_xticks(ns)

    fig.suptitle("OOPS / Towers of Hanoi — search cost collapses once the "
                 "recursive subroutine is found",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Plot 2: found programs disassembled
# ----------------------------------------------------------------------

TOKEN_COLORS = {
    "M":  "#d62728",
    "SD": "#1f77b4",
    "SA": "#9467bd",
    "C":  "#2ca02c",
}


def plot_found_programs(frozen, history, out_path: str) -> None:
    n_subs = len(frozen)
    fig, ax = plt.subplots(figsize=(9.5, max(2.5, 0.42 * n_subs + 1.6)),
                            dpi=140)
    ax.set_axis_off()

    # Reserve a row of space at the bottom for the token-color legend.
    legend_y = -1.0
    y_top = n_subs
    row_h = 0.85
    token_w = 0.55
    label_x = -0.1
    tokens_x0 = 1.6

    # header
    ax.text(label_x, y_top + 0.55, "task", ha="right", va="center",
            fontsize=10, weight="bold")
    ax.text(tokens_x0 - 0.2, y_top + 0.55, "frozen subroutine (tokens)",
            ha="left", va="center", fontsize=10, weight="bold")
    ax.text(tokens_x0 + 12 * token_w + 0.6, y_top + 0.55, "moves   /  optimal",
            ha="left", va="center", fontsize=10, weight="bold")
    ax.text(tokens_x0 + 12 * token_w + 4.4, y_top + 0.55, "search",
            ha="left", va="center", fontsize=10, weight="bold")

    for i, sub in enumerate(frozen):
        h = history[i]
        y = y_top - i - 0.5
        # task label
        ax.text(label_x, y, f"s_{sub.n}  (n={sub.n})",
                ha="right", va="center", fontsize=10)
        # token squares
        for j, tok_id in enumerate(sub.tokens):
            tok = TOKENS[tok_id]
            x = tokens_x0 + j * token_w
            color = TOKEN_COLORS.get(tok, "#888")
            box = FancyBboxPatch((x, y - row_h / 2),
                                 token_w * 0.92, row_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.07",
                                 facecolor=color, edgecolor="black",
                                 linewidth=0.5)
            ax.add_patch(box)
            ax.text(x + token_w * 0.46, y, tok, ha="center", va="center",
                    fontsize=8, color="white", weight="bold")
        # moves
        ax.text(tokens_x0 + 12 * token_w + 0.6, y,
                f"{h['moves_made']:>5d}  /  {h['optimal_moves']:<5d}",
                ha="left", va="center", fontsize=9,
                family="monospace")
        # search tag
        if h["mode"] == "reused":
            tag = "REUSED"
            tcolor = "#2ca02c"
        elif h["mode"] == "found":
            tag = f"found ({h['nodes']} nodes, {h['elapsed_s']*1000:.1f} ms)"
            tcolor = "#1f77b4"
        else:
            tag = "FAILED"
            tcolor = "#d62728"
        ax.text(tokens_x0 + 12 * token_w + 4.4, y, tag,
                ha="left", va="center", fontsize=9, color=tcolor)

    ax.set_xlim(-3.0, tokens_x0 + 12 * token_w + 9.5)
    ax.set_ylim(legend_y - 0.6, y_top + 1.5)

    # token-color legend (below all rows)
    leg_y = legend_y
    leg_x0 = 0.5
    ax.text(leg_x0, leg_y, "tokens:", ha="left", va="center",
            fontsize=9, weight="bold")
    leg_spacing = 3.4
    for k, (tok, col) in enumerate(TOKEN_COLORS.items()):
        x = leg_x0 + 1.6 + k * leg_spacing
        box = FancyBboxPatch((x, leg_y - row_h / 2),
                             token_w * 0.92, row_h,
                             boxstyle="round,pad=0.02,rounding_size=0.07",
                             facecolor=col, edgecolor="black", linewidth=0.5)
        ax.add_patch(box)
        ax.text(x + token_w * 0.46, leg_y, tok, ha="center", va="center",
                fontsize=8, color="white", weight="bold")
        meanings = {"M": "move src->dst", "SD": "swap dst<->aux",
                    "SA": "swap src<->aux", "C": "call frozen sub"}
        ax.text(x + token_w * 0.92 + 0.15, leg_y, meanings[tok],
                ha="left", va="center", fontsize=8)

    fig.suptitle(f"Frozen subroutine library after OOPS solves Hanoi(n=1..{n_subs})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Plot 3: subroutine reuse graph
# ----------------------------------------------------------------------

def plot_reuse_graph(frozen, out_path: str) -> None:
    n_subs = len(frozen)
    fig, ax = plt.subplots(figsize=(min(12, 1.4 * n_subs + 2), 3.0),
                            dpi=140)
    ax.set_axis_off()

    y = 0.5
    spacing = 1.4
    box_w = 1.0
    box_h = 0.55

    for i, sub in enumerate(frozen):
        x = i * spacing
        color = "#fef3c7" if i == 0 else "#bfdbfe"
        box = FancyBboxPatch((x, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.03,rounding_size=0.12",
                             facecolor=color, edgecolor="black", linewidth=0.6)
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + 0.05, f"s_{sub.n}",
                ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + box_w / 2, y - 0.16,
                f"L={len(sub.tokens)}",
                ha="center", va="center", fontsize=8)
        # tokens annotated above each box
        ax.text(x + box_w / 2, y + box_h / 2 + 0.12,
                tokens_to_str(sub.tokens),
                ha="center", va="bottom", fontsize=7,
                family="monospace", color="#444")
        # arrow from s_{i+1} -> s_i (next calls previous)
        if i < n_subs - 1:
            arr = FancyArrowPatch((x + spacing, y),
                                  (x + box_w + 0.05, y),
                                  arrowstyle="->", mutation_scale=14,
                                  color="#555", linewidth=1.0)
            ax.add_patch(arr)
            ax.text(x + box_w + (spacing - box_w) / 2, y - 0.42,
                    "calls", ha="center", va="center", fontsize=7,
                    color="#555", style="italic")

    ax.set_xlim(-0.4, n_subs * spacing - spacing + box_w + 0.4)
    ax.set_ylim(-0.3, 1.3)

    fig.suptitle("Subroutine reuse chain — each task's solver invokes the previous one",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-n", type=int, default=10)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Running OOPS up to n={args.max_n} (seed={args.seed})...")
    run = oops_solve(max_n=args.max_n, verbose=False)
    print(f"  finished {len(run.frozen)}/{args.max_n} tasks")

    plot_search_cost(run.history,
                     os.path.join(args.outdir, "search_cost_vs_n.png"))
    plot_found_programs(run.frozen, run.history,
                        os.path.join(args.outdir, "found_programs.png"))
    plot_reuse_graph(run.frozen,
                     os.path.join(args.outdir, "subroutine_reuse_graph.png"))


if __name__ == "__main__":
    main()
