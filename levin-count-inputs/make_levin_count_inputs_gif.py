"""
Render an animated GIF showing Levin search finding the popcount program.

Layout per frame:
  Top:    big counter -- programs enumerated so far + current Levin round k
          + current program length L being explored.
  Middle: cumulative-programs-vs-round line plot (grows as the search runs).
  Bottom: when the program is found, the disassembly is revealed and held
          for a few frames; then a quick VM trace of the found program on a
          small input runs to completion.

Usage:
    python3 make_levin_count_inputs_gif.py
    python3 make_levin_count_inputs_gif.py --seed 0 --fps 14
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from levin_count_inputs import (
    OPS, BITS_PER_OP, NUM_OPS,
    levin_search, run, disassemble,
    make_training_examples,
)
from visualize_levin_count_inputs import trace_run


# --- helpers ---------------------------------------------------------------

def fig_to_image(fig) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64)


def render_search_frame(snap, history_so_far, max_progs, found_program=None,
                        found_at_idx=None):
    fig = plt.figure(figsize=(8.5, 5.0), dpi=80)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.45)

    # ---- top: counter banner ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    title = "Levin search: enumerate (program, runtime) pairs in order of $|p| + \\log_2 t$"
    ax.text(5, 3.5, title, fontsize=11, ha="center", color="#222")

    if found_program is None:
        ax.text(2.5, 1.8, f"k = {snap['k']}", fontsize=22, ha="center",
                fontweight="bold", color="#1f77b4")
        ax.text(2.5, 1.0, f"round  (cost cap $2^k$)", fontsize=9, ha="center",
                color="#666")

        ax.text(5.0, 1.8, f"L = {snap['L']}", fontsize=22, ha="center",
                fontweight="bold", color="#2ca02c")
        ax.text(5.0, 1.0, f"current length    t-budget = {snap['t_budget']}",
                fontsize=9, ha="center", color="#666")

        ax.text(7.5, 1.8, f"{snap['cumulative_progs_run']:,}", fontsize=22,
                ha="center", fontweight="bold", color="#d62728")
        ax.text(7.5, 1.0, "programs enumerated", fontsize=9, ha="center",
                color="#666")
    else:
        ax.text(5.0, 2.2, "FOUND", fontsize=26, ha="center", fontweight="bold",
                color="#0a7e1c")
        ax.text(5.0, 1.3, disassemble(found_program), fontsize=13, ha="center",
                fontfamily="monospace", color="#222")
        ax.text(5.0, 0.6,
                f"length = {len(found_program)} ops = "
                f"{BITS_PER_OP * len(found_program)} bits   "
                f"after {snap['cumulative_progs_run']:,} programs   "
                f"({snap['elapsed']:.2f} s)",
                fontsize=10, ha="center", color="#444")

    # ---- bottom: search progression -----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ks = [h["k"] for h in history_so_far]
    progs = [h["cumulative_progs_run"] for h in history_so_far]
    if ks:
        ax.plot(ks, progs, color="#1f77b4", marker="o", markersize=4,
                linewidth=1.3)
    ax.set_xlim(-0.5, 26)
    ax.set_ylim(0.5, max_progs * 2)
    ax.set_yscale("log")
    ax.set_xlabel("Levin round k")
    ax.set_ylabel("programs enumerated (cumulative)")
    ax.grid(alpha=0.3)

    # length-introduction markers
    L_max = 6
    for L in range(1, L_max + 1):
        k_intro = BITS_PER_OP * L
        ax.axvline(k_intro, color="#888", linestyle=":", linewidth=0.8)
        ax.text(k_intro + 0.1, 1.5, f"L={L}", fontsize=8, color="#666")
    if found_at_idx is not None and found_at_idx < len(history_so_far):
        k_found = history_so_far[found_at_idx]["k"]
        ax.axvline(k_found, color="green", linestyle="--", linewidth=1.2)
        ax.text(k_found + 0.2, max_progs * 0.5,
                f"FOUND @ k={k_found}", fontsize=9, color="green",
                fontweight="bold")

    return fig_to_image(fig)


def render_trace_frame(program, trace, step_idx, inp_str: str):
    """Frame showing the VM running the found popcount program on a small input."""
    fig = plt.figure(figsize=(8.5, 5.0), dpi=80)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.45)

    pc, op_name, stack, ptr, _ = trace[step_idx]

    # ---- top: program with current pc highlighted --------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, len(program) * 1.5 + 2)
    ax.set_ylim(-0.5, 4)

    ax.text(0.5 * (len(program) * 1.5 + 2), 3.5,
            "Running the found program: PUSH0 HERE BIT ADD LOOP",
            fontsize=11, ha="center", color="#222")

    for i, op in enumerate(program):
        x = i * 1.5 + 1
        name = OPS[op]
        is_current = (i == pc and step_idx > 0)
        color = "#ffd700" if is_current else (
            "#cce5ff" if name in ("HERE", "LOOP") else "#ffe6cc")
        edge = "black" if not is_current else "#cc6600"
        lw = 1.0 if not is_current else 2.5
        ax.add_patch(FancyBboxPatch((x - 0.6, 1.4), 1.2, 1.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor=edge,
                                    linewidth=lw))
        ax.text(x, 2.0, name, fontsize=11, ha="center", va="center",
                fontfamily="monospace", fontweight="bold")
        ax.text(x, 0.7, str(i), fontsize=8, ha="center", va="center",
                color="#666")

    # ---- bottom: stack + input pointer --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1, 6)
    ax.axis("off")

    # stack visualisation (bottom up)
    ax.text(0.5, 5.5, "stack:", fontsize=10, fontweight="bold")
    for i, val in enumerate(stack[-8:]):  # show top 8
        y = 4.2 - 0.6 * i
        ax.add_patch(Rectangle((1.5, y - 0.25), 1.2, 0.5,
                               facecolor="#cce5ff", edgecolor="black",
                               linewidth=0.8))
        ax.text(2.1, y, str(val), fontsize=10, ha="center", va="center",
                fontfamily="monospace")
        if i == 0:
            ax.text(3.0, y, "<- top", fontsize=8, va="center", color="#666")

    # ptr / input bar
    ax.text(5.5, 5.5, "input bits  (ptr ↓):", fontsize=10, fontweight="bold")
    for i, b in enumerate(inp_str):
        x = 5.5 + i * 0.65
        is_read = (i < ptr)
        color = "#666666" if is_read else "#ffffff"
        text_color = "white" if is_read else "black"
        ax.add_patch(Rectangle((x - 0.3, 4.0), 0.55, 0.55,
                               facecolor=color, edgecolor="black",
                               linewidth=0.8))
        ax.text(x - 0.05, 4.27, b, fontsize=10, ha="center", va="center",
                color=text_color, fontfamily="monospace")
        if i == ptr and ptr < len(inp_str):
            ax.text(x - 0.05, 4.85, "↓", fontsize=14, ha="center", color="red")

    # status line
    status = ""
    if step_idx > 0 and op_name:
        status = f"step {step_idx}: just executed {op_name}"
    else:
        status = "step 0: start"
    ax.text(0.5, 0.5, status, fontsize=10, color="#444")
    ax.text(0.5, -0.2,
            f"pc = {pc}    ptr = {ptr}    stack-top = "
            f"{stack[-1] if stack else 'empty'}",
            fontsize=9, color="#666", fontfamily="monospace")

    return fig_to_image(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bits", type=int, default=100)
    parser.add_argument("--max-program-bits", type=int, default=18)
    parser.add_argument("--max-log2-runtime", type=int, default=11)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default="levin_count_inputs.gif")
    args = parser.parse_args()

    train = make_training_examples(args.seed, n_bits=args.n_bits)

    # We capture per-round summaries from levin_search via its `history`,
    # plus a snapshot at first introduction of each L (those snapshots are
    # the visually interesting moments).
    print("Running search to collect history...")
    result = levin_search(
        train,
        max_program_bits=args.max_program_bits,
        max_log2_runtime=args.max_log2_runtime,
        verbose=False,
    )
    if not result["found"]:
        raise SystemExit("Search did not find a program; cannot render GIF.")

    program = result["program"]
    history = result["history"]
    final_progs = result["cumulative_progs_run"]

    # Build per-round snapshots for the search-progression part of the GIF.
    # For each round k, pick the last L visited in that round.
    L_max = args.max_program_bits // BITS_PER_OP
    found_round_idx = None
    for i, h in enumerate(history):
        if h["cumulative_progs_run"] == final_progs:
            found_round_idx = i
            break
    if found_round_idx is None:
        found_round_idx = len(history) - 1

    frames = []

    # Phase A: walk through the rounds, accumulating history
    for round_idx in range(found_round_idx + 1):
        h = history[round_idx]
        # determine L most recently first-seen at or before this round
        L_at_round = max(1, min(L_max, h["k"] // BITS_PER_OP))
        # synthesize a snapshot
        snap = {
            "k": h["k"],
            "L": L_at_round,
            "t_budget": 1 << max(0, h["k"] - BITS_PER_OP * L_at_round),
            "cumulative_progs_run": h["cumulative_progs_run"],
            "elapsed": h["elapsed"],
            "cumulative_steps": h["cumulative_steps"],
        }
        frame = render_search_frame(snap, history[: round_idx + 1],
                                     final_progs)
        frames.append(frame)

    # Phase B: 'FOUND' banner with cumulative line, hold for ~1 second
    final_snap = {
        "k": history[found_round_idx]["k"],
        "cumulative_progs_run": final_progs,
        "elapsed": result["elapsed_seconds"],
    }
    found_frame = render_search_frame(final_snap,
                                       history[: found_round_idx + 1],
                                       final_progs,
                                       found_program=program,
                                       found_at_idx=found_round_idx)
    for _ in range(args.fps):  # 1s hold
        frames.append(found_frame)

    # Phase C: VM trace of the found program on an 8-bit input.
    rng = np.random.default_rng(123)
    inp = tuple(int(b) for b in (rng.random(8) < 0.5).astype(int).tolist())
    inp_str = "".join(map(str, inp))
    trace = trace_run(program, inp)
    # Limit trace frames so the GIF stays small.
    trace_steps = min(len(trace), 24)
    for s in range(trace_steps):
        frames.append(render_trace_frame(program, trace, s, inp_str))

    # final hold
    for _ in range(args.fps):
        frames.append(frames[-1])

    print(f"Rendered {len(frames)} frames; saving to {args.out}")
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    sz = os.path.getsize(args.out)
    print(f"  size: {sz:,} bytes ({sz / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
