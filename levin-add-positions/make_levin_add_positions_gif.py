"""
Render an animated GIF of Levin search visiting programs in lex order.

Each frame shows:
  Top: the current program being tested and its outputs on the 3 training
       examples vs targets.
  Bottom: a heatmap. Each row is a program length L. Each cell is a program
          (in lex order). Cell color: gray = not yet visited, red = visited
          but failed, green = visited and matched all training examples.

Usage:
  python3 make_levin_add_positions_gif.py --seed 0
  python3 make_levin_add_positions_gif.py --seed 0 --snapshot-every 4 --fps 10
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

from levin_add_positions import (
    ALPHABET,
    N_OPS,
    all_programs_of_length,
    make_examples,
    prog_str,
    run_body,
)


def render_frame(grid_data: dict[int, list[int]],
                 current_prog: str,
                 current_outputs: list[int],
                 targets: list[int],
                 evals: int,
                 found_at: int | None) -> Image.Image:
    fig = plt.figure(figsize=(9, 5.0), dpi=90)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.5], hspace=0.35)

    # ---- top: text panel ----
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis("off")
    lines = []
    lines.append(f"Levin search   evaluations so far: {evals}")
    lines.append("")
    lines.append(f"Now testing:  {current_prog!r}    (length {len(current_prog)})")
    for i, (out, tgt) in enumerate(zip(current_outputs, targets)):
        mark = "MATCH" if out == tgt else "     "
        lines.append(f"  example {i}:  output = {out:>5}    "
                     f"target = {tgt:>5}    {mark}")
    if found_at is not None:
        lines.append("")
        lines.append(f"FOUND. Program {current_prog!r} matches all 3 training examples.")
    ax_text.text(0.01, 0.98, "\n".join(lines),
                 va="top", ha="left", family="monospace", fontsize=10)

    # ---- bottom: program-grid heatmap ----
    ax_grid = fig.add_subplot(gs[1, 0])
    lengths = sorted(grid_data.keys())
    max_count = max(len(grid_data[L]) for L in lengths)
    grid = np.full((len(lengths), max_count), -1, dtype=int)
    for r, L in enumerate(lengths):
        cells = grid_data[L]
        for c, v in enumerate(cells):
            grid[r, c] = v

    cmap = ListedColormap(["#dddddd", "#d62728", "#2ca02c"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax_grid.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    ax_grid.set_yticks(range(len(lengths)))
    ax_grid.set_yticklabels([f"L = {L}\n({N_OPS**L} progs)" for L in lengths],
                             fontsize=9)
    ax_grid.set_xticks([])
    ax_grid.set_xlabel("programs in lex order   "
                       "(gray = not yet visited, red = failed, green = passed)")
    ax_grid.set_title("Levin search frontier", fontsize=10)

    fig.suptitle("levin-add-positions: searching for index-sum on 100-bit input",
                 fontsize=11, y=0.98)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bits", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--max-length", type=int, default=4,
                   help="cap on program length for the GIF (default 4)")
    p.add_argument("--snapshot-every", type=int, default=4,
                   help="snapshot one frame every N program evaluations")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="levin_add_positions.gif")
    p.add_argument("--hold-final", type=int, default=12,
                   help="repeat the final frame this many times")
    p.add_argument("--palette-colors", type=int, default=64,
                   help="colors in the GIF palette (smaller = smaller file)")
    args = p.parse_args()

    examples = make_examples(args.n_examples, args.n_bits, args.seed)
    targets = [t for _, t in examples]

    grid_data: dict[int, list[int]] = {}
    for L in range(1, args.max_length + 1):
        grid_data[L] = [-1] * (N_OPS ** L)

    frames: list[Image.Image] = []
    evals = 0
    found_at: int | None = None
    found_prog_str: str | None = None
    found_outputs: list[int] | None = None

    print(f"Searching with seed={args.seed}, max_length={args.max_length}...")
    for L in range(1, args.max_length + 1):
        for c, prog in enumerate(all_programs_of_length(L)):
            outputs = [run_body(prog, x) for x, _ in examples]
            ok = all(o == t for o, t in zip(outputs, targets))
            grid_data[L][c] = 1 if ok else 0
            evals += 1

            new_found = ok and found_at is None
            if new_found:
                found_at = evals
                found_prog_str = prog_str(prog)
                found_outputs = outputs

            should_snap = (evals % args.snapshot_every == 0) or new_found
            if should_snap:
                frames.append(
                    render_frame(grid_data, prog_str(prog), outputs,
                                 targets, evals, found_at)
                )
            if found_at is not None:
                break
        if found_at is not None:
            break

    if found_at is None:
        print("  no program found within max_length; dumping a final frame.")
        frames.append(render_frame(grid_data, "(none)",
                                   [0] * len(targets), targets,
                                   evals, None))
    else:
        print(f"  found {found_prog_str!r} after {found_at} evaluations.")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    # palettize for size
    palette_frames = [
        f.convert("P", palette=Image.Palette.ADAPTIVE,
                  colors=args.palette_colors)
        for f in frames
    ]

    duration_ms = max(1000 // max(args.fps, 1), 40)
    palette_frames[0].save(
        args.out,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out}: {len(palette_frames)} frames, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
