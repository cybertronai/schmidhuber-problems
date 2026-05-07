"""
Animate OOPS solving Towers of Hanoi with a discovered recursive program.

Layout per frame:
  Top:    three pegs with disks (current state).
  Middle: program tape with the currently-executing token highlighted, and
          the call-depth indicator (s_n calling s_{n-1} calling ...).
  Bottom: move counter and frame indicator (src, dst, aux pegs).

Default behavior: run OOPS up to n=5, animate the discovered recursive
program executing on the n=5 puzzle (31 moves -> ~31 frames + intermediate
"swap/call" frames -> small GIF).

Usage:
    python3 make_oops_towers_of_hanoi_gif.py
    python3 make_oops_towers_of_hanoi_gif.py --animate-n 6 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from oops_towers_of_hanoi import (oops_solve, Hanoi, TOKENS, T_M, T_SD, T_SA,
                                   T_C, _step_budget_for, tokens_to_str)


PEG_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
DISK_PALETTE = ["#fde68a", "#fcd34d", "#fbbf24", "#f59e0b", "#d97706",
                "#b45309", "#92400e", "#78350f", "#451a03", "#1c1917"]
TOKEN_COLORS = {"M": "#d62728", "SD": "#1f77b4",
                "SA": "#9467bd", "C": "#2ca02c"}


# ----------------------------------------------------------------------
# Trace a program with frame-aware snapshots (records all token events,
# including swaps and calls — not just disk moves).
# ----------------------------------------------------------------------

def trace_full(tokens: tuple[int, ...], frozen, n: int) -> list[dict]:
    """Return a list of frame dicts describing every token execution event.

    Each event = {state_pegs, frame, depth, call_stack (list of (sub_n, ip)),
    token_str, move (src,dst) or None, n_moves_so_far}.
    """
    state = Hanoi(n)
    steps_left = [_step_budget_for(n) * 4]  # ample budget for tracing
    events: list[dict] = []
    # initial frame, before any token
    events.append({
        "pegs": state.snapshot(),
        "frame": state.frame,
        "depth": 0,
        "call_stack": [(0, 0)],   # (sub_n=0 means "main"); ip=0
        "token": "(start)",
        "move": None,
        "n_moves_so_far": 0,
    })

    def run(toks: tuple[int, ...], call_target: int, sub_label: int):
        for ip, tok in enumerate(toks):
            if steps_left[0] <= 0:
                return False
            steps_left[0] -= 1
            move = None
            if tok == T_M:
                before = len(state.moves)
                state.move()
                if len(state.moves) > before:
                    move = state.moves[-1]
            elif tok == T_SD:
                state.swap_dst_aux()
            elif tok == T_SA:
                state.swap_src_aux()
            elif tok == T_C:
                if call_target < 0:
                    pass
                else:
                    sub = frozen[call_target]
                    saved = state.frame
                    # push a stack frame for visualization
                    if not run(sub.tokens, sub.call_target, sub.n):
                        return False
                    state.frame = saved
            events.append({
                "pegs": state.snapshot(),
                "frame": state.frame,
                "depth": _stack_depth(events),
                "call_stack": _current_stack(events, sub_label, ip + 1),
                "token": TOKENS[tok],
                "move": move,
                "n_moves_so_far": len(state.moves),
            })
        return True

    # We can't actually maintain a "live" call stack from outside, so we
    # emit events with a simpler stack: we only track depth via recursion.
    # Reset events and re-run with a proper push/pop model.
    events.clear()
    events.append({
        "pegs": state.snapshot(),
        "frame": state.frame,
        "depth": 0,
        "call_stack": [(0, 0)],
        "token": "(start)",
        "move": None,
        "n_moves_so_far": 0,
    })
    state = Hanoi(n)
    stack: list[list] = [[0, 0, list(tokens)]]  # [sub_n, ip, tokens]
    steps_left = [_step_budget_for(n) * 4]

    def emit(token_str, move):
        events.append({
            "pegs": state.snapshot(),
            "frame": state.frame,
            "depth": len(stack) - 1,
            "call_stack": [(s[0], s[1]) for s in stack],
            "token": token_str,
            "move": move,
            "n_moves_so_far": len(state.moves),
        })

    # Iterative interpreter so we can record the call stack at every step.
    while stack and steps_left[0] > 0:
        sub_n, ip, toks = stack[-1]
        if ip >= len(toks):
            stack.pop()
            continue
        tok = toks[ip]
        stack[-1][1] = ip + 1
        steps_left[0] -= 1
        move = None
        if tok == T_M:
            before = len(state.moves)
            state.move()
            if len(state.moves) > before:
                move = state.moves[-1]
            emit("M", move)
        elif tok == T_SD:
            state.swap_dst_aux()
            emit("SD", None)
        elif tok == T_SA:
            state.swap_src_aux()
            emit("SA", None)
        elif tok == T_C:
            # determine current call_target from the active sub
            if sub_n == 0:
                # main program's call_target = len(frozen) - 1
                call_target = len(frozen) - 1
            else:
                # find the frozen sub with this n, get its call_target
                for fs in frozen:
                    if fs.n == sub_n:
                        call_target = fs.call_target
                        break
                else:
                    call_target = -1
            if call_target >= 0:
                callee = frozen[call_target]
                # save the current frame on stack itself (we restore via stack semantics)
                stack[-1].append(("saved_frame", state.frame))
                stack.append([callee.n, 0, list(callee.tokens)])
                emit("C", None)
            else:
                emit("C", None)  # no-op
            continue
        # If we just popped after a call's tokens are exhausted, restore frame.
        # Detect this by looking at the parent's saved frame slot.
        while len(stack) >= 2 and stack[-1][1] >= len(stack[-1][2]):
            stack.pop()
            if stack and len(stack[-1]) > 3 and stack[-1][-1][0] == "saved_frame":
                state.frame = stack[-1][-1][1]
                stack[-1].pop()  # remove saved_frame marker

    # Final pop / frame restore loop
    while len(stack) >= 1 and stack[-1][1] >= len(stack[-1][2]):
        if len(stack[-1]) > 3 and stack[-1][-1][0] == "saved_frame":
            state.frame = stack[-1][-1][1]
        stack.pop()

    return events


def _stack_depth(events):
    return events[-1]["depth"] if events else 0


def _current_stack(events, sub_label, ip):
    # not used in iterative interpreter
    return []


# ----------------------------------------------------------------------
# Frame rendering
# ----------------------------------------------------------------------

def render_frame(event: dict,
                 main_program: tuple[int, ...],
                 main_program_idx: int,
                 n_disks: int,
                 max_moves: int,
                 sub_n_for_main: int) -> Image.Image:
    fig = plt.figure(figsize=(9, 5.4), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 0.7, 0.7], hspace=0.38)

    # ---- pegs ----
    ax = fig.add_subplot(gs[0, 0])
    pegs = event["pegs"]
    src, dst, aux = event["frame"]
    peg_xs = [0.5, 1.5, 2.5]
    peg_h = max(n_disks + 0.5, 4)
    for i in range(3):
        # peg post
        ax.add_patch(Rectangle((peg_xs[i] - 0.025, 0), 0.05, peg_h,
                               facecolor="#555", edgecolor="none"))
        # base label
        roles = []
        if i == src: roles.append("src")
        if i == dst: roles.append("dst")
        if i == aux: roles.append("aux")
        role_str = "/".join(roles) if roles else ""
        ax.text(peg_xs[i], -0.55, f"peg {i}\n{role_str}", ha="center",
                va="top", fontsize=9, color=PEG_COLORS[i],
                weight="bold" if roles else "normal")

    # disks
    for peg_idx, peg in enumerate(pegs):
        for j, d in enumerate(peg):
            color = DISK_PALETTE[(d - 1) % len(DISK_PALETTE)]
            # Scale so the largest disk fits within the per-peg slot (1.0 wide).
            half = 0.06 + 0.36 * (d / max(n_disks, 1))
            ax.add_patch(FancyBboxPatch((peg_xs[peg_idx] - half, j + 0.05),
                                        2 * half, 0.85,
                                        boxstyle="round,pad=0.005,rounding_size=0.06",
                                        facecolor=color, edgecolor="black",
                                        linewidth=0.5))
            ax.text(peg_xs[peg_idx], j + 0.5, str(d), ha="center", va="center",
                    fontsize=8, weight="bold")

    # base
    ax.add_patch(Rectangle((0.1, -0.05), 2.8, 0.05, facecolor="#222"))

    ax.set_xlim(0, 3)
    ax.set_ylim(-1.2, peg_h + 0.5)
    ax.set_axis_off()
    move_str = ""
    if event["move"] is not None:
        a, b = event["move"]
        move_str = f"  ->  moved disk peg{a} -> peg{b}"
    ax.set_title(f"Hanoi(n={n_disks})  -  move {event['n_moves_so_far']}/{max_moves}"
                 f"  -  call depth {event['depth']}{move_str}",
                 fontsize=11)

    # ---- token tape (main program) ----
    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    L = len(main_program)
    cell_w = 0.8
    x0 = 0.4
    y = 0.5
    for i, tok_id in enumerate(main_program):
        tok = TOKENS[tok_id]
        x = x0 + i * cell_w
        is_current = (i == main_program_idx - 1)
        ec = "black" if is_current else "#bbb"
        lw = 2.0 if is_current else 0.5
        box = FancyBboxPatch((x, y - 0.28), cell_w * 0.85, 0.56,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=TOKEN_COLORS.get(tok, "#888"),
                             edgecolor=ec, linewidth=lw)
        ax.add_patch(box)
        ax.text(x + cell_w * 0.425, y, tok, ha="center", va="center",
                fontsize=10, color="white", weight="bold")
    ax.text(x0 - 0.1, y, f"s_{sub_n_for_main}:",
            ha="right", va="center", fontsize=10, weight="bold")
    ax.set_xlim(0, x0 + L * cell_w + 0.5)
    ax.set_ylim(0, 1)

    # ---- call stack visualization ----
    ax = fig.add_subplot(gs[2, 0])
    ax.set_axis_off()
    stack = event["call_stack"]
    msg_parts = []
    for sub_n, ip in stack:
        label = "main" if sub_n == 0 else f"s_{sub_n}"
        msg_parts.append(f"{label}@{ip}")
    msg = "  ->  ".join(msg_parts) if msg_parts else "(done)"
    ax.text(0.02, 0.5, "call stack:  " + msg, ha="left", va="center",
            fontsize=9, family="monospace")
    ax.text(0.02, 0.05, f"current token: {event['token']}",
            ha="left", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-n", type=int, default=5,
                   help="Largest n OOPS solves (and the n we animate).")
    p.add_argument("--animate-n", type=int, default=None,
                   help="If set, animate this n's solver instead of --max-n.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="oops_towers_of_hanoi.gif")
    p.add_argument("--hold-final", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=400,
                   help="Subsample event trace to at most this many frames.")
    args = p.parse_args()

    print(f"Running OOPS up to n={args.max_n} (seed={args.seed})...")
    run = oops_solve(max_n=args.max_n, verbose=False)
    if not run.frozen:
        raise SystemExit("OOPS produced no subroutines")

    n_animate = args.animate_n if args.animate_n is not None else args.max_n
    if n_animate > len(run.frozen):
        raise SystemExit(f"animate-n={n_animate} not solved (max solved = {len(run.frozen)})")
    sub = run.frozen[n_animate - 1]
    print(f"Animating Hanoi(n={n_animate}) with s_{sub.n}: "
          f"[{tokens_to_str(sub.tokens)}]")

    # Trace
    events = trace_full(sub.tokens, run.frozen[:n_animate - 1], n_animate)
    print(f"  trace produced {len(events)} events; final move count = "
          f"{events[-1]['n_moves_so_far']} (optimal = {2**n_animate - 1})")

    # Subsample if too long
    if len(events) > args.max_frames:
        idxs = np.linspace(0, len(events) - 1, args.max_frames).astype(int)
        # Always include final state
        idxs = sorted(set(idxs.tolist() + [len(events) - 1]))
        events = [events[i] for i in idxs]
        print(f"  subsampled to {len(events)} frames")

    # Map each event to "which token of the main program is currently
    # executing" for the highlight bar. For nested calls, the highlighted
    # token is the position in the main program that triggered the call.
    main_ips = []
    for ev in events:
        if not ev["call_stack"]:
            main_ips.append(0)
        else:
            main_ips.append(ev["call_stack"][0][1])

    max_moves = 2 ** n_animate - 1
    frames = []
    for k, ev in enumerate(events):
        frame = render_frame(ev, sub.tokens, main_ips[k],
                             n_animate, max_moves, sub.n)
        frames.append(frame)
        if (k + 1) % 20 == 0 or k + 1 == len(events):
            print(f"  rendered {k+1}/{len(events)}")

    if args.hold_final > 0:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
