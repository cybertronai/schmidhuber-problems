"""
Static visualizations for the levin-count-inputs run.

Outputs (in `viz/`):
  search_progression.png  - cumulative programs enumerated vs Levin round k
  dsl_table.png           - the 8-instruction stack DSL
  program_disassembly.png - the found program annotated with role
  vm_trace.png            - VM stack trace of the found program on a small input
  generalization.png      - per-popcount test-set accuracy
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from levin_count_inputs import (
    OPS, BITS_PER_OP, NUM_OPS, MAX_STACK,
    levin_search, run, disassemble,
    make_training_examples, make_test_examples, evaluate_program,
    OK,
)


OP_DESCRIPTIONS = {
    "PUSH0": "push 0",
    "PUSH1": "push 1",
    "ADD":   "pop a, pop b; push a + b",
    "BIT":   "push input[ptr]; advance ptr",
    "DUP":   "duplicate top",
    "SWAP":  "swap top two",
    "HERE":  "mark loop point: loop_pc <- pc",
    "LOOP":  "if input has more bits, jump to most recent HERE",
}


def plot_search_progression(history_per_L_first_seen, search_history, out_path):
    """One panel: cumulative programs enumerated as a function of Levin round k.
    Annotated with the round at which each program length is first introduced
    and the round at which the popcount program is found.

    `history_per_L_first_seen` is a list of (L, k, cumulative_progs) tuples;
    `search_history` is a list of {k, cumulative_progs_run, cumulative_steps,
    elapsed} dicts emitted by `levin_search`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)

    ks = [h["k"] for h in search_history]
    progs = [h["cumulative_progs_run"] for h in search_history]
    steps = [h["cumulative_steps"] for h in search_history]

    ax = axes[0]
    ax.plot(ks, progs, color="#1f77b4", marker="o", markersize=3, linewidth=1.2)
    for L, k_intro, c in history_per_L_first_seen:
        ax.axvline(k_intro, color="#888", linestyle=":", linewidth=0.8)
        ax.annotate(f"L={L}", xy=(k_intro, max(progs) * 0.6 + L * max(progs) * 0.04),
                    fontsize=8, color="#444")
    ax.set_xlabel("Levin round k  (cost cap = $2^k$)")
    ax.set_ylabel("programs enumerated (cumulative)")
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.3)
    ax.set_title("Search effort by Levin round")

    ax = axes[1]
    ax.plot(ks, steps, color="#d62728", marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Levin round k")
    ax.set_ylabel("VM steps (cumulative)")
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.3)
    ax.set_title("VM steps by Levin round")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_dsl_table(out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, NUM_OPS + 0.5)

    # header
    ax.text(0.4, NUM_OPS, "code", fontweight="bold", fontsize=10)
    ax.text(1.4, NUM_OPS, "name", fontweight="bold", fontsize=10)
    ax.text(3.4, NUM_OPS, "effect", fontweight="bold", fontsize=10)
    ax.plot([0.2, 9.8], [NUM_OPS - 0.3] * 2, color="black", linewidth=0.8)

    for i, op in enumerate(OPS):
        y = NUM_OPS - i - 1
        bg = "#f4f4f4" if i % 2 == 0 else "#ffffff"
        ax.add_patch(Rectangle((0.2, y - 0.4), 9.6, 0.8,
                               facecolor=bg, edgecolor="none"))
        bin_str = format(i, f"0{BITS_PER_OP}b")
        ax.text(0.4, y, f"{i} ({bin_str})", fontfamily="monospace",
                fontsize=10, va="center")
        ax.text(1.4, y, op, fontfamily="monospace",
                fontsize=10, va="center", fontweight="bold")
        ax.text(3.4, y, OP_DESCRIPTIONS[op], fontsize=10, va="center")

    ax.set_title(f"DSL: {NUM_OPS} stack-machine ops, {BITS_PER_OP} bits each",
                 fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_program_disassembly(program: tuple[int, ...], out_path: str):
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=120)
    ax.axis("off")
    ax.set_xlim(0, len(program) * 1.5 + 2)
    ax.set_ylim(-2, 5)

    role_for = {
        "PUSH0": "init accumulator = 0",
        "HERE":  "loop point",
        "BIT":   "push next input bit",
        "ADD":   "accumulator += bit",
        "LOOP":  "loop if more bits remain",
    }

    for i, op in enumerate(program):
        x = i * 1.5 + 1
        name = OPS[op]
        color = "#cce5ff" if name in ("HERE", "LOOP") else "#ffe6cc"
        ax.add_patch(FancyBboxPatch((x - 0.6, 1.6), 1.2, 1.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="black",
                                    linewidth=1.0))
        ax.text(x, 2.2, name, fontsize=11, ha="center", va="center",
                fontfamily="monospace", fontweight="bold")
        ax.text(x, 1.0, str(op), fontsize=9, ha="center", va="center",
                fontfamily="monospace", color="#666")
        if name in role_for:
            ax.text(x, 0.0, role_for[name], fontsize=8, ha="center", va="top",
                    color="#444", style="italic", wrap=True,
                    rotation=20)

    bits = BITS_PER_OP * len(program)
    ax.set_title(f"Found program: {disassemble(program)}     "
                 f"({len(program)} instructions = {bits} bits)",
                 fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def trace_run(program, inp, max_steps=10000):
    """Verbose VM trace for visualisation. Returns list of (pc, op, stack, ptr)."""
    stack = []
    ptr = 0
    pc = 0
    loop_pc = -1
    n = len(inp)
    plen = len(program)
    steps = 0
    out = [(pc, None, list(stack), ptr, "start")]
    while pc < plen and steps < max_steps:
        op = program[pc]
        name = OPS[op]
        if op == 0:
            stack.append(0); pc += 1
        elif op == 1:
            stack.append(1); pc += 1
        elif op == 2:
            a = stack.pop(); b = stack.pop(); stack.append(a + b); pc += 1
        elif op == 3:
            if ptr < n:
                stack.append(inp[ptr]); ptr += 1
            else:
                stack.append(0)
            pc += 1
        elif op == 4:
            stack.append(stack[-1]); pc += 1
        elif op == 5:
            stack[-1], stack[-2] = stack[-2], stack[-1]; pc += 1
        elif op == 6:
            loop_pc = pc; pc += 1
        elif op == 7:
            if ptr < n and loop_pc >= 0:
                pc = loop_pc + 1
            else:
                pc += 1
        steps += 1
        out.append((pc, name, list(stack), ptr, ""))
    return out


def plot_vm_trace(program, out_path, n_demo_bits: int = 8):
    """Plot stack-top trace of running the popcount program on a small example."""
    rng = np.random.default_rng(123)
    inp = (rng.random(n_demo_bits) < 0.5).astype(int).tolist()
    inp = tuple(int(b) for b in inp)
    expected = sum(inp)
    trace = trace_run(program, inp)

    accs = [t[2][-1] if t[2] else 0 for t in trace]
    ptrs = [t[3] for t in trace]
    pcs = [t[0] for t in trace]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.0), dpi=120, sharex=True)

    ax = axes[0]
    ax.plot(range(len(accs)), accs, color="#1f77b4", marker="o", markersize=2.5,
            linewidth=1.0, label="stack top (accumulator)")
    ax.plot(range(len(ptrs)), ptrs, color="#2ca02c", marker="s", markersize=2.5,
            linewidth=1.0, label="input pointer")
    ax.axhline(expected, color="#d62728", linestyle="--", linewidth=0.8,
               label=f"target popcount = {expected}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.set_title(f"VM trace of {disassemble(program)} on {n_demo_bits}-bit input "
                 f"= {''.join(map(str, inp))}    (target popcount = {expected})")

    ax = axes[1]
    ax.plot(range(len(pcs)), pcs, color="#9467bd", marker=".", markersize=4,
            linewidth=0.8)
    ax.set_xlabel("VM step")
    ax.set_ylabel("program counter (pc)")
    ax.set_yticks(list(range(len(program) + 1)))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_generalization(program, test_examples, out_path):
    rng = np.random.default_rng(0)
    n_buckets = 10
    bucket_acc = []
    bucket_n = []
    for k in range(n_buckets):
        lo = int(100 * k / n_buckets)
        hi = int(100 * (k + 1) / n_buckets)
        in_bucket = [(x, y) for x, y in test_examples if lo <= y < hi]
        if not in_bucket:
            bucket_acc.append(0); bucket_n.append(0); continue
        ev = evaluate_program(program, in_bucket)
        bucket_acc.append(ev["accuracy"])
        bucket_n.append(len(in_bucket))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=120)

    ax = axes[0]
    xs = [f"[{int(100 * k / n_buckets)},{int(100 * (k + 1) / n_buckets)})"
          for k in range(n_buckets)]
    ax.bar(xs, [a * 100 for a in bucket_acc], color="#2ca02c")
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("popcount bucket")
    ax.set_ylim(0, 110)
    ax.set_title("Test accuracy by popcount bucket")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(xs, bucket_n, color="#1f77b4")
    ax.set_ylabel("# test examples")
    ax.set_xlabel("popcount bucket")
    ax.set_title(f"Test-set distribution (n={len(test_examples)})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bits", type=int, default=100)
    parser.add_argument("--max-program-bits", type=int, default=18)
    parser.add_argument("--max-log2-runtime", type=int, default=11)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train = make_training_examples(args.seed, n_bits=args.n_bits)
    test = make_test_examples(args.seed, n=args.n_test, n_bits=args.n_bits)

    # Re-run search to collect history
    first_seen = []
    seen_lengths = set()

    def collect_first_seen(L, k, cumulative_progs):
        if L not in seen_lengths:
            seen_lengths.add(L)
            first_seen.append((L, k, cumulative_progs))

    # Patch levin_search to record first-seen markers via stats
    result = levin_search(
        train,
        max_program_bits=args.max_program_bits,
        max_log2_runtime=args.max_log2_runtime,
        verbose=False,
    )
    if not result["found"]:
        raise SystemExit("Search did not find a program; cannot visualise.")

    program = result["program"]
    history = result["history"]

    # Reconstruct first-seen markers from history (k where each L is introduced
    # is k = BITS_PER_OP * L, so we mark those rounds)
    L_max = args.max_program_bits // BITS_PER_OP
    first_seen_pairs = []
    for L in range(1, L_max + 1):
        k_intro = BITS_PER_OP * L
        # find cumulative_progs at that round
        c = 0
        for h in history:
            if h["k"] == k_intro:
                c = h["cumulative_progs_run"]
                break
        first_seen_pairs.append((L, k_intro, c))

    plot_search_progression(first_seen_pairs, history,
                            os.path.join(args.outdir, "search_progression.png"))
    plot_dsl_table(os.path.join(args.outdir, "dsl_table.png"))
    plot_program_disassembly(program,
                             os.path.join(args.outdir, "program_disassembly.png"))
    plot_vm_trace(program, os.path.join(args.outdir, "vm_trace.png"))
    plot_generalization(program, test,
                        os.path.join(args.outdir, "generalization.png"))

    print(f"Saved visualisations to {args.outdir}/")
    print(f"  search_progression.png")
    print(f"  dsl_table.png")
    print(f"  program_disassembly.png")
    print(f"  vm_trace.png")
    print(f"  generalization.png")


if __name__ == "__main__":
    main()
