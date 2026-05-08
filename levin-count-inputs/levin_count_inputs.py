"""
levin-count-inputs -- Levin search (universal search) for a program that
computes the popcount of a 100-bit input from only 3 training examples.

Schmidhuber, "Discovering solutions with low Kolmogorov complexity and high
generalization capability", ICML 1995; Neural Networks 10(5):857-873, 1997.

The 1995/1997 paper instantiates Levin search on a Forth-like assembler
(13 instructions) where the search target is the *weight vector* of a
linear unit f(x) = w . x; the solution program emits w_i = 1 for all i and
the linear unit then computes the popcount of x. We adopt the same
universal-search machinery but search directly over programs that map a
100-bit input to its popcount in a small stack-based DSL (8 instructions,
3 bits each). The algorithmic content is the same: enumerate programs in
order of |p| + log(t(p)), where t(p) is the runtime budget. See README
section "Deviations from the original" for the full list.

DSL (3 bits per instruction, 8 ops):
  0  PUSH0     push 0
  1  PUSH1     push 1
  2  ADD       pop a, pop b; push a + b
  3  BIT       push input[ptr]; advance ptr (no-op at end of input)
  4  DUP       duplicate top
  5  SWAP      swap top two
  6  HERE      mark loop point: loop_pc <- pc
  7  LOOP      if input has more bits, jump to most recent HERE
              (otherwise fall through)

The output of a program is the value left at the top of the stack when
control falls off the end of the program (or 0 if the stack is empty).

Reference popcount program (5 instructions, 15 bits):
  PUSH0  HERE  BIT  ADD  LOOP
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# DSL
# ----------------------------------------------------------------------

OPS = ("PUSH0", "PUSH1", "ADD", "BIT", "DUP", "SWAP", "HERE", "LOOP")
NUM_OPS = len(OPS)
BITS_PER_OP = 3  # ceil(log2(NUM_OPS))
MAX_STACK = 32   # cap to prevent DUP-spam blowing memory


def disassemble(program: tuple[int, ...]) -> str:
    return " ".join(OPS[op] for op in program)


# ----------------------------------------------------------------------
# Virtual machine
# ----------------------------------------------------------------------

# VM exit statuses
OK = 0           # halted normally (fell off the end)
TIMEOUT = 1      # exceeded max_steps -- could be revisited at a larger budget
ABORTED = 2     # underflow / overflow -- definitive failure, no point in
                # rerunning with more budget


def run(program: tuple[int, ...], inp: tuple[int, ...],
        max_steps: int) -> tuple[int, int, int]:
    """Execute one program against one input.

    Returns (output, status, steps_used). When status != OK, output is 0.
    """
    stack = []
    ptr = 0
    pc = 0
    loop_pc = -1  # -1 means "no HERE seen yet"; LOOP without HERE is a no-op
    steps = 0
    n = len(inp)
    plen = len(program)
    while pc < plen:
        if steps >= max_steps:
            return 0, TIMEOUT, steps
        op = program[pc]
        if op == 0:    # PUSH0
            if len(stack) >= MAX_STACK:
                return 0, ABORTED, steps
            stack.append(0)
            pc += 1
        elif op == 1:  # PUSH1
            if len(stack) >= MAX_STACK:
                return 0, ABORTED, steps
            stack.append(1)
            pc += 1
        elif op == 2:  # ADD
            if len(stack) < 2:
                return 0, ABORTED, steps
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
            pc += 1
        elif op == 3:  # BIT
            if len(stack) >= MAX_STACK:
                return 0, ABORTED, steps
            if ptr < n:
                stack.append(inp[ptr])
                ptr += 1
            else:
                stack.append(0)
            pc += 1
        elif op == 4:  # DUP
            if not stack or len(stack) >= MAX_STACK:
                return 0, ABORTED, steps
            stack.append(stack[-1])
            pc += 1
        elif op == 5:  # SWAP
            if len(stack) < 2:
                return 0, ABORTED, steps
            stack[-1], stack[-2] = stack[-2], stack[-1]
            pc += 1
        elif op == 6:  # HERE
            loop_pc = pc
            pc += 1
        elif op == 7:  # LOOP
            if ptr < n and loop_pc >= 0:
                pc = loop_pc + 1  # jump to instruction after HERE
            else:
                pc += 1
        else:
            return 0, ABORTED, steps
        steps += 1
    out = stack[-1] if stack else 0
    return out, OK, steps


# ----------------------------------------------------------------------
# Levin search
# ----------------------------------------------------------------------

def enumerate_programs(L: int):
    """Yield all programs of length L (tuples of ints in [0, NUM_OPS))."""
    if L == 0:
        yield tuple()
        return
    p = [0] * L
    while True:
        yield tuple(p)
        i = L - 1
        while i >= 0 and p[i] == NUM_OPS - 1:
            p[i] = 0
            i -= 1
        if i < 0:
            return
        p[i] += 1


def levin_search(training_examples: list[tuple[tuple[int, ...], int]],
                 max_program_bits: int,
                 max_log2_runtime: int,
                 verbose: bool = False,
                 snapshot_callback=None):
    """Universal-search ordering: enumerate (program, runtime) pairs in
    order of |p| + log2(t).

    At round k (cost cap 2^k), for each length L (in instructions, with
    description-length 3L bits), give each program of that length runtime
    budget 2^(k - 3L) ops.

    Programs that produce a definitive wrong answer are cached so they
    are not re-tested at later rounds.

    Returns a dict with {found, program, history, stats}.
    """
    max_L = max_program_bits // BITS_PER_OP
    # `dead`: programs that produced a definitive (non-timeout) wrong answer
    # at any earlier round. Indexed by length.
    dead = [set() for _ in range(max_L + 1)]
    history = []
    cumulative_progs_run = 0
    cumulative_steps = 0
    t0 = time.time()
    stats = {"rounds": 0, "max_round": 0,
             "programs_first_seen_per_L": [0] * (max_L + 1)}
    snap_idx = 0

    max_k = max_program_bits + max_log2_runtime
    for k in range(max_k + 1):
        stats["rounds"] += 1
        stats["max_round"] = k
        for L in range(1, max_L + 1):
            cost_per_prog = k - BITS_PER_OP * L  # log2 of runtime budget
            if cost_per_prog < 0:
                # Programs too long for this round; skip.
                continue
            t_budget = 1 << cost_per_prog
            seen_first = (cost_per_prog == 0)  # first round where length L appears
            for p in enumerate_programs(L):
                if p in dead[L]:
                    continue
                cumulative_progs_run += 1
                if seen_first:
                    stats["programs_first_seen_per_L"][L] += 1
                ok = True
                indecisive = False
                used_steps = 0
                for x, y in training_examples:
                    out, status, steps = run(p, x, max_steps=t_budget)
                    used_steps += steps
                    if status == TIMEOUT:
                        indecisive = True
                        ok = False
                        break
                    if status == ABORTED:
                        # definitive failure; no later round will fix it
                        dead[L].add(p)
                        ok = False
                        break
                    if out != y:
                        # definitive wrong answer
                        dead[L].add(p)
                        ok = False
                        break
                cumulative_steps += used_steps
                if snapshot_callback is not None and (cumulative_progs_run & 0xFFF) == 0:
                    snapshot_callback({
                        "k": k, "L": L, "t_budget": t_budget,
                        "cumulative_progs_run": cumulative_progs_run,
                        "cumulative_steps": cumulative_steps,
                        "elapsed": time.time() - t0,
                        "snap_idx": snap_idx,
                    })
                    snap_idx += 1
                if ok:
                    elapsed = time.time() - t0
                    if verbose:
                        print(f"[k={k}] FOUND length={L} bits={BITS_PER_OP*L} "
                              f"t_budget={t_budget} cumulative_progs={cumulative_progs_run:,} "
                              f"elapsed={elapsed:.2f}s")
                        print(f"  program: {disassemble(p)}")
                    return {
                        "found": True,
                        "program": p,
                        "program_length_instructions": L,
                        "program_length_bits": BITS_PER_OP * L,
                        "runtime_budget_at_find": t_budget,
                        "round_k_at_find": k,
                        "cumulative_progs_run": cumulative_progs_run,
                        "cumulative_steps": cumulative_steps,
                        "elapsed_seconds": elapsed,
                        "history": history,
                        "stats": stats,
                    }
            if verbose and seen_first:
                elapsed = time.time() - t0
                print(f"[k={k}] L={L} ({BITS_PER_OP*L} bits) introduced; "
                      f"cumulative_progs={cumulative_progs_run:,} "
                      f"cumulative_steps={cumulative_steps:,} "
                      f"elapsed={elapsed:.2f}s")
        history.append({
            "k": k,
            "cumulative_progs_run": cumulative_progs_run,
            "cumulative_steps": cumulative_steps,
            "elapsed": time.time() - t0,
        })
    return {
        "found": False,
        "history": history,
        "stats": stats,
        "cumulative_progs_run": cumulative_progs_run,
        "cumulative_steps": cumulative_steps,
        "elapsed_seconds": time.time() - t0,
    }


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_training_examples(seed: int, n_bits: int = 100,
                           popcounts=(25, 50, 75)
                           ) -> list[tuple[tuple[int, ...], int]]:
    """3 training examples: random 100-bit strings whose popcounts span the
    range so that no constant / short-prefix program can pass.

    The Schmidhuber 1995/1997 paper claims the search succeeds with only 3
    training examples; we honour that with 3 deliberately diverse examples.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in popcounts:
        bits = [0] * n_bits
        idx = rng.permutation(n_bits)[:k]
        for i in idx:
            bits[i] = 1
        out.append((tuple(int(b) for b in bits), int(k)))
    return out


def make_test_examples(seed: int, n: int = 200, n_bits: int = 100
                       ) -> list[tuple[tuple[int, ...], int]]:
    """Held-out test set: n random 100-bit strings with random popcounts."""
    rng = np.random.default_rng(seed + 0xBEEF)
    out = []
    for _ in range(n):
        bits = (rng.random(n_bits) < 0.5).astype(int)
        out.append((tuple(int(b) for b in bits), int(bits.sum())))
    return out


def evaluate_program(program: tuple[int, ...],
                     examples: list[tuple[tuple[int, ...], int]],
                     max_steps: int = 100_000) -> dict:
    correct = 0
    total = len(examples)
    for x, y in examples:
        out, status, _ = run(program, x, max_steps=max_steps)
        if status == OK and out == y:
            correct += 1
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0}


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

def collect_env() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": commit,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bits", type=int, default=100,
                        help="number of input bits (default 100, paper value)")
    parser.add_argument("--max-program-bits", type=int, default=18,
                        help="cap on program description length, in bits "
                             "(default 18 = 6 ops; popcount fits at 15 bits)")
    parser.add_argument("--max-log2-runtime", type=int, default=11,
                        help="cap on log2(runtime budget) per program "
                             "(default 11 -> 2^11 = 2048 ops max)")
    parser.add_argument("--n-test", type=int, default=200,
                        help="held-out generalization test size")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train = make_training_examples(args.seed, n_bits=args.n_bits)
    test = make_test_examples(args.seed, n=args.n_test, n_bits=args.n_bits)

    if not args.quiet:
        print(f"# levin-count-inputs (seed={args.seed})")
        print(f"# DSL: {NUM_OPS} ops at {BITS_PER_OP} bits each")
        print(f"# training examples ({len(train)}): "
              f"popcounts = {[y for _, y in train]}")
        print(f"# test examples: {len(test)}")
        print(f"# search bounds: max_program_bits={args.max_program_bits} "
              f"max_log2_runtime={args.max_log2_runtime}")
        print(f"# total Levin cost cap: 2^{args.max_program_bits + args.max_log2_runtime}")
        print()

    t0 = time.time()
    result = levin_search(
        train,
        max_program_bits=args.max_program_bits,
        max_log2_runtime=args.max_log2_runtime,
        verbose=not args.quiet,
    )
    wallclock = time.time() - t0

    out = {
        "seed": args.seed,
        "n_bits": args.n_bits,
        "max_program_bits": args.max_program_bits,
        "max_log2_runtime": args.max_log2_runtime,
        "training_popcounts": [y for _, y in train],
        "wallclock_seconds": wallclock,
        "env": collect_env(),
        "search": {k: v for k, v in result.items() if k != "history"},
    }

    if result["found"]:
        program = result["program"]
        train_eval = evaluate_program(program, train)
        test_eval = evaluate_program(program, test)
        out["program"] = list(program)
        out["disassembly"] = disassemble(program)
        out["program_length_instructions"] = result["program_length_instructions"]
        out["program_length_bits"] = result["program_length_bits"]
        out["train_accuracy"] = train_eval["accuracy"]
        out["test_accuracy"] = test_eval["accuracy"]
        if not args.quiet:
            print()
            print("Found program (Schmidhuber 1995/1997-style universal search):")
            print(f"  bytes (op codes):   {list(program)}")
            print(f"  disassembly:        {disassemble(program)}")
            print(f"  length:             {result['program_length_instructions']} "
                  f"instructions = {result['program_length_bits']} bits")
            print(f"  Levin round k:      {result['round_k_at_find']}")
            print(f"  runtime budget:     {result['runtime_budget_at_find']} ops")
            print(f"  programs enumerated: {result['cumulative_progs_run']:,}")
            print(f"  VM steps total:     {result['cumulative_steps']:,}")
            print(f"  wallclock:          {wallclock:.2f}s")
            print()
            print(f"  training accuracy:  {train_eval['correct']}/{train_eval['total']} "
                  f"= {train_eval['accuracy']*100:.0f}%")
            print(f"  test accuracy:      {test_eval['correct']}/{test_eval['total']} "
                  f"= {test_eval['accuracy']*100:.0f}%")
    else:
        if not args.quiet:
            print()
            print("No program found within the bounds.")
            print(f"  programs enumerated: {result['cumulative_progs_run']:,}")
            print(f"  VM steps total:     {result['cumulative_steps']:,}")
            print(f"  wallclock:          {wallclock:.2f}s")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
