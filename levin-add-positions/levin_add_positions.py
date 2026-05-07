"""
levin_add_positions.py -- Levin search for index-sum on a 100-bit input.

Schmidhuber, "Discovering solutions with low Kolmogorov complexity and high
generalization capability", ICML 1995 / Neural Networks 10(5):857-873, 1997.

Task
----
Input: 100-bit binary string. Target: sum of indices i where input[i] == 1.
For example, input with 1s at positions 0 and 2 -> target = 0 + 2 = 2.

Schmidhuber's claim: Levin universal search, given just 3 random training
examples, finds a short program whose induced weight vector w_i = i is the
canonical "ramp" solution. The bias toward short programs in Levin search
implements Occam's razor; gradient descent on a linear unit with sparse data
overfits and does not recover the ramp.

DSL (6 ops, see table below)
----------------------------
A "body" program of length L is executed once per (B = bit, I = index) where
B = input[I]. The interpreter has two integer registers:
  A -- accumulator. Initialized to 0 at program start, persists across all
       100 iterations, and is the final output.
  T -- temp. Reset to 0 at the start of every iteration.

  Op | Effect       | Comment
  ---|--------------|--------------------------------
   + | A := A + T   | accumulate temp into output
   * | A := A * T   | multiply output by temp
   m | T := T * B   | gate temp by current bit
   i | T := I       | load current index into temp
   b | T := B       | load current bit into temp
   1 | T := 1       | load constant 1 into temp

Optimal solution: "im+" (length 3) -- T:=I; T:=T*B; A:=A+T -- which gives
A_final = sum_{I where B=1} I = sum of indices where the bit is 1.

Levin search (LSEARCH)
----------------------
Universal search ordering: programs are visited in order of
  Kt(p) = len(p) + log2(time(p))
where time(p) is the number of interpreter steps p uses to halt. The
phase-based outer loop allocates 2^(phase - len(p)) interpreter steps to
each program at phase phi.

For our straight-line DSL, every program halts in exactly
  time(p) = len(p) * n_bits * n_examples
steps, so the time term reduces to a constant offset per length. The
phase-based search degenerates into iterative-deepening on length, but we
keep the LSEARCH structure (with the time-budget check) for fidelity to the
1995 paper. v2 hook: add a JUMP_BACK / IF_T primitive so programs can loop,
and the time term genuinely matters.

Reproducibility
---------------
`python3 levin_add_positions.py --seed N` is fully deterministic. Records
seed, numpy version, Python version, platform, and git commit in the
result dict (`describe_env()` below).

CLI
---
  python3 levin_add_positions.py --seed 0
  python3 levin_add_positions.py --seed 0 --verbose
  python3 levin_add_positions.py --seed 0 --max-length 6 --n-test 200
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess
import sys
import time

import numpy as np


# ---- DSL ----------------------------------------------------------------

ALPHABET = ('+', '*', 'm', 'i', 'b', '1')
N_OPS = len(ALPHABET)

OP_PLUS  = 0  # A := A + T
OP_TIMES = 1  # A := A * T
OP_GATE  = 2  # T := T * B
OP_IDX   = 3  # T := I
OP_BIT   = 4  # T := B
OP_ONE   = 5  # T := 1


def prog_str(prog) -> str:
    return ''.join(ALPHABET[i] for i in prog)


def parse_prog(s: str):
    return tuple(ALPHABET.index(c) for c in s)


def run_body(prog, x) -> int:
    """Execute the body once per bit of x; return final A.

    A persists across iterations, T resets to 0 at the start of each.
    """
    A = 0
    n = len(x)
    for I in range(n):
        B = int(x[I])
        T = 0
        for op in prog:
            if op == OP_PLUS:
                A = A + T
            elif op == OP_TIMES:
                A = A * T
            elif op == OP_GATE:
                T = T * B
            elif op == OP_IDX:
                T = I
            elif op == OP_BIT:
                T = B
            else:  # OP_ONE
                T = 1
    return A


def run_body_traced(prog, x):
    """Same as run_body but also returns a per-iteration trace.

    Returns (A_final, trace) where trace[i] = (I, B, [(A_after_op, T_after_op), ...]).
    """
    A = 0
    n = len(x)
    trace = []
    for I in range(n):
        B = int(x[I])
        T = 0
        per_tick = []
        for op in prog:
            if op == OP_PLUS:
                A = A + T
            elif op == OP_TIMES:
                A = A * T
            elif op == OP_GATE:
                T = T * B
            elif op == OP_IDX:
                T = I
            elif op == OP_BIT:
                T = B
            else:
                T = 1
            per_tick.append((A, T))
        trace.append((I, B, per_tick))
    return A, trace


# ---- task ---------------------------------------------------------------

def make_examples(n_examples: int, n_bits: int, seed: int):
    """Generate n_examples random 100-bit inputs with their index-sum targets."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_examples):
        x = rng.integers(0, 2, size=n_bits, dtype=np.int8)
        target = int(np.sum(np.arange(n_bits)[x == 1]))
        out.append((x, target))
    return out


# ---- Levin search -------------------------------------------------------

def all_programs_of_length(L: int):
    """Yield every length-L program as a tuple of op indices, in lex order."""
    for idx in range(N_OPS ** L):
        prog = []
        x = idx
        for _ in range(L):
            prog.append(x % N_OPS)
            x //= N_OPS
        yield tuple(prog)


def levin_search(examples,
                 max_length: int = 6,
                 max_phase: int = 25,
                 log: dict | None = None,
                 verbose: bool = False):
    """LSEARCH ordered by Kt(p) = len(p) + log2(time(p)).

    Returns (prog, info). info is a dict with keys:
      n_visited, phase_found, length_found, wallclock_s, Kt_cost.
    On failure prog is None and length_found / phase_found are -1.

    log, if provided, gets {"evals": [...]} appended with one record per
    program evaluation, and {"phase_summary": [...]} with one record per
    phase that contributed new evaluations.
    """
    n_bits = len(examples[0][0])
    n_ex = len(examples)
    visited = set()
    n_total = 0
    t0 = time.time()

    if log is not None:
        log.setdefault("evals", [])
        log.setdefault("phase_summary", [])

    for phase in range(1, max_phase + 1):
        phase_evals = 0
        phase_lengths_run = []
        for L in range(1, phase + 1):
            if L > max_length:
                continue
            log_t = phase - L
            time_budget = 1 << log_t  # 2 ** log_t
            min_time_needed = L * n_bits * n_ex
            if time_budget < min_time_needed:
                continue
            phase_lengths_run.append(L)
            for prog in all_programs_of_length(L):
                if prog in visited:
                    continue
                visited.add(prog)
                n_total += 1
                phase_evals += 1
                ok = True
                for x, target in examples:
                    if run_body(prog, x) != target:
                        ok = False
                        break
                if log is not None:
                    log["evals"].append({
                        "phase": phase, "length": L,
                        "prog": prog_str(prog), "ok": ok,
                    })
                if ok:
                    elapsed = time.time() - t0
                    info = {
                        "n_visited": n_total,
                        "phase_found": phase,
                        "length_found": L,
                        "wallclock_s": elapsed,
                        "Kt_cost": L + math.log2(min_time_needed),
                    }
                    if verbose:
                        print(f"  found {prog_str(prog)!r} (length {L}) at "
                              f"phase {phase} after {n_total} evaluations, "
                              f"{elapsed:.3f}s")
                    if log is not None:
                        log["phase_summary"].append({
                            "phase": phase, "new_evals": phase_evals,
                            "lengths_run": phase_lengths_run,
                        })
                    return prog, info
        if log is not None and phase_evals > 0:
            log["phase_summary"].append({
                "phase": phase, "new_evals": phase_evals,
                "lengths_run": phase_lengths_run,
            })
        if verbose and phase_evals > 0:
            print(f"  phase {phase}: {phase_evals} new programs (total "
                  f"{n_total}), lengths fully evaluated: {phase_lengths_run}")
    elapsed = time.time() - t0
    return None, {
        "n_visited": n_total,
        "phase_found": -1,
        "length_found": -1,
        "wallclock_s": elapsed,
        "Kt_cost": float("inf"),
    }


# ---- generalization test ------------------------------------------------

def test_generalization(prog, n_test: int, n_bits: int, seed: int):
    """Test prog on n_test fresh random 100-bit inputs (seed-derived)."""
    rng = np.random.default_rng(seed + 99999)
    correct = 0
    for _ in range(n_test):
        x = rng.integers(0, 2, size=n_bits, dtype=np.int8)
        target = int(np.sum(np.arange(n_bits)[x == 1]))
        if run_body(prog, x) == target:
            correct += 1
    return correct, n_test


def induced_weight_vector(prog, n_bits: int):
    """The implicit linear weight w_k that prog computes on standard basis e_k."""
    weights = []
    for k in range(n_bits):
        x = np.zeros(n_bits, dtype=np.int8)
        x[k] = 1
        weights.append(run_body(prog, x))
    return weights


# ---- environment record -------------------------------------------------

def describe_env(seed: int) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "seed": seed,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": commit,
    }


# ---- CLI ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bits", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--max-length", type=int, default=6)
    p.add_argument("--max-phase", type=int, default=25)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    env = describe_env(args.seed)
    print("Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print()

    print(f"Generating {args.n_examples} training examples "
          f"({args.n_bits} bits each, seed={args.seed})...")
    examples = make_examples(args.n_examples, args.n_bits, args.seed)
    for i, (x, t) in enumerate(examples):
        print(f"  example {i}: popcount={int(x.sum())}, target={t}")
    print()

    print(f"Running Levin search (alphabet={list(ALPHABET)}, "
          f"max_length={args.max_length}, max_phase={args.max_phase})...")
    prog, info = levin_search(examples, max_length=args.max_length,
                               max_phase=args.max_phase, verbose=args.verbose)

    print()
    if prog is None:
        print(f"NO PROGRAM FOUND within max_length={args.max_length}, "
              f"max_phase={args.max_phase} ({info['n_visited']} evaluations).")
        return

    print(f"Found program:    {prog_str(prog)!r}")
    print(f"Length:           {info['length_found']}")
    print(f"Phase:            {info['phase_found']}")
    print(f"Kt-cost (approx): {info['Kt_cost']:.2f}")
    print(f"Evaluations:      {info['n_visited']}")
    print(f"Wallclock:        {info['wallclock_s']:.3f} s")
    print()

    # Show the induced weight vector to verify the ramp
    weights = induced_weight_vector(prog, args.n_bits)
    is_ramp = weights == list(range(args.n_bits))
    print(f"Induced weight vector (first 10): {weights[:10]}")
    print(f"Induced weight vector (last 10):  {weights[-10:]}")
    print(f"Matches ramp w_i = i: {is_ramp}")
    print()

    print(f"Testing on {args.n_test} held-out 100-bit inputs (seed-derived)...")
    correct, total = test_generalization(prog, args.n_test, args.n_bits, args.seed)
    print(f"  generalization: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
