"""
oops-towers-of-hanoi - Schmidhuber's Optimal Ordered Problem Solver applied
to the Towers of Hanoi puzzle for varying disk count n.

Reference:
  J. Schmidhuber, "Optimal Ordered Problem Solver", Machine Learning 54
  (3):211-254, 2004 (TR IDSIA-12-02; arXiv:cs/0207097).

What OOPS does:
  Levin-style universal search ordered by program length, **augmented with
  reusable subroutines**. Whenever OOPS finds a program for task k, it
  freezes that program as a callable primitive in the DSL, then searches
  for task k+1. The very next task can call the freshly frozen subroutine,
  which dramatically shrinks the description length of recursive families
  like the Hanoi solver.

  For a sequence of related tasks Hanoi(n=1, 2, 3, ...), OOPS first finds
  a 1-token solver for n=1 (`M`), then a 7-token solver for n=2 that
  invokes the n=1 program via `C` (CALL last subroutine). For n>=3, the
  *same* 7-token program already solves the task: `C` now calls the n=2
  program, which itself calls n=1, and so on - the recursion bottoms out
  at the right depth. OOPS therefore reuses the n=2 program directly
  for every higher n, never re-searching.

  Headline: the program found is constant in length while the puzzle's
  optimal move count grows as 2**n - 1.

DSL (4 tokens, log2 4 = 2 bits / token):
  M   - move top disk from peg `src` to peg `dst` (no-op if illegal)
  SD  - swap the `dst` and `aux` peg in the current frame
  SA  - swap the `src` and `aux` peg in the current frame
  C   - call the most-recently-frozen subroutine; no-op if none.
        The caller's frame is **saved before the call and restored after**,
        so callees mutate frame freely without polluting the caller.

A "frame" (src, dst, aux) is a permutation of (0, 1, 2) carried during
execution. Initial frame is (0, 2, 1) - move all disks from peg 0 to peg 2,
using peg 1 as auxiliary. Frame save/restore on CALL is the one piece of
"interpreter sugar" we need so that a single recursive program generalizes
across all n - this matches the canonical recursive Hanoi solver, where
`hanoi(n-1, src, aux, dst)` and `hanoi(n-1, aux, dst, src)` are evaluated
with their own argument bindings rather than mutating the parent's.

Levin search vs OOPS:
  Plain Levin search re-enumerates from scratch for every new task. With
  our 4-token alphabet, n=2 needs a 7-token program (4**7 = 16384
  candidates) but n=3 needs ~21 tokens directly (4**21 ~= 4e12,
  infeasible). OOPS's reuse mechanism makes n=3 trivial because the
  search already found the recursive step at n=2. This file demonstrates
  both regimes.
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# DSL
# ----------------------------------------------------------------------

TOKENS = ("M", "SD", "SA", "C")
T_M, T_SD, T_SA, T_C = 0, 1, 2, 3
ALPHABET = len(TOKENS)


def tokens_to_str(tokens: tuple[int, ...]) -> str:
    return " ".join(TOKENS[t] for t in tokens)


def idx_to_tokens(idx: int, length: int) -> tuple[int, ...]:
    """Decode an integer in [0, ALPHABET**length) as a length-`length` token sequence."""
    out = [0] * length
    for i in range(length - 1, -1, -1):
        out[i] = idx % ALPHABET
        idx //= ALPHABET
    return tuple(out)


# ----------------------------------------------------------------------
# Hanoi simulator
# ----------------------------------------------------------------------

class Hanoi:
    """Three-peg Towers of Hanoi state with the OOPS interpreter's `frame`."""

    __slots__ = ("n", "pegs", "frame", "moves")

    def __init__(self, n: int):
        self.n = n
        # peg as a list, bottom-to-top; disk size is the int (larger = larger)
        self.pegs = [list(range(n, 0, -1)), [], []]
        self.frame = (0, 2, 1)  # (src, dst, aux)
        self.moves: list[tuple[int, int]] = []

    def is_solved(self) -> bool:
        return (len(self.pegs[2]) == self.n
                and self.pegs[2] == list(range(self.n, 0, -1)))

    def move(self) -> bool:
        """Execute one disk move from frame[0] to frame[1]. Return True if legal."""
        src, dst, _ = self.frame
        if not self.pegs[src]:
            return False
        top = self.pegs[src][-1]
        if self.pegs[dst] and self.pegs[dst][-1] < top:
            return False
        self.pegs[dst].append(self.pegs[src].pop())
        self.moves.append((src, dst))
        return True

    def swap_dst_aux(self) -> None:
        s, d, a = self.frame
        self.frame = (s, a, d)

    def swap_src_aux(self) -> None:
        s, d, a = self.frame
        self.frame = (a, d, s)

    def snapshot(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(p) for p in self.pegs)


# ----------------------------------------------------------------------
# Frozen subroutines
# ----------------------------------------------------------------------

@dataclass
class Subroutine:
    """A frozen program plus the call-target index it captured at freeze time.

    When this subroutine executes a `C` token, it calls subroutines[`call_target`]
    in the current frozen library - i.e. the previous one. This is what makes
    the recursive Hanoi program work: s_k calls s_{k-1} which calls s_{k-2} ...
    """
    tokens: tuple[int, ...]
    call_target: int  # index into the frozen list at the moment of freezing
    n: int            # the task it was frozen for


# ----------------------------------------------------------------------
# Interpreter
# ----------------------------------------------------------------------

@dataclass
class ExecResult:
    solved: bool
    steps: int            # number of tokens consumed (across all call frames)
    illegal_move: bool    # True if any `M` failed
    timeout: bool         # True if step budget exceeded
    final: Hanoi


def _exec(tokens: tuple[int, ...],
          state: Hanoi,
          frozen: list[Subroutine],
          call_target: int,
          steps_left: list[int]) -> tuple[bool, bool]:
    """Execute `tokens` on `state`. Returns (illegal_move_seen, timeout)."""
    illegal = False
    for tok in tokens:
        if steps_left[0] <= 0:
            return illegal, True
        steps_left[0] -= 1
        if tok == T_M:
            if not state.move():
                illegal = True
                # Continue executing - illegal moves are no-ops in this DSL.
        elif tok == T_SD:
            state.swap_dst_aux()
        elif tok == T_SA:
            state.swap_src_aux()
        elif tok == T_C:
            if call_target < 0:
                # No subroutine available; treat as no-op so search-space
                # programs that include `C` early can still progress.
                continue
            sub = frozen[call_target]
            saved_frame = state.frame  # save caller's frame
            sub_illegal, timed_out = _exec(sub.tokens, state, frozen,
                                           sub.call_target, steps_left)
            state.frame = saved_frame  # restore on return
            if sub_illegal:
                illegal = True
            if timed_out:
                return illegal, True
    return illegal, False


def execute(tokens: tuple[int, ...],
            frozen: list[Subroutine],
            n: int,
            step_budget: int) -> ExecResult:
    """Execute `tokens` as the top-level program for Hanoi(n)."""
    state = Hanoi(n)
    steps_left = [step_budget]
    call_target = len(frozen) - 1  # top-level CALL targets most recent frozen
    illegal, timed_out = _exec(tokens, state, frozen, call_target, steps_left)
    consumed = step_budget - steps_left[0]
    return ExecResult(solved=state.is_solved(),
                      steps=consumed,
                      illegal_move=illegal,
                      timeout=timed_out,
                      final=state)


# ----------------------------------------------------------------------
# Levin search (length-first enumeration with subroutine reuse)
# ----------------------------------------------------------------------

@dataclass
class SearchResult:
    tokens: Optional[tuple[int, ...]]
    mode: str                     # "reused" | "found" | "failed"
    nodes_expanded: int
    elapsed_s: float
    program_length: int
    moves_made: int               # number of disk moves the program performs


def _step_budget_for(n: int) -> int:
    """Bound on token-steps we let any candidate program run.

    A correct recursive Hanoi program performs ~2**n moves and a constant
    number of swap+call tokens per move. Give a generous multiplier so
    longer-than-optimal programs still get a fair shot.
    """
    return max(64, (2 ** n) * 8)


def levin_search(n: int,
                 frozen: list[Subroutine],
                 max_program_length: int = 12,
                 max_nodes: int = 200_000,
                 verbose: bool = False) -> SearchResult:
    """OOPS-style search for a program that solves Hanoi(n).

    Step 1: try the most recently frozen program first. If it solves the
    new task, we're done with zero search - this is OOPS's reuse mechanism
    that gives the dramatic speedup on related-task sequences.

    Step 2: enumerate programs by ascending length. With a 4-token alphabet
    this is equivalent to Levin search under a uniform code (each token
    contributes log2(4) = 2 bits, so length-L programs share the same prior
    weight 4**(-L)).
    """
    t0 = time.time()
    budget = _step_budget_for(n)

    # --- OOPS reuse step: try the previous program directly -------------
    if frozen:
        prev = frozen[-1]
        result = execute(prev.tokens, frozen, n, step_budget=budget)
        if result.solved and not result.illegal_move:
            return SearchResult(tokens=prev.tokens, mode="reused",
                                nodes_expanded=0,
                                elapsed_s=time.time() - t0,
                                program_length=len(prev.tokens),
                                moves_made=len(result.final.moves))

    # --- Length-first Levin enumeration ---------------------------------
    nodes = 0
    for length in range(1, max_program_length + 1):
        n_candidates = ALPHABET ** length
        for idx in range(n_candidates):
            if nodes >= max_nodes:
                return SearchResult(tokens=None, mode="failed",
                                    nodes_expanded=nodes,
                                    elapsed_s=time.time() - t0,
                                    program_length=0, moves_made=0)
            tokens = idx_to_tokens(idx, length)
            nodes += 1
            result = execute(tokens, frozen, n, step_budget=budget)
            if result.solved and not result.illegal_move:
                if verbose:
                    print(f"  n={n}: found {tokens_to_str(tokens)} "
                          f"({len(tokens)} tokens, {nodes} nodes)")
                return SearchResult(tokens=tokens, mode="found",
                                    nodes_expanded=nodes,
                                    elapsed_s=time.time() - t0,
                                    program_length=length,
                                    moves_made=len(result.final.moves))
        if verbose:
            print(f"  n={n}: length {length} exhausted, {nodes} nodes, "
                  f"{time.time()-t0:.3f}s elapsed")

    return SearchResult(tokens=None, mode="failed",
                        nodes_expanded=nodes,
                        elapsed_s=time.time() - t0,
                        program_length=0, moves_made=0)


# ----------------------------------------------------------------------
# OOPS top-level loop
# ----------------------------------------------------------------------

@dataclass
class OOPSRun:
    frozen: list[Subroutine] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def oops_solve(max_n: int,
               max_program_length: int = 12,
               max_nodes: int = 200_000,
               verbose: bool = True) -> OOPSRun:
    """Solve Hanoi(n) for n = 1, 2, ..., max_n in order, reusing each
    discovered program as a frozen subroutine for subsequent tasks."""
    run = OOPSRun()
    for n in range(1, max_n + 1):
        result = levin_search(n, run.frozen,
                              max_program_length=max_program_length,
                              max_nodes=max_nodes,
                              verbose=verbose)
        if result.tokens is None:
            if verbose:
                print(f"n={n}: FAILED ({result.nodes_expanded} nodes, "
                      f"{result.elapsed_s:.3f}s)")
            run.history.append({"n": n, "mode": "failed",
                                "nodes": result.nodes_expanded,
                                "elapsed_s": result.elapsed_s,
                                "program": None,
                                "program_length": 0,
                                "moves_made": 0,
                                "optimal_moves": 2 ** n - 1})
            break
        sub = Subroutine(tokens=result.tokens,
                         call_target=len(run.frozen) - 1,
                         n=n)
        run.frozen.append(sub)
        if verbose:
            mark = "REUSED" if result.mode == "reused" else "found "
            print(f"n={n:2d}  {mark}  L={result.program_length:2d}  "
                  f"moves={result.moves_made:4d}/{2**n - 1:<4d}  "
                  f"nodes={result.nodes_expanded:6d}  "
                  f"time={result.elapsed_s*1000:7.2f}ms  "
                  f"prog=[{tokens_to_str(result.tokens)}]")
        run.history.append({"n": n, "mode": result.mode,
                            "nodes": result.nodes_expanded,
                            "elapsed_s": result.elapsed_s,
                            "program": tokens_to_str(result.tokens),
                            "program_length": result.program_length,
                            "moves_made": result.moves_made,
                            "optimal_moves": 2 ** n - 1})
    return run


# ----------------------------------------------------------------------
# Verification helpers
# ----------------------------------------------------------------------

def verify_program(tokens: tuple[int, ...],
                   frozen: list[Subroutine],
                   n: int) -> tuple[bool, int, int]:
    """Re-run the program and return (solved, n_moves, optimal_moves)."""
    result = execute(tokens, frozen, n, step_budget=_step_budget_for(n))
    return (result.solved and not result.illegal_move,
            len(result.final.moves),
            2 ** n - 1)


def trace_moves(tokens: tuple[int, ...],
                frozen: list[Subroutine],
                n: int) -> list[tuple[tuple[tuple[int, ...], ...],
                                      tuple[int, int]]]:
    """Run program, return [(state-after, (src, dst)), ...] per move."""
    state = Hanoi(n)
    steps_left = [_step_budget_for(n)]
    snapshots: list[tuple[tuple[tuple[int, ...], ...],
                          tuple[int, int]]] = []
    snapshots.append((state.snapshot(), (-1, -1)))  # initial state

    def run(toks: tuple[int, ...], call_target: int) -> bool:
        for tok in toks:
            if steps_left[0] <= 0:
                return False
            steps_left[0] -= 1
            if tok == T_M:
                before_moves = len(state.moves)
                state.move()
                if len(state.moves) > before_moves:
                    snapshots.append((state.snapshot(), state.moves[-1]))
            elif tok == T_SD:
                state.swap_dst_aux()
            elif tok == T_SA:
                state.swap_src_aux()
            elif tok == T_C:
                if call_target < 0:
                    continue
                sub = frozen[call_target]
                saved_frame = state.frame
                ok_inner = run(sub.tokens, sub.call_target)
                state.frame = saved_frame
                if not ok_inner:
                    return False
        return True

    run(tokens, len(frozen) - 1)
    return snapshots


# ----------------------------------------------------------------------
# Environment metadata (for §Reproducibility in the README)
# ----------------------------------------------------------------------

def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed (kept for the reproducibility contract; "
                        "search is deterministic regardless).")
    p.add_argument("--max-n", type=int, default=8,
                   help="Largest n to attempt. Defaults to 8 (laptop-runnable).")
    p.add_argument("--max-program-length", type=int, default=10,
                   help="Cap on Levin-enumerated program length. "
                        "Set higher only if you also raise --max-nodes.")
    p.add_argument("--max-nodes", type=int, default=200_000,
                   help="Per-task cap on number of programs enumerated.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # `seed` is part of the reproducibility contract; we don't actually use
    # randomness (search is deterministic), but record it.
    info = env_info()
    if not args.quiet:
        print(f"# OOPS / Towers of Hanoi  (seed={args.seed})")
        print(f"# python {info['python']}  {info['platform']}")
        print(f"# alphabet = {TOKENS}  ({ALPHABET} tokens, "
              f"{2:.0f} bits/token)\n")

    t_total = time.time()
    run = oops_solve(max_n=args.max_n,
                     max_program_length=args.max_program_length,
                     max_nodes=args.max_nodes,
                     verbose=not args.quiet)
    t_total = time.time() - t_total

    if not args.quiet:
        print()
        print("=" * 68)
        print(f"OOPS finished {len(run.frozen)}/{args.max_n} tasks in "
              f"{t_total*1000:.1f}ms total")
        print("=" * 68)
        print(f"\nSubroutine library (frozen, in order):")
        for sub in run.frozen:
            print(f"  s_{sub.n} = [{tokens_to_str(sub.tokens)}]  "
                  f"(L={len(sub.tokens)}, calls s_{sub.call_target + 1 if sub.call_target >= 0 else 0})")

    # Independently verify each frozen subroutine on its task, using the
    # frozen library available at the time it was created.
    if not args.quiet:
        print(f"\nVerification:")
    all_ok = True
    for i, sub in enumerate(run.frozen):
        # Re-run with the prefix of frozen subroutines that existed when
        # this one was created (i.e. frozen[:i]).
        ok, moves, opt = verify_program(sub.tokens, run.frozen[:i], sub.n)
        marker = "OK " if ok else "FAIL"
        if not args.quiet:
            print(f"  [{marker}] n={sub.n}: solved={ok}, "
                  f"moves={moves}, optimal={opt}, "
                  f"excess={moves - opt}")
        all_ok = all_ok and ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
