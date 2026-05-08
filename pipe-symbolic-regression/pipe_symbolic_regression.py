"""pipe-symbolic-regression — Salustowicz & Schmidhuber 1997 (Evol Comp 5(2)).

PIPE = Probabilistic Incremental Program Evolution. Symbolic regression
target is Koza's classic f(x) = x^4 + x^3 + x^2 + x evaluated on 20
fitness cases x ∈ linspace(-1, 1, 20).

Algorithm (single file):

* Maintain a Probabilistic Prototype Tree (PPT). Every PPT node carries:
    - a probability distribution P over the instruction set
      I = F ∪ T  (functions + terminals)
    - a private random constant R (re-estimated whenever 'R' is sampled
      and lands in the elite).
    - up to MAX_ARITY children, lazily allocated.

* Each generation:
    1. Sample N programs by descending the PPT from the root. At depth
       MAX_DEPTH−1 only terminals can be sampled (depth-cap).
    2. Evaluate fitness on all 20 fitness cases:
           SSE = sum_i (y_i − ŷ_i)^2 (with protected ops; NaN/Inf → huge)
           Fit = 1 / (1 + SSE)            (PIPE's standardised fitness).
    3. Track the best-of-generation and the elite (best ever).
    4. Population-Based Incremental Learning update at every PPT node
       visited by the chosen target (best-of-gen with prob 1−P_EL,
       elite with prob P_EL). For symbol s* used at that node:
           P(s*) ← P(s*) + lr · P_TARGET · (1 − P(s*))
       repeated until P(s*) ≥ P_TARGET, then re-normalised. The schedule
       P_TARGET = P_T + (1 − P_T) · lr · (eps + Fit_best)/(eps + Fit_elite)
       is the one Salustowicz & Schmidhuber 1997 §3 uses.
    5. Mutation: at every visited PPT node, with per-symbol probability
           P_Mp = P_M / (|I| · sqrt(n_visited))
       shift P(s) ← P(s) + mr · (1 − P(s)) and re-normalise.

* Stop when the elite reaches Fit ≥ FIT_TARGET (effectively SSE < 1e−6),
  or after MAX_GEN generations.

Pure numpy. Deterministic for fixed --seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Instruction set
# ---------------------------------------------------------------------------

# Protected arithmetic so a sampled program can never raise.

def _pdiv(a: float, b: float) -> float:
    if abs(b) < 1e-9:
        return 1.0
    return a / b


def _plog(a: float) -> float:
    return math.log(abs(a) + 1e-9)


def _pexp(a: float) -> float:
    return math.exp(max(-50.0, min(50.0, a)))


# All available functions. The active set is selected at runtime via
# --funcs in the CLI (default = the original PIPE paper's arithmetic
# instruction set for Koza's symbolic regression).
ALL_FUNCS: dict[str, tuple[int, Callable]] = {
    "+": (2, lambda a, b: a + b),
    "-": (2, lambda a, b: a - b),
    "*": (2, lambda a, b: a * b),
    "/": (2, _pdiv),
    "sin": (1, math.sin),
    "cos": (1, math.cos),
    "exp": (1, _pexp),
    "log": (1, _plog),
}

ARITH_KEYS = ("+", "-", "*", "/")
FULL_KEYS = ("+", "-", "*", "/", "sin", "cos", "exp", "log")
TERMS: list[str] = ["x", "R"]  # 'R' is a node-local random constant.

# These four module-level constants are populated by `set_function_set` so
# the rest of the module (PPT shapes, sampling, evaluation) can stay simple
# and not thread the instruction set through every function.
FUNCS: dict[str, tuple[int, Callable]] = {}
INSTRUCTIONS: list[str] = []
NF: int = 0
NI: int = 0
TERM_IDX: list[int] = []
MAX_ARITY: int = 2


def set_function_set(keys: tuple[str, ...]) -> None:
    """Choose which functions are in the instruction set. Must be called
    before `train` (the CLI calls it from --funcs)."""
    global FUNCS, INSTRUCTIONS, NF, NI, TERM_IDX, MAX_ARITY
    FUNCS = {k: ALL_FUNCS[k] for k in keys}
    INSTRUCTIONS = list(FUNCS.keys()) + TERMS
    NF = len(FUNCS)
    NI = len(INSTRUCTIONS)
    TERM_IDX = [INSTRUCTIONS.index(t) for t in TERMS]
    MAX_ARITY = max(a for a, _ in FUNCS.values())


# Default to the paper's arithmetic-only set so importers get a usable module.
set_function_set(ARITH_KEYS)


# ---------------------------------------------------------------------------
# PPT — Probabilistic Prototype Tree
# ---------------------------------------------------------------------------


@dataclass
class PPTNode:
    """One node of the prototype tree.

    `probs` is a length-NI distribution over INSTRUCTIONS. `constant` is
    the node-local random constant R. `children` is the list of up to
    MAX_ARITY child nodes, allocated lazily."""

    probs: np.ndarray
    constant: float
    children: list[Optional["PPTNode"]] = field(default_factory=lambda: [None] * MAX_ARITY)

    @staticmethod
    def fresh(rng: np.random.Generator, p_terminal_init: float = 0.6) -> "PPTNode":
        """Initial PPT node.

        The paper biases the initial distribution so terminals and
        functions split p_terminal_init / (1 − p_terminal_init), within
        each split uniform. This keeps early generations from drowning
        in 50-deep random functions."""
        probs = np.empty(NI, dtype=np.float64)
        probs[:NF] = (1.0 - p_terminal_init) / NF
        probs[NF:] = p_terminal_init / len(TERMS)
        # gaussian random constant in [-2, 2] approx
        const = float(rng.normal(0.0, 1.0))
        return PPTNode(probs=probs, constant=const)


class PPT:
    """Lazily-grown PPT, depth-bounded at MAX_DEPTH."""

    def __init__(self, max_depth: int, rng: np.random.Generator,
                 p_terminal_init: float = 0.6):
        self.max_depth = max_depth
        self.rng = rng
        self.p_terminal_init = p_terminal_init
        self.root = PPTNode.fresh(rng, p_terminal_init)

    def child(self, parent: PPTNode, idx: int) -> PPTNode:
        if parent.children[idx] is None:
            parent.children[idx] = PPTNode.fresh(self.rng, self.p_terminal_init)
        return parent.children[idx]


# ---------------------------------------------------------------------------
# Program (sampled tree)
# ---------------------------------------------------------------------------


@dataclass
class ProgNode:
    """A node of an actual sampled program.

    `symbol` is one of INSTRUCTIONS. `children` holds arity-many
    sub-programs. `ppt_node` is a back-pointer to the PPT node that
    spawned this sample (used by the elite-update step). `value` is set
    only when symbol == 'R' (the constant value used by *this* program)."""

    symbol: str
    children: list["ProgNode"]
    ppt_node: PPTNode
    value: Optional[float] = None

    def evaluate(self, x: float) -> float:
        s = self.symbol
        if s == "x":
            return x
        if s == "R":
            return self.value if self.value is not None else self.ppt_node.constant
        arity, fn = FUNCS[s]
        if arity == 1:
            return fn(self.children[0].evaluate(x))
        else:
            return fn(self.children[0].evaluate(x),
                      self.children[1].evaluate(x))

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def to_str(self) -> str:
        s = self.symbol
        if s == "x":
            return "x"
        if s == "R":
            v = self.value if self.value is not None else self.ppt_node.constant
            return f"{v:.4g}"
        arity, _ = FUNCS[s]
        if arity == 1:
            return f"{s}({self.children[0].to_str()})"
        else:
            return f"({self.children[0].to_str()} {s} {self.children[1].to_str()})"


def sample_program(ppt: PPT, rng: np.random.Generator,
                   ppt_node: Optional[PPTNode] = None,
                   depth: int = 0) -> ProgNode:
    """Top-down PPT sampling. At depth >= max_depth−1, force a terminal."""
    if ppt_node is None:
        ppt_node = ppt.root

    p = ppt_node.probs
    if depth >= ppt.max_depth - 1:
        # Force a terminal: zero out function probs, renormalise.
        p_use = p.copy()
        p_use[:NF] = 0.0
        s = p_use.sum()
        if s <= 0:
            p_use = np.zeros(NI)
            p_use[TERM_IDX] = 1.0 / len(TERMS)
        else:
            p_use /= s
    else:
        p_use = p

    idx = int(rng.choice(NI, p=p_use))
    symbol = INSTRUCTIONS[idx]

    if symbol in FUNCS:
        arity, _ = FUNCS[symbol]
        children = []
        for i in range(arity):
            child_ppt = ppt.child(ppt_node, i)
            children.append(sample_program(ppt, rng, child_ppt, depth + 1))
        return ProgNode(symbol=symbol, children=children, ppt_node=ppt_node)
    else:
        # Terminal. If 'R', snapshot the PPT-node's constant into the program.
        value = ppt_node.constant if symbol == "R" else None
        return ProgNode(symbol=symbol, children=[], ppt_node=ppt_node, value=value)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def koza_target(X: np.ndarray) -> np.ndarray:
    return X**4 + X**3 + X**2 + X


def evaluate_program(prog: ProgNode, X: np.ndarray) -> np.ndarray:
    """Evaluate a program on an array of x values; returns array of same shape.

    Each fitness case runs through the protected ops; any overflow is
    silenced into a finite penalty so SSE stays comparable."""
    out = np.empty_like(X, dtype=np.float64)
    for i, x in enumerate(X):
        try:
            v = prog.evaluate(float(x))
        except (OverflowError, ValueError, ZeroDivisionError):
            v = 1e6
        if not np.isfinite(v):
            v = 1e6
        out[i] = v
    return out


def fitness(prog: ProgNode, X: np.ndarray, Y: np.ndarray,
            hit_eps: float = 0.01) -> tuple[float, float, int]:
    """Returns (fit, sse, hits).

    - SSE = sum of squared residuals over the 20 fitness cases.
    - fit = 1/(1+SSE), bounded in (0, 1].
    - hits = Koza's "hit" count: number of fitness cases where
      |y_i - yhat_i| < hit_eps. 20/20 hits = problem solved per Koza."""
    yhat = evaluate_program(prog, X)
    diff = Y - yhat
    diff = np.clip(diff, -1e6, 1e6)
    sse = float(np.sum(diff * diff))
    fit = 1.0 / (1.0 + sse)
    hits = int(np.sum(np.abs(diff) < hit_eps))
    return fit, sse, hits


# ---------------------------------------------------------------------------
# PPT update (PBIL-style elite learning + mutation)
# ---------------------------------------------------------------------------


def collect_visited(prog: ProgNode) -> list[tuple[PPTNode, str]]:
    """Pairs (ppt_node, symbol) at every node of the program."""
    out: list[tuple[PPTNode, str]] = [(prog.ppt_node, prog.symbol)]
    for c in prog.children:
        out.extend(collect_visited(c))
    return out


def collect_visited_with_values(prog: ProgNode) -> list[tuple[PPTNode, str, Optional[float]]]:
    out = [(prog.ppt_node, prog.symbol, prog.value)]
    for c in prog.children:
        out.extend(collect_visited_with_values(c))
    return out


def renormalise(p: np.ndarray) -> None:
    s = p.sum()
    if s <= 0:
        p[:] = 1.0 / NI
    else:
        p /= s


def update_toward_elite(elite_prog: ProgNode, fit_best: float, fit_elite: float,
                        lr: float, P_T: float, eps: float = 1e-6) -> None:
    """PBIL update of every visited PPT node toward the elite's symbol.

    The schedule
        P_TARGET = P_T + (1 − P_T) · lr · (eps + Fit_best)/(eps + Fit_elite)
    raises the target probability when this generation produced something
    close to (or better than) the elite."""
    P_TARGET = P_T + (1.0 - P_T) * lr * (eps + fit_best) / (eps + fit_elite)
    P_TARGET = min(P_TARGET, 0.999)

    visited = collect_visited_with_values(elite_prog)
    for node, symbol, value in visited:
        idx = INSTRUCTIONS.index(symbol)
        # Repeat additive update until P(s*) ≥ P_TARGET (cap iterations).
        for _ in range(50):
            if node.probs[idx] >= P_TARGET:
                break
            node.probs[idx] += lr * P_TARGET * (1.0 - node.probs[idx])
        renormalise(node.probs)
        # If we re-elected an R, lock in the elite's actual constant.
        if symbol == "R" and value is not None:
            node.constant = float(value)


def mutate_visited(elite_prog: ProgNode, rng: np.random.Generator,
                   P_M: float, mr: float) -> None:
    """At every node visited by the elite, for each instruction-symbol,
    with probability P_M / (NI · sqrt(n_visited)) shift its probability
    toward 1 by mr·(1−p). This is the per-component mutation from the
    1997 paper."""
    visited = collect_visited(elite_prog)
    n_visited = max(1, len(visited))
    p_per_sym = P_M / (NI * math.sqrt(n_visited))
    for node, _ in visited:
        any_changed = False
        for i in range(NI):
            if rng.random() < p_per_sym:
                node.probs[i] += mr * (1.0 - node.probs[i])
                any_changed = True
        if any_changed:
            renormalise(node.probs)
        # Mutate the constant too with small Gaussian, so 'R' nodes keep
        # exploring magnitudes (otherwise the only way to change R is to
        # re-elect a different sample whose value got captured).
        if rng.random() < p_per_sym * NI:  # per-node, same scale as one symbol
            node.constant = node.constant + float(rng.normal(0.0, 0.1))


# ---------------------------------------------------------------------------
# Top-level training loop
# ---------------------------------------------------------------------------


@dataclass
class Hyper:
    pop_size: int = 100
    max_gen: int = 200
    max_depth: int = 6
    lr: float = 0.2          # PBIL learning rate
    P_T: float = 0.8         # base target probability
    P_EL: float = 0.2        # prob of updating toward elite (vs gen-best)
    P_M: float = 0.4         # per-program mutation budget
    mr: float = 0.4          # per-symbol mutation magnitude
    p_terminal_init: float = 0.6
    fit_target: float = 1.0 - 1e-6  # SSE < 1e-6


def train(hyp: Hyper, seed: int, verbose: bool = False
          ) -> dict:
    rng = np.random.default_rng(seed)
    X = np.linspace(-1.0, 1.0, 20)
    Y = koza_target(X)

    ppt = PPT(max_depth=hyp.max_depth, rng=rng, p_terminal_init=hyp.p_terminal_init)

    elite_prog: Optional[ProgNode] = None
    elite_fit: float = 0.0
    elite_sse: float = float("inf")
    elite_hits: int = 0

    history: list[dict] = []
    best_yhat_per_gen: list[np.ndarray] = []
    elite_str_per_gen: list[str] = []

    t0 = time.time()
    solved_at: Optional[int] = None
    hits_solved_at: Optional[int] = None

    for gen in range(hyp.max_gen):
        # 1) sample population
        pop = [sample_program(ppt, rng) for _ in range(hyp.pop_size)]

        # 2) evaluate fitness
        fits = [fitness(p, X, Y) for p in pop]
        fit_arr = np.array([f for f, _, _ in fits])
        sse_arr = np.array([s for _, s, _ in fits])
        hits_arr = np.array([h for _, _, h in fits])
        best_idx = int(np.argmax(fit_arr))
        best_prog = pop[best_idx]
        fit_best = float(fit_arr[best_idx])
        sse_best = float(sse_arr[best_idx])
        hits_best = int(hits_arr[best_idx])

        # 3) update elite
        if best_prog is not None and fit_best > elite_fit:
            elite_prog = best_prog
            elite_fit = fit_best
            elite_sse = sse_best
            elite_hits = hits_best
            if elite_hits >= len(X) and hits_solved_at is None:
                hits_solved_at = gen

        # 4) choose target for the PPT update
        target_is_elite = rng.random() < hyp.P_EL
        if target_is_elite and elite_prog is not None:
            target_prog = elite_prog
            target_fit = elite_fit
        else:
            target_prog = best_prog
            target_fit = fit_best

        # 5) PPT update + mutation
        update_toward_elite(target_prog,
                            fit_best=fit_best,
                            fit_elite=max(elite_fit, fit_best),
                            lr=hyp.lr, P_T=hyp.P_T)
        mutate_visited(target_prog, rng, P_M=hyp.P_M, mr=hyp.mr)

        # logging
        yhat_elite = evaluate_program(elite_prog, X) if elite_prog is not None else np.zeros_like(X)
        best_yhat_per_gen.append(yhat_elite.copy())
        elite_str_per_gen.append(elite_prog.to_str() if elite_prog is not None else "?")
        history.append(dict(
            gen=gen,
            fit_best=fit_best,
            sse_best=sse_best,
            hits_best=hits_best,
            fit_elite=elite_fit,
            sse_elite=elite_sse,
            hits_elite=elite_hits,
            mean_size=float(np.mean([p.size() for p in pop])),
            mean_depth=float(np.mean([p.depth() for p in pop])),
            elite_size=elite_prog.size() if elite_prog is not None else 0,
            elite_depth=elite_prog.depth() if elite_prog is not None else 0,
        ))

        if verbose and (gen % 10 == 0 or gen == hyp.max_gen - 1):
            print(f"gen={gen:4d}  fit_best={fit_best:.6f}  sse_best={sse_best:.6e}  "
                  f"fit_elite={elite_fit:.6f}  sse_elite={elite_sse:.6e}  "
                  f"hits={elite_hits}/{len(X)}  "
                  f"size={history[-1]['elite_size']}  depth={history[-1]['elite_depth']}")

        if elite_fit >= hyp.fit_target:
            solved_at = gen
            if verbose:
                print(f"  → solved at generation {gen}")
            break

    wallclock = time.time() - t0

    return dict(
        history=history,
        elite_prog_str=elite_prog.to_str() if elite_prog is not None else "?",
        elite_yhat=evaluate_program(elite_prog, X).tolist() if elite_prog is not None else [0.0] * len(X),
        elite_fit=elite_fit,
        elite_sse=elite_sse,
        elite_hits=elite_hits,
        elite_size=elite_prog.size() if elite_prog is not None else 0,
        elite_depth=elite_prog.depth() if elite_prog is not None else 0,
        solved_at=solved_at,
        hits_solved_at=hits_solved_at,
        wallclock=wallclock,
        X=X.tolist(),
        Y=Y.tolist(),
        best_yhat_per_gen=[y.tolist() for y in best_yhat_per_gen],
        elite_str_per_gen=elite_str_per_gen,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def env_record() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        commit=commit,
    )


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--max-gen", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--P-T", type=float, default=0.8)
    ap.add_argument("--P-EL", type=float, default=0.2)
    ap.add_argument("--P-M", type=float, default=0.4)
    ap.add_argument("--mr", type=float, default=0.4)
    ap.add_argument("--p-terminal-init", type=float, default=0.6)
    ap.add_argument("--fit-target", type=float, default=1.0 - 1e-6)
    ap.add_argument("--funcs", choices=("arith", "full"), default="arith",
                    help="arith = {+,-,*,/} (paper Table 1 for Koza); "
                         "full = {+,-,*,/,sin,cos,exp,log}")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    keys = ARITH_KEYS if args.funcs == "arith" else FULL_KEYS
    set_function_set(keys)

    hyp = Hyper(
        pop_size=args.pop_size, max_gen=args.max_gen, max_depth=args.max_depth,
        lr=args.lr, P_T=args.P_T, P_EL=args.P_EL, P_M=args.P_M, mr=args.mr,
        p_terminal_init=args.p_terminal_init, fit_target=args.fit_target,
    )

    np.random.seed(args.seed)  # extra belt; sampling uses default_rng(seed) inside
    out = train(hyp, seed=args.seed, verbose=not args.quiet)

    summary = dict(
        env=env_record(), args=vars(args),
        function_set=list(FUNCS.keys()),
        elite_prog_str=out["elite_prog_str"],
        elite_fit=out["elite_fit"],
        elite_sse=out["elite_sse"],
        elite_hits=out["elite_hits"],
        elite_size=out["elite_size"],
        elite_depth=out["elite_depth"],
        solved_at=out["solved_at"],
        hits_solved_at=out["hits_solved_at"],
        wallclock=out["wallclock"],
        n_generations_run=len(out["history"]),
    )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, args.out), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Function set    : {list(FUNCS.keys())}")
    print(f"Elite program   : {out['elite_prog_str']}")
    print(f"Elite SSE       : {out['elite_sse']:.6e}")
    print(f"Elite fit       : {out['elite_fit']:.6f}")
    print(f"Elite hits      : {out['elite_hits']}/20  (Koza: 20/20 = solved)")
    print(f"Elite size      : {out['elite_size']} nodes, depth {out['elite_depth']}")
    print(f"Hits-solved gen : {out['hits_solved_at']}")
    print(f"Fit-solved gen  : {out['solved_at']}")
    print(f"Wallclock       : {out['wallclock']:.2f} s "
          f"({len(out['history'])} generations)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    _cli()
