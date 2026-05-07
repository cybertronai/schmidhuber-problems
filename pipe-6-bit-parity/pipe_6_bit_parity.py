"""PIPE on n-bit even parity (default n=6).

Reference: Rafal Salustowicz and Juergen Schmidhuber,
*Probabilistic Incremental Program Evolution*,
Evolutionary Computation 5(2):123-141, 1997.

PIPE evolves a population of program trees by maintaining a Probabilistic
Prototype Tree (PPT). Each PPT node holds a probability distribution over the
instruction set (here {AND, OR, NOT, IF, x0..x5}). Each generation:

    1. Sample a population of program trees from the PPT.
    2. Evaluate each program on the 64 fitness cases (all 6-bit inputs).
    3. Update the PPT toward the best-so-far program (PBIL-style update along
       the path that produced it, then renormalize).
    4. Mutate the PPT by adding small uniform noise to each component.

No gradient descent. Pure numpy + Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Instruction set (parameterised by n_bits)
# ---------------------------------------------------------------------------
# 4 boolean functions (AND, OR, NOT, IF) + n terminals (input bits)
INSTR_FUNCS: List[Tuple[str, int]] = [("AND", 2), ("OR", 2), ("NOT", 1), ("IF", 3)]
N_FUNCS = len(INSTR_FUNCS)


def _build_instr_table(n_bits: int) -> Tuple[List[Tuple[str, int]], List[str], np.ndarray]:
    terms = [(f"x{i}", 0) for i in range(n_bits)]
    table = INSTR_FUNCS + terms
    names = [n for n, _ in table]
    arities = np.array([a for _, a in table], dtype=np.int32)
    return table, names, arities


# Default: 6-bit parity (the headline experiment).
INSTR_TERMS: List[Tuple[str, int]] = [(f"x{i}", 0) for i in range(6)]
INSTR_TABLE: List[Tuple[str, int]] = INSTR_FUNCS + INSTR_TERMS
INSTR_NAMES: List[str] = [n for n, _ in INSTR_TABLE]
INSTR_ARITIES: np.ndarray = np.array([a for _, a in INSTR_TABLE], dtype=np.int32)
N_INSTR = len(INSTR_TABLE)
TERMINAL_INDICES = list(range(N_FUNCS, N_INSTR))


def configure_n_bits(n_bits: int) -> None:
    """Reconfigure the module-level instruction table for n-bit parity.

    The PPT, sampling, evaluation, and update routines all read these
    module-level globals, so calling this function before ``train()`` switches
    the whole pipeline to a different parity width. The default is 6.
    """
    global INSTR_TERMS, INSTR_TABLE, INSTR_NAMES, INSTR_ARITIES
    global N_INSTR, TERMINAL_INDICES, X_BITMASK, Y_BITMASK_PARITY, MASK_NBITS
    INSTR_TABLE, INSTR_NAMES, INSTR_ARITIES = _build_instr_table(n_bits)
    INSTR_TERMS = [(f"x{i}", 0) for i in range(n_bits)]
    N_INSTR = len(INSTR_TABLE)
    TERMINAL_INDICES = list(range(N_FUNCS, N_INSTR))
    n_cases = 1 << n_bits
    MASK_NBITS = (1 << n_cases) - 1
    X_BITMASK = np.array(
        [sum(((j >> i) & 1) << j for j in range(n_cases)) for i in range(n_bits)],
        dtype=object,
    )
    Y_BITMASK_PARITY = sum(
        int(bin(j).count("1") % 2 == 0) << j for j in range(n_cases)
    )


# A program tree is represented as a nested tuple: (instr_idx, [child_tree, ...]).
Tree = Tuple[int, List["Tree"]]


# ---------------------------------------------------------------------------
# PPT
# ---------------------------------------------------------------------------
class PPTNode:
    """A node of the Probabilistic Prototype Tree.

    Each PPT node holds a probability vector of length N_INSTR, plus a dict of
    PPT children indexed by argument position (0, 1, 2). Children are created
    lazily the first time the position is visited during sampling.
    """

    __slots__ = ("probs", "children")

    def __init__(self, n_instr: Optional[int] = None) -> None:
        n = N_INSTR if n_instr is None else n_instr
        self.probs: np.ndarray = np.ones(n, dtype=np.float64) / n
        self.children: Dict[int, "PPTNode"] = {}


def _depth_prior(depth: int, max_depth: int) -> np.ndarray:
    """Depth-dependent prior multiplier (Salustowicz & Schmidhuber 1997).

    Linearly shifts mass from functions toward terminals as depth grows. At
    depth 0 the prior is uniform; at depth ``max_depth`` functions are fully
    suppressed. This is what keeps PIPE's sampled programs from growing
    without bound and is the paper's substitute for an explicit size penalty.
    """
    frac = min(1.0, depth / max(max_depth, 1))
    prior = np.ones(N_INSTR, dtype=np.float64)
    # functions multiplier shrinks linearly to 0 at max_depth
    prior[:N_FUNCS] = 1.0 - frac
    # terminals multiplier grows linearly to (1 + 1) = 2 at max_depth
    prior[N_FUNCS:] = 1.0 + frac
    return prior


def sample_tree(
    ppt: PPTNode,
    rng: np.random.Generator,
    depth: int = 0,
    max_depth: int = 8,
) -> Tuple[Tree, List[Tuple[PPTNode, int]]]:
    """Sample a program tree from the PPT.

    Returns the tree and the list of (PPTNode, chosen_instr_idx) pairs visited
    along the way (used for the PBIL update). Sampling combines the PPT's
    learned distribution with a depth-dependent prior so deep nodes are biased
    toward terminals.
    """
    prior = _depth_prior(depth, max_depth)
    probs = ppt.probs * prior
    if depth >= max_depth:
        # at the cap, force a terminal regardless of PPT bias
        probs = ppt.probs.copy()
        probs[:N_FUNCS] = 0.0
    s = probs.sum()
    if s <= 0:
        probs = np.zeros(N_INSTR, dtype=np.float64)
        probs[N_FUNCS:] = 1.0 / len(TERMINAL_INDICES)
    else:
        probs = probs / s

    idx = int(rng.choice(N_INSTR, p=probs))
    arity = int(INSTR_ARITIES[idx])
    children: List[Tree] = []
    paths: List[Tuple[PPTNode, int]] = [(ppt, idx)]
    for i in range(arity):
        if i not in ppt.children:
            ppt.children[i] = PPTNode()
        sub_tree, sub_paths = sample_tree(ppt.children[i], rng, depth + 1, max_depth)
        children.append(sub_tree)
        paths.extend(sub_paths)
    return (idx, children), paths


def _path_p(paths: List[Tuple[PPTNode, int]]) -> float:
    """P(program | PPT) = product of P(I_d | d) along the program's path."""
    p = 1.0
    for node, idx in paths:
        p *= float(node.probs[idx])
        if p == 0.0:
            return 0.0
    return p


def _clamp_renorm(probs: np.ndarray, eps: float) -> None:
    """Clamp probabilities to [eps, 1-eps] then renormalise so they sum to 1.

    Keeping a non-zero floor on every entry preserves exploration: even a
    saturated PPT node still has a small chance of sampling alternative
    instructions, so the mutation step can grow them again.
    """
    np.clip(probs, eps, 1.0 - eps, out=probs)
    probs /= probs.sum()


def update_ppt(
    paths: List[Tuple[PPTNode, int]],
    lr: float = 0.2,
    eps: float = 0.01,
) -> None:
    """PIPE update toward the elite program (Salustowicz & Schmidhuber 1997).

    For each (node, idx) on the elite tree's path, push the probability of
    the chosen instruction toward 1 by ``lr * (1 - p)`` and renormalise. After
    the update we clamp every entry to ``[eps, 1-eps]`` so the distribution
    never saturates fully; clamping preserves a small probability of sampling
    alternative instructions, which is what lets the mutation step introduce
    structural variation.

    This is a single PBIL step (eq. 4 of the paper); the paper's iterative
    "drive P(BS) to a target" form (eq. 5) is approximately equivalent and
    given ``lr`` already large here, one step suffices per generation.
    """
    seen: set = set()
    for node, idx in paths:
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        p_old = float(node.probs[idx])
        node.probs[idx] = p_old + lr * (1.0 - p_old)
        denom = 1.0 - p_old
        scale = (1.0 - node.probs[idx]) / denom if denom > 1e-12 else 0.0
        for j in range(N_INSTR):
            if j != idx:
                node.probs[j] *= scale
        _clamp_renorm(node.probs, eps)


def mutate_ppt_path(
    paths: List[Tuple[PPTNode, int]],
    p_mut: float,
    mut_rate: float,
    rng: np.random.Generator,
    prog_size: int,
) -> None:
    """Faithful Salustowicz & Schmidhuber 1997 mutation.

    Mutates only the PPT nodes on the best-so-far program's path. Per-component
    mutation probability is ``p_mut / (N_INSTR * sqrt(|program|))``. When a
    component is selected, its probability is *increased* toward 1 by
    ``mut_rate * (1 - p)`` before renormalising. This matches eq. (7) of the
    paper: small, structured nudges that explore around the elite, rather than
    bulk uniform noise.
    """
    p_per = p_mut / (N_INSTR * max(float(np.sqrt(prog_size)), 1.0))
    seen: set = set()
    for node, _idx in paths:
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        for j in range(N_INSTR):
            if rng.random() < p_per:
                node.probs[j] += mut_rate * (1.0 - node.probs[j])
        s = node.probs.sum()
        if s > 0:
            node.probs /= s


def ppt_size(node: PPTNode) -> int:
    return 1 + sum(ppt_size(c) for c in node.children.values())


def _partial_reset_ppt(node: PPTNode, alpha: float = 0.5) -> None:
    """Pull every PPT node's distribution back toward uniform by ``alpha``.

    ``alpha=0`` is a no-op; ``alpha=1`` is a full reset to uniform. PIPE uses
    this kind of partial-reset mechanism to escape local optima once the
    elite has saturated and mutation alone can't find better structures. With
    ``alpha=1`` this is equivalent to a multi-start PIPE: drop the old PPT
    and begin a fresh search.
    """
    if N_INSTR != node.probs.shape[0]:
        # PPT was created under a different n_bits config (e.g. parametric
        # tests). Resize to current N_INSTR.
        node.probs = np.ones(N_INSTR, dtype=np.float64) / N_INSTR
    else:
        uniform = np.ones(N_INSTR, dtype=np.float64) / N_INSTR
        node.probs = (1.0 - alpha) * node.probs + alpha * uniform
        node.probs /= node.probs.sum()
    for child in node.children.values():
        _partial_reset_ppt(child, alpha)


def ppt_max_prob_mean(node: PPTNode) -> Tuple[float, int]:
    """Mean of max-probability across all instantiated PPT nodes (sharpness)."""
    total = float(node.probs.max())
    count = 1
    for c in node.children.values():
        t, n = ppt_max_prob_mean(c)
        total += t * n
        count += n
    return total / count, count


# ---------------------------------------------------------------------------
# Program evaluation (bitmask trick)
# ---------------------------------------------------------------------------
# All 64 fitness cases (rows of X) are evaluated in parallel by representing
# each variable as a 64-bit Python int whose j-th bit is the variable's value
# on input j. AND/OR/NOT/IF then map directly to bitwise ops, so a whole tree
# evaluation across the entire 64-case table takes O(tree_size) bitwise ops.
# This is ~100x faster than the per-row Python loop and makes 5-min PIPE runs
# tractable.
MASK64 = (1 << 64) - 1
# X_BITMASK[i] is the bitmask for terminal x_i: bit j is set iff input j has
# x_i == 1. For n_bits=6 we use 64 bits; configure_n_bits() rebuilds these.
MASK_NBITS: int = (1 << 64) - 1  # default for n_bits=6 (rebuilt on configure)
X_BITMASK: np.ndarray = np.array(
    [sum(((j >> i) & 1) << j for j in range(64)) for i in range(6)],
    dtype=object,
)
# Bitmask of all even-parity inputs (Y as a single bitfield).
Y_BITMASK_PARITY: int = sum(
    int(bin(j).count("1") % 2 == 0) << j for j in range(64)
)


def evaluate_tree_bitmask(tree: Tree) -> int:
    """Evaluate the tree across all 2^n_bits inputs in one pass. Returns int."""
    idx, children = tree
    name = INSTR_NAMES[idx]
    if INSTR_ARITIES[idx] == 0:
        return int(X_BITMASK[int(name[1:])])
    if name == "NOT":
        return (~evaluate_tree_bitmask(children[0])) & MASK_NBITS
    if name == "AND":
        return evaluate_tree_bitmask(children[0]) & evaluate_tree_bitmask(children[1])
    if name == "OR":
        return evaluate_tree_bitmask(children[0]) | evaluate_tree_bitmask(children[1])
    if name == "IF":
        a = evaluate_tree_bitmask(children[0])
        b = evaluate_tree_bitmask(children[1])
        c = evaluate_tree_bitmask(children[2])
        return ((a & b) | ((~a) & c)) & MASK_NBITS
    raise ValueError(f"unknown instruction {name}")


def fitness_bitmask(tree: Tree, y_bitmask: Optional[int] = None) -> int:
    """Fitness = number of inputs where tree output matches the parity target."""
    if y_bitmask is None:
        y_bitmask = Y_BITMASK_PARITY
    out = evaluate_tree_bitmask(tree)
    n_cases = MASK_NBITS.bit_length() if MASK_NBITS else 0
    return n_cases - bin((out ^ y_bitmask) & MASK_NBITS).count("1")


# Slow per-row evaluator kept for cross-checking.
def evaluate_tree(tree: Tree, x: np.ndarray) -> bool:
    idx, children = tree
    name = INSTR_NAMES[idx]
    if INSTR_ARITIES[idx] == 0:
        return bool(x[int(name[1:])])
    if name == "NOT":
        return not evaluate_tree(children[0], x)
    if name == "AND":
        return evaluate_tree(children[0], x) and evaluate_tree(children[1], x)
    if name == "OR":
        return evaluate_tree(children[0], x) or evaluate_tree(children[1], x)
    if name == "IF":
        if evaluate_tree(children[0], x):
            return evaluate_tree(children[1], x)
        return evaluate_tree(children[2], x)
    raise ValueError(f"unknown instruction {name}")


def evaluate_tree_all(tree: Tree, X: np.ndarray) -> np.ndarray:
    """Slow per-row evaluator (for cross-checking the bitmask path)."""
    out = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        out[i] = evaluate_tree(tree, X[i])
    return out


def fitness(tree: Tree, X: np.ndarray, Y: np.ndarray) -> int:
    """Slow fitness for cross-checking; agrees with fitness_bitmask on parity."""
    return int((evaluate_tree_all(tree, X) == Y).sum())


def tree_str(tree: Tree) -> str:
    idx, children = tree
    name = INSTR_NAMES[idx]
    if INSTR_ARITIES[idx] == 0:
        return name
    return f"{name}({', '.join(tree_str(c) for c in children)})"


def tree_size(tree: Tree) -> int:
    _, children = tree
    return 1 + sum(tree_size(c) for c in children)


def tree_depth(tree: Tree) -> int:
    _, children = tree
    if not children:
        return 1
    return 1 + max(tree_depth(c) for c in children)


# ---------------------------------------------------------------------------
# Dataset: 6-bit even parity
# ---------------------------------------------------------------------------
def parity_dataset(n_bits: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    n_cases = 1 << n_bits
    X = np.array(
        [[(i >> b) & 1 for b in range(n_bits)] for i in range(n_cases)],
        dtype=np.int32,
    )
    # even parity: True iff number of 1-bits is even
    Y = (X.sum(axis=1) % 2 == 0)
    return X, Y


# ---------------------------------------------------------------------------
# PIPE training loop
# ---------------------------------------------------------------------------
def train(
    seed: int = 0,
    n_bits: int = 6,
    pop_size: int = 30,
    max_gens: int = 2000,
    lr: float = 0.1,
    p_mut: float = 0.2,
    mut_rate: float = 0.4,
    max_depth: int = 8,
    elitist_prob: float = 1.0,
    eps: float = 0.01,
    stagnation_window: int = 200,
    reset_alpha: float = 1.0,
    max_time_s: Optional[float] = None,
    log_every: int = 25,
    verbose: bool = True,
    early_stop: bool = True,
    snapshot_callback: Optional[Any] = None,
    snapshot_every: int = 0,
) -> Dict[str, Any]:
    """Run PIPE on 6-bit parity.

    Returns a dict with the best tree, its fitness, the per-generation
    history, and the final PPT.
    """
    configure_n_bits(n_bits)
    n_cases = 1 << n_bits

    rng = np.random.default_rng(seed)
    X, Y = parity_dataset(n_bits)
    y_mask = Y_BITMASK_PARITY  # bitmask fitness target

    ppt = PPTNode()
    best_tree: Optional[Tree] = None
    best_paths: Optional[List[Tuple[PPTNode, int]]] = None
    best_fitness: int = -1
    # cross-restart best (so multi-start PIPE never forgets its overall champ)
    overall_best_tree: Optional[Tree] = None
    overall_best_fitness: int = -1

    history = {
        "gen": [],
        "gen_best_fit": [],
        "gen_mean_fit": [],
        "best_fit_so_far": [],
        "best_size": [],
        "ppt_max_prob": [],
        "elapsed_s": [],
        "restarts": [],
    }

    t0 = time.time()
    solved_at: Optional[int] = None
    n_restarts: int = 0
    last_improvement_gen: int = 0

    for gen in range(max_gens):
        gen_best_fit = -1
        gen_best_tree: Optional[Tree] = None
        gen_best_paths: Optional[List[Tuple[PPTNode, int]]] = None
        fits: List[int] = []

        for _ in range(pop_size):
            tree, paths = sample_tree(ppt, rng, max_depth=max_depth)
            f = fitness_bitmask(tree, y_mask)
            fits.append(f)
            if f > gen_best_fit:
                gen_best_fit = f
                gen_best_tree = tree
                gen_best_paths = paths

        # elitist: keep best-so-far across generations (within current restart)
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_tree = gen_best_tree
            best_paths = gen_best_paths
            last_improvement_gen = gen
        if gen_best_fit > overall_best_fitness:
            overall_best_fitness = gen_best_fit
            overall_best_tree = gen_best_tree

        # stagnation handling: if no improvement for `stagnation_window`
        # generations, partially reset the PPT toward uniform. This is the
        # "restart" mechanism mentioned in Salustowicz & Schmidhuber 1997 to
        # escape local optima after PIPE saturates.
        if gen - last_improvement_gen >= stagnation_window and best_fitness < n_cases:
            _partial_reset_ppt(ppt, alpha=reset_alpha)
            # also drop the elite path so the fresh PPT can be biased by a
            # genuinely new sample rather than re-locked to the old elite.
            best_paths = None
            best_tree = None
            best_fitness = -1
            last_improvement_gen = gen
            n_restarts += 1
            history["restarts"].append(gen)
            if verbose:
                print(f"  [stagnation reset @ gen {gen}, "
                      f"best={best_fitness}/{n_cases}]", flush=True)

        # PIPE update: with prob elitist_prob update toward best-so-far,
        # else toward best-of-generation. Faithful to Salustowicz & Schmidhuber's
        # P_el parameter.
        if rng.random() < elitist_prob and best_paths is not None:
            target_paths = best_paths
            target_size = tree_size(best_tree) if best_tree is not None else 1
        else:
            target_paths = gen_best_paths if gen_best_paths is not None else best_paths
            target_size = (
                tree_size(gen_best_tree) if gen_best_tree is not None
                else (tree_size(best_tree) if best_tree else 1)
            )
        if target_paths is not None:
            update_ppt(target_paths, lr=lr)
            # mutate only the elite path's nodes (Salustowicz & Schmidhuber 1997)
            mutate_ppt_path(
                target_paths,
                p_mut=p_mut,
                mut_rate=mut_rate,
                rng=rng,
                prog_size=target_size,
            )

        max_prob_mean, _ = ppt_max_prob_mean(ppt)
        history["gen"].append(gen)
        history["gen_best_fit"].append(int(gen_best_fit))
        history["gen_mean_fit"].append(float(np.mean(fits)))
        history["best_fit_so_far"].append(int(overall_best_fitness))
        history["best_size"].append(
            int(tree_size(overall_best_tree)) if overall_best_tree else 0
        )
        history["ppt_max_prob"].append(float(max_prob_mean))
        history["elapsed_s"].append(float(time.time() - t0))

        if (
            snapshot_callback is not None
            and snapshot_every > 0
            and (gen % snapshot_every == 0 or overall_best_fitness == n_cases)
        ):
            snapshot_callback(
                gen,
                gen_best_fit,
                overall_best_fitness,
                overall_best_tree,
                n_restarts,
            )

        if verbose and (gen % log_every == 0 or overall_best_fitness == n_cases):
            print(
                f"gen {gen:4d}  gen_best={gen_best_fit:3d}/{n_cases}"
                f"  best={overall_best_fitness:3d}/{n_cases}"
                f"  best_size={tree_size(overall_best_tree) if overall_best_tree else 0:3d}"
                f"  ppt_max_p={max_prob_mean:.3f}"
                f"  rest={n_restarts}  t={time.time() - t0:5.1f}s",
                flush=True,
            )

        if overall_best_fitness == n_cases:
            solved_at = gen
            if early_stop:
                break

        if max_time_s is not None and (time.time() - t0) >= max_time_s:
            if verbose:
                print(f"  [time budget {max_time_s:.0f}s reached @ gen {gen}, "
                      f"best={overall_best_fitness}/{n_cases}]", flush=True)
            break

    return {
        "best_tree": overall_best_tree,
        "best_fitness": overall_best_fitness,
        "best_size": tree_size(overall_best_tree) if overall_best_tree else 0,
        "best_depth": tree_depth(overall_best_tree) if overall_best_tree else 0,
        "solved_at": solved_at,
        "history": history,
        "ppt": ppt,
        "n_restarts": n_restarts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def serialize_tree(tree: Tree) -> Any:
    """Recursive json-friendly tree dump."""
    idx, children = tree
    return {"instr": INSTR_NAMES[idx], "args": [serialize_tree(c) for c in children]}


def main() -> None:
    parser = argparse.ArgumentParser(description="PIPE on n-bit even parity.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bits", type=int, default=6,
                        help="Parity width (default 6 = headline experiment).")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-gens", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--p-mut", type=float, default=0.2)
    parser.add_argument("--mut-rate", type=float, default=0.4)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--elitist-prob", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Probability floor / ceiling per PPT entry.")
    parser.add_argument("--stagnation-window", type=int, default=200,
                        help="Generations without improvement before partial reset.")
    parser.add_argument("--reset-alpha", type=float, default=1.0,
                        help="0 = no reset, 1 = full multi-start; 0.5 = half-way.")
    parser.add_argument("--max-time-s", type=float, default=None,
                        help="Stop after this many seconds, even if not solved.")
    parser.add_argument("--out", type=str, default="results.json",
                        help="Where to write the run record (set to '' to skip).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"PIPE on {args.n_bits}-bit even parity  seed={args.seed}"
          f"  pop={args.pop_size}  lr={args.lr}  p_mut={args.p_mut}"
          f"  mut_rate={args.mut_rate}  max_depth={args.max_depth}", flush=True)

    t0 = time.time()
    result = train(
        seed=args.seed,
        n_bits=args.n_bits,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        lr=args.lr,
        p_mut=args.p_mut,
        mut_rate=args.mut_rate,
        max_depth=args.max_depth,
        elitist_prob=args.elitist_prob,
        eps=args.eps,
        stagnation_window=args.stagnation_window,
        reset_alpha=args.reset_alpha,
        max_time_s=args.max_time_s,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    best_tree = result["best_tree"]
    n_cases = 1 << args.n_bits
    print(
        f"\nFinal: best={result['best_fitness']}/{n_cases}  size={result['best_size']}"
        f"  depth={result['best_depth']}  solved_at={result['solved_at']}"
        f"  restarts={result.get('n_restarts', 0)}  wallclock={elapsed:.1f}s",
        flush=True,
    )
    if best_tree is not None:
        s = tree_str(best_tree)
        if len(s) > 600:
            s = s[:600] + "..."
        print(f"Best program: {s}", flush=True)

    if args.out:
        record = {
            "args": vars(args),
            "env": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "git_hash": _git_hash(),
            },
            "best_fitness": result["best_fitness"],
            "best_size": result["best_size"],
            "best_depth": result["best_depth"],
            "solved_at": result["solved_at"],
            "wallclock_s": elapsed,
            "best_tree": serialize_tree(best_tree) if best_tree else None,
            "best_tree_str": tree_str(best_tree) if best_tree else None,
            "history": result["history"],
        }
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
        with open(out_path, "w") as f:
            json.dump(record, f)
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
