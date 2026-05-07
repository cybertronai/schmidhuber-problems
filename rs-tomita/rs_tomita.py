"""
rs-tomita -- random-weight-guessing on Tomita grammars #1, #2, #4.

Reproduction of the random-search baseline from Hochreiter & Schmidhuber,
"LSTM can solve hard long time lag problems," NIPS 9 (1996/1997). The point
of that baseline is uncomfortable: the Tomita-grammar testbed (Tomita 1982,
Miller & Giles 1993), often cited as a hard recurrent-net benchmark, can be
attacked by sampling weights iid and keeping the first sample that classifies
the training set. No gradient. No BPTT. Just keep rolling.

What this script reproduces:
  * grammar #1: language a^*  (only sequences of all a's are accepted)
  * grammar #2: language (ab)^*  (alternating ab, even length)
  * grammar #4: no string contains the substring 'aaa'

Setup:
  * vocab = {a, b}, one-hot encoded
  * 5 hidden units, fully recurrent, tanh activation
  * binary classifier (sigmoid output) read from the final hidden state
  * weights sampled iid uniform[-scale, scale]; keep the first sample that
    classifies the full training set perfectly

CLI:
  python3 rs_tomita.py --seed 0 --grammar all
  python3 rs_tomita.py --seed 0 --grammar 4 --max-trials 200000

Outputs:
  results/rs_tomita_seed{N}.npz  (training/test sets, history, best weights)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


# -----------------------------------------------------------------------------
# Tomita grammars (Tomita 1982 -- the seven regular languages on {a, b}).
# -----------------------------------------------------------------------------

def _accept_g1(s: str) -> bool:
    """Tomita #1: a^* -- only strings of all a's."""
    return all(c == "a" for c in s)


def _accept_g2(s: str) -> bool:
    """Tomita #2: (ab)^* -- alternating ab, even length."""
    if len(s) % 2 != 0:
        return False
    for i in range(0, len(s), 2):
        if s[i:i + 2] != "ab":
            return False
    return True


def _accept_g4(s: str) -> bool:
    """Tomita #4: no string containing 'aaa'."""
    return "aaa" not in s


GRAMMARS: dict[int, Callable[[str], bool]] = {
    1: _accept_g1,
    2: _accept_g2,
    4: _accept_g4,
}


def in_tomita(grammar: int, s: str) -> bool:
    return GRAMMARS[grammar](s)


# -----------------------------------------------------------------------------
# Dataset construction.
#
# Tomita's standard testbed (1982) trains on short strings and tests on
# longer strings of the same language to check that the learner generalises.
# We mirror that: train on length 0..10 (2047 strings), test on 11..14
# (sampled 4096 per length when the full enumeration is small enough).
# Final train/test sets are class-balanced so chance accuracy is 50%.
# -----------------------------------------------------------------------------

def _enumerate_strings(L: int) -> list[str]:
    """All 2^L strings of length L over {a, b}."""
    if L == 0:
        return [""]
    return [
        "".join("a" if (i >> j) & 1 == 0 else "b" for j in range(L))
        for i in range(2 ** L)
    ]


def _sample_strings(L: int, n: int, rng: np.random.Generator) -> list[str]:
    """Random strings of length L; deduplicated, deterministic order."""
    seen: set[str] = set()
    out: list[str] = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        s = "".join(rng.choice(["a", "b"]) for _ in range(L))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _positive_strings_of_length(grammar: int, L: int) -> list[str]:
    """Direct enumeration of accepted strings of length L (cheap for #1/#2)."""
    if grammar == 1:
        return ["a" * L]
    if grammar == 2:
        return ["ab" * (L // 2)] if L % 2 == 0 else []
    # grammar 4: enumerate (acceptable up to length ~12)
    if 2 ** L <= 8192:
        return [s for s in _enumerate_strings(L) if "aaa" not in s]
    # for very long L, sample
    return []


def make_dataset(
    grammar: int,
    rng: np.random.Generator,
    train_lens: range = range(0, 11),
    test_lens: range = range(11, 15),
    n_train_per_class: int = 8,
    n_test_per_class: int = 32,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Class-balanced train and test sets.

    Returns lists of (string, label) where label is 1 for grammar-accepted, 0 otherwise.
    """
    accept = GRAMMARS[grammar]

    # Train pool: full enumeration of train_lens
    train_pool: list[str] = []
    for L in train_lens:
        train_pool.extend(_enumerate_strings(L))

    train_pos = [s for s in train_pool if accept(s)]
    train_neg = [s for s in train_pool if not accept(s)]

    # Test pool: enumerate when feasible, otherwise sample
    test_pool: list[str] = []
    for L in test_lens:
        if 2 ** L <= 8192:
            test_pool.extend(_enumerate_strings(L))
        else:
            test_pool.extend(_sample_strings(L, 4096, rng))

    test_pos = [s for s in test_pool if accept(s)]
    test_neg = [s for s in test_pool if not accept(s)]

    # Augment Tomita #2 positives: only one accepted string per even length,
    # so add explicit (ab)^k for all even k in train_lens / test_lens.
    if grammar == 2:
        seen = set(train_pos)
        for L in train_lens:
            if L % 2 == 0:
                cand = "ab" * (L // 2)
                if cand not in seen:
                    train_pos.append(cand)
                    seen.add(cand)
        seen = set(test_pos)
        for L in test_lens:
            if L % 2 == 0:
                cand = "ab" * (L // 2)
                if cand not in seen:
                    test_pos.append(cand)
                    seen.add(cand)

    # Permute deterministically via rng
    train_pos = [train_pos[i] for i in rng.permutation(len(train_pos))]
    train_neg = [train_neg[i] for i in rng.permutation(len(train_neg))]
    test_pos = [test_pos[i] for i in rng.permutation(len(test_pos))]
    test_neg = [test_neg[i] for i in rng.permutation(len(test_neg))]

    n_tr_pos = min(n_train_per_class, len(train_pos))
    n_tr_neg = min(n_train_per_class, len(train_neg))
    n_te_pos = min(n_test_per_class, len(test_pos))
    n_te_neg = min(n_test_per_class, len(test_neg))

    train = (
        [(s, 1) for s in train_pos[:n_tr_pos]]
        + [(s, 0) for s in train_neg[:n_tr_neg]]
    )
    test = (
        [(s, 1) for s in test_pos[:n_te_pos]]
        + [(s, 0) for s in test_neg[:n_te_neg]]
    )

    # Shuffle so positives and negatives are mixed
    train_idx = rng.permutation(len(train))
    test_idx = rng.permutation(len(test))
    train = [train[i] for i in train_idx]
    test = [test[i] for i in test_idx]
    return train, test


# -----------------------------------------------------------------------------
# Encoding strings to padded numeric tensors.
# -----------------------------------------------------------------------------

def encode_batch(
    strings: list[str], max_len: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode a batch of strings.

    Returns (X, lengths) where X has shape (B, T, 2) and lengths has shape (B,).
    """
    if max_len is None:
        max_len = max((len(s) for s in strings), default=1)
    max_len = max(max_len, 1)
    B = len(strings)
    X = np.zeros((B, max_len, 2), dtype=np.float32)
    lens = np.zeros(B, dtype=np.int32)
    for i, s in enumerate(strings):
        for t, c in enumerate(s):
            X[i, t, 0 if c == "a" else 1] = 1.0
        lens[i] = len(s)
    return X, lens


# -----------------------------------------------------------------------------
# Model: small fully-recurrent net with binary-classifier head.
# -----------------------------------------------------------------------------

def sample_weights(
    rng: np.random.Generator, hidden: int = 5, input_size: int = 2, scale: float = 1.0
) -> dict[str, np.ndarray]:
    """iid uniform[-scale, scale] weights and biases."""
    return {
        "W_xh": rng.uniform(-scale, scale, (input_size, hidden)).astype(np.float32),
        "W_hh": rng.uniform(-scale, scale, (hidden, hidden)).astype(np.float32),
        "W_hy": rng.uniform(-scale, scale, (hidden, 1)).astype(np.float32),
        "b_h": rng.uniform(-scale, scale, (hidden,)).astype(np.float32),
        "b_y": rng.uniform(-scale, scale, (1,)).astype(np.float32),
    }


def forward(
    W: dict[str, np.ndarray], X: np.ndarray, lens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run the RNN over a padded batch.

    Returns (preds, h_final) where preds is (B,) int32 in {0, 1} and h_final
    is the hidden state at each example's actual end-of-sequence.
    For empty strings (length 0), h_final is the zero initial state.
    """
    B, T, _ = X.shape
    H = W["W_hh"].shape[0]
    h = np.zeros((B, H), dtype=np.float32)
    h_final = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        active = lens > t
        h_new = np.tanh(X[:, t] @ W["W_xh"] + h @ W["W_hh"] + W["b_h"])
        # Only update h for examples whose sequence is still going.
        h = np.where(active[:, None], h_new, h)
        # Capture the post-update h as h_final for examples ending at this step.
        ending = (lens - 1) == t
        h_final = np.where(ending[:, None], h, h_final)
    logits = h_final @ W["W_hy"] + W["b_y"]
    preds = (logits[:, 0] > 0).astype(np.int32)
    return preds, h_final


def hidden_trajectory(W: dict[str, np.ndarray], s: str) -> np.ndarray:
    """Return the hidden-state trajectory for one string, shape (T+1, H)."""
    H = W["W_hh"].shape[0]
    h = np.zeros(H, dtype=np.float32)
    out = [h.copy()]
    for c in s:
        x = np.array(
            [1.0 if c == "a" else 0.0, 1.0 if c == "b" else 0.0],
            dtype=np.float32,
        )
        h = np.tanh(x @ W["W_xh"] + h @ W["W_hh"] + W["b_h"])
        out.append(h.copy())
    return np.stack(out)


# -----------------------------------------------------------------------------
# Random-weight-guessing search.
# -----------------------------------------------------------------------------

def random_search(
    train: list[tuple[str, int]],
    test: list[tuple[str, int]],
    max_trials: int,
    scale: float,
    hidden: int,
    rng: np.random.Generator,
    train_threshold: float = 1.0,
):
    """Sample weights iid and stop at the first trial that perfectly fits train.

    Returns (solved_at, best_train, best_test, best_W, history). solved_at is
    the trial index of the first perfect-fit; None if budget exhausted.
    history is a list of (trial, train_acc, test_acc) tuples recording every
    new best train_acc seen so far (used for plotting).
    """
    train_strings = [s for s, _ in train]
    train_y = np.array([y for _, y in train], dtype=np.int32)
    test_strings = [s for s, _ in test]
    test_y = np.array([y for _, y in test], dtype=np.int32)

    max_len_all = max(
        max(len(s) for s in train_strings),
        max(len(s) for s in test_strings),
    )
    train_X, train_lens = encode_batch(train_strings, max_len=max_len_all)
    test_X, test_lens = encode_batch(test_strings, max_len=max_len_all)

    best_train, best_test = 0.0, 0.0
    best_W: dict[str, np.ndarray] | None = None
    history: list[tuple[int, float, float]] = [(0, 0.0, 0.0)]

    for trial in range(1, max_trials + 1):
        W = sample_weights(rng, hidden=hidden, scale=scale)
        train_pred, _ = forward(W, train_X, train_lens)
        train_acc = float(np.mean(train_pred == train_y))

        if train_acc > best_train:
            test_pred, _ = forward(W, test_X, test_lens)
            test_acc = float(np.mean(test_pred == test_y))
            best_train = train_acc
            best_test = test_acc
            best_W = {k: v.copy() for k, v in W.items()}
            history.append((trial, best_train, best_test))

            if train_acc >= train_threshold:
                return trial, best_train, best_test, best_W, history

    return None, best_train, best_test, best_W, history


# -----------------------------------------------------------------------------
# Per-grammar runner.
# -----------------------------------------------------------------------------

def run_grammar(
    grammar: int,
    seed: int,
    max_trials: int,
    scale: float,
    hidden: int,
    n_train_per_class: int = 8,
    n_test_per_class: int = 32,
) -> dict:
    rng = np.random.default_rng(seed)
    train, test = make_dataset(
        grammar, rng,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
    )
    t0 = time.time()
    solved_at, best_tr, best_te, best_W, history = random_search(
        train, test, max_trials, scale, hidden, rng,
    )
    wallclock = time.time() - t0
    return {
        "grammar": grammar,
        "seed": seed,
        "solved_at": solved_at,
        "best_train": best_tr,
        "best_test": best_te,
        "wallclock": wallclock,
        "n_train": len(train),
        "n_test": len(test),
        "history": history,
        "train": train,
        "test": test,
        "weights": best_W,
        "config": {"scale": scale, "hidden": hidden, "max_trials": max_trials},
    }


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------

# Default trial budgets per grammar -- generous compared to the H&S 1996
# medians (#1: 182, #2: 1,511, #4: 13,833) so seeds that get unlucky still
# resolve.
DEFAULT_MAX_TRIALS: dict[int, int] = {1: 5_000, 2: 50_000, 4: 200_000}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grammar", type=str, default="all", choices=["1", "2", "4", "all"])
    p.add_argument("--max-trials", type=int, default=None,
                   help="Cap on RS trials per grammar. Default: 5k/50k/200k for #1/#2/#4.")
    p.add_argument("--scale", type=float, default=2.0,
                   help="Half-width of the uniform weight distribution.")
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--save", type=str, default="results/rs_tomita_seed{seed}.npz",
                   help="Where to save results NPZ (use {seed} for seed substitution).")
    return p.parse_args()


def save_results(path: Path, results: list[dict], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, np.ndarray] = {"seed": np.array([seed])}
    for r in results:
        g = r["grammar"]
        h = np.array(r["history"], dtype=np.float64) if r["history"] else np.zeros((0, 3))
        save_dict[f"g{g}_history"] = h
        save_dict[f"g{g}_solved_at"] = np.array(
            [r["solved_at"] if r["solved_at"] is not None else -1]
        )
        save_dict[f"g{g}_best_train"] = np.array([r["best_train"]])
        save_dict[f"g{g}_best_test"] = np.array([r["best_test"]])
        save_dict[f"g{g}_wallclock"] = np.array([r["wallclock"]])
        save_dict[f"g{g}_train_strings"] = np.array(
            [s for s, _ in r["train"]], dtype=object
        )
        save_dict[f"g{g}_train_y"] = np.array([y for _, y in r["train"]], dtype=np.int32)
        save_dict[f"g{g}_test_strings"] = np.array(
            [s for s, _ in r["test"]], dtype=object
        )
        save_dict[f"g{g}_test_y"] = np.array([y for _, y in r["test"]], dtype=np.int32)
        if r["weights"]:
            for k, v in r["weights"].items():
                save_dict[f"g{g}_W_{k}"] = v
    np.savez(path, **save_dict, allow_pickle=True)


def main() -> None:
    args = parse_args()
    if args.grammar == "all":
        grammars = [1, 2, 4]
    else:
        grammars = [int(args.grammar)]

    results = []
    for g in grammars:
        max_trials = args.max_trials if args.max_trials else DEFAULT_MAX_TRIALS[g]
        print(
            f"=== Tomita #{g} | seed={args.seed} | "
            f"max_trials={max_trials} | scale={args.scale} | hidden={args.hidden} ==="
        )
        r = run_grammar(g, args.seed, max_trials, args.scale, args.hidden)
        if r["solved_at"] is not None:
            print(
                f"  SOLVED at trial {r['solved_at']:>6}  | "
                f"train={r['best_train']:.3f}  test={r['best_test']:.3f}  | "
                f"{r['wallclock']:.2f}s"
            )
        else:
            print(
                f"  unsolved (budget {max_trials}) | "
                f"best_train={r['best_train']:.3f}  best_test={r['best_test']:.3f}  | "
                f"{r['wallclock']:.2f}s"
            )
        results.append(r)

    print("\nSummary:")
    print(f"{'Grammar':<10}{'Solved at':<12}{'Train':<8}{'Test':<8}{'Wallclock':<10}{'N_train':<10}{'N_test':<10}")
    for r in results:
        sa = str(r["solved_at"]) if r["solved_at"] is not None else "unsolved"
        print(
            f"#{r['grammar']:<9}{sa:<12}{r['best_train']:<8.3f}{r['best_test']:<8.3f}"
            f"{r['wallclock']:<10.2f}{r['n_train']:<10d}{r['n_test']:<10d}"
        )

    save_path = Path(args.save.format(seed=args.seed))
    save_results(save_path, results, args.seed)
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
