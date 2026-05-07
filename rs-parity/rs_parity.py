"""
rs-parity — Random-weight-guessing on N-bit sequence parity.

Hochreiter & Schmidhuber, *Bridging Long Time Lags by Weight Guessing and
"Long Short-Term Memory"*, NIPS 9 workshop, 1996. (Also reported in the
literature review of the 1997 LSTM paper and in Hochreiter, Bengio,
Frasconi & Schmidhuber 2001, *Gradient flow in recurrent nets*.)

Problem:
  A bit sequence x_1, ..., x_N in {-1, +1} is fed to a small fully-recurrent
  net one bit per timestep. After the final input the readout unit must
  predict the parity (XOR of all bits, equivalently the product of the
  inputs in {-1, +1}). The classic long-time-lag failure mode of BPTT/RTRL:
  the credit-assignment signal must traverse the full sequence backwards,
  and vanishes long before it reaches the early bits.

Algorithm (random-weight guessing):
  Sample every weight in the network uniformly from a fixed wide range
  [-r, r], run the RNN forward through every training sequence, score on
  parity-correct, repeat. There is no gradient descent and no mutation /
  crossover — each trial is independent. The point of the paper is that
  with a small enough net the basin of weights that solves parity, while
  rare, is large enough that uniform sampling hits it in roughly thousands
  of trials, while BPTT/RTRL fail to converge from gradient information.

Architecture (A2 from Schmidhuber's 1996 family):
  - 1 input unit, H hidden units, 1 output unit
  - hidden units fully recurrent **without** self-connections
    (diag(W_hh) = 0)
  - tanh hidden, tanh output
  - h_0 = 0
  - prediction = sign(y_T)

Reproducibility:
  python3 rs_parity.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def make_parity_dataset(N: int) -> tuple[np.ndarray, np.ndarray]:
    """All 2^N sequences of +/-1 of length N, with parity targets in {-1, +1}.

    Target = product of the bits = +1 if the count of -1s is even, else -1.
    Equivalently: -1 if XOR of the (bit==-1) indicators is 1.

    Returns
    -------
    X : (2**N, N) float32, values in {-1, +1}
    y : (2**N,) float32, values in {-1, +1}
    """
    if N > 22:
        raise ValueError(f"N={N} is too large for full enumeration "
                         f"(2^{N} = {2**N} patterns). Use --sample-size.")
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N, dtype=np.int64)[None, :]) & 1)
    X = (bits * 2 - 1).astype(np.float32)            # {0,1} -> {-1,+1}
    y = X.prod(axis=1).astype(np.float32)            # parity in {-1,+1}
    return X, y


def sample_parity_dataset(N: int, n_samples: int, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Random subset of length-N parity patterns (for N too large to enumerate)."""
    bits = rng.integers(0, 2, size=(n_samples, N))
    X = (bits * 2 - 1).astype(np.float32)
    y = X.prod(axis=1).astype(np.float32)
    return X, y


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

@dataclass
class RNNParams:
    W_xh: np.ndarray   # (1, H)
    W_hh: np.ndarray   # (H, H), diag = 0   (A2: no self-connections)
    b_h:  np.ndarray   # (H,)
    W_hy: np.ndarray   # (H, 1)
    b_y:  np.ndarray   # (1,)

    def n_params(self) -> int:
        H = self.b_h.shape[0]
        # exclude the H zero diagonal entries that A2 fixes by definition
        return self.W_xh.size + (H * H - H) + self.b_h.size + self.W_hy.size + self.b_y.size


def sample_params(H: int, weight_scale: float,
                  rng: np.random.Generator,
                  no_self_connections: bool = False) -> RNNParams:
    """Draw all weights uniformly from [-weight_scale, +weight_scale].

    With ``no_self_connections=True`` the diagonal of ``W_hh`` is zeroed
    (the "A2" architecture in Schmidhuber's 1992 Sequence Chunker family).
    Default is the standard fully-recurrent net used in the H&S 1996 RS
    experiments.
    """
    r = weight_scale
    W_xh = rng.uniform(-r, r, size=(1, H)).astype(np.float32)
    W_hh = rng.uniform(-r, r, size=(H, H)).astype(np.float32)
    if no_self_connections:
        np.fill_diagonal(W_hh, 0.0)
    b_h  = rng.uniform(-r, r, size=(H,)).astype(np.float32)
    W_hy = rng.uniform(-r, r, size=(H, 1)).astype(np.float32)
    b_y  = rng.uniform(-r, r, size=(1,)).astype(np.float32)
    return RNNParams(W_xh, W_hh, b_h, W_hy, b_y)


def forward(params: RNNParams, X: np.ndarray, return_states: bool = False
            ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run the RNN on a batch of sequences.

    Parameters
    ----------
    X : (B, N) float32 in {-1, +1}

    Returns
    -------
    y : (B,) float32 in (-1, 1)        if return_states=False
    (y, H_traj) : H_traj is (B, N+1, H) including h_0 = 0 if return_states=True
    """
    B, N = X.shape
    H = params.b_h.shape[0]
    h = np.zeros((B, H), dtype=np.float32)
    if return_states:
        traj = np.zeros((B, N + 1, H), dtype=np.float32)
    for t in range(N):
        x_t = X[:, t:t + 1]                                    # (B, 1)
        h = np.tanh(x_t @ params.W_xh + h @ params.W_hh + params.b_h)
        if return_states:
            traj[:, t + 1] = h
    y = np.tanh((h @ params.W_hy + params.b_y).squeeze(-1))    # (B,)
    if return_states:
        return y, traj
    return y


def accuracy(params: RNNParams, X: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of sequences where sign(y_pred) == y_true."""
    y_pred = forward(params, X)
    # both y_true and y_pred are nonzero by construction (y_true in {-1,+1};
    # y_pred = tanh(.) which is exactly 0 only if its preactivation is 0).
    return float((np.sign(y_pred) == y_true).mean())


# ----------------------------------------------------------------------
# Random search
# ----------------------------------------------------------------------

def random_search(N: int = 10,
                  H: int = 5,
                  weight_scale: float = 10.0,
                  max_trials: int = 200_000,
                  target_acc: float = 1.0,
                  log_every: int = 1000,
                  seed: int = 0,
                  sample_size: int | None = None,
                  no_self_connections: bool = False,
                  verbose: bool = True
                  ) -> tuple[RNNParams, dict]:
    """Random-weight guessing on N-bit sequence parity.

    Returns the best parameters found and a history dict. Stops as soon as
    accuracy on the training set reaches target_acc, or after max_trials.
    """
    rng = np.random.default_rng(seed)
    if sample_size is None:
        X, y = make_parity_dataset(N)
    else:
        X, y = sample_parity_dataset(N, sample_size, rng)

    if verbose:
        n_pat = X.shape[0]
        arch = "A2 (no self-connections)" if no_self_connections else "fully-recurrent"
        print(f"# rs-parity: N={N} bits, H={H} hidden ({arch}), "
              f"weight_scale={weight_scale:g}, "
              f"{n_pat} patterns ({'enumerated' if sample_size is None else 'sampled'}), "
              f"seed={seed}")

    best_acc = 0.0
    best_params: RNNParams | None = None
    history = {
        "best_trial":   [],
        "best_acc":     [],
        "all_trial":    [],     # subsampled at log_every
        "all_acc":      [],
        "found_trial":  None,
        "n_trials":     0,
    }

    t0 = time.time()
    for trial in range(1, max_trials + 1):
        params = sample_params(H, weight_scale, rng,
                               no_self_connections=no_self_connections)
        acc = accuracy(params, X, y)

        if trial % log_every == 0 or trial == 1:
            history["all_trial"].append(trial)
            history["all_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_params = params
            history["best_trial"].append(trial)
            history["best_acc"].append(acc)
            if verbose:
                print(f"  trial {trial:>7d}  acc={acc*100:6.2f}%  "
                      f"({time.time() - t0:5.2f}s)")
            if best_acc >= target_acc:
                history["found_trial"] = trial
                break

    history["n_trials"] = trial
    history["wallclock_s"] = time.time() - t0
    history["final_acc"] = best_acc

    if verbose:
        if history["found_trial"] is not None:
            print(f"\n# SOLVED in {history['found_trial']} trials "
                  f"({history['wallclock_s']:.2f}s wallclock)")
        else:
            print(f"\n# NOT SOLVED in {max_trials} trials "
                  f"(best acc {best_acc*100:.2f}%, "
                  f"{history['wallclock_s']:.2f}s wallclock)")

    return best_params, history


# ----------------------------------------------------------------------
# Environment logging  (per .claude/rules/experiment-reproducibility.md)
# ----------------------------------------------------------------------

def env_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "python":   sys.version.split()[0],
        "numpy":    np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": commit,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50,
                   help="sequence length / number of input bits (default 50)")
    p.add_argument("--hidden", type=int, default=2,
                   help="number of hidden units (default 2; matches the "
                        "2-state parity automaton)")
    p.add_argument("--weight-scale", type=float, default=30.0,
                   help="weights drawn uniform[-r, +r] (default 30.0; wide "
                        "range puts tanh deeply into saturation, so the "
                        "recurrence behaves like a discrete FSM)")
    p.add_argument("--max-trials", type=int, default=200_000,
                   help="abort after this many guesses (default 200k)")
    p.add_argument("--target-acc", type=float, default=1.0,
                   help="stop when training accuracy reaches this (default 1.0)")
    p.add_argument("--sample-size", type=int, default=2048,
                   help="train on this many random length-N sequences "
                        "(default 2048). Pass 0 to enumerate all 2^N patterns "
                        "(only viable for N <= ~22).")
    p.add_argument("--no-self-connections", action="store_true",
                   help="zero the diagonal of W_hh (Schmidhuber 1992 'A2' "
                        "architecture). Default: standard fully-recurrent net.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-json", type=str, default=None,
                   help="if set, dump full config + result + env to this path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    sample_size = None if args.sample_size in (0, None) else args.sample_size
    best_params, history = random_search(
        N=args.n,
        H=args.hidden,
        weight_scale=args.weight_scale,
        max_trials=args.max_trials,
        target_acc=args.target_acc,
        sample_size=sample_size,
        no_self_connections=args.no_self_connections,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Independent verification on a held-out random sample (only meaningful
    # if we enumerated training; otherwise it's a separate sample of the
    # same distribution).
    if best_params is not None:
        rng_eval = np.random.default_rng(args.seed + 10_000)
        X_eval, y_eval = sample_parity_dataset(args.n, 4096, rng_eval)
        eval_acc = accuracy(best_params, X_eval, y_eval)
        if not args.quiet:
            print(f"# held-out sample acc (4096 random sequences, "
                  f"seed={args.seed + 10000}): {eval_acc * 100:.2f}%")
        history["holdout_acc"] = eval_acc

    if args.results_json:
        results = {
            "config": {
                "N": args.n, "H": args.hidden,
                "weight_scale": args.weight_scale,
                "max_trials": args.max_trials,
                "target_acc": args.target_acc,
                "sample_size": args.sample_size,
                "seed": args.seed,
            },
            "result": {
                "found_trial": history["found_trial"],
                "n_trials": history["n_trials"],
                "wallclock_s": history["wallclock_s"],
                "final_acc": history["final_acc"],
                "holdout_acc": history.get("holdout_acc"),
            },
            "env": env_info(),
        }
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"# wrote {args.results_json}")


if __name__ == "__main__":
    main()
