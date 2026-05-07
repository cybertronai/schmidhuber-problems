"""
rs-two-sequence — random-weight-guessing on the Bengio-94 latch task.

Hochreiter & Schmidhuber, "LSTM can solve hard long time lag problems",
NIPS 9 (1996), pp. 473-479.

The Bengio-94 latch ("two-sequence problem"): a single real-valued input is
presented over T timesteps. The first symbol is +1 or -1 and determines the
target class; the remaining T-1 inputs are Gaussian noise (distractors). The
network sees the whole sequence and must report the class at the final step.

The provocative claim of the H&S 1996 paper: this widely-cited "long time
lag" benchmark is solvable by **random search over weight space** (RS) on a
small fully-recurrent net, without any gradient method. We sample a weight
vector iid from U[-r, r], run a forward pass through the entire sequence,
and accept if classification accuracy on a small training set crosses a
threshold. No BPTT, no RTRL, no evolution -- just sample-and-evaluate.

Architecture (per paper; Section 4.1):
  Fully-recurrent net, ~5 hidden units, tanh activations.
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    y   = sigmoid(W_hy * h_T + b_y)
  All weights and biases sampled iid from U[-r, r] each trial.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_two_sequence_data(n_samples: int, T: int, noise_std: float,
                           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate Bengio-94 two-sequence latch data.

    X[:, 0]  in {-1, +1}  -- the class indicator (the only informative signal)
    X[:, 1:] ~ N(0, noise_std^2) -- distractors
    y[i] = 1 if X[i, 0] == +1 else 0
    """
    y = rng.integers(0, 2, size=n_samples).astype(np.int32)
    X = np.zeros((n_samples, T), dtype=np.float32)
    X[:, 0] = (2.0 * y - 1.0).astype(np.float32)
    if T > 1:
        X[:, 1:] = rng.normal(0.0, noise_std, size=(n_samples, T - 1)).astype(np.float32)
    return X, y


# ----------------------------------------------------------------------
# Recurrent network (fully recurrent, ~5 hidden units, tanh)
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def sample_weights(rng: np.random.Generator, H: int, weight_range: float) -> dict:
    """Sample a fresh parameter set iid from U[-weight_range, +weight_range]."""
    return {
        "W_xh": rng.uniform(-weight_range, weight_range, size=(1, H)).astype(np.float32),
        "W_hh": rng.uniform(-weight_range, weight_range, size=(H, H)).astype(np.float32),
        "b_h":  rng.uniform(-weight_range, weight_range, size=(H,)).astype(np.float32),
        "W_hy": rng.uniform(-weight_range, weight_range, size=(H, 1)).astype(np.float32),
        "b_y":  rng.uniform(-weight_range, weight_range, size=(1,)).astype(np.float32),
    }


def forward_rnn(X: np.ndarray, theta: dict) -> np.ndarray:
    """Forward pass through fully-recurrent net.

    X: (B, T) input sequences
    Returns: (B,) sigmoid output at the FINAL timestep.
    """
    B, T = X.shape
    H = theta["W_hh"].shape[0]
    W_xh, W_hh, b_h = theta["W_xh"], theta["W_hh"], theta["b_h"]
    W_hy, b_y = theta["W_hy"], theta["b_y"]
    h = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        x_t = X[:, t:t + 1]                         # (B, 1)
        h = np.tanh(x_t @ W_xh + h @ W_hh + b_h)    # (B, H)
    z = (h @ W_hy + b_y).reshape(-1)                # (B,)
    return sigmoid(z)


def accuracy(X: np.ndarray, y: np.ndarray, theta: dict) -> float:
    yhat = forward_rnn(X, theta)
    pred = (yhat >= 0.5).astype(np.int32)
    return float((pred == y).mean())


# ----------------------------------------------------------------------
# Random search loop
# ----------------------------------------------------------------------

def random_search(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray,
                  H: int, weight_range: float,
                  max_trials: int, threshold: float,
                  rng: np.random.Generator,
                  log_every: int = 5000) -> dict:
    """Iid weight sampling. Stop when both train and test accuracy cross threshold.

    Returns a dict with the trace of accepted trials and the final solution.
    """
    best_train = 0.0
    best_test = 0.0
    best_theta = None
    trace_trial: list[int] = []
    trace_train: list[float] = []
    trace_test: list[float] = []
    accepted_trial: list[int] = []
    accepted_train: list[float] = []
    accepted_test: list[float] = []
    t0 = time.time()

    for trial in range(1, max_trials + 1):
        theta = sample_weights(rng, H, weight_range)
        a_tr = accuracy(X_tr, y_tr, theta)
        if a_tr > best_train:
            best_train = a_tr
            trace_trial.append(trial)
            trace_train.append(a_tr)
            trace_test.append(float("nan"))

        if a_tr >= threshold:
            a_te = accuracy(X_te, y_te, theta)
            accepted_trial.append(trial)
            accepted_train.append(a_tr)
            accepted_test.append(a_te)
            trace_trial.append(trial)
            trace_train.append(a_tr)
            trace_test.append(a_te)
            if a_te >= threshold:
                wallclock = time.time() - t0
                return {
                    "solved": True,
                    "trial": trial,
                    "wallclock": wallclock,
                    "best_train_acc": a_tr,
                    "best_test_acc": a_te,
                    "theta": theta,
                    "trace_trial": trace_trial,
                    "trace_train": trace_train,
                    "trace_test": trace_test,
                    "accepted_trial": accepted_trial,
                    "accepted_train": accepted_train,
                    "accepted_test": accepted_test,
                }
            if a_te > best_test:
                best_test = a_te
                best_theta = theta

        if trial % log_every == 0:
            elapsed = time.time() - t0
            rate = trial / max(elapsed, 1e-9)
            print(f"  trial {trial:>7d} | best_train {best_train:.3f} | "
                  f"best_test {best_test:.3f} | accepted {len(accepted_trial):>4d} | "
                  f"{rate:>6.0f} trials/s | {elapsed:.1f}s")

    wallclock = time.time() - t0
    return {
        "solved": False,
        "trial": max_trials,
        "wallclock": wallclock,
        "best_train_acc": best_train,
        "best_test_acc": best_test,
        "theta": best_theta,
        "trace_trial": trace_trial,
        "trace_train": trace_train,
        "trace_test": trace_test,
        "accepted_trial": accepted_trial,
        "accepted_train": accepted_train,
        "accepted_test": accepted_test,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run(seed: int, lag: int, hidden: int, noise_std: float,
        n_train: int, n_test: int, weight_range: float,
        threshold: float, max_trials: int, verbose: bool = True) -> dict:
    """Run one RS attempt with the given seed and hyperparameters."""
    seed_seq = np.random.SeedSequence(seed)
    data_seed, search_seed = seed_seq.spawn(2)
    data_rng = np.random.default_rng(data_seed)
    search_rng = np.random.default_rng(search_seed)

    X_tr, y_tr = make_two_sequence_data(n_train, lag, noise_std, data_rng)
    X_te, y_te = make_two_sequence_data(n_test, lag, noise_std, data_rng)

    if verbose:
        print(f"Bengio-94 latch | T={lag} | hidden={hidden} | "
              f"weight_range=±{weight_range} | noise_std={noise_std}")
        print(f"  train: {n_train} sequences | test: {n_test} sequences | "
              f"threshold={threshold}")
        print(f"  seed={seed} | max_trials={max_trials}")

    result = random_search(
        X_tr, y_tr, X_te, y_te,
        H=hidden, weight_range=weight_range,
        max_trials=max_trials, threshold=threshold,
        rng=search_rng,
    )

    result["config"] = {
        "seed": seed, "lag": lag, "hidden": hidden, "noise_std": noise_std,
        "n_train": n_train, "n_test": n_test, "weight_range": weight_range,
        "threshold": threshold, "max_trials": max_trials,
    }
    result["env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }

    if verbose:
        print()
        if result["solved"]:
            print(f"  SOLVED at trial {result['trial']:,} in "
                  f"{result['wallclock']:.2f}s")
            print(f"  train_acc {result['best_train_acc']:.3f}  "
                  f"test_acc {result['best_test_acc']:.3f}")
        else:
            print(f"  UNSOLVED after {max_trials:,} trials "
                  f"({result['wallclock']:.2f}s)")
            print(f"  best train_acc {result['best_train_acc']:.3f}  "
                  f"best test_acc {result['best_test_acc']:.3f}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lag", type=int, default=100,
                   help="Sequence length T (paper: 50-100). Default 100.")
    p.add_argument("--hidden", type=int, default=5,
                   help="Hidden units (paper: 5).")
    p.add_argument("--noise-std", type=float, default=0.2,
                   help="Std of distractor noise (paper: 0.2).")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--weight-range", type=float, default=1.0,
                   help="Weights drawn from U[-r, r]. Paper used r up to 100; "
                        "v1 uses r=1 (linear regime) for higher latch density and "
                        "interpretable accepted weights. See README §Deviations.")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Stop when train AND test accuracy >= threshold.")
    p.add_argument("--max-trials", type=int, default=200_000)
    p.add_argument("--out", type=str, default=None,
                   help="Write result JSON (without weight matrices) to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        seed=args.seed, lag=args.lag, hidden=args.hidden, noise_std=args.noise_std,
        n_train=args.n_train, n_test=args.n_test, weight_range=args.weight_range,
        threshold=args.threshold, max_trials=args.max_trials, verbose=True,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Strip non-JSON-serializable fields
        slim = {k: v for k, v in result.items() if k != "theta"}
        with open(args.out, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
