"""
linear-transformers-fwp --- Schlag, Irie, Schmidhuber, *Linear Transformers
Are Secretly Fast Weight Programmers*, ICML 2021 (arXiv:2102.11174).

The 2021 paper observes that *unnormalised linear self-attention* and the
1991/1992 fast-weight programmer compute the *same numpy expression*:

    linear-attention :  y_q  =  V^T (K q)
                              =  sum_t  v_t  <k_t, q>

    1992 FWP read    :  W    =  sum_t  v_t  k_t^T   (outer-product writes)
                       y_q   =  W q

By matrix-multiplication associativity, V^T (K q) = (V^T K) q and the
matrix V^T K is exactly W. They differ only in the *schedule*: linear-
attention re-fetches every stored key on every read; the FWP writes once
into a fixed-size scratchpad and reads with a single matrix-vector multiply.

This stub demonstrates the equivalence on a synthetic key/value retrieval
task and adds the *delta-rule* update from Schlag et al. 2021:

    sum rule (1992)    :  W <- W + outer(v_t, k_t)
    delta rule (2021)  :  W <- W + outer(v_t - W k_t, k_t)

The delta rule subtracts the value the network would currently retrieve for
k_t before writing the correction. On capacity-limited memories with
correlated keys this materially reduces interference -- the headline claim
of the 2021 paper.

Architecture
------------

A learnable slow projector W_K (d_key x d_key) plays the role of the
"slow net" that programs the fast weights (Schmidhuber 1992). Per
episode:

    K = keys @ W_K^T                       # projected keys, (N, d_key)
    --- write ---
    sum-rule    : W_fast = V^T @ K         # one outer-product per pair
    delta-rule  : W_fast accumulated step by step with delta correction
    --- read ---
    k_q = W_K @ q_key                      # project the query the same way
    y   = W_fast @ k_q                     # 1992 FWP read = linear-attn read

Loss L = 0.5 ||y - v_match||^2 is back-propagated through the sum-rule
write into W_K. SGD trains W_K to project out a shared bias direction
present in every raw key, exposing the slow-net role of the 1992 paper.

CLI
---

    python3 linear_transformers_fwp.py --seed 0
    # ~1 s on an M-series laptop CPU; deterministic.

    python3 linear_transformers_fwp.py --equivalence-check
    # Verifies that linear-attention and FWP produce numerically identical
    # outputs on random K, V, q. Prints max abs difference (~1e-15).

    python3 linear_transformers_fwp.py --capacity-sweep
    # Compares sum-rule vs delta-rule retrieval cosine vs N stored pairs.

    python3 linear_transformers_fwp.py --grad-check
    # Numerical-vs-analytic gradient check on the slow projector.
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
# Utilities
# ----------------------------------------------------------------------

def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_BIAS_DIR_CACHE: dict[int, np.ndarray] = {}


def fixed_bias_direction(d_key: int) -> np.ndarray:
    """Unit vector that every raw key in every episode contains as a shared
    component. Deterministic given d_key.
    """
    if d_key not in _BIAS_DIR_CACHE:
        rng = np.random.default_rng(13)
        v = rng.standard_normal(d_key)
        _BIAS_DIR_CACHE[d_key] = v / np.linalg.norm(v)
    return _BIAS_DIR_CACHE[d_key]


# ----------------------------------------------------------------------
# THE TWO READS THAT ARE THE SAME NUMPY OPERATION
# ----------------------------------------------------------------------

def linear_attention(keys: np.ndarray, values: np.ndarray,
                     query: np.ndarray) -> np.ndarray:
    """Unnormalised linear self-attention.

        y  =  sum_t  v_t  <k_t, q>
            =  V^T (K q)        # the "Q K^T V" formulation, written for one query

    keys   : (N, d_key)
    values : (N, d_val)
    query  : (d_key,)
    returns: (d_val,)

    This is the read schedule that touches every stored key.
    """
    scores = keys @ query                    # (N,)  -- one inner product per pair
    return values.T @ scores                 # (d_val,)


def fwp_outer_product_write(keys: np.ndarray,
                            values: np.ndarray) -> np.ndarray:
    """1992 FWP write: accumulate outer products into a fixed-size matrix.

        W_fast  =  sum_t  outer(v_t, k_t)  =  V^T K

    keys   : (N, d_key)
    values : (N, d_val)
    returns: W_fast (d_val, d_key)
    """
    return values.T @ keys                   # = sum_t v_t k_t^T


def fwp_read(W_fast: np.ndarray, query: np.ndarray) -> np.ndarray:
    """1992 FWP read: y = W_fast @ q.  Matrix-vector multiply, one shot.

    W_fast : (d_val, d_key)
    query  : (d_key,)
    returns: (d_val,)
    """
    return W_fast @ query


def linear_attention_via_fwp(keys: np.ndarray, values: np.ndarray,
                             query: np.ndarray) -> np.ndarray:
    """The 2021 identity:  V^T (K q)  ==  (V^T K) q  ==  W_fast q.

    Compute linear attention by first building the fast-weight matrix and
    then reading it once. Algebraically identical to `linear_attention`,
    only the schedule differs.
    """
    W_fast = fwp_outer_product_write(keys, values)
    return fwp_read(W_fast, query)


def equivalence_check(seed: int = 0, n_trials: int = 20,
                      d_key: int = 16, d_val: int = 16,
                      max_n_pairs: int = 32) -> dict:
    """Verify on random inputs that the two reads agree to floating-point
    round-off. Returns the worst-case max abs diff across trials.
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_trials):
        n_pairs = int(rng.integers(1, max_n_pairs + 1))
        keys = rng.standard_normal((n_pairs, d_key)) / np.sqrt(d_key)
        values = rng.standard_normal((n_pairs, d_val)) / np.sqrt(d_val)
        query = rng.standard_normal(d_key) / np.sqrt(d_key)
        y_attn = linear_attention(keys, values, query)
        y_fwp = linear_attention_via_fwp(keys, values, query)
        diffs.append(float(np.max(np.abs(y_attn - y_fwp))))
    diffs = np.asarray(diffs)
    return {
        "n_trials": n_trials,
        "d_key": d_key,
        "d_val": d_val,
        "max_diff": float(diffs.max()),
        "mean_diff": float(diffs.mean()),
    }


# ----------------------------------------------------------------------
# Schlag 2021 delta-rule write
# ----------------------------------------------------------------------

def delta_rule_write(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Delta-rule fast-weight write (Schlag et al. 2021, eq. (5)/(11)).

        W_0 = 0
        W_t = W_{t-1} + outer(v_t - W_{t-1} k_t,  k_t)

    Equivalently: write the *correction* between the new value and what
    the current memory would retrieve at this key. Reduces interference
    when the key sequence is non-orthogonal -- the headline contribution
    of the 2021 paper.

    Returns W (d_val, d_key).
    """
    n_pairs, d_key = keys.shape
    d_val = values.shape[1]
    W = np.zeros((d_val, d_key))
    for k, v in zip(keys, values):
        v_now = W @ k                # current retrieval at this key
        W = W + np.outer(v - v_now, k)
    return W


# ----------------------------------------------------------------------
# Episodic dataset (correlated keys -- slow-net has work to do)
# ----------------------------------------------------------------------

def generate_episode(rng: np.random.Generator, n_pairs: int,
                     d_key: int, d_val: int,
                     bias_alpha: float = 1.0, bias_beta: float = 0.4):
    """Sample N (key, value) pairs with shared-bias keys.

    k_t  =  alpha * b  +  beta * iid_t

    With (1.0, 0.4) every raw key is dominated by the same direction `b`;
    untrained retrieval (W_K = I) is swamped by cross-key interference.
    The slow projector W_K must learn to project b out.
    """
    b = fixed_bias_direction(d_key)
    iid = rng.standard_normal((n_pairs, d_key)) / np.sqrt(d_key)
    keys = bias_alpha * b[None, :] + bias_beta * iid
    values = rng.standard_normal((n_pairs, d_val)) / np.sqrt(d_val)
    q_idx = int(rng.integers(0, n_pairs))
    return keys, values, q_idx


# ----------------------------------------------------------------------
# Slow-projector forward / backward (sum-rule write, differentiable)
# ----------------------------------------------------------------------

def slow_net_forward(W_K: np.ndarray, keys: np.ndarray,
                     values: np.ndarray, q_key: np.ndarray):
    """Linear-attention / sum-rule FWP forward through learnable W_K.

        K       = keys @ W_K^T              # (N, d_key)  projected keys
        W_fast  = values^T @ K              # (d_val, d_key)  = sum_t v_t (W_K k_t)^T
        k_q     = W_K @ q_key               # (d_key,)
        y       = W_fast @ k_q              # (d_val,) retrieved value
    """
    K = keys @ W_K.T
    W_fast = values.T @ K
    k_q = W_K @ q_key
    y = W_fast @ k_q
    return y, W_fast, K, k_q


def slow_net_backward(W_K: np.ndarray, keys: np.ndarray,
                      values: np.ndarray, q_key: np.ndarray,
                      target_v: np.ndarray, y: np.ndarray,
                      W_fast: np.ndarray, K: np.ndarray,
                      k_q: np.ndarray) -> np.ndarray:
    """Gradient of L = 0.5 ||y - target_v||^2 wrt W_K."""
    dy = y - target_v
    dW_fast = np.outer(dy, k_q)
    dk_q = W_fast.T @ dy
    dK = values @ dW_fast
    dW_K = dK.T @ keys + np.outer(dk_q, q_key)
    return dW_K


def numerical_grad_check(seed: int = 0, eps: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    d_key, d_val, n_pairs = 4, 4, 3
    W_K = rng.standard_normal((d_key, d_key)) * 0.3
    keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
    q_key = keys[q_idx]
    target_v = values[q_idx]
    y, W_fast, K, k_q = slow_net_forward(W_K, keys, values, q_key)
    dW_K_an = slow_net_backward(W_K, keys, values, q_key, target_v,
                                y, W_fast, K, k_q)
    dW_K_num = np.zeros_like(W_K)
    for a in range(d_key):
        for b in range(d_key):
            W_K_p = W_K.copy(); W_K_p[a, b] += eps
            W_K_m = W_K.copy(); W_K_m[a, b] -= eps
            yp, *_ = slow_net_forward(W_K_p, keys, values, q_key)
            ym, *_ = slow_net_forward(W_K_m, keys, values, q_key)
            Lp = 0.5 * np.sum((yp - target_v) ** 2)
            Lm = 0.5 * np.sum((ym - target_v) ** 2)
            dW_K_num[a, b] = (Lp - Lm) / (2 * eps)
    return float(np.max(np.abs(dW_K_an - dW_K_num)))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(seed: int = 0, n_pairs: int = 5, d_key: int = 8, d_val: int = 8,
          n_steps: int = 1500, lr: float = 0.05):
    rng = np.random.default_rng(seed)
    W_K = np.eye(d_key) + 0.05 * rng.standard_normal((d_key, d_key))
    history = {"step": [], "loss": [], "cos": []}
    for step in range(n_steps):
        keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
        q_key = keys[q_idx]
        target_v = values[q_idx]
        y, W_fast, K, k_q = slow_net_forward(W_K, keys, values, q_key)
        loss = 0.5 * float(np.sum((y - target_v) ** 2))
        history["step"].append(step)
        history["loss"].append(loss)
        history["cos"].append(cosine_sim(y, target_v))
        dW_K = slow_net_backward(W_K, keys, values, q_key, target_v,
                                 y, W_fast, K, k_q)
        gnorm = float(np.linalg.norm(dW_K))
        if gnorm > 1.0:
            dW_K = dW_K / gnorm
        W_K -= lr * dW_K
    return W_K, history


def evaluate_two_ways(W_K: np.ndarray, seed: int, n_pairs: int,
                      d_key: int, d_val: int, n_test: int = 200):
    """Evaluate retrieval cosine using BOTH the linear-attention schedule
    and the FWP outer-product schedule. They MUST agree to round-off.
    Returns the post-projection mean cosine and the schedule-equivalence
    diff statistics.
    """
    rng = np.random.default_rng(seed)
    cos_attn, cos_fwp = [], []
    diffs = []
    for _ in range(n_test):
        keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
        q_key = keys[q_idx]
        target_v = values[q_idx]
        # Slow projection
        K = keys @ W_K.T
        k_q = W_K @ q_key
        # Schedule A: linear-attention (re-fetch every key)
        y_attn = linear_attention(K, values, k_q)
        # Schedule B: FWP (build matrix once, read once)
        y_fwp = linear_attention_via_fwp(K, values, k_q)
        cos_attn.append(cosine_sim(y_attn, target_v))
        cos_fwp.append(cosine_sim(y_fwp, target_v))
        diffs.append(float(np.max(np.abs(y_attn - y_fwp))))
    return {
        "n_test": n_test,
        "mean_cos_linear_attention": float(np.mean(cos_attn)),
        "mean_cos_fwp_outer_product": float(np.mean(cos_fwp)),
        "schedule_max_diff": float(max(diffs)),
        "schedule_mean_diff": float(np.mean(diffs)),
    }


# ----------------------------------------------------------------------
# Capacity sweep: sum-rule vs delta-rule
# ----------------------------------------------------------------------

def capacity_sweep_rules(W_K: np.ndarray, seed: int, d_key: int,
                         d_val: int, max_pairs: int = 16,
                         n_test: int = 100):
    """For each N in [1, max_pairs], measure mean retrieval cosine using
    sum-rule (1992 FWP / linear-attention) vs delta-rule (Schlag 2021).
    The delta rule trades a slightly more expensive write for cleaner
    retrieval at higher N.
    """
    rng = np.random.default_rng(seed + 9999)
    out = []
    for n_pairs in range(1, max_pairs + 1):
        cos_sum, cos_delta = [], []
        for _ in range(n_test):
            keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
            q_key = keys[q_idx]
            target_v = values[q_idx]
            K = keys @ W_K.T
            k_q = W_K @ q_key
            # Sum rule = 1992 FWP outer-product = linear attention
            W_sum = fwp_outer_product_write(K, values)
            y_sum = fwp_read(W_sum, k_q)
            # Delta rule (Schlag 2021)
            W_delta = delta_rule_write(K, values)
            y_delta = fwp_read(W_delta, k_q)
            cos_sum.append(cosine_sim(y_sum, target_v))
            cos_delta.append(cosine_sim(y_delta, target_v))
        out.append({
            "n_pairs": n_pairs,
            "mean_cos_sum_rule": float(np.mean(cos_sum)),
            "mean_cos_delta_rule": float(np.mean(cos_delta)),
        })
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--d-key", type=int, default=8)
    parser.add_argument("--d-val", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--equivalence-check", action="store_true",
                        help="Verify linear-attention == FWP outer-product on random inputs.")
    parser.add_argument("--capacity-sweep", action="store_true",
                        help="After training, sweep N=1..16 comparing sum-rule vs delta-rule.")
    parser.add_argument("--grad-check", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON results here.")
    args = parser.parse_args()

    if args.grad_check:
        err = numerical_grad_check(seed=args.seed)
        print(f"Max |analytic - numerical| dW_K = {err:.3e}")
        return

    if args.equivalence_check:
        result = equivalence_check(seed=args.seed)
        print(json.dumps(result, indent=2))
        if result["max_diff"] > 1e-10:
            print(f"FAIL: max_diff {result['max_diff']:.3e} > 1e-10")
            sys.exit(1)
        print("OK: linear-attention and 1992-FWP outer-product agree to floating-point round-off.")
        return

    t0 = time.time()
    W_K, history = train(seed=args.seed, n_pairs=args.n_pairs,
                         d_key=args.d_key, d_val=args.d_val,
                         n_steps=args.n_steps, lr=args.lr)
    eval_seed = args.seed + 12345
    pre_eval = evaluate_two_ways(np.eye(args.d_key), eval_seed,
                                 args.n_pairs, args.d_key, args.d_val,
                                 n_test=args.n_test)
    post_eval = evaluate_two_ways(W_K, eval_seed, args.n_pairs,
                                  args.d_key, args.d_val,
                                  n_test=args.n_test)
    elapsed = time.time() - t0

    eq = equivalence_check(seed=args.seed)

    result = {
        "config": {
            "seed": args.seed,
            "n_pairs": args.n_pairs,
            "d_key": args.d_key,
            "d_val": args.d_val,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "n_test": args.n_test,
        },
        "equivalence_check": eq,
        "pretraining_eval": pre_eval,
        "posttraining_eval": post_eval,
        "final_train_loss": history["loss"][-1],
        "final_train_cos": history["cos"][-1],
        "wallclock_s": elapsed,
        "env": {
            "git_commit": git_hash(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    if args.capacity_sweep:
        result["capacity_sweep"] = capacity_sweep_rules(
            W_K, eval_seed, args.d_key, args.d_val,
            max_pairs=16, n_test=100,
        )

    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "result": result,
                       "history": history}, f)


if __name__ == "__main__":
    main()
