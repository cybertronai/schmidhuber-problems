"""
fast-weights-key-value --- Schmidhuber, *Learning to control fast-weight
memories: An alternative to dynamic recurrent networks*, Neural Computation
4(1):131--139, 1992.

The 1992 paper introduced a slow programmer net `S` that produces outer-
product updates to a fast weight matrix:

    W_fast := sum_t  v_t  (W_K k_t)^T

To retrieve the value bound to a query key `k_q`, present `k_q` and read out

    y = W_fast @ (W_K k_q)

That is exactly the unnormalised linear-attention math later formalised in
Schlag, Irie, Schmidhuber (2021) "Linear Transformers are Secretly Fast
Weight Programmers". This stub demonstrates the 1992 origin: a sequence of
`(key, value)` pairs is presented (one outer-product update per pair), then
a single query key is presented and the network must retrieve the bound
value.

Architecture in this stub
-------------------------

    o  Slow net S = a single learnable linear projector W_K (d_key x d_key).
    o  Values pass through identity (no learnable W_V); the loss is computed
       on raw values. This keeps the fixed point of training clearly
       interpretable: W_K should converge to a projector that makes the
       N projected keys `K_t = W_K k_t` near-orthogonal so that
       y = sum_t v_t (k_t W_K^T W_K k_q) collapses to v_match.
    o  Fast weight matrix W_fast (d_val x d_key) is recomputed from scratch
       per episode (it is the "scratchpad" of the 1992 paper).

Key distribution
----------------

To make the slow projector matter, raw keys are drawn from a *correlated*
distribution: every raw key shares a fixed bias direction `b` (a unit
vector picked once, deterministically, given d_key). This makes raw keys
look like

    k_t  =  alpha * b  +  beta * iid_t                (all t share alpha*b)

so the cross-key inner product `k_t . k_t'` is dominated by `alpha**2`.
Untrained retrieval (W_K = I) is therefore poor -- every read is swamped
by the shared-bias contribution of all stored values. The slow projector
W_K can fix this by learning to project out the bias direction; on the
projected space, only the idiosyncratic component beta*iid_t survives,
and idiosyncratic keys are near-orthogonal in d_key=8.

This is exactly the role Schmidhuber 1992 envisioned for S: a slow net
that writes a useful address space into the fast weights.

Training: per episode, sample N random (key, value) pairs (with shared
bias) and one query index; compute the loss between the retrieved `y` and
the true bound value; back-propagate through `W_fast` into `W_K`. The slow
programmer S learns to project the bias out.

CLI
---

    python3 fast_weights_key_value.py --seed 0
    # ~1 s on an M-series laptop CPU; reproducible.

    python3 fast_weights_key_value.py --capacity-sweep
    # extra knob: sweep N from 1 to 12 and report retrieval cosine.

The default recipe (seed 0, n_pairs=5, d_key=d_val=8, n_steps=1500,
lr=0.05) reproduces the headline number reported in README.md / §Results.
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


# Bias direction shared by every raw key (deterministic given d_key).
_BIAS_DIR_CACHE: dict[int, np.ndarray] = {}


def fixed_bias_direction(d_key: int) -> np.ndarray:
    """Return a unit vector that is THE shared-bias direction for raw keys.

    Deterministic given d_key so the dataset distribution is fixed across
    runs and seeds; only the per-episode iid noise depends on the user
    seed. Cached because it is queried in every episode.
    """
    if d_key not in _BIAS_DIR_CACHE:
        rng = np.random.default_rng(13)
        v = rng.standard_normal(d_key)
        _BIAS_DIR_CACHE[d_key] = v / np.linalg.norm(v)
    return _BIAS_DIR_CACHE[d_key]


# ----------------------------------------------------------------------
# Fast-weight forward / backward
# ----------------------------------------------------------------------

def generate_episode(rng: np.random.Generator, n_pairs: int,
                     d_key: int, d_val: int,
                     bias_alpha: float = 1.0, bias_beta: float = 0.4):
    """Sample N (key, value) pairs from the shared-bias distribution.

    Each raw key is

        k_t  =  alpha * b  +  beta * iid_t

    where `b` is the fixed unit bias direction and iid_t is N(0, I)
    rescaled by 1/sqrt(d_key). With (alpha=1.0, beta=0.4) every raw key
    has the same dominant component along b -- untrained retrieval suffers
    badly from the shared component, and the slow projector W_K must
    learn to suppress it.

    Values are iid Gaussian, no bias.
    """
    b = fixed_bias_direction(d_key)
    iid = rng.standard_normal((n_pairs, d_key)) / np.sqrt(d_key)
    keys = bias_alpha * b[None, :] + bias_beta * iid
    values = rng.standard_normal((n_pairs, d_val)) / np.sqrt(d_val)
    q_idx = int(rng.integers(0, n_pairs))
    return keys, values, q_idx


def fast_weight_forward(W_K: np.ndarray, keys: np.ndarray,
                        values: np.ndarray, q_key: np.ndarray):
    """Forward pass through one episode of the fast-weight programmer.

    keys    : (N, d_key)
    values  : (N, d_val)
    q_key   : (d_key,)

    Computes:
        K       = keys @ W_K^T              # projected keys, (N, d_key)
        W_fast  = values^T @ K              # (d_val, d_key)
        k_q     = W_K @ q_key               # (d_key,)
        y       = W_fast @ k_q              # (d_val,) retrieved value

    Returns y, W_fast, K (the latter two cached for the backward pass).
    """
    K = keys @ W_K.T
    W_fast = values.T @ K
    k_q = W_K @ q_key
    y = W_fast @ k_q
    return y, W_fast, K, k_q


def fast_weight_backward(W_K: np.ndarray, keys: np.ndarray,
                         values: np.ndarray, q_key: np.ndarray,
                         target_v: np.ndarray, y: np.ndarray,
                         W_fast: np.ndarray, K: np.ndarray,
                         k_q: np.ndarray) -> np.ndarray:
    """Gradient of L = 0.5 * ||y - target_v||^2 wrt W_K.

    Walk the same chain as forward, in reverse:

        dy        = y - target_v
        dW_fast   = outer(dy, k_q)               # because y = W_fast @ k_q
        dk_q      = W_fast^T @ dy
        dK        = values @ dW_fast             # because W_fast = V^T K
        dW_K  +=  dK^T @ keys                    # because K = keys @ W_K^T
        dW_K  +=  outer(dk_q, q_key)             # because k_q = W_K @ q_key
    """
    dy = y - target_v
    dW_fast = np.outer(dy, k_q)
    dk_q = W_fast.T @ dy
    dK = values @ dW_fast
    dW_K = dK.T @ keys + np.outer(dk_q, q_key)
    return dW_K


def numerical_grad_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Sanity check: max abs difference between analytic and numerical
    gradient of the loss wrt W_K. Returns the worst-case error.
    """
    rng = np.random.default_rng(seed)
    d_key, d_val, n_pairs = 4, 4, 3
    W_K = rng.standard_normal((d_key, d_key)) * 0.3
    keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
    q_key = keys[q_idx]
    target_v = values[q_idx]

    y, W_fast, K, k_q = fast_weight_forward(W_K, keys, values, q_key)
    dW_K_an = fast_weight_backward(W_K, keys, values, q_key, target_v,
                                   y, W_fast, K, k_q)

    dW_K_num = np.zeros_like(W_K)
    for a in range(d_key):
        for b in range(d_key):
            W_K_p = W_K.copy(); W_K_p[a, b] += eps
            W_K_m = W_K.copy(); W_K_m[a, b] -= eps
            yp, *_ = fast_weight_forward(W_K_p, keys, values, q_key)
            ym, *_ = fast_weight_forward(W_K_m, keys, values, q_key)
            Lp = 0.5 * np.sum((yp - target_v) ** 2)
            Lm = 0.5 * np.sum((ym - target_v) ** 2)
            dW_K_num[a, b] = (Lp - Lm) / (2 * eps)

    return float(np.max(np.abs(dW_K_an - dW_K_num)))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(seed: int = 0, n_pairs: int = 5, d_key: int = 8, d_val: int = 8,
          n_steps: int = 1500, lr: float = 0.05, snapshot_every: int = 0):
    """Train W_K via SGD on episodic key/value retrieval loss.

    Returns:
        W_K        : final (d_key, d_key) projector
        history    : dict of per-step loss / cosine similarity
        snapshots  : list of (step, W_K_copy) tuples (empty if snapshot_every=0)
    """
    rng = np.random.default_rng(seed)

    # Init W_K close to identity. With pure identity + random Gaussian keys
    # in d=8, retrieval already works modestly; training pushes the
    # projected keys toward orthonormality which dramatically tightens
    # retrieval.
    W_K = np.eye(d_key) + 0.05 * rng.standard_normal((d_key, d_key))

    history = {"step": [], "loss": [], "cos": []}
    snapshots = []

    for step in range(n_steps):
        keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
        q_key = keys[q_idx]
        target_v = values[q_idx]

        y, W_fast, K, k_q = fast_weight_forward(W_K, keys, values, q_key)
        loss = 0.5 * float(np.sum((y - target_v) ** 2))
        cos = cosine_sim(y, target_v)
        history["step"].append(step)
        history["loss"].append(loss)
        history["cos"].append(cos)

        dW_K = fast_weight_backward(W_K, keys, values, q_key, target_v,
                                    y, W_fast, K, k_q)

        # Plain SGD with gradient norm clip at 1.0. The 1992 paper used
        # a bespoke fast-weight learning rule on a much larger architecture;
        # for a one-projector demo, vanilla SGD converges cleanly.
        gnorm = float(np.linalg.norm(dW_K))
        if gnorm > 1.0:
            dW_K = dW_K / gnorm
        W_K -= lr * dW_K

        if snapshot_every and (step % snapshot_every == 0 or step == n_steps - 1):
            snapshots.append((step, W_K.copy()))

    return W_K, history, snapshots


def evaluate(W_K: np.ndarray, seed: int, n_pairs: int, d_key: int,
             d_val: int, n_test: int = 200):
    """Evaluate on `n_test` fresh episodes. Returns mean cosine + per-episode
    list and the success rate at cosine > 0.9 / > 0.95 thresholds.
    """
    rng = np.random.default_rng(seed)
    cos_list = []
    err_list = []
    for _ in range(n_test):
        keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
        q_key = keys[q_idx]
        target_v = values[q_idx]
        y, *_ = fast_weight_forward(W_K, keys, values, q_key)
        cos_list.append(cosine_sim(y, target_v))
        err_list.append(float(np.linalg.norm(y - target_v)))
    cos_list = np.asarray(cos_list)
    err_list = np.asarray(err_list)
    return {
        "mean_cos": float(cos_list.mean()),
        "std_cos": float(cos_list.std()),
        "frac_cos_gt_0p9": float(np.mean(cos_list > 0.9)),
        "frac_cos_gt_0p95": float(np.mean(cos_list > 0.95)),
        "mean_err": float(err_list.mean()),
        "cos_list": cos_list.tolist(),
    }


def capacity_sweep(W_K: np.ndarray, seed: int, d_key: int, d_val: int,
                   max_pairs: int = 12, n_test: int = 100):
    """Without retraining, measure retrieval cosine as a function of the
    number of stored pairs. Demonstrates the classical capacity break:
    once N exceeds ~ d_key, retrieval drops sharply due to interference.
    """
    rng = np.random.default_rng(seed + 9999)
    out = []
    for n_pairs in range(1, max_pairs + 1):
        cos_vals = []
        for _ in range(n_test):
            keys, values, q_idx = generate_episode(rng, n_pairs, d_key, d_val)
            q_key = keys[q_idx]
            target_v = values[q_idx]
            y, *_ = fast_weight_forward(W_K, keys, values, q_key)
            cos_vals.append(cosine_sim(y, target_v))
        out.append({"n_pairs": n_pairs, "mean_cos": float(np.mean(cos_vals))})
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
    parser.add_argument("--capacity-sweep", action="store_true",
                        help="After training, sweep N=1..12 and report cosine.")
    parser.add_argument("--grad-check", action="store_true",
                        help="Run analytic-vs-numerical gradient check and exit.")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON results here.")
    args = parser.parse_args()

    if args.grad_check:
        err = numerical_grad_check(seed=args.seed)
        print(f"Max |analytic - numerical| dW_K = {err:.3e}")
        return

    t0 = time.time()
    W_K, history, _ = train(seed=args.seed, n_pairs=args.n_pairs,
                            d_key=args.d_key, d_val=args.d_val,
                            n_steps=args.n_steps, lr=args.lr)
    eval_seed = args.seed + 12345
    pre_W_K = np.eye(args.d_key)
    pre_eval = evaluate(pre_W_K, eval_seed, args.n_pairs, args.d_key,
                        args.d_val, n_test=args.n_test)
    post_eval = evaluate(W_K, eval_seed, args.n_pairs, args.d_key,
                         args.d_val, n_test=args.n_test)
    elapsed = time.time() - t0

    # Drop the cosine list from the on-screen JSON; keep it for --out.
    pre_summary = {k: v for k, v in pre_eval.items() if k != "cos_list"}
    post_summary = {k: v for k, v in post_eval.items() if k != "cos_list"}

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
        "pretraining_eval": pre_summary,
        "posttraining_eval": post_summary,
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
        result["capacity_sweep"] = capacity_sweep(
            W_K, eval_seed, args.d_key, args.d_val,
            max_pairs=12, n_test=100,
        )

    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            payload = {
                "args": vars(args),
                "result": result,
                "history": history,
                "pre_cos_list": pre_eval["cos_list"],
                "post_cos_list": post_eval["cos_list"],
            }
            json.dump(payload, f)


if __name__ == "__main__":
    main()
