"""
fast-weights-unknown-delay --- Schmidhuber, *Learning to control fast-weight
memories: an alternative to dynamic recurrent networks*, Neural Computation
4(1):131-139 (1992).

A pattern P is presented at step 0 with a store flag.  Some unknown number K
of distractor steps later, a recall flag fires and the network must output P.
The slow programmer net S (purely feedforward) only ever sees the current
input bits and the two control flags --- it has no recurrent state of its
own.  All memory lives in a fast weight matrix W_fast that S writes into and
reads from via key/value/query/gate heads:

    h_t       = tanh(W_xh x_t + b_h)
    k_t       = tanh(W_hk h_t + b_k)            # FROM-address (key)
    v_t       = tanh(W_hv h_t + b_v)            # TO-content   (value)
    q_t       = tanh(W_hq h_t + b_q)            # read query
    g_t       = sigmoid(w_hg h_t + b_g)         # write gate
    W_fast_t  = W_fast_{t-1} + eta * g_t * outer(v_t, k_t)
    y_t       = W_fast_t @ q_t                  # used only at recall

S has zero recurrent connections.  The only path that carries information
across the K-step gap is W_fast.  This is the 1992 setup that the 2021
*Linear Transformers are secretly fast weight programmers* paper later
reframed as unnormalized linear self-attention.

CLI
---

    python3 fast_weights_unknown_delay.py --seed 0
    python3 fast_weights_unknown_delay.py --seed 0 --d-min 5 --d-max 30

Single seed, default settings: ~30-50 s on an M-series laptop CPU.
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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


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


# ----------------------------------------------------------------------
# Slow programmer net S (feedforward, no recurrence)
# ----------------------------------------------------------------------

class SlowNet:
    """One hidden tanh layer; four output heads (key, value, query, gate).

    Inputs : x_t = [pattern_bits (P_dim), store_bit, recall_bit].
    Hidden : tanh, size H.
    Heads  : key k_t, value v_t, query q_t (all tanh), and write-gate g_t
             (sigmoid scalar).

    There is no recurrent connection inside S.  The only memory is W_fast.
    """

    def __init__(self,
                 p_dim: int = 4,
                 hidden: int = 32,
                 d_k: int = 8,
                 init_scale: float = 0.5,
                 rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        n_in = p_dim + 2
        # Hidden
        self.W_xh = rng.standard_normal((hidden, n_in)) * (init_scale / np.sqrt(n_in))
        self.b_h = np.zeros(hidden)
        # Heads
        self.W_hk = rng.standard_normal((d_k, hidden)) * (init_scale / np.sqrt(hidden))
        self.b_k = np.zeros(d_k)
        self.W_hv = rng.standard_normal((p_dim, hidden)) * (init_scale / np.sqrt(hidden))
        self.b_v = np.zeros(p_dim)
        self.W_hq = rng.standard_normal((d_k, hidden)) * (init_scale / np.sqrt(hidden))
        self.b_q = np.zeros(d_k)
        self.W_hg = rng.standard_normal((1, hidden)) * (init_scale / np.sqrt(hidden))
        self.b_g = np.zeros(1)
        self.p_dim = p_dim
        self.hidden = hidden
        self.d_k = d_k

    def params(self):
        return [self.W_xh, self.b_h,
                self.W_hk, self.b_k,
                self.W_hv, self.b_v,
                self.W_hq, self.b_q,
                self.W_hg, self.b_g]

    def param_names(self):
        return ["W_xh", "b_h",
                "W_hk", "b_k",
                "W_hv", "b_v",
                "W_hq", "b_q",
                "W_hg", "b_g"]

    def n_params(self) -> int:
        return sum(p.size for p in self.params())


# ----------------------------------------------------------------------
# Episode generation
# ----------------------------------------------------------------------

def make_batch(batch: int,
               p_dim: int,
               delay: int,
               rng: np.random.Generator):
    """Build a batch of episodes that all share the same delay length.

    Episode timeline (T = delay + 2 steps):
        t = 0          : pattern P, store_bit = 1, recall_bit = 0
        t = 1..delay   : random distractor pattern, both bits = 0
        t = delay + 1  : zeros for the pattern slot, recall_bit = 1

    Pattern P is drawn from {-1, +1}^p_dim per episode.

    Returns
    -------
    x         : (B, T, p_dim + 2) float
    target    : (B, p_dim) float --- the pattern P, repeated per episode
    recall_t  : int           --- the recall step (= T - 1)
    """
    T = delay + 2
    # Pattern
    P = rng.choice([-1.0, 1.0], size=(batch, p_dim))
    x = np.zeros((batch, T, p_dim + 2), dtype=np.float64)
    # Step 0: store
    x[:, 0, :p_dim] = P
    x[:, 0, p_dim] = 1.0
    # Steps 1..delay: distractors, no flags
    for t in range(1, delay + 1):
        d = rng.choice([-1.0, 1.0], size=(batch, p_dim))
        x[:, t, :p_dim] = d
    # Step delay+1: recall, pattern slot zero, recall flag
    x[:, delay + 1, p_dim + 1] = 1.0
    recall_t = T - 1
    return x, P, recall_t


# ----------------------------------------------------------------------
# Forward pass over one batch episode
# ----------------------------------------------------------------------

def forward_episode(S: SlowNet,
                    x: np.ndarray,
                    recall_t: int,
                    eta: float):
    """Run the slow net over the episode while accumulating W_fast.

    All hidden states are cached for backprop.

    Returns
    -------
    y          : (B, p_dim)        --- output at the recall step
    cache      : dict               --- inputs and per-step activations needed
                                        by `backward_episode`
    """
    B, T, _ = x.shape
    p_dim = S.p_dim
    d_k = S.d_k
    hidden = S.hidden

    h_seq = np.zeros((T, B, hidden))
    k_seq = np.zeros((T, B, d_k))
    v_seq = np.zeros((T, B, p_dim))
    q_seq = np.zeros((T, B, d_k))
    g_seq = np.zeros((T, B))

    W_fast = np.zeros((B, p_dim, d_k))     # one fast-weight matrix per episode
    Wfast_history = []                      # for later analysis (not for grad)

    for t in range(T):
        x_t = x[:, t, :]                    # (B, p_dim+2)
        h_t = np.tanh(x_t @ S.W_xh.T + S.b_h)
        k_t = np.tanh(h_t @ S.W_hk.T + S.b_k)
        v_t = np.tanh(h_t @ S.W_hv.T + S.b_v)
        q_t = np.tanh(h_t @ S.W_hq.T + S.b_q)
        g_t = sigmoid(h_t @ S.W_hg.T + S.b_g).reshape(B)        # (B,)

        # Fast-weight write: W_fast += eta * g * outer(v, k)
        W_fast = W_fast + eta * g_t[:, None, None] * v_t[:, :, None] * k_t[:, None, :]

        h_seq[t] = h_t
        k_seq[t] = k_t
        v_seq[t] = v_t
        q_seq[t] = q_t
        g_seq[t] = g_t
        Wfast_history.append(W_fast.copy())

    # Read at recall step using the W_fast accumulated so far.
    q_recall = q_seq[recall_t]                          # (B, d_k)
    y = np.einsum("bpk,bk->bp", W_fast, q_recall)       # (B, p_dim)

    cache = dict(
        x=x,
        h_seq=h_seq, k_seq=k_seq, v_seq=v_seq, q_seq=q_seq, g_seq=g_seq,
        W_fast_final=W_fast,
        Wfast_history=Wfast_history,
        recall_t=recall_t,
        eta=eta,
    )
    return y, cache


# ----------------------------------------------------------------------
# Backward pass
# ----------------------------------------------------------------------

def backward_episode(S: SlowNet,
                     cache: dict,
                     y: np.ndarray,
                     P: np.ndarray) -> dict:
    """Compute gradients of the recall-step MSE loss w.r.t. all params of S.

    Returns a dict mapping parameter name -> gradient (same shape as the
    parameter), averaged over the batch.
    """
    x = cache["x"]
    h_seq = cache["h_seq"]
    k_seq = cache["k_seq"]
    v_seq = cache["v_seq"]
    q_seq = cache["q_seq"]
    g_seq = cache["g_seq"]
    W_fast_T = cache["W_fast_final"]
    recall_t = cache["recall_t"]
    eta = cache["eta"]

    B, T, _ = x.shape
    p_dim = S.p_dim
    d_k = S.d_k

    # MSE: L = (1 / (B * p_dim)) * sum (y - P)^2
    dL_dy = 2.0 * (y - P) / (B * p_dim)                  # (B, p_dim)

    # y = einsum('bpk,bk->bp', W_fast_T, q_recall)
    # dL_dW_fast_T = outer(dL_dy, q_recall)
    q_recall = q_seq[recall_t]                           # (B, d_k)
    dW_fast = dL_dy[:, :, None] * q_recall[:, None, :]   # (B, p_dim, d_k)
    dq_recall = np.einsum("bpk,bp->bk", W_fast_T, dL_dy) # (B, d_k)

    # Param-gradient accumulators
    dW_xh = np.zeros_like(S.W_xh)
    db_h = np.zeros_like(S.b_h)
    dW_hk = np.zeros_like(S.W_hk)
    db_k = np.zeros_like(S.b_k)
    dW_hv = np.zeros_like(S.W_hv)
    db_v = np.zeros_like(S.b_v)
    dW_hq = np.zeros_like(S.W_hq)
    db_q = np.zeros_like(S.b_q)
    dW_hg = np.zeros_like(S.W_hg)
    db_g = np.zeros_like(S.b_g)

    # Walk every step and accumulate.
    for t in range(T):
        h_t = h_seq[t]                       # (B, H)
        k_t = k_seq[t]                       # (B, d_k)
        v_t = v_seq[t]                       # (B, p_dim)
        q_t = q_seq[t]                       # (B, d_k)
        g_t = g_seq[t]                       # (B,)
        x_t = x[:, t, :]                     # (B, p_dim+2)

        # Derivatives flowing in from W_fast_T = sum_t eta * g_t * outer(v_t, k_t)
        # dL/dg_t = eta * sum_{i,j} dW_fast[i,j] * v_t[i] * k_t[j]
        dg_t = eta * np.einsum("bpk,bp,bk->b", dW_fast, v_t, k_t)
        # dL/dv_t = eta * g_t * dW_fast @ k_t
        dv_t = eta * g_t[:, None] * np.einsum("bpk,bk->bp", dW_fast, k_t)
        # dL/dk_t = eta * g_t * dW_fast.T @ v_t
        dk_t = eta * g_t[:, None] * np.einsum("bpk,bp->bk", dW_fast, v_t)

        # Query head: only the recall step contributes to L through q_recall.
        dq_t = dq_recall if t == recall_t else np.zeros_like(q_t)

        # Through tanh / sigmoid
        dpre_k = dk_t * (1.0 - k_t * k_t)
        dpre_v = dv_t * (1.0 - v_t * v_t)
        dpre_q = dq_t * (1.0 - q_t * q_t)
        dpre_g = (dg_t * g_t * (1.0 - g_t)).reshape(B, 1)    # (B, 1)

        # Head -> hidden
        dh_t = (dpre_k @ S.W_hk
                + dpre_v @ S.W_hv
                + dpre_q @ S.W_hq
                + dpre_g @ S.W_hg)                            # (B, hidden)

        # Hidden tanh
        dpre_h = dh_t * (1.0 - h_t * h_t)

        # Accumulate parameter gradients (averaged-over-batch is already in dL_dy)
        dW_hk += dpre_k.T @ h_t
        db_k += dpre_k.sum(axis=0)
        dW_hv += dpre_v.T @ h_t
        db_v += dpre_v.sum(axis=0)
        dW_hq += dpre_q.T @ h_t
        db_q += dpre_q.sum(axis=0)
        dW_hg += dpre_g.T @ h_t
        db_g += dpre_g.sum(axis=0)
        dW_xh += dpre_h.T @ x_t
        db_h += dpre_h.sum(axis=0)

    grads = {
        "W_xh": dW_xh, "b_h": db_h,
        "W_hk": dW_hk, "b_k": db_k,
        "W_hv": dW_hv, "b_v": db_v,
        "W_hq": dW_hq, "b_q": db_q,
        "W_hg": dW_hg, "b_g": db_g,
    }
    return grads


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params, names, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = {n: np.zeros_like(p) for p, n in zip(params, names)}
        self.v = {n: np.zeros_like(p) for p, n in zip(params, names)}

    def step(self, params, names, grads, clip: float | None = None):
        # Optional gradient clipping by global norm.
        if clip is not None:
            total = 0.0
            for n in names:
                total += float((grads[n] ** 2).sum())
            norm = np.sqrt(total)
            if norm > clip:
                scale = clip / (norm + 1e-12)
                for n in names:
                    grads[n] = grads[n] * scale
        self.t += 1
        for p, n in zip(params, names):
            g = grads[n]
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * (g * g)
            m_hat = self.m[n] / (1 - self.b1 ** self.t)
            v_hat = self.v[n] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(seed: int = 0,
          p_dim: int = 4,
          hidden: int = 32,
          d_k: int = 8,
          eta: float = 0.5,
          d_min: int = 5,
          d_max: int = 30,
          batch: int = 32,
          iters: int = 3000,
          lr: float = 1e-2,
          clip: float = 1.0,
          log_every: int = 50,
          snapshot_every: int = 0,
          verbose: bool = True):
    """Train S; return (S, history, snapshots)."""
    rng = np.random.default_rng(seed)
    S = SlowNet(p_dim=p_dim, hidden=hidden, d_k=d_k, rng=rng)
    opt = Adam(S.params(), S.param_names(), lr=lr)

    history = dict(step=[], loss=[], bit_acc=[], delay=[],
                   wallclock=[], eta=eta, p_dim=p_dim, hidden=hidden,
                   d_k=d_k, batch=batch, lr=lr, d_min=d_min, d_max=d_max,
                   seed=seed)
    snapshots = []
    t0 = time.time()

    for it in range(iters):
        delay = int(rng.integers(d_min, d_max + 1))
        x, P, recall_t = make_batch(batch, p_dim, delay, rng)
        y, cache = forward_episode(S, x, recall_t, eta)

        loss = float(((y - P) ** 2).mean())
        bit_acc = float(((np.sign(y) == np.sign(P)) | (P == 0)).mean())

        grads = backward_episode(S, cache, y, P)
        opt.step(S.params(), S.param_names(), grads, clip=clip)

        if it % log_every == 0 or it == iters - 1:
            history["step"].append(it)
            history["loss"].append(loss)
            history["bit_acc"].append(bit_acc)
            history["delay"].append(delay)
            history["wallclock"].append(time.time() - t0)
            if verbose:
                print(f"  step {it:5d}  delay={delay:3d}  loss={loss:.5f}  "
                      f"bit_acc={bit_acc:.3f}  ({time.time() - t0:.1f}s)")

        if snapshot_every and (it % snapshot_every == 0 or it == iters - 1):
            snapshots.append(dict(
                step=it,
                params={n: p.copy() for p, n in zip(S.params(), S.param_names())},
            ))

    return S, history, snapshots


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(S: SlowNet,
             p_dim: int,
             eta: float,
             d_min: int,
             d_max: int,
             n_episodes_per_delay: int = 50,
             seed: int = 12345) -> dict:
    """Eval bit-accuracy and per-bit MSE across the full delay range."""
    rng = np.random.default_rng(seed)
    out = dict(delays=[], bit_acc=[], mse=[])
    for delay in range(d_min, d_max + 1):
        x, P, recall_t = make_batch(n_episodes_per_delay, p_dim, delay, rng)
        y, _ = forward_episode(S, x, recall_t, eta)
        bit_acc = float((np.sign(y) == np.sign(P)).mean())
        mse = float(((y - P) ** 2).mean())
        out["delays"].append(delay)
        out["bit_acc"].append(bit_acc)
        out["mse"].append(mse)
    out["mean_bit_acc"] = float(np.mean(out["bit_acc"]))
    out["mean_mse"] = float(np.mean(out["mse"]))
    return out


# ----------------------------------------------------------------------
# Numerical-gradient sanity check
# ----------------------------------------------------------------------

def numerical_gradcheck(seed: int = 0, eps: float = 1e-5,
                        p_dim: int = 3, hidden: int = 5, d_k: int = 4,
                        delay: int = 4, batch: int = 2,
                        eta: float = 0.5,
                        verbose: bool = True) -> float:
    """Compare analytical to numerical gradient on a single random batch.

    Returns the maximum relative error across all params.
    """
    rng = np.random.default_rng(seed)
    S = SlowNet(p_dim=p_dim, hidden=hidden, d_k=d_k, rng=rng)
    x, P, recall_t = make_batch(batch, p_dim, delay, rng)

    y, cache = forward_episode(S, x, recall_t, eta)
    grads = backward_episode(S, cache, y, P)

    max_rel = 0.0
    for p, n in zip(S.params(), S.param_names()):
        flat = p.reshape(-1)
        gflat = grads[n].reshape(-1)
        # Probe up to 8 indices per param.
        idx_list = list(range(min(8, flat.size)))
        for i in idx_list:
            orig = flat[i]
            flat[i] = orig + eps
            y_p, _ = forward_episode(S, x, recall_t, eta)
            l_p = float(((y_p - P) ** 2).mean())
            flat[i] = orig - eps
            y_m, _ = forward_episode(S, x, recall_t, eta)
            l_m = float(((y_m - P) ** 2).mean())
            flat[i] = orig
            num = (l_p - l_m) / (2 * eps)
            ana = float(gflat[i])
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
    if verbose:
        print(f"  gradcheck max_rel = {max_rel:.2e}  "
              f"(should be < 1e-4)")
    return max_rel


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--d-k", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--d-min", type=int, default=5)
    p.add_argument("--d-max", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--gradcheck", action="store_true",
                   help="Run a small numerical gradient check and exit.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gradcheck:
        rel = numerical_gradcheck(seed=args.seed)
        sys.exit(0 if rel < 1e-4 else 1)

    print(f"fast-weights-unknown-delay  seed={args.seed}  "
          f"p_dim={args.p_dim}  hidden={args.hidden}  d_k={args.d_k}  "
          f"eta={args.eta}  delay~Uniform[{args.d_min},{args.d_max}]")
    print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"git {git_hash()}  {platform.platform()}")
    print()

    S, history, _ = train(
        seed=args.seed,
        p_dim=args.p_dim,
        hidden=args.hidden,
        d_k=args.d_k,
        eta=args.eta,
        d_min=args.d_min,
        d_max=args.d_max,
        batch=args.batch,
        iters=args.iters,
        lr=args.lr,
        clip=args.clip,
        log_every=args.log_every,
        verbose=not args.quiet,
    )

    print()
    print("Eval (50 episodes per delay across full range):")
    eval_out = evaluate(S, args.p_dim, args.eta, args.d_min, args.d_max)
    print(f"  mean bit-accuracy : {eval_out['mean_bit_acc']*100:.1f}%")
    print(f"  mean MSE          : {eval_out['mean_mse']:.5f}")
    # Print per-delay accuracy histogram.
    print("  per-delay bit-accuracy:")
    for d, acc in zip(eval_out["delays"], eval_out["bit_acc"]):
        bar = "#" * int(round(acc * 30))
        print(f"    K={d:3d}  {acc*100:5.1f}%  {bar}")

    summary = dict(
        final_loss=float(history["loss"][-1]),
        final_bit_acc=float(history["bit_acc"][-1]),
        eval_mean_bit_acc=eval_out["mean_bit_acc"],
        eval_mean_mse=eval_out["mean_mse"],
        wallclock_s=float(history["wallclock"][-1]),
        n_params=int(S.n_params()),
        config=dict(
            seed=args.seed, p_dim=args.p_dim, hidden=args.hidden,
            d_k=args.d_k, eta=args.eta, d_min=args.d_min, d_max=args.d_max,
            batch=args.batch, iters=args.iters, lr=args.lr, clip=args.clip,
        ),
        env=dict(
            python=sys.version.split()[0], numpy=np.__version__,
            platform=platform.platform(), git=git_hash(),
        ),
    )
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
