"""
adding-problem — Hochreiter & Schmidhuber 1997, "Long Short-Term Memory",
Neural Computation 9(8):1735-1780, Experiment 4 ("the adding problem").

Problem:
  Each timestep is a 2-D input. Channel 0 is a random real in [-1, 1].
  Channel 1 is a marker. Exactly two markers are 1.0 in the sequence:
  one in the first half, one in the second half; all others are 0.0.
  Target at the final step is the sum of the two marked channel-0 values.
  Loss: MSE.

  This is the canonical long-time-lag temporal indexing benchmark —
  the network must remember the first marked value across hundreds of
  irrelevant timesteps and add it to the second marked value.

Architecture:
  Standard LSTM cell with forget gate (Gers, Schmidhuber, Cummins 2000),
  small hidden size (8 by default; paper used 2-8). BPTT, fully manual
  (numpy only). Trained with Adam.

  The 1997 paper used the original "vanilla" LSTM cell *without* a forget
  gate (the constant error carrousel had c_t = c_{t-1} + i_t * g_t).
  We use the more common modern variant with forget gate; this is
  documented in §Deviations of README.md. The forget-gate bias is
  initialized to 1.0, which biases the cell toward "remember by default"
  early in training and is the standard recipe for long-lag tasks.

  A vanilla-RNN baseline (same hidden size, same optimizer) is included
  to demonstrate gradient vanishing — it never converges past chance MSE.

CLI:
  python3 adding_problem.py --seed 0 --T 100 --hidden 8
  python3 adding_problem.py --seed 0 --T 100 --hidden 8 --rnn   # baseline
  python3 adding_problem.py --gradcheck                         # numerical check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def make_adding_batch(rng: np.random.RandomState, T: int,
                      batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X, y for a batch of adding-problem sequences.

    X: (T, B, 2)  -- channel 0 in [-1,1], channel 1 in {0, 1}
    y: (B,)       -- sum of the two marked channel-0 values
    """
    X = np.zeros((T, batch_size, 2), dtype=np.float64)
    X[:, :, 0] = rng.uniform(-1.0, 1.0, size=(T, batch_size))
    half = T // 2
    pos1 = rng.randint(0, half, size=batch_size)
    pos2 = rng.randint(half, T, size=batch_size)
    b_idx = np.arange(batch_size)
    X[pos1, b_idx, 1] = 1.0
    X[pos2, b_idx, 1] = 1.0
    y = X[pos1, b_idx, 0] + X[pos2, b_idx, 0]
    return X, y.astype(np.float64)


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def dsigmoid_from_y(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


def dtanh_from_y(y: np.ndarray) -> np.ndarray:
    return 1.0 - y * y


# ----------------------------------------------------------------------
# LSTM with forget gate, manual BPTT
# ----------------------------------------------------------------------

@dataclass
class LSTMParams:
    Wx: np.ndarray  # (input_dim, 4H)  gate order: i, f, g, o
    Wh: np.ndarray  # (H, 4H)
    b: np.ndarray   # (4H,)
    Wy: np.ndarray  # (H, 1)
    by: np.ndarray  # (1,)

    def keys(self):
        return ["Wx", "Wh", "b", "Wy", "by"]

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_lstm(input_dim: int, H: int, rng: np.random.RandomState) -> LSTMParams:
    scale_x = 1.0 / math.sqrt(input_dim)
    scale_h = 1.0 / math.sqrt(H)
    Wx = rng.randn(input_dim, 4 * H) * scale_x * 0.5
    Wh = rng.randn(H, 4 * H) * scale_h * 0.5
    b = np.zeros(4 * H)
    # forget-gate bias = 1.0 (Gers/Schmidhuber/Cummins recipe for long-lag)
    b[H:2 * H] = 1.0
    Wy = rng.randn(H, 1) * (1.0 / math.sqrt(H))
    by = np.zeros(1)
    return LSTMParams(Wx=Wx, Wh=Wh, b=b, Wy=Wy, by=by)


def lstm_forward(p: LSTMParams, X: np.ndarray):
    """Run LSTM over the sequence and return final prediction + cache.

    X: (T, B, input_dim)
    pred: (B,)
    cache: dict of arrays for BPTT
    """
    T, B, D = X.shape
    H = p.Wh.shape[0]
    h = np.zeros((T + 1, B, H))
    c = np.zeros((T + 1, B, H))
    i_g = np.zeros((T, B, H))
    f_g = np.zeros((T, B, H))
    g_g = np.zeros((T, B, H))
    o_g = np.zeros((T, B, H))
    tc = np.zeros((T, B, H))  # tanh(c_t)
    for t in range(T):
        z = X[t] @ p.Wx + h[t] @ p.Wh + p.b  # (B, 4H)
        i_g[t] = sigmoid(z[:, 0:H])
        f_g[t] = sigmoid(z[:, H:2 * H])
        g_g[t] = np.tanh(z[:, 2 * H:3 * H])
        o_g[t] = sigmoid(z[:, 3 * H:4 * H])
        c[t + 1] = f_g[t] * c[t] + i_g[t] * g_g[t]
        tc[t] = np.tanh(c[t + 1])
        h[t + 1] = o_g[t] * tc[t]
    pred = (h[T] @ p.Wy + p.by).reshape(B)
    cache = dict(X=X, h=h, c=c, i=i_g, f=f_g, g=g_g, o=o_g, tc=tc, pred=pred)
    return pred, cache


def lstm_backward(p: LSTMParams, cache: dict, dpred: np.ndarray):
    """Backprop given dL/dpred. Returns dict of grads matching p.keys()."""
    X = cache["X"]
    h = cache["h"]
    c = cache["c"]
    i_g = cache["i"]
    f_g = cache["f"]
    g_g = cache["g"]
    o_g = cache["o"]
    tc = cache["tc"]
    T, B, D = X.shape
    H = p.Wh.shape[0]

    grads = {k: np.zeros_like(p.get(k)) for k in p.keys()}

    # Output: pred = h[T] @ Wy + by
    dpred_col = dpred.reshape(B, 1)
    grads["Wy"] = h[T].T @ dpred_col
    grads["by"] = dpred_col.sum(axis=0)
    dh_next = dpred_col @ p.Wy.T  # (B, H)
    dc_next = np.zeros((B, H))

    for t in reversed(range(T)):
        dh = dh_next  # (B, H)
        # h_t = o_t * tanh(c_t)
        do_t = dh * tc[t]
        dtc_t = dh * o_g[t]
        dc = dc_next + dtc_t * dtanh_from_y(tc[t])
        # c_t = f_t * c_{t-1} + i_t * g_t
        df_t = dc * c[t]
        dc_prev = dc * f_g[t]
        di_t = dc * g_g[t]
        dg_t = dc * i_g[t]
        # gate pre-activations
        dz_i = di_t * dsigmoid_from_y(i_g[t])
        dz_f = df_t * dsigmoid_from_y(f_g[t])
        dz_g = dg_t * dtanh_from_y(g_g[t])
        dz_o = do_t * dsigmoid_from_y(o_g[t])
        dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)  # (B, 4H)
        grads["Wx"] += X[t].T @ dz
        grads["Wh"] += h[t].T @ dz
        grads["b"] += dz.sum(axis=0)
        dh_next = dz @ p.Wh.T
        dc_next = dc_prev

    return grads


# ----------------------------------------------------------------------
# Vanilla RNN baseline (same shape, simple recurrent cell)
# ----------------------------------------------------------------------

@dataclass
class RNNParams:
    Wx: np.ndarray  # (input_dim, H)
    Wh: np.ndarray  # (H, H)
    b: np.ndarray   # (H,)
    Wy: np.ndarray  # (H, 1)
    by: np.ndarray  # (1,)

    def keys(self):
        return ["Wx", "Wh", "b", "Wy", "by"]

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_rnn(input_dim: int, H: int, rng: np.random.RandomState) -> RNNParams:
    Wx = rng.randn(input_dim, H) * (1.0 / math.sqrt(input_dim)) * 0.5
    Wh = rng.randn(H, H) * (1.0 / math.sqrt(H)) * 0.5
    b = np.zeros(H)
    Wy = rng.randn(H, 1) * (1.0 / math.sqrt(H))
    by = np.zeros(1)
    return RNNParams(Wx=Wx, Wh=Wh, b=b, Wy=Wy, by=by)


def rnn_forward(p: RNNParams, X: np.ndarray):
    T, B, D = X.shape
    H = p.Wh.shape[0]
    h = np.zeros((T + 1, B, H))
    for t in range(T):
        h[t + 1] = np.tanh(X[t] @ p.Wx + h[t] @ p.Wh + p.b)
    pred = (h[T] @ p.Wy + p.by).reshape(B)
    cache = dict(X=X, h=h, pred=pred)
    return pred, cache


def rnn_backward(p: RNNParams, cache: dict, dpred: np.ndarray):
    X = cache["X"]
    h = cache["h"]
    T, B, D = X.shape
    H = p.Wh.shape[0]
    grads = {k: np.zeros_like(p.get(k)) for k in p.keys()}
    dpred_col = dpred.reshape(B, 1)
    grads["Wy"] = h[T].T @ dpred_col
    grads["by"] = dpred_col.sum(axis=0)
    dh_next = dpred_col @ p.Wy.T  # (B, H)
    for t in reversed(range(T)):
        dz = dh_next * dtanh_from_y(h[t + 1])
        grads["Wx"] += X[t].T @ dz
        grads["Wh"] += h[t].T @ dz
        grads["b"] += dz.sum(axis=0)
        dh_next = dz @ p.Wh.T
    return grads


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=5e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip = clip
        self.t = 0
        self.m = {k: np.zeros_like(params.get(k)) for k in params.keys()}
        self.v = {k: np.zeros_like(params.get(k)) for k in params.keys()}

    def step(self, params, grads):
        # global gradient clipping by L2 norm
        if self.clip is not None:
            total = math.sqrt(sum(float((grads[k] ** 2).sum())
                                  for k in grads))
            if total > self.clip:
                scale = self.clip / (total + 1e-12)
                for k in grads:
                    grads[k] = grads[k] * scale
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for k in params.keys():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params.set(k, params.get(k) - self.lr * m_hat
                       / (np.sqrt(v_hat) + self.eps))


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

@dataclass
class TrainHistory:
    iters: list = field(default_factory=list)
    train_mse: list = field(default_factory=list)
    test_mse: list = field(default_factory=list)
    solve_rate: list = field(default_factory=list)  # |err| < 0.04
    sequences_seen: list = field(default_factory=list)

    def to_dict(self):
        return {k: list(v) for k, v in self.__dict__.items()}


def evaluate(params, forward_fn, rng, T, n_test=512, batch_size=128,
             tol=0.04):
    sse = 0.0
    n_correct = 0
    n_total = 0
    for _ in range(0, n_test, batch_size):
        b = min(batch_size, n_test - n_total)
        X, y = make_adding_batch(rng, T, b)
        pred, _ = forward_fn(params, X)
        sse += float(((pred - y) ** 2).sum())
        n_correct += int((np.abs(pred - y) < tol).sum())
        n_total += b
    return sse / n_total, n_correct / n_total


def train(model: str, T: int, hidden: int, seed: int, n_iters: int,
          batch_size: int, lr: float, eval_every: int,
          lr_decay_every: int = 1500, lr_decay_factor: float = 0.5,
          verbose: bool = True, save_snapshots: bool = False):
    """Train LSTM or vanilla RNN. Returns (params, history, snapshots).

    LR is multiplied by `lr_decay_factor` every `lr_decay_every` iterations.
    Set lr_decay_every=0 to disable decay.
    """
    train_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed + 1_000_003)
    init_rng = np.random.RandomState(seed + 7)

    if model == "lstm":
        params = init_lstm(input_dim=2, H=hidden, rng=init_rng)
        forward = lstm_forward
        backward = lstm_backward
    elif model == "rnn":
        params = init_rnn(input_dim=2, H=hidden, rng=init_rng)
        forward = rnn_forward
        backward = rnn_backward
    else:
        raise ValueError(f"unknown model: {model}")

    opt = Adam(params, lr=lr, clip=1.0)
    history = TrainHistory()
    snapshots = []  # list of (iter, params_dict, sample_inputs)
    sequences_seen = 0
    t0 = time.time()
    last_train_mse = float("nan")

    for it in range(1, n_iters + 1):
        if lr_decay_every and it > 1 and (it - 1) % lr_decay_every == 0:
            opt.lr *= lr_decay_factor
        X, y = make_adding_batch(train_rng, T, batch_size)
        pred, cache = forward(params, X)
        err = pred - y
        loss = 0.5 * float((err * err).mean())
        last_train_mse = 2.0 * loss  # MSE = 2 * 0.5 * mean(err^2)
        # dL/dpred = (pred - y) / B  (since loss is mean, not sum)
        dpred = err / batch_size
        grads = backward(params, cache, dpred)
        opt.step(params, grads)
        sequences_seen += batch_size

        if it == 1 or it % eval_every == 0 or it == n_iters:
            test_mse, solve = evaluate(params, forward, test_rng, T,
                                       n_test=512, batch_size=128)
            history.iters.append(it)
            history.train_mse.append(last_train_mse)
            history.test_mse.append(test_mse)
            history.solve_rate.append(solve)
            history.sequences_seen.append(sequences_seen)
            if verbose:
                el = time.time() - t0
                print(f"  iter {it:5d}  seq {sequences_seen:7d}  "
                      f"train_mse {last_train_mse:.4f}  "
                      f"test_mse {test_mse:.4f}  "
                      f"solve_rate {solve:.3f}  "
                      f"({el:.1f}s)")
            if save_snapshots:
                # Snapshot for GIF: copy params + a fixed test sample
                snap_rng = np.random.RandomState(seed + 99)
                Xs, ys = make_adding_batch(snap_rng, T, 4)
                preds, snap_cache = forward(params, Xs)
                snapshot = dict(
                    iter=it,
                    sequences=sequences_seen,
                    train_mse=last_train_mse,
                    test_mse=test_mse,
                    solve_rate=solve,
                    Xs=Xs.copy(),
                    ys=ys.copy(),
                    preds=preds.copy(),
                )
                if model == "lstm":
                    snapshot["c"] = snap_cache["c"].copy()
                    snapshot["h"] = snap_cache["h"].copy()
                else:
                    snapshot["h"] = snap_cache["h"].copy()
                snapshots.append(snapshot)

    return params, history, snapshots


# ----------------------------------------------------------------------
# Numerical gradient check (for confidence in manual BPTT)
# ----------------------------------------------------------------------

def gradcheck(model: str = "lstm", T: int = 8, H: int = 4, B: int = 3,
              seed: int = 0, eps: float = 1e-5, n_checks: int = 20):
    rng = np.random.RandomState(seed)
    if model == "lstm":
        params = init_lstm(2, H, rng)
        forward = lstm_forward
        backward = lstm_backward
    else:
        params = init_rnn(2, H, rng)
        forward = rnn_forward
        backward = rnn_backward
    X, y = make_adding_batch(rng, T, B)
    pred, cache = forward(params, X)
    err = pred - y
    dpred = err / B
    grads = backward(params, cache, dpred)

    def total_loss(p):
        pr, _ = forward(p, X)
        return 0.5 * float(((pr - y) ** 2).mean())

    rel_errs = []
    check_rng = np.random.RandomState(seed + 1)
    for k in params.keys():
        W = params.get(k)
        flat = W.reshape(-1)
        analytic = grads[k].reshape(-1)
        idxs = check_rng.choice(flat.size,
                                size=min(n_checks, flat.size),
                                replace=False)
        for i in idxs:
            saved = flat[i]
            flat[i] = saved + eps
            lp = total_loss(params)
            flat[i] = saved - eps
            lm = total_loss(params)
            flat[i] = saved
            num = (lp - lm) / (2 * eps)
            an = analytic[i]
            denom = max(1e-12, abs(num) + abs(an))
            rel = abs(num - an) / denom
            rel_errs.append((k, i, num, an, rel))
    max_rel = max(r[-1] for r in rel_errs)
    print(f"[{model}] gradcheck: max relative error = {max_rel:.2e} "
          f"over {len(rel_errs)} samples")
    return max_rel


# ----------------------------------------------------------------------
# Reproducibility metadata
# ----------------------------------------------------------------------

def env_info():
    import platform
    import sys
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=100, help="sequence length")
    ap.add_argument("--hidden", type=int, default=8, help="hidden units")
    ap.add_argument("--iters", type=int, default=3000, help="training iters")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-decay-every", type=int, default=1500,
                    help="halve LR every N iters (0 disables)")
    ap.add_argument("--lr-decay-factor", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--rnn", action="store_true",
                    help="train vanilla RNN baseline instead of LSTM")
    ap.add_argument("--gradcheck", action="store_true",
                    help="run numerical gradient check and exit")
    ap.add_argument("--save-history", type=str, default=None,
                    help="path to write training history JSON")
    args = ap.parse_args()

    if args.gradcheck:
        for m in ("lstm", "rnn"):
            gradcheck(model=m)
        return

    model = "rnn" if args.rnn else "lstm"
    print(f"[{model}] T={args.T} hidden={args.hidden} batch={args.batch} "
          f"lr={args.lr} iters={args.iters} seed={args.seed}")
    print(f"  env: {env_info()}")
    t0 = time.time()
    params, history, _ = train(
        model=model, T=args.T, hidden=args.hidden, seed=args.seed,
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
        eval_every=args.eval_every,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
    )
    elapsed = time.time() - t0
    final_mse = history.test_mse[-1]
    final_solve = history.solve_rate[-1]
    print(f"[{model}] final test MSE = {final_mse:.4f}  "
          f"solve_rate = {final_solve:.3f}  "
          f"({elapsed:.1f}s, {history.sequences_seen[-1]} sequences)")
    if args.save_history:
        out = {
            "model": model,
            "args": vars(args),
            "env": env_info(),
            "history": history.to_dict(),
            "elapsed_sec": elapsed,
        }
        with open(args.save_history, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.save_history}")


if __name__ == "__main__":
    main()
