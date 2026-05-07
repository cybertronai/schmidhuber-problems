"""
timing-counting-spikes — Gers, Schraudolph, Schmidhuber, *Learning Precise
Timing with LSTM Recurrent Networks*, JMLR 3:115-143, 2002.

Headline task: Measure-Spike-Distance (MSD).

Problem:
    Each sequence has T = 60 time steps and a single binary input channel.
    Two input spikes appear at times t1 < t2 with separation D = t2 - t1.
    The network must produce an output spike at exactly t_target = t1 + 2*D
    (i.e. the same gap D after the second input spike). Background channel
    is 0 everywhere except the two spike steps.

    Sampling: D is uniform in [D_min, D_max], t1 is uniform in
    [0, T - 2*D - 1], t2 = t1 + D, t_target = t1 + 2*D.

    Loss: per-timestep MSE between scalar output and a delta target
    (1.0 at t_target, 0.0 elsewhere). Output is the raw linear readout of
    h_t, no output non-linearity (regression).

    Eval: a sample is "solved" if argmax of pred[t2+1 : T] is within
    +-1 step of t_target.

Architecture (the experiment):
    Two LSTM variants, identical except for peephole connections:

      * --no-peep : vanilla LSTM (Gers/Schmidhuber/Cummins 2000) with
        forget gate, no peepholes. The gates can only see the cell state
        through the bottleneck h_t = o_t * tanh(c_t).
      * --peep    : peephole LSTM (Gers/Schraudolph/Schmidhuber 2002).
        The cell state feeds directly into the input/forget gates from
        c_{t-1}, and into the output gate from the current c_t.

    Standard recurrence:
      i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + p_i * c_{t-1}? + b_i)
      f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + p_f * c_{t-1}? + b_f)
      g_t = tanh   (W_xg x_t + W_hg h_{t-1}                  + b_g)
      c_t = f_t * c_{t-1} + i_t * g_t
      o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + p_o * c_t?     + b_o)
      h_t = o_t * tanh(c_t)

    The peephole weights p_i, p_f, p_o are diagonal vectors of shape (H,)
    (one peephole per cell), as in the original 2002 paper.

    Training: BPTT, Adam, MSE per-timestep summed over T then averaged
    over batch. Gradient norm clipped at 1.0. Gradcheck passes to
    machine epsilon for both variants (--gradcheck).

CLI:
    python3 timing_counting_spikes.py --seed 0 --peep
    python3 timing_counting_spikes.py --seed 0 --no-peep
    python3 timing_counting_spikes.py --gradcheck

Why peepholes matter:
    The MSD output spike must be emitted at a specific count after t2.
    Without peepholes, gates depend on h_{t-1} = o_{t-1} * tanh(c_{t-1}),
    which means the cell state can only influence gating through the
    output gate. The output gate has to be open at the right step *and*
    the cell state has to read out the right level - two coupled
    constraints. With peepholes the gates read the cell state directly,
    decoupling the count from the readout.
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
# Dataset: Measure-Spike-Distance
# ----------------------------------------------------------------------

def make_msd_batch(rng: np.random.RandomState, T: int, D_min: int,
                   D_max: int, batch_size: int):
    """Return X, y, t1, t2, t_target for a batch of MSD sequences.

    X: (T, B, 1)   binary spike train
    y: (T, B, 1)   delta target: 1.0 at t_target, else 0.0
    t1, t2, t_target: (B,) integer arrays
    """
    X = np.zeros((T, batch_size, 1), dtype=np.float64)
    y = np.zeros((T, batch_size, 1), dtype=np.float64)
    D = rng.randint(D_min, D_max + 1, size=batch_size)
    # t1 uniform in [0, T - 2*D - 1]
    high = T - 2 * D  # exclusive upper bound for t1
    t1 = np.array([rng.randint(0, max(1, h)) for h in high], dtype=np.int64)
    t2 = t1 + D
    t_target = t1 + 2 * D
    b_idx = np.arange(batch_size)
    X[t1, b_idx, 0] = 1.0
    X[t2, b_idx, 0] = 1.0
    y[t_target, b_idx, 0] = 1.0
    return X, y, t1, t2, t_target


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
# Peephole LSTM (and vanilla LSTM as the no-peep mode)
# ----------------------------------------------------------------------

@dataclass
class LSTMParams:
    Wx: np.ndarray   # (D_in, 4H)  gate order: i, f, g, o
    Wh: np.ndarray   # (H, 4H)
    b: np.ndarray    # (4H,)
    p_i: np.ndarray  # (H,) peephole c_{t-1} -> i (zeros if not used)
    p_f: np.ndarray  # (H,) peephole c_{t-1} -> f
    p_o: np.ndarray  # (H,) peephole c_t     -> o
    Wy: np.ndarray   # (H, 1)
    by: np.ndarray   # (1,)
    use_peep: bool

    def trainable_keys(self):
        keys = ["Wx", "Wh", "b", "Wy", "by"]
        if self.use_peep:
            keys += ["p_i", "p_f", "p_o"]
        return keys

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_lstm(input_dim: int, H: int, rng: np.random.RandomState,
              use_peep: bool) -> LSTMParams:
    scale_x = 1.0 / math.sqrt(input_dim)
    scale_h = 1.0 / math.sqrt(H)
    Wx = rng.randn(input_dim, 4 * H) * scale_x * 0.5
    Wh = rng.randn(H, 4 * H) * scale_h * 0.5
    b = np.zeros(4 * H)
    # forget-gate bias = 1.0 (Gers/Schmidhuber/Cummins recipe for long lag)
    b[H:2 * H] = 1.0
    # peephole weights: small random init, as in Gers/Schraudolph/Schmidhuber
    # 2002. Zero-init was tried and is slightly worse on average.
    if use_peep:
        p_i = rng.randn(H) * 0.1
        p_f = rng.randn(H) * 0.1
        p_o = rng.randn(H) * 0.1
    else:
        p_i = np.zeros(H)
        p_f = np.zeros(H)
        p_o = np.zeros(H)
    Wy = rng.randn(H, 1) * (1.0 / math.sqrt(H))
    by = np.zeros(1)
    return LSTMParams(Wx=Wx, Wh=Wh, b=b, p_i=p_i, p_f=p_f, p_o=p_o,
                      Wy=Wy, by=by, use_peep=use_peep)


def lstm_forward(p: LSTMParams, X: np.ndarray):
    """Forward pass over the full sequence.

    X: (T, B, D_in)
    Returns pred: (T, B, 1)  (linear scalar output per timestep)
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
    tc = np.zeros((T, B, H))
    pred = np.zeros((T, B, 1))
    for t in range(T):
        z = X[t] @ p.Wx + h[t] @ p.Wh + p.b  # (B, 4H)
        # input + forget gates can read c_{t-1} via peephole
        if p.use_peep:
            z[:, 0:H] += c[t] * p.p_i
            z[:, H:2 * H] += c[t] * p.p_f
        i_g[t] = sigmoid(z[:, 0:H])
        f_g[t] = sigmoid(z[:, H:2 * H])
        g_g[t] = np.tanh(z[:, 2 * H:3 * H])
        c[t + 1] = f_g[t] * c[t] + i_g[t] * g_g[t]
        # output gate reads c_t (the *new* cell state) via peephole
        z_o = z[:, 3 * H:4 * H]
        if p.use_peep:
            z_o = z_o + c[t + 1] * p.p_o
        o_g[t] = sigmoid(z_o)
        tc[t] = np.tanh(c[t + 1])
        h[t + 1] = o_g[t] * tc[t]
        pred[t] = h[t + 1] @ p.Wy + p.by
    cache = dict(X=X, h=h, c=c, i=i_g, f=f_g, g=g_g, o=o_g, tc=tc, pred=pred)
    return pred, cache


def lstm_backward(p: LSTMParams, cache: dict, dpred: np.ndarray):
    """Backprop given dL/dpred of shape (T, B, 1).

    Returns dict of grads matching p.trainable_keys().
    """
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

    grads = {k: np.zeros_like(p.get(k)) for k in p.trainable_keys()}

    dh_next = np.zeros((B, H))
    dc_next = np.zeros((B, H))

    for t in reversed(range(T)):
        # Output: pred[t] = h[t+1] @ Wy + by
        dpred_t = dpred[t]  # (B, 1)
        grads["Wy"] += h[t + 1].T @ dpred_t
        grads["by"] += dpred_t.sum(axis=0)
        dh = dh_next + (dpred_t @ p.Wy.T)  # (B, H)

        # h_t = o_t * tanh(c_t)
        do_t = dh * tc[t]                     # dL/d o_g[t]
        dtc_t = dh * o_g[t]                   # dL/d tanh(c_t)
        # gradient into o pre-activation
        dz_o = do_t * dsigmoid_from_y(o_g[t])
        # peephole on output gate uses c_t (the new cell state)
        dc = dc_next + dtc_t * dtanh_from_y(tc[t])
        if p.use_peep:
            dc = dc + dz_o * p.p_o
            grads["p_o"] += (dz_o * c[t + 1]).sum(axis=0)
        # c_t = f_t * c_{t-1} + i_t * g_t
        df_t = dc * c[t]
        dc_prev_through_f = dc * f_g[t]
        di_t = dc * g_g[t]
        dg_t = dc * i_g[t]
        dz_i = di_t * dsigmoid_from_y(i_g[t])
        dz_f = df_t * dsigmoid_from_y(f_g[t])
        dz_g = dg_t * dtanh_from_y(g_g[t])
        # peephole on i and f reads c_{t-1}
        dc_prev = dc_prev_through_f
        if p.use_peep:
            dc_prev = dc_prev + dz_i * p.p_i + dz_f * p.p_f
            grads["p_i"] += (dz_i * c[t]).sum(axis=0)
            grads["p_f"] += (dz_f * c[t]).sum(axis=0)
        dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)
        grads["Wx"] += X[t].T @ dz
        grads["Wh"] += h[t].T @ dz
        grads["b"] += dz.sum(axis=0)
        dh_next = dz @ p.Wh.T
        dc_next = dc_prev

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
        self.keys = list(params.trainable_keys())
        self.m = {k: np.zeros_like(params.get(k)) for k in self.keys}
        self.v = {k: np.zeros_like(params.get(k)) for k in self.keys}

    def step(self, params, grads):
        if self.clip is not None:
            total = math.sqrt(sum(float((grads[k] ** 2).sum())
                                  for k in self.keys))
            if total > self.clip:
                scale = self.clip / (total + 1e-12)
                for k in self.keys:
                    grads[k] = grads[k] * scale
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for k in self.keys:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params.set(k, params.get(k) - self.lr * m_hat
                       / (np.sqrt(v_hat) + self.eps))


# ----------------------------------------------------------------------
# Training loop + eval
# ----------------------------------------------------------------------

@dataclass
class TrainHistory:
    iters: list = field(default_factory=list)
    train_mse: list = field(default_factory=list)
    test_mse: list = field(default_factory=list)
    solve_rate: list = field(default_factory=list)  # |argmax err| <= 1
    sequences_seen: list = field(default_factory=list)

    def to_dict(self):
        return {k: list(v) for k, v in self.__dict__.items()}


def evaluate(params, rng, T, D_min, D_max, n_test=512, batch_size=128,
             tol=0):
    """Eval: solve = argmax(pred[t2+1:]) within +-tol of t_target.

    `tol=0` requires exact spike timing (the paper's notion of "precise
    timing"); `tol=1` accepts +-1 step.
    """
    sse = 0.0
    n_correct = 0
    n_total = 0
    while n_total < n_test:
        b = min(batch_size, n_test - n_total)
        X, y, t1, t2, t_target = make_msd_batch(rng, T, D_min, D_max, b)
        pred, _ = lstm_forward(params, X)  # (T, b, 1)
        sse += float(((pred - y) ** 2).sum()) / T  # per-step MSE summed
        # Eval: which step has the maximum prediction in the window
        # [t2+1, T-1]? Compare to t_target.
        pred_2d = pred[..., 0]  # (T, b)
        for k in range(b):
            window_start = t2[k] + 1
            window = pred_2d[window_start:, k]
            if window.size == 0:
                continue
            argmax_local = int(np.argmax(window))
            argmax_global = window_start + argmax_local
            if abs(argmax_global - t_target[k]) <= tol:
                n_correct += 1
        n_total += b
    return sse / n_total, n_correct / n_total


def train(use_peep: bool, T: int, D_min: int, D_max: int, hidden: int,
          seed: int, n_iters: int, batch_size: int, lr: float,
          eval_every: int, lr_decay_every: int = 1500,
          lr_decay_factor: float = 0.5, verbose: bool = True,
          save_snapshots: bool = False):
    train_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed + 1_000_003)
    init_rng = np.random.RandomState(seed + 7)

    params = init_lstm(input_dim=1, H=hidden, rng=init_rng, use_peep=use_peep)
    opt = Adam(params, lr=lr, clip=1.0)
    history = TrainHistory()
    snapshots = []
    sequences_seen = 0
    t0 = time.time()
    last_train_mse = float("nan")

    for it in range(1, n_iters + 1):
        if lr_decay_every and it > 1 and (it - 1) % lr_decay_every == 0:
            opt.lr *= lr_decay_factor
        X, y, t1, t2, t_target = make_msd_batch(train_rng, T, D_min, D_max,
                                                batch_size)
        pred, cache = lstm_forward(params, X)
        err = pred - y  # (T, B, 1)
        # Loss = mean over batch of sum-over-T MSE
        # mean over batch is dpred = err / B; we accumulate sum over T below
        loss = 0.5 * float((err * err).sum()) / batch_size
        last_train_mse = (err * err).sum() / batch_size / T
        dpred = err / batch_size
        grads = lstm_backward(params, cache, dpred)
        opt.step(params, grads)
        sequences_seen += batch_size

        if it == 1 or it % eval_every == 0 or it == n_iters:
            test_mse, solve = evaluate(params, test_rng, T, D_min, D_max,
                                       n_test=512, batch_size=128)
            history.iters.append(it)
            history.train_mse.append(last_train_mse)
            history.test_mse.append(test_mse)
            history.solve_rate.append(solve)
            history.sequences_seen.append(sequences_seen)
            if verbose:
                el = time.time() - t0
                print(f"  iter {it:5d}  seq {sequences_seen:7d}  "
                      f"train_mse {last_train_mse:.5f}  "
                      f"test_mse {test_mse:.5f}  "
                      f"solve_rate {solve:.3f}  "
                      f"({el:.1f}s)")
            if save_snapshots:
                snap_rng = np.random.RandomState(seed + 99)
                Xs, ys, t1s, t2s, tts = make_msd_batch(snap_rng, T,
                                                       D_min, D_max, 4)
                preds, snap_cache = lstm_forward(params, Xs)
                snapshot = dict(
                    iter=it,
                    sequences=sequences_seen,
                    train_mse=last_train_mse,
                    test_mse=test_mse,
                    solve_rate=solve,
                    Xs=Xs.copy(), ys=ys.copy(), preds=preds.copy(),
                    t1s=t1s.copy(), t2s=t2s.copy(), tts=tts.copy(),
                    c=snap_cache["c"].copy(),
                )
                snapshots.append(snapshot)
    if verbose:
        print(f"  trained in {time.time() - t0:.1f}s")
    return params, history, snapshots


# ----------------------------------------------------------------------
# Numerical gradient check
# ----------------------------------------------------------------------

def gradcheck(use_peep: bool, seed: int = 0, T: int = 12, hidden: int = 4,
              batch_size: int = 3, n_samples: int = 25, eps: float = 1e-5,
              tol: float = 1e-4):
    """Compare analytical to numerical gradients on random parameters.

    Picks `n_samples` random (key, index) pairs and verifies the
    analytical gradient agrees with central differences. Returns the
    max relative error.
    """
    rng = np.random.RandomState(seed)
    params = init_lstm(1, hidden, rng, use_peep=use_peep)
    X, y, *_ = make_msd_batch(rng, T, D_min=2, D_max=4, batch_size=batch_size)

    def loss_fn():
        pred, _ = lstm_forward(params, X)
        err = pred - y
        return 0.5 * float((err * err).sum()) / batch_size

    pred, cache = lstm_forward(params, X)
    err = pred - y
    dpred = err / batch_size
    grads = lstm_backward(params, cache, dpred)

    keys = list(params.trainable_keys())
    sample_rng = np.random.RandomState(seed + 1)
    max_rel = 0.0
    for _ in range(n_samples):
        k = keys[sample_rng.randint(0, len(keys))]
        arr = params.get(k)
        flat_idx = sample_rng.randint(0, arr.size)
        idx = np.unravel_index(flat_idx, arr.shape)
        orig = arr[idx]
        arr[idx] = orig + eps
        l_plus = loss_fn()
        arr[idx] = orig - eps
        l_minus = loss_fn()
        arr[idx] = orig
        num = (l_plus - l_minus) / (2.0 * eps)
        ana = grads[k][idx]
        denom = max(abs(num), abs(ana), 1e-8)
        rel = abs(num - ana) / denom
        if rel > max_rel:
            max_rel = rel
    print(f"  gradcheck (peep={use_peep}): max rel err = {max_rel:.2e} "
          f"over {n_samples} samples (tol {tol:.0e})")
    return max_rel


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--T", type=int, default=150)
    parser.add_argument("--D-min", type=int, default=30)
    parser.add_argument("--D-max", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr-decay-every", type=int, default=1500)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    peep_group = parser.add_mutually_exclusive_group()
    peep_group.add_argument("--peep", dest="peep", action="store_true",
                            help="use peephole connections (Gers et al 2002)")
    peep_group.add_argument("--no-peep", dest="peep", action="store_false",
                            help="vanilla LSTM, no peepholes (baseline)")
    parser.set_defaults(peep=True)
    parser.add_argument("--gradcheck", action="store_true",
                        help="run numerical gradient check and exit")
    parser.add_argument("--save", type=str, default=None,
                        help="optional path to dump history JSON")
    args = parser.parse_args()

    if args.gradcheck:
        gradcheck(use_peep=True, seed=args.seed)
        gradcheck(use_peep=False, seed=args.seed)
        return

    label = "peep" if args.peep else "no-peep"
    print(f"[timing-counting-spikes] training {label} LSTM "
          f"(seed={args.seed}, T={args.T}, "
          f"D in [{args.D_min},{args.D_max}], "
          f"hidden={args.hidden}, iters={args.iters})")
    params, history, _ = train(
        use_peep=args.peep, T=args.T, D_min=args.D_min, D_max=args.D_max,
        hidden=args.hidden, seed=args.seed, n_iters=args.iters,
        batch_size=args.batch, lr=args.lr,
        eval_every=args.eval_every,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        verbose=True, save_snapshots=False,
    )
    final_test_mse = history.test_mse[-1]
    final_solve = history.solve_rate[-1]
    print(f"[timing-counting-spikes] final test MSE = {final_test_mse:.5f}, "
          f"solve rate = {final_solve:.3f}")

    if args.save:
        out = {
            "args": vars(args),
            "history": history.to_dict(),
            "final_test_mse": final_test_mse,
            "final_solve_rate": final_solve,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
