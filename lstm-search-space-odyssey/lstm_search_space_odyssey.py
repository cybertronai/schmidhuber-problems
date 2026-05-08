"""
lstm-search-space-odyssey — Greff, Srivastava, Koutnik, Steunebrink,
Schmidhuber (2017), "LSTM: A Search Space Odyssey", IEEE TNNLS.

The paper compares 8 LSTM variants on TIMIT, IAM, and JSB (5,400 runs,
~15 CPU-years). We approximate it on a small synthetic task: the
Hochreiter & Schmidhuber 1997 adding problem at T=50, with all 8
variants trained for the same fixed budget under identical optimizer
hyperparameters. The headline output is the variant-by-variant
ablation matrix.

The 8 variants:
  V    Vanilla LSTM (peepholes, all three gates, both activations)
  NIG  No Input Gate                 (i_t = 1)
  NFG  No Forget Gate                (f_t = 1)
  NOG  No Output Gate                (o_t = 1)
  NIAF No Input Activation Function  (g_t = z_g, skip tanh)
  NOAF No Output Activation Function (h_t = o_t * c_t, skip tanh)
  CIFG Coupled Input-Forget Gate     (i_t = 1 - f_t)
  NP   No Peepholes                  (W_ci = W_cf = W_co = 0)

All variants share one forward / backward implementation. Variant
behaviour is controlled by a small set of boolean flags evaluated at
runtime so a numerical gradient check covers every code path.
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
# Variant definitions
# ----------------------------------------------------------------------

VARIANT_NAMES = ["V", "NIG", "NFG", "NOG", "NIAF", "NOAF", "CIFG", "NP"]

VARIANT_DESCRIPTIONS = {
    "V":    "Vanilla LSTM (peepholes, all gates, both activations)",
    "NIG":  "No Input Gate (i_t = 1)",
    "NFG":  "No Forget Gate (f_t = 1)",
    "NOG":  "No Output Gate (o_t = 1)",
    "NIAF": "No Input Activation Function (g_t = z_g)",
    "NOAF": "No Output Activation Function (h_t = o_t * c_t)",
    "CIFG": "Coupled Input-Forget Gate (i_t = 1 - f_t)",
    "NP":   "No Peepholes (W_ci = W_cf = W_co = 0)",
}


@dataclass
class VariantFlags:
    name: str
    has_input_gate: bool = True
    has_forget_gate: bool = True
    has_output_gate: bool = True
    input_act: bool = True   # tanh on g_t
    output_act: bool = True  # tanh on c_t before output gate
    coupled_if: bool = False  # i_t = 1 - f_t
    has_peepholes: bool = True

    @staticmethod
    def from_name(name: str) -> "VariantFlags":
        v = VariantFlags(name=name)
        if name == "V":
            return v
        if name == "NIG":
            v.has_input_gate = False
            return v
        if name == "NFG":
            v.has_forget_gate = False
            return v
        if name == "NOG":
            v.has_output_gate = False
            return v
        if name == "NIAF":
            v.input_act = False
            return v
        if name == "NOAF":
            v.output_act = False
            return v
        if name == "CIFG":
            v.coupled_if = True
            return v
        if name == "NP":
            v.has_peepholes = False
            return v
        raise ValueError(f"unknown variant: {name}")


# ----------------------------------------------------------------------
# Adding-problem dataset (Hochreiter & Schmidhuber 1997, Experiment 4)
# ----------------------------------------------------------------------

def make_adding_batch(rng: np.random.RandomState, T: int,
                      batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X, y for a batch of adding-problem sequences.

    X: (T, B, 2)  channel 0 in [-1,1], channel 1 in {0, 1}
    y: (B,)       sum of the two marked channel-0 values
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
# Parameter container — same shape for every variant; unused params
# stay at their initial (zero) value and never receive gradients.
# ----------------------------------------------------------------------

@dataclass
class LSTMParams:
    # Gate order: i, f, g, o
    Wx: np.ndarray   # (D, 4H)
    Wh: np.ndarray   # (H, 4H)
    b: np.ndarray    # (4H,)
    Wci: np.ndarray  # (H,)  peephole into i
    Wcf: np.ndarray  # (H,)  peephole into f
    Wco: np.ndarray  # (H,)  peephole into o
    Wy: np.ndarray   # (H, 1)
    by: np.ndarray   # (1,)

    def keys(self):
        return ["Wx", "Wh", "b", "Wci", "Wcf", "Wco", "Wy", "by"]

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_lstm(input_dim: int, H: int, variant: VariantFlags,
              rng: np.random.RandomState) -> LSTMParams:
    scale_x = 0.5 / math.sqrt(input_dim)
    scale_h = 0.5 / math.sqrt(H)
    Wx = rng.randn(input_dim, 4 * H) * scale_x
    Wh = rng.randn(H, 4 * H) * scale_h
    b = np.zeros(4 * H)
    # Forget-gate bias = 1.0 (Gers/Schmidhuber/Cummins 2000) when the
    # forget gate exists. NFG variant has f_t = 1 so the bias is moot.
    if variant.has_forget_gate:
        b[H:2 * H] = 1.0
    # Peepholes — only initialized when the variant uses them
    if variant.has_peepholes:
        Wci = rng.randn(H) * 0.1
        Wcf = rng.randn(H) * 0.1
        Wco = rng.randn(H) * 0.1
    else:
        Wci = np.zeros(H)
        Wcf = np.zeros(H)
        Wco = np.zeros(H)
    # Zero out unused gate columns so they cannot accidentally drive
    # the cell (gradients are also zeroed; see lstm_backward).
    if not variant.has_input_gate or variant.coupled_if:
        Wx[:, 0:H] = 0.0
        Wh[:, 0:H] = 0.0
        b[0:H] = 0.0
    if not variant.has_forget_gate:
        Wx[:, H:2 * H] = 0.0
        Wh[:, H:2 * H] = 0.0
        b[H:2 * H] = 0.0
    if not variant.has_output_gate:
        Wx[:, 3 * H:4 * H] = 0.0
        Wh[:, 3 * H:4 * H] = 0.0
        b[3 * H:4 * H] = 0.0
    Wy = rng.randn(H, 1) * (1.0 / math.sqrt(H))
    by = np.zeros(1)
    return LSTMParams(Wx=Wx, Wh=Wh, b=b, Wci=Wci, Wcf=Wcf, Wco=Wco,
                      Wy=Wy, by=by)


# ----------------------------------------------------------------------
# Forward pass — handles every variant via VariantFlags
# ----------------------------------------------------------------------

def lstm_forward(p: LSTMParams, X: np.ndarray, variant: VariantFlags):
    T, B, D = X.shape
    H = p.Wh.shape[0]
    h = np.zeros((T + 1, B, H))
    c = np.zeros((T + 1, B, H))
    i_g = np.zeros((T, B, H))
    f_g = np.zeros((T, B, H))
    g_g = np.zeros((T, B, H))
    o_g = np.zeros((T, B, H))
    out_act = np.zeros((T, B, H))

    for t in range(T):
        z = X[t] @ p.Wx + h[t] @ p.Wh + p.b  # (B, 4H)
        z_i = z[:, 0:H]
        z_f = z[:, H:2 * H]
        z_g = z[:, 2 * H:3 * H]
        z_o = z[:, 3 * H:4 * H]
        # Peepholes for i, f use c_{t-1}
        if variant.has_peepholes:
            z_i = z_i + c[t] * p.Wci
            z_f = z_f + c[t] * p.Wcf

        # Input/forget gates
        if variant.has_forget_gate:
            f_g[t] = sigmoid(z_f)
        else:
            f_g[t] = np.ones((B, H))

        if variant.coupled_if:
            i_g[t] = 1.0 - f_g[t]
        elif variant.has_input_gate:
            i_g[t] = sigmoid(z_i)
        else:
            i_g[t] = np.ones((B, H))

        # Cell input activation
        if variant.input_act:
            g_g[t] = np.tanh(z_g)
        else:
            g_g[t] = z_g  # identity

        # Cell update
        c[t + 1] = f_g[t] * c[t] + i_g[t] * g_g[t]

        # Output gate (peephole on o uses c_t — the *new* cell state)
        if variant.has_peepholes:
            z_o = z_o + c[t + 1] * p.Wco
        if variant.has_output_gate:
            o_g[t] = sigmoid(z_o)
        else:
            o_g[t] = np.ones((B, H))

        # Output activation
        if variant.output_act:
            out_act[t] = np.tanh(c[t + 1])
        else:
            out_act[t] = c[t + 1]

        h[t + 1] = o_g[t] * out_act[t]

    pred = (h[T] @ p.Wy + p.by).reshape(B)
    cache = dict(X=X, h=h, c=c, i=i_g, f=f_g, g=g_g, o=o_g, oa=out_act,
                 pred=pred)
    return pred, cache


# ----------------------------------------------------------------------
# Backward pass
# ----------------------------------------------------------------------

def lstm_backward(p: LSTMParams, cache: dict, dpred: np.ndarray,
                  variant: VariantFlags):
    X = cache["X"]
    h = cache["h"]
    c = cache["c"]
    i_g = cache["i"]
    f_g = cache["f"]
    g_g = cache["g"]
    o_g = cache["o"]
    oa = cache["oa"]
    T, B, D = X.shape
    H = p.Wh.shape[0]

    grads = {k: np.zeros_like(p.get(k)) for k in p.keys()}

    dpred_col = dpred.reshape(B, 1)
    grads["Wy"] = h[T].T @ dpred_col
    grads["by"] = dpred_col.sum(axis=0)
    dh_next = dpred_col @ p.Wy.T  # (B, H)
    dc_next = np.zeros((B, H))

    for t in reversed(range(T)):
        dh = dh_next  # (B, H)

        # h_t = o_t * out_act_t
        if variant.has_output_gate:
            do_t = dh * oa[t]
        else:
            do_t = np.zeros_like(dh)
        d_oa = dh * o_g[t]

        # out_act -> c (with optional tanh)
        if variant.output_act:
            dc = dc_next + d_oa * dtanh_from_y(oa[t])
        else:
            dc = dc_next + d_oa

        # Output-gate peephole feeds *current* c_t
        if variant.has_output_gate:
            dz_o = do_t * dsigmoid_from_y(o_g[t])
        else:
            dz_o = np.zeros_like(dh)

        if variant.has_peepholes and variant.has_output_gate:
            grads["Wco"] += (c[t + 1] * dz_o).sum(axis=0)
            dc = dc + dz_o * p.Wco

        # c_t = f_t * c_{t-1} + i_t * g_t
        df_t = dc * c[t]
        dc_prev = dc * f_g[t]
        di_t = dc * g_g[t]
        dg_t = dc * i_g[t]

        # Coupled input-forget gate
        if variant.coupled_if:
            df_t = df_t - di_t  # i = 1 - f
            di_t = np.zeros_like(di_t)

        # Pre-activation gradients for the four gates
        if variant.has_input_gate and not variant.coupled_if:
            dz_i = di_t * dsigmoid_from_y(i_g[t])
        else:
            dz_i = np.zeros_like(di_t)
        if variant.has_forget_gate:
            dz_f = df_t * dsigmoid_from_y(f_g[t])
        else:
            dz_f = np.zeros_like(df_t)
        if variant.input_act:
            dz_g = dg_t * dtanh_from_y(g_g[t])
        else:
            dz_g = dg_t  # identity

        # Peepholes on i, f use c_{t-1}
        if variant.has_peepholes:
            if variant.has_input_gate and not variant.coupled_if:
                grads["Wci"] += (c[t] * dz_i).sum(axis=0)
                dc_prev = dc_prev + dz_i * p.Wci
            if variant.has_forget_gate:
                grads["Wcf"] += (c[t] * dz_f).sum(axis=0)
                dc_prev = dc_prev + dz_f * p.Wcf

        dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)  # (B, 4H)
        grads["Wx"] += X[t].T @ dz
        grads["Wh"] += h[t].T @ dz
        grads["b"] += dz.sum(axis=0)
        dh_next = dz @ p.Wh.T
        dc_next = dc_prev

    return grads


# ----------------------------------------------------------------------
# Adam
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
    solve_rate: list = field(default_factory=list)
    sequences_seen: list = field(default_factory=list)

    def to_dict(self):
        return {k: list(v) for k, v in self.__dict__.items()}


def evaluate(params, variant, rng, T, n_test=512, batch_size=128, tol=0.04):
    sse = 0.0
    n_correct = 0
    n_total = 0
    for _ in range(0, n_test, batch_size):
        b = min(batch_size, n_test - n_total)
        X, y = make_adding_batch(rng, T, b)
        pred, _ = lstm_forward(params, X, variant)
        sse += float(((pred - y) ** 2).sum())
        n_correct += int((np.abs(pred - y) < tol).sum())
        n_total += b
    return sse / n_total, n_correct / n_total


def train_variant(variant_name: str, T: int, hidden: int, seed: int,
                  n_iters: int, batch_size: int, lr: float,
                  eval_every: int, lr_decay_every: int = 0,
                  lr_decay_factor: float = 0.5, verbose: bool = False,
                  save_snapshots: bool = False):
    """Train one variant. Returns (params, history, snapshots)."""
    variant = VariantFlags.from_name(variant_name)
    train_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed + 1_000_003)
    init_rng = np.random.RandomState(seed + 7)

    params = init_lstm(input_dim=2, H=hidden, variant=variant, rng=init_rng)
    opt = Adam(params, lr=lr, clip=1.0)
    history = TrainHistory()
    snapshots = []
    sequences_seen = 0

    # Eval once at iter 0 so the GIF has a "before training" frame
    test_mse0, solve0 = evaluate(params, variant, test_rng, T)
    history.iters.append(0)
    history.train_mse.append(float("nan"))
    history.test_mse.append(test_mse0)
    history.solve_rate.append(solve0)
    history.sequences_seen.append(0)

    for it in range(1, n_iters + 1):
        if lr_decay_every and it > 1 and (it - 1) % lr_decay_every == 0:
            opt.lr *= lr_decay_factor
        X, y = make_adding_batch(train_rng, T, batch_size)
        pred, cache = lstm_forward(params, X, variant)
        err = pred - y
        loss = 0.5 * float((err * err).mean())
        train_mse = 2.0 * loss
        dpred = err / batch_size
        grads = lstm_backward(params, cache, dpred, variant)
        opt.step(params, grads)
        sequences_seen += batch_size

        if it % eval_every == 0 or it == n_iters:
            test_mse, solve = evaluate(params, variant, test_rng, T)
            history.iters.append(it)
            history.train_mse.append(train_mse)
            history.test_mse.append(test_mse)
            history.solve_rate.append(solve)
            history.sequences_seen.append(sequences_seen)
            if verbose:
                print(f"  [{variant_name:>4s}] iter {it:5d}  "
                      f"train_mse {train_mse:.4f}  "
                      f"test_mse {test_mse:.4f}  "
                      f"solve {solve:.3f}")
            if save_snapshots:
                snapshots.append(dict(
                    iter=it, sequences=sequences_seen,
                    test_mse=test_mse, solve_rate=solve,
                    train_mse=train_mse,
                ))

    return params, history, snapshots


def run_ablation_matrix(T: int, hidden: int, n_iters: int, batch_size: int,
                        lr: float, eval_every: int, seeds: list[int],
                        verbose: bool = True):
    """Run all 8 variants × seeds and return a results dict.

    results[variant_name] = {
        'descriptions': str,
        'seeds': [...],
        'history_per_seed': [TrainHistory.to_dict(), ...],
        'final_test_mse_per_seed': [...],
        'final_solve_rate_per_seed': [...],
        'wallclock_per_seed_sec': [...],
    }
    """
    results = {}
    for name in VARIANT_NAMES:
        results[name] = dict(
            description=VARIANT_DESCRIPTIONS[name],
            seeds=list(seeds),
            history_per_seed=[],
            final_test_mse_per_seed=[],
            final_solve_rate_per_seed=[],
            wallclock_per_seed_sec=[],
        )
        for seed in seeds:
            t0 = time.time()
            _, history, _ = train_variant(
                variant_name=name, T=T, hidden=hidden, seed=seed,
                n_iters=n_iters, batch_size=batch_size, lr=lr,
                eval_every=eval_every, verbose=False,
            )
            elapsed = time.time() - t0
            results[name]["history_per_seed"].append(history.to_dict())
            results[name]["final_test_mse_per_seed"].append(
                history.test_mse[-1])
            results[name]["final_solve_rate_per_seed"].append(
                history.solve_rate[-1])
            results[name]["wallclock_per_seed_sec"].append(elapsed)
            if verbose:
                print(f"  [{name:>4s} seed={seed}] "
                      f"final test MSE {history.test_mse[-1]:.4f}  "
                      f"solve {history.solve_rate[-1]:.3f}  "
                      f"({elapsed:.1f}s)")
    return results


# ----------------------------------------------------------------------
# Numerical gradient check — runs every variant
# ----------------------------------------------------------------------

def gradcheck(variant_name: str, T: int = 6, H: int = 4, B: int = 3,
              seed: int = 0, eps: float = 1e-5, n_checks: int = 5):
    variant = VariantFlags.from_name(variant_name)
    rng = np.random.RandomState(seed)
    params = init_lstm(2, H, variant, rng)
    X, y = make_adding_batch(rng, T, B)
    pred, cache = lstm_forward(params, X, variant)
    err = pred - y
    dpred = err / B
    grads = lstm_backward(params, cache, dpred, variant)

    def total_loss(p):
        pr, _ = lstm_forward(p, X, variant)
        return 0.5 * float(((pr - y) ** 2).mean())

    # Skip parameters that are forced to zero by the variant — their
    # gradients are zero by construction and a finite-difference check
    # would give a false positive.
    skip_keys = set()
    if not variant.has_input_gate or variant.coupled_if:
        skip_keys |= {"Wci"}  # also Wxi/Whi/bi columns, handled below
    if not variant.has_forget_gate:
        skip_keys |= {"Wcf"}
    if not variant.has_output_gate:
        skip_keys |= {"Wco"}
    if not variant.has_peepholes:
        skip_keys |= {"Wci", "Wcf", "Wco"}

    rel_errs = []
    check_rng = np.random.RandomState(seed + 1)
    for k in params.keys():
        if k in skip_keys:
            continue
        W = params.get(k)
        flat = W.reshape(-1)
        analytic = grads[k].reshape(-1)
        # For Wx / Wh / b, skip the gate columns that are forced off
        if k in ("Wx", "Wh", "b"):
            valid_idx = []
            stride = H
            stride_in = (W.shape[0] if k != "b" else 1)
            n_per_gate = stride * stride_in
            # gate slots i=0, f=1, g=2, o=3
            for slot, ok in enumerate([
                variant.has_input_gate and not variant.coupled_if,
                variant.has_forget_gate,
                True,  # g always present
                variant.has_output_gate,
            ]):
                if ok:
                    valid_idx.extend(range(slot * n_per_gate,
                                           (slot + 1) * n_per_gate))
            if not valid_idx:
                continue
            idxs = check_rng.choice(valid_idx,
                                    size=min(n_checks, len(valid_idx)),
                                    replace=False)
        else:
            if flat.size == 0:
                continue
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
            rel_errs.append((k, int(i), num, an, rel))
    if not rel_errs:
        return 0.0
    max_rel = max(r[-1] for r in rel_errs)
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
        "processor": platform.processor() or "unknown",
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated list of seeds; overrides --seed")
    ap.add_argument("--T", type=int, default=50, help="sequence length")
    ap.add_argument("--hidden", type=int, default=12, help="hidden units")
    ap.add_argument("--iters", type=int, default=1500, help="iters per run")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--variant", type=str, default=None,
                    help="run a single variant (default: full ablation)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="run numerical gradient check on every variant")
    ap.add_argument("--save-results", type=str, default=None,
                    help="path to write ablation-matrix JSON")
    args = ap.parse_args()

    if args.gradcheck:
        print("Numerical gradient check (T=6, H=4, B=3, eps=1e-5):")
        max_overall = 0.0
        for name in VARIANT_NAMES:
            r = gradcheck(name)
            max_overall = max(max_overall, r)
            print(f"  [{name:>4s}] max relative error = {r:.2e}")
        print(f"  overall max = {max_overall:.2e}")
        return

    if args.seeds is None:
        seeds = [args.seed]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"T={args.T} hidden={args.hidden} iters={args.iters} "
          f"batch={args.batch} lr={args.lr} seeds={seeds}")
    print(f"  env: {env_info()}")

    if args.variant is not None:
        names = [args.variant]
    else:
        names = VARIANT_NAMES

    if len(names) == 1 and len(seeds) == 1:
        # Single-variant single-seed run, verbose
        t0 = time.time()
        _, history, _ = train_variant(
            variant_name=names[0], T=args.T, hidden=args.hidden,
            seed=seeds[0], n_iters=args.iters, batch_size=args.batch,
            lr=args.lr, eval_every=args.eval_every, verbose=True,
        )
        elapsed = time.time() - t0
        print(f"[{names[0]}] final test MSE = {history.test_mse[-1]:.4f}  "
              f"solve = {history.solve_rate[-1]:.3f}  "
              f"({elapsed:.1f}s)")
        if args.save_results:
            with open(args.save_results, "w") as f:
                json.dump({
                    "variant": names[0],
                    "args": vars(args),
                    "env": env_info(),
                    "history": history.to_dict(),
                    "elapsed_sec": elapsed,
                }, f, indent=2)
        return

    # Full ablation matrix
    t0 = time.time()
    results = run_ablation_matrix(
        T=args.T, hidden=args.hidden, n_iters=args.iters,
        batch_size=args.batch, lr=args.lr, eval_every=args.eval_every,
        seeds=seeds, verbose=True,
    )
    total = time.time() - t0

    print(f"\n=== ablation matrix (median over {len(seeds)} seeds) ===")
    print(f"{'variant':>5}  {'test MSE':>10}  {'solve':>6}  "
          f"{'wall (s)':>8}")
    for name in VARIANT_NAMES:
        r = results[name]
        med_mse = float(np.median(r["final_test_mse_per_seed"]))
        med_solve = float(np.median(r["final_solve_rate_per_seed"]))
        med_wall = float(np.median(r["wallclock_per_seed_sec"]))
        print(f"{name:>5}  {med_mse:>10.4f}  {med_solve:>6.3f}  "
              f"{med_wall:>8.2f}")
    print(f"\nTotal wallclock: {total:.1f}s")

    if args.save_results:
        out = {
            "args": vars(args),
            "env": env_info(),
            "results": results,
            "total_sec": total,
        }
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.save_results}")


if __name__ == "__main__":
    main()
