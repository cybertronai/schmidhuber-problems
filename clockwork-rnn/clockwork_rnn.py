"""
clockwork-rnn --- Koutník, Greff, Gomez, Schmidhuber, *A Clockwork RNN*,
ICML 2014 (arXiv:1402.3511).

Architecture
------------

A standard Elman RNN with the hidden layer partitioned into G modules
(groups). Each module g has a clock period T_g; at timestep t the module
updates only when ``t mod T_g == 0``, otherwise its activations are
copied forward. Recurrent connections only flow from slower-clock
modules into faster-clock modules; equivalently, when groups are sorted
slow-to-fast the recurrent matrix is block-lower-triangular.

    h_g[t] = tanh(W_h[g, :] . h[t-1] + W_x[g, :] . x[t] + b_g)   if active
    h_g[t] = h_g[t-1]                                            otherwise
    y[t]   = W_y . h[t] + b_y

Synthetic task
--------------

*Waveform memorisation* — the same setup as Koutník et al.'s audio-
generation experiment, but on a synthetic multi-rate signal. The model
receives a constant input (``x[t] = 1`` for all ``t``); the target is a
sum of sines whose periods overlap the module clock periods. The
slowest sine should be tracked by the slowest CW-RNN module; the
fastest by the fastest module. With only a constant input, the network
has to *generate* the signal from its own dynamics — there is no
shortcut via local autocorrelation. A vanilla RNN with a hidden size
chosen so the total parameter count matches the CW-RNN serves as the
matched-capacity baseline.

CLI
---

    python3 clockwork_rnn.py --seed 0
    # ~75 s on an M-series laptop CPU; deterministic.

    python3 clockwork_rnn.py --seed 0 --grad-check
    # Numerical-vs-analytic gradient check on a small CW-RNN.

    python3 clockwork_rnn.py --seed 0 --multi-seed
    # Re-run the headline comparison across 5 seeds.
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


# ---------------------------------------------------------------------------
# Environment & reproducibility
# ---------------------------------------------------------------------------

def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": git_rev(),
    }


# ---------------------------------------------------------------------------
# Synthetic multi-rate signal
# ---------------------------------------------------------------------------

def multi_rate_signal(T: int, periods=(4, 16, 64, 256), amplitudes=None,
                      phases=None, rng=None) -> np.ndarray:
    """Sum of sines at the given periods.

    The signal is normalised to roughly unit amplitude so a (1)->1 RNN can
    fit it without rescaling. Phases are drawn from the supplied rng so that
    a fresh signal is produced each call (training/eval get different phases
    but the same period set).
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(periods)
    if phases is None:
        if rng is None:
            phases = [0.0] * len(periods)
        else:
            phases = rng.uniform(0.0, 2.0 * np.pi, size=len(periods))
    t = np.arange(T)
    out = np.zeros(T)
    for p, a, ph in zip(periods, amplitudes, phases):
        out += a * np.sin(2.0 * np.pi * t / p + ph)
    out = out / np.sqrt(sum(a * a for a in amplitudes))
    return out.astype(np.float64)


def make_dataset(n_seqs: int, T: int, periods, rng) -> np.ndarray:
    """Returns array of shape (n_seqs, T) — one signal per row, each with
    independently sampled phases."""
    return np.stack(
        [multi_rate_signal(T, periods=periods, rng=rng) for _ in range(n_seqs)]
    )


def fixed_target(T: int, periods, seed: int) -> np.ndarray:
    """A single, deterministic multi-rate target waveform — same waveform
    every time for a given (T, periods, seed) triple. Used for the
    memorisation task."""
    rng = np.random.default_rng(seed + 7919)
    return multi_rate_signal(T, periods=periods, rng=rng)


# ---------------------------------------------------------------------------
# Clockwork RNN
# ---------------------------------------------------------------------------

class ClockworkRNN:
    """Pure-numpy Clockwork RNN with manual BPTT.

    Parameters
    ----------
    in_dim, out_dim
        Input / output dimensionality (1 for the scalar-signal task).
    hidden_dim
        Total hidden size. Must be divisible by ``n_groups``.
    n_groups
        Number of modules.
    periods
        Iterable of clock periods, one per group. If None, uses [1, 2, 4,
        ..., 2^(n_groups-1)]. Internally sorted slow-to-fast (largest
        period first).
    seed
        Initialiser seed.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_groups: int, periods=None, seed: int = 0):
        assert hidden_dim % n_groups == 0, "hidden_dim must be divisible by n_groups"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.N = hidden_dim
        self.G = n_groups
        self.M = hidden_dim // n_groups

        if periods is None:
            periods = [2 ** g for g in range(n_groups)]
        periods = np.asarray(periods, dtype=np.int64)
        # Sort slow-to-fast: largest period first.
        order = np.argsort(-periods)
        self.periods = periods[order]

        rng = np.random.default_rng(seed)
        scale_h = 1.0 / np.sqrt(self.N)
        scale_x = 1.0 / np.sqrt(self.in_dim)
        scale_y = 1.0 / np.sqrt(self.N)
        self.W_h = rng.standard_normal((self.N, self.N)) * scale_h
        self.W_x = rng.standard_normal((self.N, in_dim)) * scale_x
        self.b_h = np.zeros(self.N)
        self.W_y = rng.standard_normal((out_dim, self.N)) * scale_y
        self.b_y = np.zeros(out_dim)

        # Block-lower-triangular mask on W_h: row group i (faster) reads
        # column group j (slower) iff periods[j] >= periods[i]. With
        # slow-to-fast ordering this is the lower triangle (incl. diag).
        self.mask_h = np.zeros((self.N, self.N))
        for i in range(self.G):
            for j in range(self.G):
                if self.periods[j] >= self.periods[i]:
                    self.mask_h[i * self.M:(i + 1) * self.M,
                                j * self.M:(j + 1) * self.M] = 1.0
        self.W_h *= self.mask_h

    # ---- bookkeeping --------------------------------------------------

    def n_params(self) -> int:
        return (int(self.mask_h.sum()) + self.W_x.size + self.b_h.size
                + self.W_y.size + self.b_y.size)

    def params(self):
        return [self.W_h, self.W_x, self.b_h, self.W_y, self.b_y]

    def active_groups(self, T: int) -> np.ndarray:
        """Return (T, G) boolean: group g active at time t iff (t+1) mod T_g == 0.

        We use 1-based time so that at the *first* step (t=0) every group
        with period dividing 1 (only period 1) is active, larger periods
        take their first step at later boundaries. This matches Koutník
        et al.'s convention.
        """
        t = np.arange(1, T + 1)[:, None]
        return (t % self.periods[None, :]) == 0

    # ---- forward / backward ------------------------------------------

    def forward(self, X: np.ndarray):
        """X: (T, in_dim). Returns y, cache for backward.

        Stores everything needed for BPTT: pre-activations, hidden states,
        active-group flags.
        """
        T = X.shape[0]
        h = np.zeros((T + 1, self.N))
        a = np.zeros((T, self.N))
        active = self.active_groups(T)  # (T, G)
        for t in range(T):
            h_prev = h[t]
            ht = h_prev.copy()
            # Compute pre-activation only for active rows.
            # Active row mask in the hidden vector:
            row_mask = np.repeat(active[t], self.M)  # (N,)
            if row_mask.any():
                a_t = self.W_h @ h_prev + self.W_x @ X[t] + self.b_h
                a[t] = a_t
                ht_active = np.tanh(a_t)
                ht = np.where(row_mask, ht_active, ht)
            h[t + 1] = ht
        y = h[1:] @ self.W_y.T + self.b_y
        cache = {"X": X, "h": h, "a": a, "active": active}
        return y, cache

    def backward(self, dy: np.ndarray, cache: dict):
        """dy: (T, out_dim). Returns grads dict matching params()."""
        X = cache["X"]
        h = cache["h"]
        a = cache["a"]
        active = cache["active"]
        T = X.shape[0]

        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db_h = np.zeros_like(self.b_h)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)

        # Output projection grads.
        # y[t] = W_y h[t+1] + b_y
        # dW_y += sum_t dy[t] outer h[t+1]
        # dh[t+1] += W_y.T dy[t]
        dW_y = dy.T @ h[1:]
        db_y = dy.sum(axis=0)
        dh_from_out = dy @ self.W_y  # (T, N)

        dh_next = np.zeros(self.N)
        for t in range(T - 1, -1, -1):
            dh_t = dh_from_out[t] + dh_next  # gradient at h[t+1]
            row_mask = np.repeat(active[t], self.M)
            # Active rows: h[t+1, i] = tanh(a[t, i]); gradient flows
            # through tanh, then into W_h row, W_x row, b_h, and into
            # h[t] via W_h column.
            # Inactive rows: h[t+1, i] = h[t, i]; identity gradient.
            da = np.zeros(self.N)
            if row_mask.any():
                # tanh'(a) = 1 - tanh(a)^2 = 1 - h_active^2 (only valid
                # for the active rows).
                tanh_a = np.tanh(a[t])
                da_active = dh_t * (1.0 - tanh_a * tanh_a)
                da = np.where(row_mask, da_active, 0.0)
            # Gradients to recurrent / input weights for active rows.
            if row_mask.any():
                dW_h += np.outer(da, h[t])
                dW_x += np.outer(da, X[t])
                db_h += da
            # dh[t] from active rows: W_h.T @ da. From inactive rows:
            # the upstream gradient passes straight through.
            dh_prev = self.W_h.T @ da
            inactive_mask = ~row_mask
            if inactive_mask.any():
                dh_prev = dh_prev + dh_t * inactive_mask
            dh_next = dh_prev

        # Mask off forbidden recurrent gradients (stay on the block-tri).
        dW_h *= self.mask_h
        return {
            "W_h": dW_h,
            "W_x": dW_x,
            "b_h": db_h,
            "W_y": dW_y,
            "b_y": db_y,
        }

    def step(self, grads: dict, lr: float, clip: float = 1.0):
        # Gradient-norm clip across all parameters.
        g_list = [grads["W_h"], grads["W_x"], grads["b_h"], grads["W_y"], grads["b_y"]]
        gn = np.sqrt(sum((g * g).sum() for g in g_list))
        if gn > clip:
            scale = clip / (gn + 1e-12)
            for g in g_list:
                g *= scale
        self.W_h -= lr * grads["W_h"]
        self.W_x -= lr * grads["W_x"]
        self.b_h -= lr * grads["b_h"]
        self.W_y -= lr * grads["W_y"]
        self.b_y -= lr * grads["b_y"]
        # Re-mask in case of round-off accumulation.
        self.W_h *= self.mask_h


# ---------------------------------------------------------------------------
# Vanilla RNN (matched-capacity baseline)
# ---------------------------------------------------------------------------

class VanillaRNN(ClockworkRNN):
    """Same numpy expression as ``ClockworkRNN`` but every group has
    period 1 (always active) and the recurrent mask is the full matrix.

    Implemented by passing ``n_groups=1`` so there is one group with
    period 1 — the active-group test triggers every step and the mask
    is the all-ones matrix. ``hidden_dim`` is chosen by the caller to
    match parameter counts to a target CW-RNN.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 seed: int = 0):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                         n_groups=1, periods=[1], seed=seed)


def vanilla_hidden_dim_to_match(cw: ClockworkRNN) -> int:
    """Pick the largest ``hidden_dim`` such that a vanilla RNN's total
    parameter count is <= the supplied CW-RNN's parameter count.

    Vanilla parameters at hidden size N_v are
        N_v^2 (W_h) + N_v * in_dim (W_x) + N_v (b_h)
        + N_v * out_dim (W_y) + out_dim (b_y)
    Solve for the largest N_v with that total <= cw.n_params().
    """
    target = cw.n_params()
    in_dim, out_dim = cw.in_dim, cw.out_dim
    best = 1
    for nv in range(1, cw.N + 1):
        total = (nv * nv + nv * in_dim + nv + nv * out_dim + out_dim)
        if total <= target:
            best = nv
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def memorisation_inputs(T: int) -> np.ndarray:
    """Constant input of +1 at every timestep, shape (T, 1)."""
    return np.ones((T, 1))


def train_memorise(model, target: np.ndarray, *, n_epochs: int, lr: float):
    """Train ``model`` to generate ``target`` (length T) from a constant
    input ``x[t] = 1``. Returns per-epoch MSE.

    This is the Koutník 2014 audio-generation setup: the network sees no
    input information at all, only its own recurrent dynamics, so it has
    to memorise the entire waveform. Slow clocks help by storing low-
    frequency content across many timesteps without re-deriving it.
    """
    T = target.shape[0]
    X = memorisation_inputs(T)
    y_target = target[:, None]
    losses = []
    for _ in range(n_epochs):
        y_pred, cache = model.forward(X)
        err = y_pred - y_target
        losses.append(0.5 * float(np.mean(err * err)))
        grads = model.backward(err / err.size, cache)
        model.step(grads, lr=lr)
    return losses


def eval_memorise(model, target: np.ndarray) -> float:
    """MSE of generated waveform vs target."""
    T = target.shape[0]
    X = memorisation_inputs(T)
    y_pred, _ = model.forward(X)
    err = y_pred[:, 0] - target
    return 0.5 * float(np.mean(err * err))


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def grad_check(seed: int = 0, eps: float = 1e-5, T: int = 24) -> dict:
    """Numerical vs analytic gradient on a small CW-RNN. Returns max abs
    diffs per parameter array."""
    rng = np.random.default_rng(seed)
    model = ClockworkRNN(in_dim=1, hidden_dim=8, out_dim=1, n_groups=4,
                         periods=[1, 2, 4, 8], seed=seed)
    seq = multi_rate_signal(T + 1, periods=(4, 8, 16), rng=rng)
    X = seq[:-1, None]
    y_target = seq[1:, None]

    def loss_fn(model):
        y, _ = model.forward(X)
        err = y - y_target
        return 0.5 * np.mean(err * err)

    # Analytic gradients.
    y, cache = model.forward(X)
    err = y - y_target
    analytic = model.backward(err / err.size, cache)

    out = {}
    for name, P, mask in [("W_h", model.W_h, model.mask_h),
                          ("W_x", model.W_x, None),
                          ("b_h", model.b_h, None),
                          ("W_y", model.W_y, None),
                          ("b_y", model.b_y, None)]:
        diffs = []
        flat_idx = list(np.ndindex(*P.shape))
        # Subsample for speed: at most 60 entries per array.
        rng2 = np.random.default_rng(seed + 17)
        if len(flat_idx) > 60:
            picks = rng2.choice(len(flat_idx), 60, replace=False)
            picks = [flat_idx[i] for i in picks]
        else:
            picks = flat_idx
        for idx in picks:
            if mask is not None and mask[idx] == 0:
                continue
            orig = P[idx]
            P[idx] = orig + eps
            l_plus = loss_fn(model)
            P[idx] = orig - eps
            l_minus = loss_fn(model)
            P[idx] = orig
            num = (l_plus - l_minus) / (2 * eps)
            ana = analytic[name][idx]
            diffs.append(abs(ana - num))
        out[name] = float(max(diffs)) if diffs else 0.0
    return out


# ---------------------------------------------------------------------------
# CLI / headline experiment
# ---------------------------------------------------------------------------

def run_headline(seed: int, n_epochs: int = 1500, T: int = 320,
                 hidden_dim: int = 64, n_groups: int = 8,
                 periods=(1, 2, 4, 8, 16, 32, 64, 128),
                 signal_periods=(8, 32, 80, 160),
                 lr: float = 0.02) -> dict:
    """Train CW-RNN and matched-capacity vanilla RNN to memorise a
    single multi-rate waveform. The waveform is deterministic given
    ``(T, signal_periods, seed)``; both models see the same target.

    Returns metrics for the headline table.
    """
    target = fixed_target(T, signal_periods, seed=seed)

    # CW-RNN.
    cw = ClockworkRNN(in_dim=1, hidden_dim=hidden_dim, out_dim=1,
                      n_groups=n_groups, periods=periods, seed=seed)
    # Vanilla RNN sized to match parameter count.
    nv = vanilla_hidden_dim_to_match(cw)
    vanilla = VanillaRNN(in_dim=1, hidden_dim=nv, out_dim=1, seed=seed + 1)

    t0 = time.time()
    cw_losses = train_memorise(cw, target, n_epochs=n_epochs, lr=lr)
    cw_train_time = time.time() - t0
    t0 = time.time()
    vanilla_losses = train_memorise(vanilla, target, n_epochs=n_epochs, lr=lr)
    vanilla_train_time = time.time() - t0

    cw_mse = eval_memorise(cw, target)
    vanilla_mse = eval_memorise(vanilla, target)

    return {
        "seed": seed,
        "config": {
            "n_epochs": n_epochs,
            "T": T,
            "hidden_dim": hidden_dim,
            "n_groups": n_groups,
            "periods": list(map(int, periods)),
            "signal_periods": list(map(int, signal_periods)),
            "lr": lr,
            "vanilla_hidden": int(nv),
        },
        "cw_n_params": cw.n_params(),
        "vanilla_n_params": vanilla.n_params(),
        "cw_train_loss_final": float(cw_losses[-1]),
        "vanilla_train_loss_final": float(vanilla_losses[-1]),
        "cw_mse": float(cw_mse),
        "vanilla_mse": float(vanilla_mse),
        "cw_train_time_sec": cw_train_time,
        "vanilla_train_time_sec": vanilla_train_time,
        "cw_train_curve": list(map(float, cw_losses)),
        "vanilla_train_curve": list(map(float, vanilla_losses)),
        "env": env_info(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--T", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--grad-check", action="store_true")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if args.grad_check:
        diffs = grad_check(seed=args.seed)
        print("Max |analytic - numerical| per parameter array:")
        for k, v in diffs.items():
            print(f"  {k}: {v:.3e}")
        return

    if args.multi_seed:
        rows = []
        for s in range(5):
            r = run_headline(seed=s, n_epochs=args.epochs, T=args.T,
                             hidden_dim=args.hidden, n_groups=args.groups,
                             lr=args.lr)
            rows.append(r)
            print(f"seed {s}: cw_mse={r['cw_mse']:.5f} "
                  f"vanilla_mse={r['vanilla_mse']:.5f} "
                  f"ratio={r['vanilla_mse']/r['cw_mse']:.2f}x")
        cw = np.array([r["cw_mse"] for r in rows])
        vn = np.array([r["vanilla_mse"] for r in rows])
        print(f"CW-RNN     mean MSE: {cw.mean():.5f} (sd {cw.std():.5f})")
        print(f"Vanilla    mean MSE: {vn.mean():.5f} (sd {vn.std():.5f})")
        print(f"Vanilla / CW MSE ratio:  {(vn / cw).mean():.2f}x")
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(rows, f, indent=2)
        return

    r = run_headline(seed=args.seed, n_epochs=args.epochs, T=args.T,
                     hidden_dim=args.hidden, n_groups=args.groups, lr=args.lr)
    print(json.dumps(
        {k: v for k, v in r.items()
         if k not in ("cw_train_curve", "vanilla_train_curve")},
        indent=2,
    ))
    print(f"\n  CW-RNN     params: {r['cw_n_params']}, MSE: {r['cw_mse']:.5f}")
    print(f"  Vanilla RNN params: {r['vanilla_n_params']}, MSE: {r['vanilla_mse']:.5f}")
    print(f"  Vanilla / CW MSE ratio: {r['vanilla_mse']/r['cw_mse']:.2f}x")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
