"""
noise-free-long-lag --- Hochreiter & Schmidhuber, *Long Short-Term Memory*,
Neural Computation 9(8):1735-1780 (1997), Experiment 2.

Sub-variant (a) -- the noise-free local-regularity setting

Two locally-encoded sequences over an alphabet of p+1 symbols:

    sequence A:   y, a_1, a_2, ..., a_{p-1}, y
    sequence B:   x, a_1, a_2, ..., a_{p-1}, x

The middle block `a_1 .. a_{p-1}` is identical in both -- so every symbol
inside the block is fully predictable from its predecessor. The only random
bit is the first symbol (`y` or `x` with probability 0.5). The very last
symbol *also* equals the first one, so predicting it correctly requires
remembering the choice for `p` steps. A standard RNN trained with BPTT or
RTRL fails because gradient signal at the last step decays exponentially with
`p`. The 1997 paper reports BPTT/RTRL = 0% solved at p=100, LSTM solves
within ~5,040 training sequences.

What the network is asked to do
-------------------------------

Given the one-hot symbol at step t, predict the one-hot symbol at step t+1
via a softmax over the `p+1` alphabet entries. Cross-entropy over the whole
sequence is back-propagated; the *interesting* loss is at the final step.

A sequence counts as "solved" if, at the final step, the network's argmax
matches the true terminal symbol. We track a multi-step rolling success rate
on the most recent batch of training sequences and report wallclock to first
hitting >= 95 % rolling accuracy.

v1 budget
---------

The paper uses p=100. For laptop-CPU under-5-min budget we run p=50 by
default (still ~50 distractors of lag, well past the BPTT vanishing-gradient
barrier). The same recipe can be re-run at higher p with `--p 100` if the
caller has more time.

CLI
---

    python3 noise_free_long_lag.py --seed 0
    python3 noise_free_long_lag.py --seed 0 --p 100 --max-seq 20000

Single seed, default settings: ~30-90 s wallclock on an M-series laptop.
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
# Numerics
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


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
# Data: noise-free long-lag sub-variant (a)
# ----------------------------------------------------------------------
#
# Alphabet layout, with p = number of distractors:
#   indices 0 .. p-2    : a_1 .. a_{p-1}   (the deterministic middle block)
#   index   p-1         : x   (one of the two "key" symbols)
#   index   p           : y   (the other "key" symbol)
# Total alphabet size V = p + 1.
#
# Each sequence has length T = p + 1:
#   step 0      : x or y  (random with prob 0.5)
#   step 1      : a_1
#   step 2      : a_2
#   ...
#   step p-1    : a_{p-1}
#   step p      : same key symbol as step 0
#
# The training task is to predict the symbol at step t+1 given the symbol at
# step t. There are T-1 = p prediction targets.

def alphabet_size(p: int) -> int:
    return p + 1


def x_index(p: int) -> int:
    return p - 1


def y_index(p: int) -> int:
    return p


def gen_sequence(p: int, rng: np.random.Generator):
    """Return (input_idx, target_idx) of length T = p+1 / p respectively.

    The sequence runs:  key, a_1, a_2, ..., a_{p-1}, key
    Inputs are positions 0..T-1. Targets at step t are the symbol at t+1.
    """
    key = rng.choice([x_index(p), y_index(p)])
    seq = [key] + list(range(p - 1)) + [key]   # a_1..a_{p-1} = indices 0..p-2
    inputs = np.asarray(seq[:-1], dtype=np.int64)
    targets = np.asarray(seq[1:], dtype=np.int64)
    return inputs, targets


def gen_batch_one_hot(p: int, rng: np.random.Generator):
    """Generate a single sequence in one-hot form.

    Returns
    -------
    X : (T, V) float
    Y : (T,) int   target symbol indices
    """
    inputs, targets = gen_sequence(p, rng)
    V = alphabet_size(p)
    T = inputs.shape[0]
    X = np.zeros((T, V), dtype=np.float64)
    X[np.arange(T), inputs] = 1.0
    return X, targets


# ----------------------------------------------------------------------
# Pure-numpy LSTM (single layer, with forget gate)
# ----------------------------------------------------------------------

class LSTM:
    """Single-layer LSTM with forget gate, softmax output head.

    Weight layout (single dense block, gate order = [input, forget, candidate, output]):

        W  : (4H, V_in + H)
        b  : (4H,)
        Wy : (V_out, H)
        by : (V_out,)

    Gate-bias init -- chosen empirically by sweep for the Experiment-2(a)
    long-lag setting. The combination is closer to the *original* 1997 LSTM
    CEC than the modern Gers-2000 forget-gate-only init:

        * input  gate bias = 0.0   (50 % open at init -- info flows in)
        * forget gate bias = +5.0  (>99 % retention   -- CEC effectively on)
        * output gate bias = 0.0   (50 % open at init -- cell content visible)

    The strong forget bias matters: 0.99**50 ~ 0.6 of the original gradient
    survives p=50 steps. Recurrent block W is initialised orthogonally with
    a 0.5 scale; input block is small Gaussian. With these biases the
    network rapidly learns to *close* the input gate against distractors
    and *open* it for the key symbols at the boundaries of each sequence.
    """

    def __init__(
        self,
        V_in: int,
        hidden: int,
        V_out: int,
        seed: int = 0,
        bias_in: float = 0.0,
        bias_forget: float = 5.0,
        bias_out: float = 0.0,
    ):
        rng = np.random.default_rng(seed)
        H = hidden
        # Input-to-gates init (small Gaussian)
        Wi_x = rng.normal(0, 0.1, (4 * H, V_in))
        # Recurrent-to-gates init (orthogonal stack of H x H blocks)
        Wi_h = np.zeros((4 * H, H))
        for k in range(4):
            mat = rng.normal(0, 1.0, (H, H))
            q, _ = np.linalg.qr(mat)
            Wi_h[k * H:(k + 1) * H] = 0.5 * q
        self.W = np.concatenate([Wi_x, Wi_h], axis=1)
        self.b = np.zeros(4 * H)
        self.b[:H] = bias_in
        self.b[H:2 * H] = bias_forget
        self.b[3 * H:4 * H] = bias_out
        self.Wy = rng.normal(0, 0.1, (V_out, H))
        self.by = np.zeros(V_out)

        self.H = H
        self.V_in = V_in
        self.V_out = V_out

        # Adam state
        self._adam_step = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    # -- parameter dict ----------------------------------------------

    def _params(self):
        return {"W": self.W, "b": self.b, "Wy": self.Wy, "by": self.by}

    # -- forward / backward / loss -----------------------------------

    def forward(self, X: np.ndarray):
        """X: (T, V_in) one-hot. Returns (logits (T, V_out), cache)."""
        T = X.shape[0]
        H = self.H
        h = np.zeros((T + 1, H))
        c = np.zeros((T + 1, H))
        i_t = np.zeros((T, H))
        f_t = np.zeros((T, H))
        g_t = np.zeros((T, H))
        o_t = np.zeros((T, H))
        c_tanh = np.zeros((T, H))
        logits = np.zeros((T, self.V_out))
        probs = np.zeros((T, self.V_out))
        for t in range(T):
            xh = np.concatenate([X[t], h[t]])
            z = self.W @ xh + self.b
            i_t[t] = sigmoid(z[:H])
            f_t[t] = sigmoid(z[H:2 * H])
            g_t[t] = np.tanh(z[2 * H:3 * H])
            o_t[t] = sigmoid(z[3 * H:4 * H])
            c[t + 1] = f_t[t] * c[t] + i_t[t] * g_t[t]
            c_tanh[t] = np.tanh(c[t + 1])
            h[t + 1] = o_t[t] * c_tanh[t]
            logits[t] = self.Wy @ h[t + 1] + self.by
            probs[t] = softmax(logits[t])
        cache = {
            "X": X,
            "h": h,
            "c": c,
            "i": i_t,
            "f": f_t,
            "g": g_t,
            "o": o_t,
            "c_tanh": c_tanh,
            "probs": probs,
        }
        return logits, cache

    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray, last_step_weight: float = 1.0):
        """Cross-entropy summed across the sequence, with optional last-step
        weighting.

        The locally-encoded long-lag task has a problematic optimisation
        landscape under uniform per-step cross-entropy: gradients from the
        easy a_i -> a_{i+1} transitions dominate Adam's second-moment
        normalisation and drown out the rare last-step gradient that
        actually requires long-term memory. Empirically (see README §Results)
        a last-step gradient weight of ~100 is enough to give the long-lag
        signal a foothold without destabilising the easy-step learning.

        Returns (loss_total, loss_last, grads_dict, probs)."""
        logits, cache = self.forward(X)
        probs = cache["probs"]
        T = X.shape[0]
        # Cross-entropy (reported for diagnostics, NOT scaled here)
        log_p = np.log(np.clip(probs[np.arange(T), Y], 1e-12, 1.0))
        loss_total = -log_p.sum()
        loss_last = -log_p[-1]

        # dlogits = probs - one_hot(Y), then scaled per timestep
        dlogits = probs.copy()
        dlogits[np.arange(T), Y] -= 1.0
        if last_step_weight != 1.0:
            weights = np.ones(T)
            weights[-1] = last_step_weight
            dlogits = dlogits * weights[:, None]

        H = self.H
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(H)
        dc_next = np.zeros(H)

        h = cache["h"]
        c = cache["c"]
        i_t = cache["i"]
        f_t = cache["f"]
        g_t = cache["g"]
        o_t = cache["o"]
        c_tanh = cache["c_tanh"]

        for t in range(T - 1, -1, -1):
            # Output projection
            dWy += np.outer(dlogits[t], h[t + 1])
            dby += dlogits[t]
            dh = self.Wy.T @ dlogits[t] + dh_next
            # h[t+1] = o * tanh(c[t+1])
            do_pre = dh * c_tanh[t]
            dc_tanh = dh * o_t[t]
            # tanh
            dc = dc_tanh * (1.0 - c_tanh[t] ** 2) + dc_next
            # c[t+1] = f * c[t] + i * g
            df_pre = dc * c[t]
            di_pre = dc * g_t[t]
            dg_pre = dc * i_t[t]
            dc_prev = dc * f_t[t]
            # Gate activations
            dz_i = di_pre * i_t[t] * (1.0 - i_t[t])
            dz_f = df_pre * f_t[t] * (1.0 - f_t[t])
            dz_g = dg_pre * (1.0 - g_t[t] ** 2)
            dz_o = do_pre * o_t[t] * (1.0 - o_t[t])
            dz = np.concatenate([dz_i, dz_f, dz_g, dz_o])
            xh = np.concatenate([cache["X"][t], h[t]])
            dW += np.outer(dz, xh)
            db += dz
            dxh = self.W.T @ dz
            dh_next = dxh[self.V_in:]
            dc_next = dc_prev

        grads = {"W": dW, "b": db, "Wy": dWy, "by": dby}
        return loss_total, loss_last, grads, probs

    # -- Adam ---------------------------------------------------------

    def adam_step(self, grads, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        # Global gradient clipping (per-tensor would also work)
        sq = sum((g ** 2).sum() for g in grads.values())
        norm = float(np.sqrt(sq))
        scale = 1.0 if norm < clip else (clip / (norm + 1e-12))
        self._adam_step += 1
        t = self._adam_step
        params = self._params()
        for k in params:
            g = grads[k] * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** t)
            vhat = self._v[k] / (1 - b2 ** t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        return norm


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(
    p: int = 50,
    hidden: int = 16,
    seed: int = 0,
    max_seq: int = 8000,
    lr: float = 2e-2,
    last_step_weight: float = 100.0,
    eval_every: int = 200,
    eval_batch: int = 64,
    success_threshold: float = 0.95,
    rolling_window: int = 256,
    snapshots: int = 0,
    verbose: bool = True,
):
    """Train an LSTM on noise-free long-lag (sub-variant a).

    Returns a dict with the final report and a per-eval log.
    """
    rng = np.random.default_rng(seed)
    V = alphabet_size(p)
    model = LSTM(V_in=V, hidden=hidden, V_out=V, seed=seed)

    rolling = []
    log = {"step": [], "loss_total": [], "loss_last": [], "acc_last": [],
           "acc_per_step": [], "rolling_acc_last": [], "grad_norm": []}
    snaps = []

    solved_at = None
    t_start = time.time()

    for step in range(1, max_seq + 1):
        X, Y = gen_batch_one_hot(p, rng)
        loss_total, loss_last, grads, probs = model.loss_and_grads(
            X, Y, last_step_weight=last_step_weight)
        gnorm = model.adam_step(grads, lr=lr)

        # Rolling success: did argmax match at the LAST step?
        last_correct = int(np.argmax(probs[-1]) == Y[-1])
        rolling.append(last_correct)
        if len(rolling) > rolling_window:
            rolling.pop(0)
        rolling_acc_last = float(np.mean(rolling))

        if step % eval_every == 0 or step == 1:
            # Held-out eval batch
            n_correct_last = 0
            n_correct_step = 0
            n_step_total = 0
            for _ in range(eval_batch):
                Xe, Ye = gen_batch_one_hot(p, rng)
                _, ce = model.forward(Xe)
                pe = ce["probs"]
                preds = np.argmax(pe, axis=1)
                if preds[-1] == Ye[-1]:
                    n_correct_last += 1
                n_correct_step += int((preds == Ye).sum())
                n_step_total += Ye.size
            acc_last = n_correct_last / eval_batch
            acc_per_step = n_correct_step / n_step_total

            log["step"].append(step)
            log["loss_total"].append(float(loss_total))
            log["loss_last"].append(float(loss_last))
            log["acc_last"].append(acc_last)
            log["acc_per_step"].append(acc_per_step)
            log["rolling_acc_last"].append(rolling_acc_last)
            log["grad_norm"].append(float(gnorm))

            if verbose:
                print(
                    f"step {step:5d}  loss_total {loss_total:7.3f}  "
                    f"loss_last {loss_last:6.3f}  acc_last {acc_last:.2f}  "
                    f"acc_per_step {acc_per_step:.3f}  "
                    f"rolling_last {rolling_acc_last:.2f}  gn {gnorm:.2f}"
                )

            if solved_at is None and rolling_acc_last >= success_threshold:
                solved_at = step

        if snapshots and (step in _snapshot_steps(snapshots, max_seq) or step == 1):
            snaps.append({
                "step": step,
                "W": model.W.copy(),
                "b": model.b.copy(),
                "Wy": model.Wy.copy(),
                "by": model.by.copy(),
            })

    wall = time.time() - t_start
    final_eval = _final_eval(model, p, rng, n=200)

    report = {
        "p": p,
        "alphabet": V,
        "hidden": hidden,
        "seed": seed,
        "max_seq": max_seq,
        "lr": lr,
        "last_step_weight": last_step_weight,
        "rolling_window": rolling_window,
        "success_threshold": success_threshold,
        "solved_at_seq": solved_at,
        "final_acc_last_step_200": final_eval["acc_last"],
        "final_acc_per_step_200": final_eval["acc_per_step"],
        "wallclock_sec": wall,
        "git_hash": git_hash(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    return {"report": report, "log": log, "snapshots": snaps, "model": model}


def _snapshot_steps(n_snaps: int, max_seq: int):
    steps = np.linspace(1, max_seq, n_snaps, dtype=np.int64)
    return set(int(s) for s in steps)


def _final_eval(model: LSTM, p: int, rng: np.random.Generator, n: int = 200):
    n_last = 0
    n_step_correct = 0
    n_step_total = 0
    for _ in range(n):
        X, Y = gen_batch_one_hot(p, rng)
        _, c = model.forward(X)
        preds = np.argmax(c["probs"], axis=1)
        if preds[-1] == Y[-1]:
            n_last += 1
        n_step_correct += int((preds == Y).sum())
        n_step_total += Y.size
    return {"acc_last": n_last / n, "acc_per_step": n_step_correct / n_step_total}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=50,
                    help="distractor block length (paper used p=100; v1 default 50)")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--last-step-weight", type=float, default=100.0,
                    help="multiplier on the last-step gradient (see §Deviations)")
    ap.add_argument("--max-seq", type=int, default=8000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--rolling-window", type=int, default=256)
    ap.add_argument("--success-threshold", type=float, default=0.95)
    ap.add_argument("--snapshots", type=int, default=0,
                    help="if >0, store this many parameter snapshots for viz")
    ap.add_argument("--save-log", type=str, default=None,
                    help="if given, dump the per-eval log + report as JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = train(
        p=args.p,
        hidden=args.hidden,
        seed=args.seed,
        max_seq=args.max_seq,
        lr=args.lr,
        last_step_weight=args.last_step_weight,
        eval_every=args.eval_every,
        eval_batch=args.eval_batch,
        rolling_window=args.rolling_window,
        success_threshold=args.success_threshold,
        snapshots=args.snapshots,
        verbose=not args.quiet,
    )

    print("\n=== Final report ===")
    for k, v in out["report"].items():
        print(f"  {k:30s} {v}")

    if args.save_log:
        payload = {"report": out["report"], "log": out["log"]}
        with open(args.save_log, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote log JSON to {args.save_log}")


if __name__ == "__main__":
    main()
