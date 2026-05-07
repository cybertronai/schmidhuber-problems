"""
two-sequence-noise -- Hochreiter & Schmidhuber 1997, *Long Short-Term Memory*,
Neural Computation 9(8): 1735-1780, Experiment 3 ("Noise and signal on the
same channel").

Sub-variant 3c is the one this stub implements: targets are 0.2 / 0.8 (rather
than 0 / 1) and Gaussian target noise (sigma=0.32) is added at training time.
The information-carrying input occupies the first p1 steps; afterwards the
input is pure Gaussian noise, so the network must remember the class label
across an arbitrary distractor stretch.

Two classes:

    class 0:  info phase = -1 + N(0, 0.2);   target final = 0.2
    class 1:  info phase = +1 + N(0, 0.2);   target final = 0.8

In every sequence, the post-info phase (steps p1 .. T-1) is pure N(0, 1)
noise, so the only way to solve the task is to read the first few steps and
hold the answer to step T-1 across the long-time-lag distractor.

Architecture (1997-canonical, no forget gate, no peepholes)
----------------------------------------------------------

    o  1 input unit
    o  3 memory blocks, 2 cells per block (= 6 cells total)
    o  per-block input gate iota_j  with bias 0
    o  per-block output gate omega_j with bias -2 (block 0), -4, -6
    o  output unit: sigmoid scalar  y_k(t)
    o  squashing:  g(x) = 4 sigma(x) - 2       (range (-2, 2))
                   h(x) = 2 sigma(x) - 1       (range (-1, 1))
    o  cell state update: s(t) = s(t-1) + iota * g(net_c(t))
                          y_c(t) = omega * h(s(t))

Training
--------

Online SGD (batch=1) with target noise N(0, 0.32) added to the noiseless
0.2 / 0.8 target every training step.  Loss is the squared error at step
T-1 only (the rest of the steps emit no error signal).  Gradient is full
BPTT through T=100; this is feasible because the no-forget-gate Constant
Error Carousel keeps the state-side gradient stable.

CLI
---

    python3 two_sequence_noise.py --seed 0
    python3 two_sequence_noise.py --seed 0 --T 100 --steps 30000

Single seed:  ~30-90 s on a system-python M-series laptop.
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


def g(x: np.ndarray) -> np.ndarray:
    """Cell-input squashing function: range (-2, 2)."""
    return 4.0 * sigmoid(x) - 2.0


def g_prime_from_g(gx: np.ndarray) -> np.ndarray:
    """d g(x) / dx given y = g(x).  s = sigma(x) = (gx + 2) / 4 ;
    g'(x) = 4 s (1 - s)."""
    s = (gx + 2.0) / 4.0
    return 4.0 * s * (1.0 - s)


def h(x: np.ndarray) -> np.ndarray:
    """Cell-output squashing function: range (-1, 1)."""
    return 2.0 * sigmoid(x) - 1.0


def h_prime_from_h(hx: np.ndarray) -> np.ndarray:
    """d h(x) / dx given y = h(x).  s = sigma(x) = (hx + 1) / 2 ;
    h'(x) = 2 s (1 - s)."""
    s = (hx + 1.0) / 2.0
    return 2.0 * s * (1.0 - s)


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
# Network
# ----------------------------------------------------------------------

class LSTM1997:
    """1997 LSTM (no forget gate, no peephole) with B blocks and C cells per
    block.  All gates and cell inputs receive the external input plus the
    previous cell outputs (recurrent) and a constant bias.  The output unit
    is a single sigmoid taking the current cell outputs.

    Parameter layout (everything stored as flat 2-D matrices for clarity):

        W_iota  : (B, n_in + B*C + 1)    input gate weights (incl. bias col)
        W_omega : (B, n_in + B*C + 1)    output gate weights (incl. bias)
        W_c     : (B*C, n_in + B*C + 1)  cell-input weights (incl. bias)
        W_out   : (1, B*C + 1)           output weights (incl. bias)

    The "+1" extra column at the end of each row is the bias.
    """

    def __init__(
        self,
        n_in: int = 1,
        n_blocks: int = 3,
        n_cells_per_block: int = 2,
        out_gate_biases=(-2.0, -4.0, -6.0),
        init_scale: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.n_in = n_in
        self.n_blocks = n_blocks
        self.n_cells_per_block = n_cells_per_block
        self.n_cells = n_blocks * n_cells_per_block

        # Map cell index -> block index. Cells 0..C-1 belong to block 0,
        # next C cells to block 1, etc.
        self.block_of_cell = np.repeat(
            np.arange(n_blocks), n_cells_per_block
        )

        n_x = n_in + self.n_cells + 1  # input + recurrent cell outputs + bias

        self.W_iota = rng.standard_normal((n_blocks, n_x)) * init_scale
        self.W_iota[:, -1] = 0.0  # input gate bias = 0

        self.W_omega = rng.standard_normal((n_blocks, n_x)) * init_scale
        # Asymmetric output-gate biases per the 1997 paper: -2, -4, -6.
        # If n_blocks > 3, recycle the pattern.
        biases = np.array(out_gate_biases, dtype=np.float64)
        if n_blocks <= len(biases):
            self.W_omega[:, -1] = biases[:n_blocks]
        else:
            self.W_omega[:, -1] = np.array(
                [biases[i % len(biases)] for i in range(n_blocks)]
            )

        self.W_c = rng.standard_normal((self.n_cells, n_x)) * init_scale
        self.W_c[:, -1] = 0.0  # cell-input bias = 0

        self.W_out = rng.standard_normal((1, self.n_cells + 1)) * init_scale
        self.W_out[0, -1] = 0.0  # output bias = 0

    # ------------------------------------------------------------------
    # Parameter helpers (so the optimizer can iterate)
    # ------------------------------------------------------------------

    def params(self):
        return [self.W_iota, self.W_omega, self.W_c, self.W_out]

    def param_names(self):
        return ["W_iota", "W_omega", "W_c", "W_out"]

    def n_weights(self) -> int:
        return sum(p.size for p in self.params())


# ----------------------------------------------------------------------
# Forward / backward
# ----------------------------------------------------------------------

def forward(net: LSTM1997, x_seq: np.ndarray):
    """Run the LSTM for one sequence.

    Args
    ----
    net    : LSTM1997
    x_seq  : (T, n_in) input sequence

    Returns
    -------
    cache : dict with all activation tensors for BPTT, plus y_out (T,)
    """
    T = x_seq.shape[0]
    B = net.n_blocks
    C = net.n_cells

    # storage
    iota_pre = np.zeros((T, B))
    iota = np.zeros((T, B))
    omega_pre = np.zeros((T, B))
    omega = np.zeros((T, B))
    c_pre = np.zeros((T, C))    # net_c
    c_g = np.zeros((T, C))      # g(net_c)
    s = np.zeros((T + 1, C))    # cell state, s[0] = 0
    h_s = np.zeros((T, C))      # h(s[t+1])
    y_c = np.zeros((T, C))      # cell output gated by output gate
    y_out_pre = np.zeros(T)
    y_out = np.zeros(T)

    # constant 1 to multiply the bias column
    one = np.array([1.0])

    y_c_prev = np.zeros(C)
    s_prev = np.zeros(C)
    boc = net.block_of_cell

    for t in range(T):
        x_aug = np.concatenate([x_seq[t], y_c_prev, one])

        iota_pre[t] = net.W_iota @ x_aug
        iota[t] = sigmoid(iota_pre[t])

        omega_pre[t] = net.W_omega @ x_aug
        omega[t] = sigmoid(omega_pre[t])

        c_pre[t] = net.W_c @ x_aug
        c_g[t] = g(c_pre[t])

        # cell state update (no forget gate)
        s[t + 1] = s_prev + iota[t][boc] * c_g[t]
        h_s[t] = h(s[t + 1])

        y_c[t] = omega[t][boc] * h_s[t]

        y_aug = np.concatenate([y_c[t], one])
        y_out_pre[t] = float((net.W_out @ y_aug)[0])
        y_out[t] = float(sigmoid(np.array(y_out_pre[t])))

        y_c_prev = y_c[t]
        s_prev = s[t + 1]

    return {
        "x_seq": x_seq,
        "iota_pre": iota_pre,
        "iota": iota,
        "omega_pre": omega_pre,
        "omega": omega,
        "c_pre": c_pre,
        "c_g": c_g,
        "s": s,
        "h_s": h_s,
        "y_c": y_c,
        "y_out_pre": y_out_pre,
        "y_out": y_out,
    }


def backward(net: LSTM1997, cache: dict, target: float):
    """Compute gradient of (1/2) (y_out[T-1] - target)^2 w.r.t. all params.

    Loss is applied only at the final step.  Returns dict of grads.
    """
    T = cache["x_seq"].shape[0]
    B = net.n_blocks
    C = net.n_cells
    boc = net.block_of_cell

    g_W_iota = np.zeros_like(net.W_iota)
    g_W_omega = np.zeros_like(net.W_omega)
    g_W_c = np.zeros_like(net.W_c)
    g_W_out = np.zeros_like(net.W_out)

    # gradient flowing back through time
    d_yc_next = np.zeros(C)
    d_s_next = np.zeros(C)

    one = np.array([1.0])

    for t in reversed(range(T)):
        # --- output unit (only fires at final step) ---
        if t == T - 1:
            err = cache["y_out"][t] - target
            yk = cache["y_out"][t]
            d_pre = err * yk * (1.0 - yk)

            y_aug = np.concatenate([cache["y_c"][t], one])
            g_W_out += d_pre * y_aug[None, :]
            d_yc_from_out = net.W_out[0, :C] * d_pre
        else:
            d_yc_from_out = np.zeros(C)

        # total gradient w.r.t. y_c[t]
        d_yc = d_yc_from_out + d_yc_next

        # backprop through y_c = omega[boc] * h(s)
        d_omega_per_cell = d_yc * cache["h_s"][t]   # (C,)
        d_h = d_yc * cache["omega"][t][boc]          # (C,)

        # output gate is per-block: sum across cells in same block
        d_omega = np.bincount(
            boc, weights=d_omega_per_cell, minlength=B
        )
        # backprop through sigma
        d_omega_pre = d_omega * cache["omega"][t] * (1.0 - cache["omega"][t])

        # backprop through h(s)
        h_prime = h_prime_from_h(cache["h_s"][t])
        d_s_local = d_h * h_prime
        d_s = d_s_local + d_s_next  # CEC: d L / d s flows back unchanged

        # cell state update: s[t+1] = s[t] + iota[boc] * g(c_pre)
        d_iota_per_cell = d_s * cache["c_g"][t]      # (C,)
        d_g = d_s * cache["iota"][t][boc]            # (C,)

        d_iota = np.bincount(
            boc, weights=d_iota_per_cell, minlength=B
        )
        d_iota_pre = d_iota * cache["iota"][t] * (1.0 - cache["iota"][t])

        d_c_pre = d_g * g_prime_from_g(cache["c_g"][t])

        # --- accumulate weight grads ---
        if t == 0:
            y_c_prev = np.zeros(C)
        else:
            y_c_prev = cache["y_c"][t - 1]
        x_aug = np.concatenate([cache["x_seq"][t], y_c_prev, one])

        g_W_iota += np.outer(d_iota_pre, x_aug)
        g_W_omega += np.outer(d_omega_pre, x_aug)
        g_W_c += np.outer(d_c_pre, x_aug)

        # --- gradient flowing into y_c[t-1] and s[t] ---
        # x_aug[n_in:n_in+C] are the recurrent cell outputs of step t-1
        n_in = net.n_in
        d_x_aug = (
            net.W_iota.T @ d_iota_pre
            + net.W_omega.T @ d_omega_pre
            + net.W_c.T @ d_c_pre
        )
        d_yc_next = d_x_aug[n_in:n_in + C]
        # CEC: d s[t-1] = d s[t] (unchanged) -- the no-forget-gate identity.
        d_s_next = d_s

    return {
        "W_iota": g_W_iota,
        "W_omega": g_W_omega,
        "W_c": g_W_c,
        "W_out": g_W_out,
    }


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def make_sequence(
    rng: np.random.Generator,
    T: int = 100,
    p1: int = 10,
    info_amp: float = 1.0,
    info_sigma: float = 0.2,
    distractor_sigma: float = 1.0,
):
    """Generate one (x, label) example for variant 3c.

    Steps 0..p1-1 contain the information signal: +info_amp or -info_amp,
    plus N(0, info_sigma).  Steps p1..T-1 are pure N(0, distractor_sigma).
    The final step's noiseless target is 0.2 (class 0) or 0.8 (class 1).

    Returns:
        x_seq : (T, 1)
        label : 0 or 1
    """
    label = int(rng.integers(0, 2))  # 0 or 1
    sign = +1.0 if label == 1 else -1.0

    x = rng.standard_normal(T) * distractor_sigma
    info_noise = rng.standard_normal(p1) * info_sigma
    x[:p1] = sign * info_amp + info_noise

    return x[:, None], label


def label_to_target(label: int) -> float:
    return 0.8 if label == 1 else 0.2


# ----------------------------------------------------------------------
# Optimizer (Adam, hand-rolled)
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, clip: float = 1.0):
        self.t += 1
        if clip is not None and clip > 0:
            tot = float(sum(float(np.sum(g * g)) for g in grads)) ** 0.5
            scale = 1.0 if tot <= clip else clip / (tot + 1e-12)
        else:
            scale = 1.0
        for p, gr, m, v in zip(self.params, grads, self.m, self.v):
            gr = gr * scale
            m[...] = self.b1 * m + (1 - self.b1) * gr
            v[...] = self.b2 * v + (1 - self.b2) * (gr * gr)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def train(
    seed: int = 0,
    n_steps: int = 30000,
    T: int = 100,
    p1: int = 10,
    n_blocks: int = 3,
    n_cells_per_block: int = 2,
    target_noise_sigma: float = 0.32,
    distractor_sigma: float = 1.0,
    info_amp: float = 1.0,
    info_sigma: float = 0.2,
    lr: float = 5e-3,
    init_scale: float = 0.1,
    log_every: int = 1000,
    snapshot_every: int = 0,
    snapshot_callback=None,
    verbose: bool = True,
):
    """Train LSTM-1997 on Two-Sequence with Target Noise (variant 3c).

    Online SGD (batch=1) with Adam.  Target noise is added at each training
    step but NOT at evaluation time.
    """
    seed_seq = np.random.SeedSequence(seed)
    rng_init, rng_data = (np.random.default_rng(s) for s in seed_seq.spawn(2))

    net = LSTM1997(
        n_in=1,
        n_blocks=n_blocks,
        n_cells_per_block=n_cells_per_block,
        init_scale=init_scale,
        rng=rng_init,
    )
    opt = Adam(net.params(), lr=lr)

    history = {
        "step": [],
        "loss": [],          # noiseless final-step squared error
        "acc_train": [],     # fraction correctly classified (rolling window)
    }

    rolling_correct = 0
    rolling_count = 0

    if snapshot_callback is not None:
        snapshot_callback(-1, net, history, rng_data)

    for step in range(n_steps):
        x_seq, label = make_sequence(
            rng_data, T=T, p1=p1, info_amp=info_amp,
            info_sigma=info_sigma, distractor_sigma=distractor_sigma,
        )
        clean_target = label_to_target(label)
        # 3c: target noise at training time only
        noisy_target = clean_target + float(
            rng_data.standard_normal() * target_noise_sigma
        )

        cache = forward(net, x_seq)
        grads = backward(net, cache, noisy_target)
        opt.step([grads[n] for n in net.param_names()])

        # rolling stats use the CLEAN target for sanity
        clean_err = (cache["y_out"][-1] - clean_target) ** 2
        pred_label = 1 if cache["y_out"][-1] > 0.5 else 0
        rolling_correct += int(pred_label == label)
        rolling_count += 1

        if (step + 1) % log_every == 0:
            acc = rolling_correct / rolling_count
            history["step"].append(step + 1)
            history["loss"].append(float(clean_err))
            history["acc_train"].append(float(acc))
            if verbose:
                print(
                    f"step {step + 1:7d}  err={clean_err:.4f}  "
                    f"rolling_acc={acc * 100:5.1f}%  "
                    f"y_out[T-1]={cache['y_out'][-1]:.3f}  "
                    f"target={clean_target:.1f}"
                )
            rolling_correct = 0
            rolling_count = 0

        if (snapshot_callback is not None and snapshot_every > 0
                and ((step + 1) % snapshot_every == 0
                     or step == n_steps - 1)):
            snapshot_callback(step, net, history, rng_data)

    return net, history


# ----------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------

def evaluate(
    net: LSTM1997,
    n_episodes: int = 200,
    T: int = 100,
    p1: int = 10,
    info_amp: float = 1.0,
    info_sigma: float = 0.2,
    distractor_sigma: float = 1.0,
    seed: int = 12345,
):
    """Run on noiseless-target test sequences; report mean abs err and acc."""
    rng = np.random.default_rng(seed)
    abs_errs = []
    correct = 0
    y_outs = []
    labels = []
    for _ in range(n_episodes):
        x_seq, label = make_sequence(
            rng, T=T, p1=p1, info_amp=info_amp,
            info_sigma=info_sigma, distractor_sigma=distractor_sigma,
        )
        cache = forward(net, x_seq)
        target = label_to_target(label)
        y_T = float(cache["y_out"][-1])
        abs_errs.append(abs(y_T - target))
        if (y_T > 0.5) == (label == 1):
            correct += 1
        y_outs.append(y_T)
        labels.append(label)
    return {
        "abs_err_mean": float(np.mean(abs_errs)),
        "abs_err_max": float(np.max(abs_errs)),
        "acc": correct / n_episodes,
        "n_episodes": n_episodes,
        "y_outs": y_outs,
        "labels": labels,
    }


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git": git_hash(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--p1", type=int, default=10)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--cells", type=int, default=2,
                   help="cells per block")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--target-noise", type=float, default=0.32)
    p.add_argument("--save", type=str, default="")
    args = p.parse_args()

    print(f"# two-sequence-noise (variant 3c)  seed={args.seed}")
    for k, v in env_info().items():
        print(f"#   {k}: {v}")

    t0 = time.time()
    net, history = train(
        seed=args.seed,
        n_steps=args.steps,
        T=args.T,
        p1=args.p1,
        n_blocks=args.blocks,
        n_cells_per_block=args.cells,
        target_noise_sigma=args.target_noise,
        lr=args.lr,
        init_scale=args.init_scale,
    )
    train_time = time.time() - t0

    final = evaluate(net, n_episodes=200, T=args.T, p1=args.p1, seed=12345)
    print(
        f"\nFinal eval (200 noiseless test sequences, seed=12345):"
        f"\n  acc            : {final['acc'] * 100:5.1f}%"
        f"\n  mean |err|     : {final['abs_err_mean']:.4f}"
        f"\n  max  |err|     : {final['abs_err_max']:.4f}"
        f"\n  train time     : {train_time:.1f}s"
        f"\n  n_weights      : {net.n_weights()}"
    )

    if args.save:
        out = {
            "args": vars(args),
            "env": env_info(),
            "history": history,
            "final": {k: v for k, v in final.items()
                      if k not in {"y_outs", "labels"}},
            "train_time_s": train_time,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.save}")

    return net, history, final


if __name__ == "__main__":
    main()
