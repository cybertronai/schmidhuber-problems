"""multiplication-problem (Hochreiter & Schmidhuber 1997, Experiment 5).

Task: at each timestep the network sees (x_real, x_marker) where
    x_real ~ U[0, 1]
    x_marker = -1 at first and last position
              = +1 at exactly two earlier positions (one in the first 10
                steps, one in the first T/2-1 steps)
              = 0 elsewhere
At the final step the network must emit the *product* of the two real
values that were marked. The adding-problem (Experiment 4) emits the
sum; only the target function differs.

Architecture: vanilla LSTM with a forget gate, single block of H cells,
sigmoid output, MSE loss. BPTT with Adam.

Pure numpy + matplotlib. Deterministic given --seed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_sequence(T: int, rng: np.random.Generator):
    """One sequence of length T: returns (X, target).

    X has shape (T, 2): column 0 = real value, column 1 = marker.
    target is a Python float = X[p1, 0] * X[p2, 0].
    """
    if T < 12:
        raise ValueError("T must be at least 12 (need room for both markers)")
    x_real = rng.uniform(0.0, 1.0, size=T).astype(np.float32)
    x_mark = np.zeros(T, dtype=np.float32)
    x_mark[0] = -1.0
    x_mark[-1] = -1.0
    # Marker positions: paper says one in the first 10 steps and one in the
    # first T/2 - 1 steps. We exclude position 0 (already a -1 sentinel).
    p1 = int(rng.integers(1, 10))
    # Second marker strictly later than the first, in [10, T//2 - 1].
    p2_high = max(11, T // 2)
    p2 = int(rng.integers(10, p2_high))
    if p2 == p1:
        p2 = (p2 + 1) % T
    x_mark[p1] = 1.0
    x_mark[p2] = 1.0
    target = float(x_real[p1]) * float(x_real[p2])
    X = np.stack([x_real, x_mark], axis=1).astype(np.float32)
    return X, target


def make_batch(batch_size: int, T_min: int, T_max: int, rng: np.random.Generator):
    """Variable-length batch: every sequence shares one T sampled from [T_min, T_max]."""
    T = int(rng.integers(T_min, T_max + 1))
    X = np.zeros((batch_size, T, 2), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    for b in range(batch_size):
        Xb, yb = make_sequence(T, rng)
        X[b] = Xb
        y[b] = yb
    return X, y, T


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


@dataclass
class LSTM:
    """Single-layer LSTM with forget gate. No peepholes, no projections.

    Gates packed in order [i, f, o, g]. Output is a sigmoid linear projection
    of the final hidden state.
    """

    D: int = 2
    H: int = 8

    def __post_init__(self):
        rng = np.random.default_rng(0)
        self._init_params(rng)

    def _init_params(self, rng: np.random.Generator):
        D, H = self.D, self.H
        # Xavier-ish init.
        scale_x = np.sqrt(1.0 / D)
        scale_h = np.sqrt(1.0 / H)
        self.Wx = rng.uniform(-scale_x, scale_x, size=(D, 4 * H)).astype(np.float32)
        self.Wh = rng.uniform(-scale_h, scale_h, size=(H, 4 * H)).astype(np.float32)
        self.b = np.zeros(4 * H, dtype=np.float32)
        # Forget-gate bias = 1 (Gers 1999): keeps memory by default.
        self.b[H : 2 * H] = 1.0
        scale_y = np.sqrt(1.0 / H)
        self.Wy = rng.uniform(-scale_y, scale_y, size=(H, 1)).astype(np.float32)
        self.by = np.zeros(1, dtype=np.float32)

    def reset_with_seed(self, seed: int):
        self._init_params(np.random.default_rng(seed))

    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wy, self.by]

    def param_names(self):
        return ["Wx", "Wh", "b", "Wy", "by"]

    # ----- forward / backward ---------------------------------------------------

    def forward(self, X):
        """X: (B, T, D). Returns (y, cache).

        y has shape (B,), squashed through sigmoid (target ∈ [0, 1]).
        cache stores everything needed for BPTT.
        """
        B, T, D = X.shape
        H = self.H
        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)

        ifog = np.zeros((T, B, 4 * H), dtype=np.float32)
        c_hist = np.zeros((T + 1, B, H), dtype=np.float32)
        h_hist = np.zeros((T + 1, B, H), dtype=np.float32)
        c_hist[0] = c
        h_hist[0] = h

        for t in range(T):
            x_t = X[:, t, :]
            pre = x_t @ self.Wx + h @ self.Wh + self.b
            i = sigmoid(pre[:, 0:H])
            f = sigmoid(pre[:, H : 2 * H])
            o = sigmoid(pre[:, 2 * H : 3 * H])
            g = np.tanh(pre[:, 3 * H : 4 * H])
            ifog[t, :, 0:H] = i
            ifog[t, :, H : 2 * H] = f
            ifog[t, :, 2 * H : 3 * H] = o
            ifog[t, :, 3 * H : 4 * H] = g
            c = f * c + i * g
            h = o * np.tanh(c)
            c_hist[t + 1] = c
            h_hist[t + 1] = h

        y_pre = h @ self.Wy + self.by  # (B, 1)
        y_pre = y_pre.squeeze(-1)
        y = sigmoid(y_pre)
        cache = (X, ifog, c_hist, h_hist, y, T)
        return y, cache

    def backward(self, target, cache):
        X, ifog, c_hist, h_hist, y, T = cache
        B = X.shape[0]
        H = self.H
        # Loss = mean squared error, sigmoid output.
        # dL/dy_pre = (2/B) * (y - t) * y * (1 - y)
        d_y = (2.0 / B) * (y - target) * y * (1.0 - y)  # (B,)

        h_T = h_hist[T]
        dWy = h_T.T @ d_y.reshape(-1, 1)
        dby = np.array([d_y.sum()], dtype=np.float32)
        dh = d_y.reshape(-1, 1) @ self.Wy.T  # (B, H)

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dc = np.zeros((B, H), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            i = ifog[t, :, 0:H]
            f = ifog[t, :, H : 2 * H]
            o = ifog[t, :, 2 * H : 3 * H]
            g = ifog[t, :, 3 * H : 4 * H]
            c_t = c_hist[t + 1]
            c_prev = c_hist[t]
            h_prev = h_hist[t]
            tanh_c = np.tanh(c_t)

            do = dh * tanh_c
            dc_total = dc + dh * o * (1.0 - tanh_c ** 2)
            df = dc_total * c_prev
            dc_prev = dc_total * f
            di = dc_total * g
            dg = dc_total * i

            di_pre = di * i * (1.0 - i)
            df_pre = df * f * (1.0 - f)
            do_pre = do * o * (1.0 - o)
            dg_pre = dg * (1.0 - g ** 2)
            dpre = np.concatenate([di_pre, df_pre, do_pre, dg_pre], axis=1)

            x_t = X[:, t, :]
            dWx += x_t.T @ dpre
            dWh += h_prev.T @ dpre
            db += dpre.sum(axis=0)
            dh = dpre @ self.Wh.T
            dc = dc_prev

        return [dWx, dWh, db, dWy, dby]


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads, clip=1.0):
        self.t += 1
        # Global gradient clipping.
        if clip is not None:
            total = 0.0
            for g in grads:
                total += float((g * g).sum())
            norm = np.sqrt(total)
            if norm > clip:
                scale = clip / (norm + 1e-12)
                grads = [g * scale for g in grads]
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1.0 - self.b1) * g
            v[:] = self.b2 * v + (1.0 - self.b2) * (g * g)
            m_hat = m / (1.0 - self.b1 ** self.t)
            v_hat = v / (1.0 - self.b2 ** self.t)
            p -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.dtype)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int = 0
    T_min: int = 20
    T_max: int = 30
    hidden: int = 8
    lr: float = 5e-3
    batch_size: int = 32
    max_iters: int = 6000
    log_every: int = 200
    eval_every: int = 500
    eval_batch: int = 512
    eval_T: int = 30
    target_test_mse: float = 0.030


def evaluate(model, n: int, T: int, rng: np.random.Generator):
    X, y_true, _ = make_batch(n, T, T, rng)
    y_pred, _ = model.forward(X)
    mse = float(((y_pred - y_true) ** 2).mean())
    return mse, y_pred, y_true


def train(cfg: TrainConfig, save_dir=None, verbose=True):
    rng = np.random.default_rng(cfg.seed)
    model = LSTM(D=2, H=cfg.hidden)
    model.reset_with_seed(cfg.seed)
    opt = Adam(model.params(), lr=cfg.lr)

    train_losses = []
    test_curve = []  # list of (iter, test_mse)
    eval_rng = np.random.default_rng(cfg.seed + 1_000_003)

    t0 = time.perf_counter()
    final_mse = None
    for it in range(1, cfg.max_iters + 1):
        X, y_true, T = make_batch(cfg.batch_size, cfg.T_min, cfg.T_max, rng)
        y_pred, cache = model.forward(X)
        loss = float(((y_pred - y_true) ** 2).mean())
        train_losses.append(loss)
        grads = model.backward(y_true, cache)
        opt.step(model.params(), grads, clip=1.0)

        if it % cfg.log_every == 0 and verbose:
            avg = float(np.mean(train_losses[-cfg.log_every:]))
            print(f"iter {it:5d}  T={T:3d}  train_mse={avg:.4f}")

        if it % cfg.eval_every == 0 or it == cfg.max_iters:
            mse, _, _ = evaluate(model, cfg.eval_batch, cfg.eval_T, eval_rng)
            test_curve.append((it, mse))
            if verbose:
                print(f"  >>> test_mse @ T={cfg.eval_T}: {mse:.4f}")
            final_mse = mse
            if mse < cfg.target_test_mse:
                if verbose:
                    print(f"  reached target {cfg.target_test_mse}, stopping early.")
                break

    elapsed = time.perf_counter() - t0
    history = {
        "train_losses": train_losses,
        "test_curve": test_curve,
        "elapsed_sec": elapsed,
        "final_test_mse": final_mse,
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "weights.npz"),
            Wx=model.Wx, Wh=model.Wh, b=model.b, Wy=model.Wy, by=model.by,
        )
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "test_curve": test_curve,
                    "elapsed_sec": elapsed,
                    "final_test_mse": final_mse,
                    "config": asdict(cfg),
                    "env": {
                        "python": sys.version.split()[0],
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                    },
                },
                f,
                indent=2,
            )
    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Hochreiter & Schmidhuber 1997 multiplication-problem (Experiment 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T-min", type=int, default=20)
    p.add_argument("--T-max", type=int, default=30)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=6000)
    p.add_argument("--target-test-mse", type=float, default=0.030)
    p.add_argument("--save-dir", type=str, default="run")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        T_min=args.T_min,
        T_max=args.T_max,
        hidden=args.hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        target_test_mse=args.target_test_mse,
    )
    model, hist = train(cfg, save_dir=args.save_dir, verbose=not args.quiet)
    print("\n=== summary ===")
    print(f"seed                = {cfg.seed}")
    print(f"final test MSE @T={cfg.eval_T} = {hist['final_test_mse']:.4f}")
    print(f"sequences seen      = {len(hist['train_losses']) * cfg.batch_size}")
    print(f"wallclock           = {hist['elapsed_sec']:.1f} s")


if __name__ == "__main__":
    main()
