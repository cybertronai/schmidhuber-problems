"""temporal-order-4bit — Hochreiter & Schmidhuber 1997 (NC) Experiment 6b.

Eight-class temporal-order task. A sequence carries *three* "important"
symbols drawn from {X, Y} embedded at unknown positions; everything else
is a random distractor in {a, b, c, d}. The target class encodes the
joint order of the three important symbols across 2^3 = 8 possibilities:
XXX, XXY, XYX, XYY, YXX, YXY, YYX, YYY. Distance between consecutive
markers spans roughly a third of the sequence each, so a vanilla RNN
cannot bridge the gap through the distractor stream.

This is the harder companion to `temporal-order-3bit` (wave-6, Exp 6a).
The 1997 paper reports it as the hardest LSTM benchmark in the original
battery (≈ 571 k sequences with 3 cell blocks of size 2).

This file holds the entire pipeline: dataset generator, vanilla LSTM
with BPTT (no forget gate, matching the 1997 NC paper), vanilla
recurrent net baseline, training loops for both, and a CLI that
reproduces the headline result. Pure numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Token alphabet: 8 symbols.
#   0..3  = distractors a, b, c, d
#   4     = X
#   5     = Y
#   6     = B (begin marker)
#   7     = E (end marker)
DISTRACTORS = (0, 1, 2, 3)
X_TOK, Y_TOK, B_TOK, E_TOK = 4, 5, 6, 7
VOCAB = 8
N_CLASSES = 8

# Class labels: (s1, s2, s3) -> class id, where each s_i ∈ {X, Y}.
#   bit 2 = first, bit 1 = second, bit 0 = third
#   X -> 0, Y -> 1
#   id 0 = XXX, 1 = XXY, 2 = XYX, 3 = XYY,
#   id 4 = YXX, 5 = YXY, 6 = YYX, 7 = YYY
CLASS_NAMES = ("XXX", "XXY", "XYX", "XYY", "YXX", "YXY", "YYX", "YYY")


def _label(s1: int, s2: int, s3: int) -> int:
    a = 0 if s1 == X_TOK else 1
    b = 0 if s2 == X_TOK else 1
    c = 0 if s3 == X_TOK else 1
    return 4 * a + 2 * b + c


def make_sequence(rng: np.random.Generator, *, T: int = 50,
                  t1_range=(3, 9), t2_range=(18, 26),
                  t3_range=(33, 40)) -> tuple[np.ndarray, int]:
    """Build one (one-hot sequence, class) pair.

    Position 0 is B, position T-1 is E. Three slots t1 < t2 < t3 carry
    independently drawn symbols from {X, Y}; every other interior position
    is a random distractor from {a, b, c, d}. The class encodes the
    ordered identity of the three markers as a base-2 number.
    """
    seq = np.empty(T, dtype=np.int64)
    seq[0] = B_TOK
    seq[T - 1] = E_TOK
    for t in range(1, T - 1):
        seq[t] = rng.choice(DISTRACTORS)
    t1 = rng.integers(t1_range[0], t1_range[1] + 1)
    t2 = rng.integers(t2_range[0], t2_range[1] + 1)
    t3 = rng.integers(t3_range[0], t3_range[1] + 1)
    s1 = X_TOK if rng.integers(2) == 0 else Y_TOK
    s2 = X_TOK if rng.integers(2) == 0 else Y_TOK
    s3 = X_TOK if rng.integers(2) == 0 else Y_TOK
    seq[t1] = s1
    seq[t2] = s2
    seq[t3] = s3
    label = _label(s1, s2, s3)
    onehot = np.zeros((T, VOCAB), dtype=np.float64)
    onehot[np.arange(T), seq] = 1.0
    return onehot, label


def make_batch(rng: np.random.Generator, *, n: int, T: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return X of shape (n, T, VOCAB) and y of shape (n,)."""
    X = np.empty((n, T, VOCAB), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        X[i], y[i] = make_sequence(rng, T=T)
    return X, y


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.log(p[np.arange(len(y)), y] + eps).mean())


# ---------------------------------------------------------------------------
# LSTM (1997 NC formulation: input gate, output gate, no forget gate)
# ---------------------------------------------------------------------------

@dataclass
class LSTMParams:
    Wi: np.ndarray  # (V+H, H)   input gate
    bi: np.ndarray  # (H,)
    Wo: np.ndarray  # (V+H, H)   output gate
    bo: np.ndarray  # (H,)
    Wg: np.ndarray  # (V+H, H)   cell input
    bg: np.ndarray  # (H,)
    Why: np.ndarray  # (H, n_classes)
    by: np.ndarray  # (n_classes,)


def init_lstm(rng: np.random.Generator, *, hidden: int = 6,
              vocab: int = VOCAB, n_classes: int = N_CLASSES) -> LSTMParams:
    s = 0.1
    Wi = rng.standard_normal((vocab + hidden, hidden)) * s
    Wo = rng.standard_normal((vocab + hidden, hidden)) * s
    Wg = rng.standard_normal((vocab + hidden, hidden)) * s
    # Bias the input gate negative so the cell stays empty until X/Y arrives.
    bi = -1.0 * np.ones(hidden)
    bo = np.zeros(hidden)
    bg = np.zeros(hidden)
    Why = rng.standard_normal((hidden, n_classes)) * s
    by = np.zeros(n_classes)
    return LSTMParams(Wi, bi, Wo, bo, Wg, bg, Why, by)


def lstm_forward(p: LSTMParams, X: np.ndarray) -> dict:
    """Forward through one batch.

    X: (B, T, V).  Returns cache for backward + final logits.
    """
    B, T, V = X.shape
    H = p.bi.shape[0]
    h = np.zeros((B, H))
    c = np.zeros((B, H))
    i_seq = np.empty((T, B, H))
    o_seq = np.empty((T, B, H))
    g_seq = np.empty((T, B, H))
    c_seq = np.empty((T, B, H))
    h_seq = np.empty((T, B, H))
    tanh_c_seq = np.empty((T, B, H))
    z_seq = np.empty((T, B, V + H))

    for t in range(T):
        z = np.concatenate([X[:, t, :], h], axis=1)
        i = sigmoid(z @ p.Wi + p.bi)
        o = sigmoid(z @ p.Wo + p.bo)
        g = np.tanh(z @ p.Wg + p.bg)
        c = c + i * g  # 1997 NC: no forget gate; pure CEC.
        tanh_c = np.tanh(c)
        h = o * tanh_c
        i_seq[t] = i; o_seq[t] = o; g_seq[t] = g
        c_seq[t] = c; h_seq[t] = h; tanh_c_seq[t] = tanh_c; z_seq[t] = z

    logits = h @ p.Why + p.by  # readout from last hidden
    probs = softmax(logits)
    cache = dict(X=X, i=i_seq, o=o_seq, g=g_seq, c=c_seq, h=h_seq,
                 tanh_c=tanh_c_seq, z=z_seq, probs=probs)
    return cache


def lstm_backward(p: LSTMParams, cache: dict, y: np.ndarray) -> tuple[float, dict]:
    X = cache["X"]
    i_seq, o_seq, g_seq = cache["i"], cache["o"], cache["g"]
    c_seq, h_seq = cache["c"], cache["h"]
    tanh_c_seq, z_seq = cache["tanh_c"], cache["z"]
    probs = cache["probs"]
    B, T, V = X.shape
    H = p.bi.shape[0]

    loss = cross_entropy(probs, y)

    # Output layer gradient.
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    h_T = h_seq[T - 1]
    dWhy = h_T.T @ dlogits
    dby = dlogits.sum(axis=0)
    dh_next = dlogits @ p.Why.T  # gradient flowing into h_T

    dc_next = np.zeros((B, H))
    dWi = np.zeros_like(p.Wi); dbi = np.zeros_like(p.bi)
    dWo = np.zeros_like(p.Wo); dbo = np.zeros_like(p.bo)
    dWg = np.zeros_like(p.Wg); dbg = np.zeros_like(p.bg)

    for t in reversed(range(T)):
        i_t = i_seq[t]; o_t = o_seq[t]; g_t = g_seq[t]
        c_t = c_seq[t]; tanh_c = tanh_c_seq[t]; z_t = z_seq[t]
        c_prev = c_seq[t - 1] if t > 0 else np.zeros_like(c_t)

        # h_t = o_t * tanh(c_t)
        do = dh_next * tanh_c
        dtanh_c = dh_next * o_t
        dc = dc_next + dtanh_c * (1.0 - tanh_c ** 2)
        # c_t = c_{t-1} + i_t * g_t
        di = dc * g_t
        dg = dc * i_t
        dc_prev = dc.copy()  # straight-through, no forget gate

        # gate pre-activations
        di_pre = di * i_t * (1.0 - i_t)
        do_pre = do * o_t * (1.0 - o_t)
        dg_pre = dg * (1.0 - g_t ** 2)

        # weight grads
        dWi += z_t.T @ di_pre
        dbi += di_pre.sum(axis=0)
        dWo += z_t.T @ do_pre
        dbo += do_pre.sum(axis=0)
        dWg += z_t.T @ dg_pre
        dbg += dg_pre.sum(axis=0)

        # gradient on z_t = [x_t; h_{t-1}]
        dz = di_pre @ p.Wi.T + do_pre @ p.Wo.T + dg_pre @ p.Wg.T
        dh_prev = dz[:, V:]

        dh_next = dh_prev
        dc_next = dc_prev

    grads = dict(Wi=dWi, bi=dbi, Wo=dWo, bo=dbo, Wg=dWg, bg=dbg,
                 Why=dWhy, by=dby)
    return loss, grads


# ---------------------------------------------------------------------------
# Vanilla recurrent net baseline: h_t = tanh(W_x x_t + W_h h_{t-1} + b)
# ---------------------------------------------------------------------------

@dataclass
class RNNParams:
    Wx: np.ndarray  # (V, H)
    Wh: np.ndarray  # (H, H)
    bh: np.ndarray  # (H,)
    Why: np.ndarray  # (H, n_classes)
    by: np.ndarray  # (n_classes,)


def init_rnn(rng: np.random.Generator, *, hidden: int = 6,
             vocab: int = VOCAB, n_classes: int = N_CLASSES) -> RNNParams:
    s = 0.1
    Wx = rng.standard_normal((vocab, hidden)) * s
    Wh = rng.standard_normal((hidden, hidden)) * s
    bh = np.zeros(hidden)
    Why = rng.standard_normal((hidden, n_classes)) * s
    by = np.zeros(n_classes)
    return RNNParams(Wx, Wh, bh, Why, by)


def rnn_forward(p: RNNParams, X: np.ndarray) -> dict:
    B, T, V = X.shape
    H = p.bh.shape[0]
    h = np.zeros((B, H))
    h_seq = np.empty((T, B, H))
    for t in range(T):
        h = np.tanh(X[:, t, :] @ p.Wx + h @ p.Wh + p.bh)
        h_seq[t] = h
    logits = h @ p.Why + p.by
    probs = softmax(logits)
    return dict(X=X, h=h_seq, probs=probs)


def rnn_backward(p: RNNParams, cache: dict, y: np.ndarray) -> tuple[float, dict]:
    X = cache["X"]; h_seq = cache["h"]; probs = cache["probs"]
    B, T, V = X.shape
    H = p.bh.shape[0]
    loss = cross_entropy(probs, y)
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    h_T = h_seq[T - 1]
    dWhy = h_T.T @ dlogits
    dby = dlogits.sum(axis=0)
    dh = dlogits @ p.Why.T

    dWx = np.zeros_like(p.Wx); dWh = np.zeros_like(p.Wh); dbh = np.zeros_like(p.bh)
    for t in reversed(range(T)):
        h_t = h_seq[t]
        h_prev = h_seq[t - 1] if t > 0 else np.zeros_like(h_t)
        dpre = dh * (1.0 - h_t ** 2)
        dWx += X[:, t, :].T @ dpre
        dWh += h_prev.T @ dpre
        dbh += dpre.sum(axis=0)
        dh = dpre @ p.Wh.T
    return loss, dict(Wx=dWx, Wh=dWh, bh=dbh, Why=dWhy, by=dby)


# ---------------------------------------------------------------------------
# Adam optimiser (small; works on a dict of arrays)
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(g); self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def lstm_to_dict(p: LSTMParams) -> dict:
    return {k: getattr(p, k) for k in ("Wi", "bi", "Wo", "bo", "Wg", "bg", "Why", "by")}


def dict_to_lstm(d: dict) -> LSTMParams:
    return LSTMParams(**{k: d[k] for k in ("Wi", "bi", "Wo", "bo", "Wg", "bg", "Why", "by")})


def rnn_to_dict(p: RNNParams) -> dict:
    return {k: getattr(p, k) for k in ("Wx", "Wh", "bh", "Why", "by")}


def dict_to_rnn(d: dict) -> RNNParams:
    return RNNParams(**{k: d[k] for k in ("Wx", "Wh", "bh", "Why", "by")})


def grad_clip(grads: dict, max_norm: float = 1.0) -> None:
    total = 0.0
    for g in grads.values():
        total += float((g * g).sum())
    norm = np.sqrt(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for g in grads.values():
            g *= scale


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def evaluate_lstm(p: LSTMParams, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    cache = lstm_forward(p, X)
    pred = cache["probs"].argmax(axis=-1)
    acc = float((pred == y).mean())
    return acc, pred


def evaluate_rnn(p: RNNParams, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    cache = rnn_forward(p, X)
    pred = cache["probs"].argmax(axis=-1)
    acc = float((pred == y).mean())
    return acc, pred


def train_lstm(rng: np.random.Generator, *, T: int, hidden: int, lr: float,
               n_steps: int, batch: int, eval_every: int, val_X: np.ndarray,
               val_y: np.ndarray, record_hidden: bool = False,
               record_indices: tuple[int, ...] = ()) -> dict:
    p = init_lstm(rng, hidden=hidden)
    pdict = lstm_to_dict(p)
    opt = Adam(lr=lr)
    steps = []; train_loss = []; val_acc = []
    snapshots = []  # for GIF
    for step in range(1, n_steps + 1):
        X, y = make_batch(rng, n=batch, T=T)
        cache = lstm_forward(dict_to_lstm(pdict), X)
        loss, grads = lstm_backward(dict_to_lstm(pdict), cache, y)
        grad_clip(grads, max_norm=1.0)
        opt.step(pdict, grads)
        if step == 1 or step % eval_every == 0:
            acc, _ = evaluate_lstm(dict_to_lstm(pdict), val_X, val_y)
            steps.append(step); train_loss.append(loss); val_acc.append(acc)
            if record_hidden:
                rec_X = val_X[list(record_indices)]
                rec_cache = lstm_forward(dict_to_lstm(pdict), rec_X)
                snapshots.append(dict(step=step,
                                       acc=acc,
                                       loss=loss,
                                       c=rec_cache["c"].copy(),
                                       h=rec_cache["h"].copy(),
                                       i=rec_cache["i"].copy(),
                                       o=rec_cache["o"].copy()))
    return dict(params=pdict, steps=steps, train_loss=train_loss,
                val_acc=val_acc, snapshots=snapshots)


def train_rnn(rng: np.random.Generator, *, T: int, hidden: int, lr: float,
              n_steps: int, batch: int, eval_every: int, val_X: np.ndarray,
              val_y: np.ndarray) -> dict:
    p = init_rnn(rng, hidden=hidden)
    pdict = rnn_to_dict(p)
    opt = Adam(lr=lr)
    steps = []; train_loss = []; val_acc = []
    for step in range(1, n_steps + 1):
        X, y = make_batch(rng, n=batch, T=T)
        cache = rnn_forward(dict_to_rnn(pdict), X)
        loss, grads = rnn_backward(dict_to_rnn(pdict), cache, y)
        grad_clip(grads, max_norm=1.0)
        opt.step(pdict, grads)
        if step == 1 or step % eval_every == 0:
            acc, _ = evaluate_rnn(dict_to_rnn(pdict), val_X, val_y)
            steps.append(step); train_loss.append(loss); val_acc.append(acc)
    return dict(params=pdict, steps=steps, train_loss=train_loss, val_acc=val_acc)


# ---------------------------------------------------------------------------
# Gradient check (tiny, used for self-test only)
# ---------------------------------------------------------------------------

def _gradcheck(seed: int = 0, T: int = 8, B: int = 3, H: int = 3) -> float:
    rng = np.random.default_rng(seed)
    p = init_lstm(rng, hidden=H, vocab=VOCAB, n_classes=N_CLASSES)
    X = np.zeros((B, T, VOCAB))
    for b in range(B):
        for t in range(T):
            X[b, t, rng.integers(VOCAB)] = 1.0
    y = rng.integers(0, N_CLASSES, size=B)
    cache = lstm_forward(p, X)
    loss, grads = lstm_backward(p, cache, y)
    eps = 1e-5
    max_err = 0.0
    pdict = lstm_to_dict(p)
    for k in pdict:
        flat = pdict[k].reshape(-1)
        for idx in range(0, flat.size, max(1, flat.size // 5)):
            orig = flat[idx]
            flat[idx] = orig + eps
            l_pos = cross_entropy(lstm_forward(dict_to_lstm(pdict), X)["probs"], y)
            flat[idx] = orig - eps
            l_neg = cross_entropy(lstm_forward(dict_to_lstm(pdict), X)["probs"], y)
            flat[idx] = orig
            num = (l_pos - l_neg) / (2 * eps)
            ana = grads[k].reshape(-1)[idx]
            err = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train LSTM and RNN baseline on temporal-order-4bit.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=50,
                    help="Sequence length (B at 0, E at T-1).")
    ap.add_argument("--hidden", type=int, default=6,
                    help="Hidden / cell count for both LSTM and RNN.")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n_steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val_n", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--record_hidden", action="store_true",
                    help="Capture snapshots of hidden states on a fixed mini-batch (for GIF).")
    ap.add_argument("--snap_path", type=str, default="snapshots.npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.gradcheck:
        err = _gradcheck()
        print(f"[gradcheck] max relative error = {err:.3e}")
        return

    val_rng = np.random.default_rng(args.seed + 1_000_003)
    val_X, val_y = make_batch(val_rng, n=args.val_n, T=args.T)

    record_indices = tuple(_pick_one_per_class(val_y))

    t0 = time.perf_counter()
    lstm_log = train_lstm(np.random.default_rng(args.seed),
                          T=args.T, hidden=args.hidden, lr=args.lr,
                          n_steps=args.n_steps, batch=args.batch,
                          eval_every=args.eval_every, val_X=val_X, val_y=val_y,
                          record_hidden=args.record_hidden,
                          record_indices=record_indices)
    lstm_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rnn_log = train_rnn(np.random.default_rng(args.seed + 7),
                        T=args.T, hidden=args.hidden, lr=args.lr,
                        n_steps=args.n_steps, batch=args.batch,
                        eval_every=args.eval_every, val_X=val_X, val_y=val_y)
    rnn_time = time.perf_counter() - t0

    final_lstm = lstm_log["val_acc"][-1]
    final_rnn = rnn_log["val_acc"][-1]
    best_lstm = max(lstm_log["val_acc"])
    best_rnn = max(rnn_log["val_acc"])
    n_seq_to_solve = _first_solve(lstm_log, threshold=0.95)

    # Confusion matrix on validation set for the LSTM.
    p_final = dict_to_lstm(lstm_log["params"])
    _, lstm_pred = evaluate_lstm(p_final, val_X, val_y)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, pcls in zip(val_y, lstm_pred):
        cm[t, pcls] += 1

    summary = dict(
        seed=args.seed, T=args.T, hidden=args.hidden, lr=args.lr,
        n_steps=args.n_steps, batch=args.batch, val_n=args.val_n,
        n_classes=N_CLASSES,
        lstm_final_acc=final_lstm, rnn_final_acc=final_rnn,
        lstm_best_acc=best_lstm, rnn_best_acc=best_rnn,
        sequences_to_95pct=n_seq_to_solve,
        lstm_wallclock_s=round(lstm_time, 3),
        rnn_wallclock_s=round(rnn_time, 3),
        confusion_lstm=cm.tolist(),
        steps=lstm_log["steps"],
        lstm_loss=lstm_log["train_loss"],
        lstm_acc=lstm_log["val_acc"],
        rnn_loss=rnn_log["train_loss"],
        rnn_acc=rnn_log["val_acc"],
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("steps", "lstm_loss", "lstm_acc",
                                    "rnn_loss", "rnn_acc", "confusion_lstm")},
                     indent=2))
    print("confusion (rows=true class, cols=pred class):")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>3s} -> {row.tolist()}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    if args.record_hidden and lstm_log["snapshots"]:
        snap = lstm_log["snapshots"]
        # internal arrays are (T, n_examples, H); transpose to (n_examples, T, H)
        # so each saved tensor is (n_snapshots, n_examples, T, H).
        def _stack(key: str) -> np.ndarray:
            return np.stack([s[key].transpose(1, 0, 2) for s in snap])
        np.savez(args.snap_path,
                 steps=np.array([s["step"] for s in snap]),
                 acc=np.array([s["acc"] for s in snap]),
                 loss=np.array([s["loss"] for s in snap]),
                 c=_stack("c"), h=_stack("h"),
                 i=_stack("i"), o=_stack("o"),
                 record_X=val_X[list(record_indices)],
                 record_y=val_y[list(record_indices)])


def _first_solve(log: dict, threshold: float = 0.95) -> int | None:
    for s, a in zip(log["steps"], log["val_acc"]):
        if a >= threshold:
            return int(s)
    return None


def _pick_one_per_class(y: np.ndarray) -> list[int]:
    out = []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) > 0:
            out.append(int(idx[0]))
    while len(out) < N_CLASSES:
        out.append(0)
    return out


if __name__ == "__main__":
    main()
