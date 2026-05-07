"""
continual-embedded-reber --- Gers, Schmidhuber, Cummins,
*Learning to forget: continual prediction with LSTM*,
Neural Computation 12(10):2451-2471, 2000.

Setup
-----
A single never-ending symbol stream produced by concatenating embedded
Reber strings without any episode reset. The model must predict the
next symbol at every step. Predicting the second-to-last symbol of each
embedded string requires remembering the outer T/P chosen 6-17 steps
earlier *of that string*, while ignoring outer T/Ps from previous
strings.

Architecture contrast (the headline)
------------------------------------
1. ``LSTMNoForget`` -- Hochreiter & Schmidhuber 1997: input gate +
   output gate, additive cell update with no decay
   (s_t = s_{t-1} + i_t * g_t). On a continual stream the cell state
   accumulates indefinitely; once h_squash saturates, the gates can no
   longer carry distinguishable signals and the long-range outer-T/P
   prediction collapses.
2. ``LSTMForget``   -- Gers, Schmidhuber, Cummins 2000 ("Vanilla LSTM"):
   adds a forget gate f_t so the cell update becomes
   s_t = f_t * s_{t-1} + i_t * g_t. The gate learns to drop towards 0
   at end-of-string markers ('E'), letting the cell silently reset
   between embedded Reber strings while still carrying the outer T/P
   inside each string.

Both networks are trained with identical optimizer/hyperparameters on
identical streams; the only difference is the architecture.

CLI
---
    python3 continual_embedded_reber.py --seed 0
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
# Reber grammar (same as wave-6 embedded-reber)
# ----------------------------------------------------------------------

ALPHABET = ["B", "T", "P", "S", "X", "V", "E"]
SYM2IDX = {s: i for i, s in enumerate(ALPHABET)}
N_SYM = len(ALPHABET)

INNER_REBER = {
    0: [("T", 1), ("P", 2)],
    1: [("S", 1), ("X", 3)],
    2: [("T", 2), ("V", 4)],
    3: [("X", 2), ("S", 5)],
    4: [("P", 3), ("V", 5)],
    5: [("E", None)],
}


def reber_legal_set(state: int) -> set:
    return {sym for sym, _ in INNER_REBER[state]}


def gen_inner_reber(rng: np.random.Generator) -> str:
    out = ["B"]
    state = 0
    while True:
        choices = INNER_REBER[state]
        sym, nxt = choices[rng.integers(len(choices))]
        out.append(sym)
        if nxt is None:
            break
        state = nxt
    return "".join(out)


def gen_embedded_reber(rng: np.random.Generator) -> str:
    """One embedded Reber string: B + outer + inner + outer + E."""
    outer = "T" if rng.integers(2) == 0 else "P"
    inner = gen_inner_reber(rng)
    return "B" + outer + inner + outer + "E"


def gen_continual_stream(rng: np.random.Generator, n_strings: int) -> tuple:
    """Generate a continual stream: ``n_strings`` embedded Reber strings
    concatenated end-to-end.

    Returns (stream_str, boundaries) where ``boundaries`` is a list of
    (start, end) index pairs into ``stream_str`` -- one pair per string,
    half-open. ``stream_str[start:end]`` is one embedded Reber string.
    """
    parts = []
    bounds = []
    pos = 0
    for _ in range(n_strings):
        s = gen_embedded_reber(rng)
        parts.append(s)
        bounds.append((pos, pos + len(s)))
        pos += len(s)
    return "".join(parts), bounds


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------

def encode(string: str) -> np.ndarray:
    arr = np.zeros((len(string), N_SYM), dtype=np.float64)
    for t, c in enumerate(string):
        arr[t, SYM2IDX[c]] = 1.0
    return arr


def make_io(string: str) -> tuple:
    enc = encode(string)
    return enc[:-1], np.argmax(enc[1:], axis=-1)


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def g_squash(z: np.ndarray) -> np.ndarray:
    return 4.0 * sigmoid(z) - 2.0


def g_squash_grad_from_g(g: np.ndarray) -> np.ndarray:
    return (g + 2.0) * (2.0 - g) / 4.0


def h_squash(s: np.ndarray) -> np.ndarray:
    return 2.0 * sigmoid(s) - 1.0


def h_squash_grad_from_h(h: np.ndarray) -> np.ndarray:
    return (h + 1.0) * (1.0 - h) / 2.0


# ----------------------------------------------------------------------
# Original 1997 LSTM (no forget gate)
# ----------------------------------------------------------------------

class LSTMNoForget:
    """Input gate + output gate, additive cell update, no decay.

    Identical to the wave-6 ``LSTM1997`` class. Cell state accumulates
    monotonically along the stream because there is nothing to subtract.
    """

    HAS_FORGET = False

    def __init__(self, n_in=N_SYM, n_hidden=8, n_out=N_SYM,
                 init_scale=0.2, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        nx = n_in + n_hidden
        def W(rows, cols):
            return rng.standard_normal((rows, cols)) * (init_scale / np.sqrt(cols))

        self.W_in = W(n_hidden, nx)
        self.W_out = W(n_hidden, nx)
        self.W_c = W(n_hidden, nx)
        self.b_in = np.full(n_hidden, -1.0)
        self.b_out = np.full(n_hidden, -1.0)
        self.b_c = np.zeros(n_hidden)
        self.W_y = W(n_out, n_hidden)
        self.b_y = np.zeros(n_out)

    def params(self):
        return [self.W_in, self.W_out, self.W_c,
                self.b_in, self.b_out, self.b_c,
                self.W_y, self.b_y]

    def param_names(self):
        return ["W_in", "W_out", "W_c", "b_in", "b_out", "b_c", "W_y", "b_y"]

    def set_params(self, plist):
        (self.W_in, self.W_out, self.W_c,
         self.b_in, self.b_out, self.b_c,
         self.W_y, self.b_y) = plist

    def initial_state(self):
        return np.zeros(self.n_hidden), np.zeros(self.n_hidden)

    def forward(self, X, h0=None, s0=None):
        T = X.shape[0]
        H = self.n_hidden
        h = h0 if h0 is not None else np.zeros(H)
        s = s0 if s0 is not None else np.zeros(H)

        cache = {
            "X": X,
            "h": np.zeros((T + 1, H)), "s": np.zeros((T + 1, H)),
            "i": np.zeros((T, H)), "o": np.zeros((T, H)),
            "g": np.zeros((T, H)), "hs": np.zeros((T, H)),
            "z": np.zeros((T, self.n_in + H)),
            "logits": np.zeros((T, self.n_out)),
        }
        cache["h"][0] = h
        cache["s"][0] = s

        for t in range(T):
            z = np.concatenate([X[t], h])
            i = sigmoid(self.W_in @ z + self.b_in)
            o = sigmoid(self.W_out @ z + self.b_out)
            g = g_squash(self.W_c @ z + self.b_c)
            s = s + i * g
            hs = h_squash(s)
            h = o * hs
            logits = self.W_y @ h + self.b_y

            cache["z"][t] = z
            cache["i"][t] = i
            cache["o"][t] = o
            cache["g"][t] = g
            cache["s"][t + 1] = s
            cache["hs"][t] = hs
            cache["h"][t + 1] = h
            cache["logits"][t] = logits

        return cache

    def loss_and_grads(self, X, y, h0=None, s0=None):
        cache = self.forward(X, h0, s0)
        logits = cache["logits"]
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / ex.sum(axis=1, keepdims=True)
        T = X.shape[0]
        loss = -np.log(probs[np.arange(T), y] + 1e-12).sum()

        dlogits = probs.copy()
        dlogits[np.arange(T), y] -= 1.0
        H = self.n_hidden

        dW_y = dlogits.T @ cache["h"][1:]
        db_y = dlogits.sum(axis=0)

        dW_in = np.zeros_like(self.W_in)
        dW_out = np.zeros_like(self.W_out)
        dW_c = np.zeros_like(self.W_c)
        db_in = np.zeros_like(self.b_in)
        db_out = np.zeros_like(self.b_out)
        db_c = np.zeros_like(self.b_c)

        dh_next = np.zeros(H)
        ds_next = np.zeros(H)
        for t in reversed(range(T)):
            dh = dlogits[t] @ self.W_y + dh_next
            o = cache["o"][t]
            hs = cache["hs"][t]
            i = cache["i"][t]
            g = cache["g"][t]

            do_pre = dh * hs * o * (1.0 - o)
            dhs = dh * o
            ds = dhs * h_squash_grad_from_h(hs) + ds_next

            di_pre = ds * g * i * (1.0 - i)
            dg_pre = ds * i * g_squash_grad_from_g(g)

            z = cache["z"][t]
            dW_in += np.outer(di_pre, z)
            dW_out += np.outer(do_pre, z)
            dW_c += np.outer(dg_pre, z)
            db_in += di_pre
            db_out += do_pre
            db_c += dg_pre

            dz = self.W_in.T @ di_pre + self.W_out.T @ do_pre + self.W_c.T @ dg_pre
            dh_next = dz[self.n_in:]
            ds_next = ds  # additive update -> grad flows unchanged

        grads = {
            "W_in": dW_in, "W_out": dW_out, "W_c": dW_c,
            "b_in": db_in, "b_out": db_out, "b_c": db_c,
            "W_y": dW_y, "b_y": db_y,
        }
        return loss, grads, cache, probs

    def predict(self, X, h0=None, s0=None):
        cache = self.forward(X, h0, s0)
        logits = cache["logits"]
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        return ex / ex.sum(axis=1, keepdims=True), cache


# ----------------------------------------------------------------------
# Vanilla LSTM with forget gate (Gers, Schmidhuber, Cummins 2000)
# ----------------------------------------------------------------------

class LSTMForget:
    """Adds a forget gate. Cell update: s_t = f_t * s_{t-1} + i_t * g_t.

    Forget-gate bias is initialized at +1 ("remember by default"); the
    network is expected to learn to drop f_t towards 0 at end-of-string
    markers in order to reset cell state between embedded strings.
    """

    HAS_FORGET = True

    def __init__(self, n_in=N_SYM, n_hidden=8, n_out=N_SYM,
                 init_scale=0.2, b_forget_init=1.0, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        nx = n_in + n_hidden
        def W(rows, cols):
            return rng.standard_normal((rows, cols)) * (init_scale / np.sqrt(cols))

        self.W_in = W(n_hidden, nx)
        self.W_out = W(n_hidden, nx)
        self.W_f = W(n_hidden, nx)
        self.W_c = W(n_hidden, nx)
        self.b_in = np.full(n_hidden, -1.0)
        self.b_out = np.full(n_hidden, -1.0)
        self.b_f = np.full(n_hidden, b_forget_init)
        self.b_c = np.zeros(n_hidden)
        self.W_y = W(n_out, n_hidden)
        self.b_y = np.zeros(n_out)

    def params(self):
        return [self.W_in, self.W_out, self.W_f, self.W_c,
                self.b_in, self.b_out, self.b_f, self.b_c,
                self.W_y, self.b_y]

    def param_names(self):
        return ["W_in", "W_out", "W_f", "W_c",
                "b_in", "b_out", "b_f", "b_c",
                "W_y", "b_y"]

    def set_params(self, plist):
        (self.W_in, self.W_out, self.W_f, self.W_c,
         self.b_in, self.b_out, self.b_f, self.b_c,
         self.W_y, self.b_y) = plist

    def initial_state(self):
        return np.zeros(self.n_hidden), np.zeros(self.n_hidden)

    def forward(self, X, h0=None, s0=None):
        T = X.shape[0]
        H = self.n_hidden
        h = h0 if h0 is not None else np.zeros(H)
        s = s0 if s0 is not None else np.zeros(H)

        cache = {
            "X": X,
            "h": np.zeros((T + 1, H)), "s": np.zeros((T + 1, H)),
            "i": np.zeros((T, H)), "o": np.zeros((T, H)),
            "f": np.zeros((T, H)),
            "g": np.zeros((T, H)), "hs": np.zeros((T, H)),
            "z": np.zeros((T, self.n_in + H)),
            "logits": np.zeros((T, self.n_out)),
        }
        cache["h"][0] = h
        cache["s"][0] = s

        for t in range(T):
            z = np.concatenate([X[t], h])
            i = sigmoid(self.W_in @ z + self.b_in)
            o = sigmoid(self.W_out @ z + self.b_out)
            f = sigmoid(self.W_f @ z + self.b_f)
            g = g_squash(self.W_c @ z + self.b_c)
            s = f * s + i * g
            hs = h_squash(s)
            h = o * hs
            logits = self.W_y @ h + self.b_y

            cache["z"][t] = z
            cache["i"][t] = i
            cache["o"][t] = o
            cache["f"][t] = f
            cache["g"][t] = g
            cache["s"][t + 1] = s
            cache["hs"][t] = hs
            cache["h"][t + 1] = h
            cache["logits"][t] = logits

        return cache

    def loss_and_grads(self, X, y, h0=None, s0=None):
        cache = self.forward(X, h0, s0)
        logits = cache["logits"]
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / ex.sum(axis=1, keepdims=True)
        T = X.shape[0]
        loss = -np.log(probs[np.arange(T), y] + 1e-12).sum()

        dlogits = probs.copy()
        dlogits[np.arange(T), y] -= 1.0
        H = self.n_hidden

        dW_y = dlogits.T @ cache["h"][1:]
        db_y = dlogits.sum(axis=0)

        dW_in = np.zeros_like(self.W_in)
        dW_out = np.zeros_like(self.W_out)
        dW_f = np.zeros_like(self.W_f)
        dW_c = np.zeros_like(self.W_c)
        db_in = np.zeros_like(self.b_in)
        db_out = np.zeros_like(self.b_out)
        db_f = np.zeros_like(self.b_f)
        db_c = np.zeros_like(self.b_c)

        dh_next = np.zeros(H)
        ds_next = np.zeros(H)
        for t in reversed(range(T)):
            dh = dlogits[t] @ self.W_y + dh_next
            o = cache["o"][t]
            hs = cache["hs"][t]
            i = cache["i"][t]
            f = cache["f"][t]
            g = cache["g"][t]
            s_prev = cache["s"][t]

            do_pre = dh * hs * o * (1.0 - o)
            dhs = dh * o
            ds = dhs * h_squash_grad_from_h(hs) + ds_next

            # s_t = f * s_{t-1} + i * g
            df_pre = ds * s_prev * f * (1.0 - f)
            di_pre = ds * g * i * (1.0 - i)
            dg_pre = ds * i * g_squash_grad_from_g(g)

            z = cache["z"][t]
            dW_in += np.outer(di_pre, z)
            dW_out += np.outer(do_pre, z)
            dW_f += np.outer(df_pre, z)
            dW_c += np.outer(dg_pre, z)
            db_in += di_pre
            db_out += do_pre
            db_f += df_pre
            db_c += dg_pre

            dz = (self.W_in.T @ di_pre
                  + self.W_out.T @ do_pre
                  + self.W_f.T @ df_pre
                  + self.W_c.T @ dg_pre)
            dh_next = dz[self.n_in:]
            ds_next = ds * f  # forget gate scales gradient flowing through CEC

        grads = {
            "W_in": dW_in, "W_out": dW_out, "W_f": dW_f, "W_c": dW_c,
            "b_in": db_in, "b_out": db_out, "b_f": db_f, "b_c": db_c,
            "W_y": dW_y, "b_y": db_y,
        }
        return loss, grads, cache, probs

    def predict(self, X, h0=None, s0=None):
        cache = self.forward(X, h0, s0)
        logits = cache["logits"]
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        return ex / ex.sum(axis=1, keepdims=True), cache


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads_list):
        self.t += 1
        for p, g, m, v in zip(params, grads_list, self.m, self.v):
            m[...] = self.b1 * m + (1.0 - self.b1) * g
            v[...] = self.b2 * v + (1.0 - self.b2) * (g * g)
            mh = m / (1.0 - self.b1 ** self.t)
            vh = v / (1.0 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# Continual evaluation
# ----------------------------------------------------------------------

def outer_acc_by_position(net, n_strings: int, rng: np.random.Generator):
    """Run ``net`` on one fresh continual stream of ``n_strings``
    embedded-Reber strings without any state reset. Return an array of
    length ``n_strings`` whose i-th entry is 1 if the prediction at the
    second-to-last position of the i-th string matched the outer T/P,
    else 0.

    Also returns the cell-state magnitude trace and the full string and
    boundary list for downstream visualization.
    """
    stream, bounds = gen_continual_stream(rng, n_strings)
    X, _ = make_io(stream)
    probs, cache = net.predict(X)

    hits = np.zeros(n_strings, dtype=np.float64)
    legal_hits = np.zeros(n_strings, dtype=np.float64)
    legal_total = np.zeros(n_strings, dtype=np.float64)
    for k, (start, end) in enumerate(bounds):
        # Outer T/P prediction is at the position predicting stream[end-2].
        # Input X[t] predicts stream[t+1]; predict stream[end-2] uses t = end-3.
        t_outer = end - 3
        if 0 <= t_outer < probs.shape[0]:
            pred = ALPHABET[int(np.argmax(probs[t_outer]))]
            hits[k] = 1.0 if pred == stream[start + 1] else 0.0
        # legal-symbol accuracy across the inner positions of this string
        for t in range(start, min(end - 1, probs.shape[0])):
            allowed = _legal_next_in_substring(stream, bounds, t)
            arg = ALPHABET[int(np.argmax(probs[t]))]
            if arg in allowed:
                legal_hits[k] += 1.0
            legal_total[k] += 1.0

    s_trace = cache["s"]   # (T+1, H)
    cell_norm = np.linalg.norm(s_trace, axis=1)
    return {
        "stream": stream,
        "bounds": bounds,
        "outer_hits": hits,
        "legal_hits": legal_hits,
        "legal_total": legal_total,
        "cell_norm": cell_norm,
        "probs": probs,
        "cache": cache,
    }


def _legal_next_in_substring(stream: str, bounds, t: int) -> set:
    """Reber-legal next-symbol set for the position predicting stream[t+1].

    Identifies which embedded-Reber substring t lies in, and what the
    automaton allows after the prefix stream[start:t+1].
    """
    pos = t + 1
    for (start, end) in bounds:
        if start <= pos < end:
            local = pos - start
            L = end - start
            if local == 0:
                # We are at the very first position of a new string. The
                # next symbol is the outer B; the model can't know that
                # boundary, so treat any of the start-of-string set as
                # legal: just B (continual streams start each string with
                # B).
                return {"B"}
            if local == 1:
                return {"T", "P"}
            if local == 2:
                return {"B"}
            if local == L - 1:
                return {"E"}
            if local == L - 2:
                return {stream[start + 1]}
            # inner Reber: replay automaton from state 0
            state = 0
            for s in stream[start + 3:start + local]:
                for sym, nxt in INNER_REBER[state]:
                    if sym == s and nxt is not None:
                        state = nxt
                        break
            return reber_legal_set(state)
    return set()


# ----------------------------------------------------------------------
# Training (truncated BPTT, state carried across chunks)
# ----------------------------------------------------------------------

def train(
    net_class,
    seed: int = 0,
    n_hidden: int = 12,
    lr: float = 1e-2,
    n_chunks: int = 2000,
    chunk_strings: int = 6,
    eval_every: int = 200,
    eval_strings: int = 60,
    grad_clip: float = 5.0,
    snapshot_every: int = 0,
    state_clip: float = 50.0,
    verbose: bool = True,
):
    """Train ``net_class`` (a class, e.g. ``LSTMForget``) on a continual
    stream via truncated BPTT.

    Each chunk is ``chunk_strings`` freshly-sampled embedded-Reber
    strings concatenated. State is carried between chunks; gradient is
    truncated at chunk boundaries.

    ``state_clip`` clips ``|s_t|`` after each chunk to a finite value to
    keep the no-forget-gate run from overflowing the sigmoid clamp; this
    only changes the loss in the saturated regime where the cell is
    already useless, so it does not rescue the no-forget net.
    """
    rng = np.random.default_rng(seed)
    eval_rng_seed = seed + 99999

    net = net_class(n_hidden=n_hidden, rng=rng)
    opt = Adam(net.params(), lr=lr)
    names = net.param_names()

    losses = []
    legal_curve = []
    outer_curve = []
    chunk_index = []

    snapshots = []

    h, s = net.initial_state()

    for c in range(1, n_chunks + 1):
        stream, _ = gen_continual_stream(rng, chunk_strings)
        X, y = make_io(stream)

        loss, grads, cache, _ = net.loss_and_grads(X, y, h0=h, s0=s)
        losses.append(loss / max(1, len(y)))

        # carry state forward; clip s to keep finite without changing topology
        h = cache["h"][-1].copy()
        s = cache["s"][-1].copy()
        if state_clip is not None:
            np.clip(s, -state_clip, state_clip, out=s)

        # gradient clipping (global L2)
        gnorm2 = sum((grads[n] ** 2).sum() for n in names)
        gnorm = float(np.sqrt(gnorm2))
        if gnorm > grad_clip:
            scale = grad_clip / (gnorm + 1e-12)
            for n in names:
                grads[n] *= scale

        opt.step(net.params(), [grads[n] for n in names])

        if c % eval_every == 0 or c == n_chunks:
            ev_rng = np.random.default_rng(eval_rng_seed)
            stats = outer_acc_by_position(net, eval_strings, ev_rng)
            outer = float(stats["outer_hits"].mean())
            legal = float(stats["legal_hits"].sum() / max(1, stats["legal_total"].sum()))
            outer_curve.append(outer)
            legal_curve.append(legal)
            chunk_index.append(c)
            if verbose:
                tail_outer = float(stats["outer_hits"][-min(20, eval_strings):].mean())
                first_outer = float(stats["outer_hits"][:min(5, eval_strings)].mean())
                print(f"  chunk {c:4d}/{n_chunks}  loss/step {np.mean(losses[-eval_every:]):.4f}  "
                      f"legal {legal:.3f}  outer {outer:.3f}  "
                      f"first5 {first_outer:.2f}  last20 {tail_outer:.2f}")

        if snapshot_every > 0 and c % snapshot_every == 0:
            snapshots.append({
                "chunk": c,
                "params": [p.copy() for p in net.params()],
            })

    # final measurement on a long stream (the headline plot)
    long_rng = np.random.default_rng(eval_rng_seed + 1)
    long_stats = outer_acc_by_position(net, eval_strings, long_rng)

    return {
        "net": net,
        "losses": losses,
        "legal_curve": legal_curve,
        "outer_curve": outer_curve,
        "chunk_index": chunk_index,
        "snapshots": snapshots,
        "final_outer": outer_curve[-1] if outer_curve else 0.0,
        "final_legal": legal_curve[-1] if legal_curve else 0.0,
        "long_stats": long_stats,
    }


# ----------------------------------------------------------------------
# Environment / git utilities
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
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-chunks", type=int, default=2000)
    p.add_argument("--chunk-strings", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-strings", type=int, default=60)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--only", choices=["forget", "noforget", "both"], default="both")
    args = p.parse_args()

    results = {}

    if args.only in ("forget", "both"):
        print(f"# continual-embedded-reber  net=LSTMForget  seed={args.seed}")
        t0 = time.time()
        out_f = train(
            LSTMForget,
            seed=args.seed,
            n_hidden=args.hidden,
            lr=args.lr,
            n_chunks=args.n_chunks,
            chunk_strings=args.chunk_strings,
            eval_every=args.eval_every,
            eval_strings=args.eval_strings,
            verbose=not args.quiet,
        )
        dt_f = time.time() - t0
        results["forget"] = {
            "final_outer_acc": out_f["final_outer"],
            "final_legal_acc": out_f["final_legal"],
            "wallclock_sec": round(dt_f, 2),
            "tail20_outer_acc": float(out_f["long_stats"]["outer_hits"][-20:].mean()),
            "first5_outer_acc": float(out_f["long_stats"]["outer_hits"][:5].mean()),
            "mean_cell_norm_late": float(
                out_f["long_stats"]["cell_norm"][-200:].mean()),
        }

    if args.only in ("noforget", "both"):
        print(f"# continual-embedded-reber  net=LSTMNoForget  seed={args.seed}")
        t0 = time.time()
        out_nf = train(
            LSTMNoForget,
            seed=args.seed,
            n_hidden=args.hidden,
            lr=args.lr,
            n_chunks=args.n_chunks,
            chunk_strings=args.chunk_strings,
            eval_every=args.eval_every,
            eval_strings=args.eval_strings,
            verbose=not args.quiet,
        )
        dt_nf = time.time() - t0
        results["noforget"] = {
            "final_outer_acc": out_nf["final_outer"],
            "final_legal_acc": out_nf["final_legal"],
            "wallclock_sec": round(dt_nf, 2),
            "tail20_outer_acc": float(out_nf["long_stats"]["outer_hits"][-20:].mean()),
            "first5_outer_acc": float(out_nf["long_stats"]["outer_hits"][:5].mean()),
            "mean_cell_norm_late": float(
                out_nf["long_stats"]["cell_norm"][-200:].mean()),
        }

    summary = {
        "seed": args.seed,
        "hidden": args.hidden,
        "lr": args.lr,
        "n_chunks": args.n_chunks,
        "chunk_strings": args.chunk_strings,
        "results": results,
        "env": env_info(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
