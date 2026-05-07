"""
blues-improvisation — Eck & Schmidhuber 2002, "A First Look at Music Composition
Using LSTM Recurrent Networks", IDSIA technical report (also NNSP/IJCNN 2002).

Problem:
  12-bar bebop blues. Standard chord progression (4 bars C7, 2 bars F7, 2 bars C7,
  1 bar G7, 1 bar F7, 2 bars C7) repeated. The 1997 LSTM cell has to learn both
  the long-range chord progression (period 12 bars = 96 eighth-notes) AND the
  short-range melody phrasing on top.

  Symbolic vocabulary at each eighth-note step:
    - chord one-hot, |C| = 3   (0=C7, 1=F7, 2=G7)
    - pitch one-hot, |P| = 8   (6 blues-scale tones across two octaves + REST)
                               (0=C3, 1=Eb3, 2=F3, 3=G3, 4=Bb3, 5=C4, 6=Eb4, 7=REST)

  Training corpus (synthesized internally, no external dataset):
    - 8 hand-constructed 12-bar choruses, each 96 steps long
    - All share the canonical bebop-blues chord progression
    - Each has a distinct pentatonic/blues melody that emphasises chord tones on
      strong beats (1, 3) and uses passing tones / rests on weak beats

Architecture:
  Two-layer stacked LSTM (1997 LSTM cell, with forget gate à la Gers/Schmidhuber/
  Cummins 2000):
    - Layer 1 (chord layer, H1=20):  receives raw input,           predicts chord
    - Layer 2 (melody layer, H2=24): receives layer-1 hidden state, predicts pitch
  Two output heads, each a softmax. Cross-entropy loss summed over both heads
  across all timesteps.

  Why this stack mirrors the paper: Eck & Schmidhuber 2002 split the LSTM into
  two memory banks, one biased to long time-scales (chord), one to short
  (melody). A stacked LSTM gives the lower layer a fast pathway to its own
  output while the upper layer uses both x_t and h1_t as input. We document
  this as a "stack instead of partition" deviation in §Deviations.

  BPTT through the full 96-step sequence, all weights manual numpy. Trained
  with Adam + global gradient clipping. Forget-gate bias initialised to 1.0.

CLI:
  python3 blues_improvisation.py --seed 0
  python3 blues_improvisation.py --seed 0 --epochs 200 --save-history history.json
  python3 blues_improvisation.py --gradcheck         # numerical BPTT check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Symbolic blues vocabulary
# ----------------------------------------------------------------------

# Chord vocabulary
CHORDS = ["C7", "F7", "G7"]
N_CHORDS = len(CHORDS)
CHORD_IDX = {c: i for i, c in enumerate(CHORDS)}

# Pitch vocabulary: C blues scale across two octaves + rest
# Indices: 0=C3, 1=Eb3, 2=F3, 3=G3, 4=Bb3, 5=C4, 6=Eb4, 7=REST
PITCH_NAMES = ["C3", "Eb3", "F3", "G3", "Bb3", "C4", "Eb4", "REST"]
N_PITCHES = len(PITCH_NAMES)
REST = N_PITCHES - 1  # last index is rest

# Standard 12-bar bebop blues chord progression, one chord per bar
BLUES_PROGRESSION = ["C7", "C7", "C7", "C7",
                     "F7", "F7", "C7", "C7",
                     "G7", "F7", "C7", "C7"]
BARS_PER_CHORUS = len(BLUES_PROGRESSION)         # 12
STEPS_PER_BAR = 8                                 # eighth notes
STEPS_PER_CHORUS = BARS_PER_CHORUS * STEPS_PER_BAR  # 96

INPUT_DIM = N_CHORDS + N_PITCHES   # 11


# ----------------------------------------------------------------------
# Synthetic training corpus
# ----------------------------------------------------------------------

# Pitch palette per chord. Strong-beat preference is the chord-root tones;
# weak-beat preference includes the b7 / passing notes; rest is a low-prob
# alternative. These are eight hand-constructed melodies: deliberately not
# random, so the LSTM's free-running output can be visually compared to them.

# Chord-tone choices (heavily weighted on beat 1 / 5 of each bar)
ROOT_TONES = {
    "C7": [0, 5, 4, 6],     # C3, C4, Bb3, Eb4
    "F7": [2, 1, 0, 4],     # F3, Eb3, C3, Bb3
    "G7": [3, 2, 4, 1],     # G3, F3, Bb3, Eb3
}
# Passing tones (any blues-scale note, weighted toward downward motion)
PASS_TONES = list(range(N_PITCHES - 1))  # all pitches except REST


def synth_chorus_chord_seq() -> np.ndarray:
    """Return the canonical chord track (96,) of chord indices."""
    c = np.zeros(STEPS_PER_CHORUS, dtype=np.int64)
    for bar, name in enumerate(BLUES_PROGRESSION):
        c[bar * STEPS_PER_BAR:(bar + 1) * STEPS_PER_BAR] = CHORD_IDX[name]
    return c


def synth_chorus_melody(rng: np.random.RandomState,
                        rest_prob_weak: float = 0.20,
                        chord_tone_strength: float = 1.0) -> np.ndarray:
    """Return one (96,) pitch-index sequence over the standard 12-bar progression.

    The melody alternates strong beats (positions 0, 4 within a bar) which
    take a chord-root tone, and weak beats (positions 1, 2, 3, 5, 6, 7) which
    take either a passing tone, a chord tone, or a rest.

    `chord_tone_strength`: 1.0 = paper-faithful, lower = more random,
    higher = stickier on chord tones.
    """
    chords = synth_chorus_chord_seq()
    pitches = np.zeros(STEPS_PER_CHORUS, dtype=np.int64)
    for t in range(STEPS_PER_CHORUS):
        beat_in_bar = t % STEPS_PER_BAR
        chord_name = CHORDS[chords[t]]
        roots = ROOT_TONES[chord_name]
        if beat_in_bar in (0, 4):
            # strong beat → chord root, mostly the root pitch
            weights = np.array([0.55, 0.20, 0.15, 0.10])[:len(roots)]
            weights = weights ** chord_tone_strength
            weights = weights / weights.sum()
            pitches[t] = rng.choice(roots, p=weights)
        else:
            r = rng.random()
            if r < rest_prob_weak:
                pitches[t] = REST
            elif r < rest_prob_weak + 0.45:
                pitches[t] = rng.choice(roots)
            else:
                pitches[t] = rng.choice(PASS_TONES)
    return pitches


def synth_corpus(n_pieces: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a small corpus of (n_pieces, 96) chord/pitch index sequences.

    All pieces share BLUES_PROGRESSION; melodies differ.
    """
    rng = np.random.RandomState(seed)
    chords = np.tile(synth_chorus_chord_seq(), (n_pieces, 1))
    pitches = np.zeros((n_pieces, STEPS_PER_CHORUS), dtype=np.int64)
    for i in range(n_pieces):
        pitches[i] = synth_chorus_melody(rng)
    return chords, pitches


def make_input_array(chords: np.ndarray, pitches: np.ndarray) -> np.ndarray:
    """Convert (B, T) index arrays to (T, B, INPUT_DIM) one-hot input.

    Layout: concatenated [chord_one_hot (3), pitch_one_hot (8)].
    """
    B, T = chords.shape
    X = np.zeros((T, B, INPUT_DIM), dtype=np.float64)
    bb = np.arange(B)
    for t in range(T):
        X[t, bb, chords[:, t]] = 1.0
        X[t, bb, N_CHORDS + pitches[:, t]] = 1.0
    return X


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def dsig(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


def dtanh(y: np.ndarray) -> np.ndarray:
    return 1.0 - y * y


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax along the last axis."""
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# Two-layer LSTM with output heads (manual BPTT)
# ----------------------------------------------------------------------

@dataclass
class Model:
    # Layer 1
    W1x: np.ndarray
    W1h: np.ndarray
    b1: np.ndarray
    # Layer 2
    W2x: np.ndarray  # input is layer-1 hidden
    W2h: np.ndarray
    b2: np.ndarray
    # Chord head (from layer 1)
    Wc: np.ndarray
    bc: np.ndarray
    # Pitch head (from layer 2)
    Wp: np.ndarray
    bp: np.ndarray

    def keys(self):
        return ["W1x", "W1h", "b1",
                "W2x", "W2h", "b2",
                "Wc", "bc", "Wp", "bp"]

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_model(input_dim: int, h1: int, h2: int,
               rng: np.random.RandomState,
               init_scale: float = 0.5) -> Model:
    sx1 = (1.0 / math.sqrt(input_dim)) * init_scale
    sh1 = (1.0 / math.sqrt(h1)) * init_scale
    sx2 = (1.0 / math.sqrt(h1)) * init_scale
    sh2 = (1.0 / math.sqrt(h2)) * init_scale
    W1x = rng.randn(input_dim, 4 * h1) * sx1
    W1h = rng.randn(h1, 4 * h1) * sh1
    b1 = np.zeros(4 * h1)
    b1[h1:2 * h1] = 1.0     # forget-gate bias
    W2x = rng.randn(h1, 4 * h2) * sx2
    W2h = rng.randn(h2, 4 * h2) * sh2
    b2 = np.zeros(4 * h2)
    b2[h2:2 * h2] = 1.0     # forget-gate bias
    Wc = rng.randn(h1, N_CHORDS) * (1.0 / math.sqrt(h1)) * 0.5
    bc = np.zeros(N_CHORDS)
    Wp = rng.randn(h2, N_PITCHES) * (1.0 / math.sqrt(h2)) * 0.5
    bp = np.zeros(N_PITCHES)
    return Model(W1x=W1x, W1h=W1h, b1=b1,
                 W2x=W2x, W2h=W2h, b2=b2,
                 Wc=Wc, bc=bc, Wp=Wp, bp=bp)


def forward(p: Model, X: np.ndarray):
    """Forward pass. X: (T, B, D_in).

    Returns:
      logits_c: (T, B, N_CHORDS)
      logits_p: (T, B, N_PITCHES)
      cache for BPTT
    """
    T, B, D = X.shape
    H1 = p.W1h.shape[0]
    H2 = p.W2h.shape[0]
    h1 = np.zeros((T + 1, B, H1))
    c1 = np.zeros((T + 1, B, H1))
    h2 = np.zeros((T + 1, B, H2))
    c2 = np.zeros((T + 1, B, H2))
    i1 = np.zeros((T, B, H1)); f1 = np.zeros((T, B, H1))
    g1 = np.zeros((T, B, H1)); o1 = np.zeros((T, B, H1))
    tc1 = np.zeros((T, B, H1))
    i2 = np.zeros((T, B, H2)); f2 = np.zeros((T, B, H2))
    g2 = np.zeros((T, B, H2)); o2 = np.zeros((T, B, H2))
    tc2 = np.zeros((T, B, H2))

    for t in range(T):
        z1 = X[t] @ p.W1x + h1[t] @ p.W1h + p.b1   # (B, 4H1)
        i1[t] = sigmoid(z1[:, 0:H1])
        f1[t] = sigmoid(z1[:, H1:2 * H1])
        g1[t] = np.tanh(z1[:, 2 * H1:3 * H1])
        o1[t] = sigmoid(z1[:, 3 * H1:4 * H1])
        c1[t + 1] = f1[t] * c1[t] + i1[t] * g1[t]
        tc1[t] = np.tanh(c1[t + 1])
        h1[t + 1] = o1[t] * tc1[t]

        z2 = h1[t + 1] @ p.W2x + h2[t] @ p.W2h + p.b2
        i2[t] = sigmoid(z2[:, 0:H2])
        f2[t] = sigmoid(z2[:, H2:2 * H2])
        g2[t] = np.tanh(z2[:, 2 * H2:3 * H2])
        o2[t] = sigmoid(z2[:, 3 * H2:4 * H2])
        c2[t + 1] = f2[t] * c2[t] + i2[t] * g2[t]
        tc2[t] = np.tanh(c2[t + 1])
        h2[t + 1] = o2[t] * tc2[t]

    # Predict chord_{t+1} from h1[t+1], pitch_{t+1} from h2[t+1].
    H1_seq = h1[1:]   # (T, B, H1)
    H2_seq = h2[1:]   # (T, B, H2)
    logits_c = H1_seq @ p.Wc + p.bc          # (T, B, N_CHORDS)
    logits_p = H2_seq @ p.Wp + p.bp          # (T, B, N_PITCHES)

    cache = dict(X=X, h1=h1, c1=c1, i1=i1, f1=f1, g1=g1, o1=o1, tc1=tc1,
                 h2=h2, c2=c2, i2=i2, f2=f2, g2=g2, o2=o2, tc2=tc2,
                 logits_c=logits_c, logits_p=logits_p)
    return logits_c, logits_p, cache


def loss_and_grads(p: Model, cache: dict, target_c: np.ndarray,
                   target_p: np.ndarray):
    """Cross-entropy loss summed over chord and pitch heads, mean over (T,B).

    target_c: (T, B) chord indices,  target_p: (T, B) pitch indices
    Returns scalar loss and grads dict.
    """
    X = cache["X"]
    h1 = cache["h1"]; c1 = cache["c1"]
    i1 = cache["i1"]; f1 = cache["f1"]; g1 = cache["g1"]
    o1 = cache["o1"]; tc1 = cache["tc1"]
    h2 = cache["h2"]; c2 = cache["c2"]
    i2 = cache["i2"]; f2 = cache["f2"]; g2 = cache["g2"]
    o2 = cache["o2"]; tc2 = cache["tc2"]
    logits_c = cache["logits_c"]
    logits_p = cache["logits_p"]
    T, B, _ = X.shape
    H1 = p.W1h.shape[0]
    H2 = p.W2h.shape[0]

    pc = softmax(logits_c)   # (T, B, N_CHORDS)
    pp = softmax(logits_p)   # (T, B, N_PITCHES)

    # Average cross-entropy per (timestep, batch element)
    bb = np.arange(B)
    loss_c = 0.0
    loss_p = 0.0
    for t in range(T):
        loss_c -= np.log(np.maximum(pc[t, bb, target_c[t]], 1e-12)).mean()
        loss_p -= np.log(np.maximum(pp[t, bb, target_p[t]], 1e-12)).mean()
    loss_c /= T
    loss_p /= T
    loss = loss_c + loss_p

    # Backward
    grads = {k: np.zeros_like(p.get(k)) for k in p.keys()}

    # dL/dlogits_c, dL/dlogits_p
    dlogits_c = pc.copy()
    dlogits_p = pp.copy()
    for t in range(T):
        dlogits_c[t, bb, target_c[t]] -= 1.0
        dlogits_p[t, bb, target_p[t]] -= 1.0
    # divide by T*B for mean and head normalization (each head was 1/(T*B))
    dlogits_c /= (T * B)
    dlogits_p /= (T * B)

    H1_seq = h1[1:]
    H2_seq = h2[1:]
    # Heads: logits = h @ W + b
    grads["Wc"] = np.einsum("tbh,tbk->hk", H1_seq, dlogits_c)
    grads["bc"] = dlogits_c.sum(axis=(0, 1))
    grads["Wp"] = np.einsum("tbh,tbk->hk", H2_seq, dlogits_p)
    grads["bp"] = dlogits_p.sum(axis=(0, 1))

    # dh per timestep coming from heads
    dh1_from_head = dlogits_c @ p.Wc.T   # (T, B, H1)
    dh2_from_head = dlogits_p @ p.Wp.T   # (T, B, H2)

    dh2_next = np.zeros((B, H2))
    dc2_next = np.zeros((B, H2))
    dh1_next = np.zeros((B, H1))
    dc1_next = np.zeros((B, H1))
    # We'll accumulate dh1 from layer-2 input contributions across time
    # by collecting them as we backprop layer 2.
    dh1_from_layer2 = np.zeros((T, B, H1))

    for t in reversed(range(T)):
        # Layer 2 backward
        dh2 = dh2_next + dh2_from_head[t]
        do2 = dh2 * tc2[t]
        dtc2 = dh2 * o2[t]
        dc2 = dc2_next + dtc2 * dtanh(tc2[t])
        df2 = dc2 * c2[t]
        dc2_prev = dc2 * f2[t]
        di2 = dc2 * g2[t]
        dg2 = dc2 * i2[t]
        dz2 = np.concatenate([
            di2 * dsig(i2[t]),
            df2 * dsig(f2[t]),
            dg2 * dtanh(g2[t]),
            do2 * dsig(o2[t]),
        ], axis=1)
        # Layer-2 gets x = h1[t+1]; record gradient w.r.t. that into the
        # right time-step slot.
        grads["W2x"] += h1[t + 1].T @ dz2
        grads["W2h"] += h2[t].T @ dz2
        grads["b2"] += dz2.sum(axis=0)
        dh1_from_layer2[t] = dz2 @ p.W2x.T
        dh2_next = dz2 @ p.W2h.T
        dc2_next = dc2_prev

        # Layer 1 backward
        dh1 = dh1_next + dh1_from_head[t] + dh1_from_layer2[t]
        do1 = dh1 * tc1[t]
        dtc1 = dh1 * o1[t]
        dc1 = dc1_next + dtc1 * dtanh(tc1[t])
        df1 = dc1 * c1[t]
        dc1_prev = dc1 * f1[t]
        di1 = dc1 * g1[t]
        dg1 = dc1 * i1[t]
        dz1 = np.concatenate([
            di1 * dsig(i1[t]),
            df1 * dsig(f1[t]),
            dg1 * dtanh(g1[t]),
            do1 * dsig(o1[t]),
        ], axis=1)
        grads["W1x"] += X[t].T @ dz1
        grads["W1h"] += h1[t].T @ dz1
        grads["b1"] += dz1.sum(axis=0)
        dh1_next = dz1 @ p.W1h.T
        dc1_next = dc1_prev

    return loss, grads, loss_c, loss_p


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: Model, lr=5e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, clip=2.0):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.clip = clip; self.t = 0
        self.m = {k: np.zeros_like(params.get(k)) for k in params.keys()}
        self.v = {k: np.zeros_like(params.get(k)) for k in params.keys()}

    def step(self, params: Model, grads: dict):
        if self.clip is not None:
            tot = math.sqrt(sum(float((grads[k] ** 2).sum())
                                for k in grads))
            if tot > self.clip:
                s = self.clip / (tot + 1e-12)
                for k in grads:
                    grads[k] = grads[k] * s
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in params.keys():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / bc1
            vh = self.v[k] / bc2
            params.set(k, params.get(k) - self.lr * mh
                       / (np.sqrt(vh) + self.eps))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

@dataclass
class History:
    epochs: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    loss_c: List[float] = field(default_factory=list)
    loss_p: List[float] = field(default_factory=list)
    chord_acc: List[float] = field(default_factory=list)
    pitch_acc: List[float] = field(default_factory=list)

    def to_dict(self):
        return {k: list(v) for k, v in self.__dict__.items()}


def make_targets(chords: np.ndarray, pitches: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Build next-step targets given (B, T) chord/pitch index arrays.

    target_c[t] = chord at step t+1 (we predict the next step from h_{t+1}).
    For the final timestep, we wrap around (the corpus is periodic by design).
    """
    B, T = chords.shape
    chords_next = np.roll(chords, -1, axis=1)
    pitches_next = np.roll(pitches, -1, axis=1)
    # (B, T) → (T, B)
    return chords_next.T.copy(), pitches_next.T.copy()


def evaluate_seqaccuracy(p: Model, X: np.ndarray,
                         target_c: np.ndarray,
                         target_p: np.ndarray) -> Tuple[float, float]:
    logits_c, logits_p, _ = forward(p, X)
    pred_c = logits_c.argmax(axis=-1)   # (T, B)
    pred_p = logits_p.argmax(axis=-1)   # (T, B)
    acc_c = (pred_c == target_c).mean()
    acc_p = (pred_p == target_p).mean()
    return float(acc_c), float(acc_p)


def train(seed: int, h1: int = 20, h2: int = 24,
          n_pieces: int = 8, epochs: int = 200, batch_size: int = 8,
          lr: float = 8e-3, lr_decay_every: int = 80,
          lr_decay_factor: float = 0.5,
          eval_every: int = 5, save_snapshots: bool = False,
          verbose: bool = True):
    rng_corpus = np.random.RandomState(seed + 11)
    rng_init = np.random.RandomState(seed + 7)
    rng_train = np.random.RandomState(seed)

    chords, pitches = synth_corpus(n_pieces=n_pieces, seed=seed + 11)
    X = make_input_array(chords, pitches)            # (T, B, D)
    target_c, target_p = make_targets(chords, pitches)

    params = init_model(INPUT_DIM, h1=h1, h2=h2, rng=rng_init)
    opt = Adam(params, lr=lr, clip=2.0)
    history = History()
    snapshots = []

    n_train = chords.shape[0]
    for ep in range(1, epochs + 1):
        if lr_decay_every and ep > 1 and (ep - 1) % lr_decay_every == 0:
            opt.lr *= lr_decay_factor

        # full-batch training (corpus is small)
        # We can shuffle order but with fixed RNG it is reproducible.
        order = rng_train.permutation(n_train)
        # Process in batches
        ep_loss = 0.0; ep_loss_c = 0.0; ep_loss_p = 0.0; n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            Xb = X[:, idx, :]
            tcb = target_c[:, idx]
            tpb = target_p[:, idx]
            _, _, cache = forward(params, Xb)
            loss, grads, lc, lp = loss_and_grads(params, cache, tcb, tpb)
            opt.step(params, grads)
            ep_loss += loss; ep_loss_c += lc; ep_loss_p += lp; n_batches += 1
        ep_loss /= n_batches; ep_loss_c /= n_batches; ep_loss_p /= n_batches

        if ep == 1 or ep % eval_every == 0 or ep == epochs:
            acc_c, acc_p = evaluate_seqaccuracy(params, X, target_c, target_p)
            history.epochs.append(ep)
            history.loss.append(ep_loss)
            history.loss_c.append(ep_loss_c)
            history.loss_p.append(ep_loss_p)
            history.chord_acc.append(acc_c)
            history.pitch_acc.append(acc_p)
            if verbose:
                print(f"  epoch {ep:4d}  loss {ep_loss:.4f}  "
                      f"loss_c {ep_loss_c:.4f}  loss_p {ep_loss_p:.4f}  "
                      f"chord_acc {acc_c:.3f}  pitch_acc {acc_p:.3f}  "
                      f"lr {opt.lr:.1e}")
            if save_snapshots:
                gen_c, gen_p = generate(
                    params, n_steps=STEPS_PER_CHORUS,
                    seed=seed + 999, temperature=0.85,
                )
                snapshots.append(dict(
                    epoch=ep,
                    loss=ep_loss,
                    loss_c=ep_loss_c,
                    loss_p=ep_loss_p,
                    chord_acc=acc_c,
                    pitch_acc=acc_p,
                    gen_c=gen_c.copy(),
                    gen_p=gen_p.copy(),
                ))

    return params, history, snapshots, (chords, pitches)


# ----------------------------------------------------------------------
# Free-running generation
# ----------------------------------------------------------------------

def generate(p: Model, n_steps: int, seed: int = 0,
             temperature: float = 0.85,
             chord_temperature: float = None,
             primer_chord: int = 0, primer_pitch: int = 0
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Free-running: sample one step at a time using model's predicted dist.

    `temperature` controls pitch sampling.
    `chord_temperature` controls chord sampling. If None, uses `temperature`.
    A temperature of 0 means deterministic argmax.
    """
    if chord_temperature is None:
        chord_temperature = temperature
    rng = np.random.RandomState(seed)
    H1 = p.W1h.shape[0]
    H2 = p.W2h.shape[0]
    h1 = np.zeros((1, H1)); c1 = np.zeros((1, H1))
    h2 = np.zeros((1, H2)); c2 = np.zeros((1, H2))

    out_c = np.zeros(n_steps, dtype=np.int64)
    out_p = np.zeros(n_steps, dtype=np.int64)
    cur_c = primer_chord
    cur_p = primer_pitch

    for t in range(n_steps):
        x = np.zeros((1, INPUT_DIM))
        x[0, cur_c] = 1.0
        x[0, N_CHORDS + cur_p] = 1.0
        z1 = x @ p.W1x + h1 @ p.W1h + p.b1
        i1g = sigmoid(z1[:, 0:H1])
        f1g = sigmoid(z1[:, H1:2 * H1])
        g1g = np.tanh(z1[:, 2 * H1:3 * H1])
        o1g = sigmoid(z1[:, 3 * H1:4 * H1])
        c1 = f1g * c1 + i1g * g1g
        h1 = o1g * np.tanh(c1)
        z2 = h1 @ p.W2x + h2 @ p.W2h + p.b2
        i2g = sigmoid(z2[:, 0:H2])
        f2g = sigmoid(z2[:, H2:2 * H2])
        g2g = np.tanh(z2[:, 2 * H2:3 * H2])
        o2g = sigmoid(z2[:, 3 * H2:4 * H2])
        c2 = f2g * c2 + i2g * g2g
        h2 = o2g * np.tanh(c2)
        logits_c = (h1 @ p.Wc + p.bc).reshape(-1)
        logits_p = (h2 @ p.Wp + p.bp).reshape(-1)
        if chord_temperature <= 0:
            cur_c = int(np.argmax(logits_c))
        else:
            pc = softmax(logits_c / chord_temperature)
            cur_c = int(rng.choice(N_CHORDS, p=pc))
        if temperature <= 0:
            cur_p = int(np.argmax(logits_p))
        else:
            pp = softmax(logits_p / temperature)
            cur_p = int(rng.choice(N_PITCHES, p=pp))
        out_c[t] = cur_c
        out_p[t] = cur_p
    return out_c, out_p


# ----------------------------------------------------------------------
# Music-theoretic evaluation metrics
# ----------------------------------------------------------------------

def chord_progression_match(gen_c: np.ndarray) -> float:
    """Fraction of timesteps where generated chord matches BLUES_PROGRESSION."""
    target = synth_chorus_chord_seq()
    if len(gen_c) != len(target):
        m = min(len(gen_c), len(target))
        return float((gen_c[:m] == target[:m]).mean())
    return float((gen_c == target).mean())


def bar_onset_chord_match(gen_c: np.ndarray) -> float:
    """Fraction of bar-onsets (steps 0, 8, 16, …) whose chord matches the
    canonical 12-bar progression. The headline structural metric."""
    target = synth_chorus_chord_seq()
    onsets = np.arange(0, len(gen_c), STEPS_PER_BAR)
    if len(gen_c) != len(target):
        m = min(len(gen_c), len(target))
        onsets = onsets[onsets < m]
    return float((gen_c[onsets] == target[onsets]).mean())


def on_beat_note_rate(gen_p: np.ndarray) -> float:
    """Fraction of strong-beat steps (positions 0, 4 mod 8) that are NOT rest."""
    strong_idx = np.array([t for t in range(len(gen_p))
                           if t % STEPS_PER_BAR in (0, 4)])
    if len(strong_idx) == 0:
        return 0.0
    return float((gen_p[strong_idx] != REST).mean())


def chord_tone_rate(gen_c: np.ndarray, gen_p: np.ndarray) -> float:
    """Fraction of non-rest steps where the pitch belongs to the current chord's
    root-tone palette."""
    n_total = 0; n_match = 0
    for t in range(len(gen_p)):
        if gen_p[t] == REST:
            continue
        chord_name = CHORDS[gen_c[t]]
        if gen_p[t] in ROOT_TONES[chord_name]:
            n_match += 1
        n_total += 1
    if n_total == 0:
        return 0.0
    return n_match / n_total


# ----------------------------------------------------------------------
# Numerical gradient check
# ----------------------------------------------------------------------

def gradcheck(seed: int = 0, h1: int = 4, h2: int = 5, T: int = 12,
              B: int = 2, eps: float = 1e-5, n_checks: int = 12):
    rng = np.random.RandomState(seed)
    p = init_model(INPUT_DIM, h1=h1, h2=h2, rng=rng)
    # Random short sequence
    chords = rng.randint(0, N_CHORDS, size=(B, T))
    pitches = rng.randint(0, N_PITCHES, size=(B, T))
    X = make_input_array(chords, pitches)
    tc, tp = make_targets(chords, pitches)
    _, _, cache = forward(p, X)
    loss, grads, _, _ = loss_and_grads(p, cache, tc, tp)

    def total_loss(p_):
        _, _, cc = forward(p_, X)
        l, _, _, _ = loss_and_grads(p_, cc, tc, tp)
        return l

    rel_errs = []
    crng = np.random.RandomState(seed + 1)
    for k in p.keys():
        W = p.get(k)
        flat = W.reshape(-1)
        analytic = grads[k].reshape(-1)
        idxs = crng.choice(flat.size,
                           size=min(n_checks, flat.size), replace=False)
        for i in idxs:
            saved = flat[i]
            flat[i] = saved + eps
            lp = total_loss(p)
            flat[i] = saved - eps
            lm = total_loss(p)
            flat[i] = saved
            num = (lp - lm) / (2 * eps)
            an = analytic[i]
            denom = max(1e-12, abs(num) + abs(an))
            rel = abs(num - an) / denom
            rel_errs.append((k, i, num, an, rel))
    max_rel = max(r[-1] for r in rel_errs)
    print(f"gradcheck: max relative error = {max_rel:.2e} "
          f"over {len(rel_errs)} samples (loss = {loss:.4f})")
    return max_rel


# ----------------------------------------------------------------------
# Reproducibility metadata
# ----------------------------------------------------------------------

def env_info():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def pretty_print_chorus(chords: np.ndarray, pitches: np.ndarray,
                        title: str = "") -> None:
    if title:
        print(title)
    grid = []
    for bar in range(BARS_PER_CHORUS):
        bar_str = ""
        for step in range(STEPS_PER_BAR):
            t = bar * STEPS_PER_BAR + step
            ch = CHORDS[chords[t]]
            pn = PITCH_NAMES[pitches[t]]
            if pn == "REST":
                pn_disp = "."
            else:
                pn_disp = pn
            bar_str += f"{pn_disp:>4} "
        chord_label = CHORDS[chords[bar * STEPS_PER_BAR]]
        grid.append(f"  bar {bar + 1:2d} [{chord_label:>3}] | {bar_str}")
    for row in grid:
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h1", type=int, default=20, help="layer-1 hidden size (chord)")
    ap.add_argument("--h2", type=int, default=24, help="layer-2 hidden size (melody)")
    ap.add_argument("--n-pieces", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--lr-decay-every", type=int, default=80)
    ap.add_argument("--lr-decay-factor", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--save-history", type=str, default=None)
    ap.add_argument("--print-corpus", action="store_true",
                    help="print the synthesized first chorus then exit")
    args = ap.parse_args()

    if args.gradcheck:
        gradcheck()
        return

    if args.print_corpus:
        chords, pitches = synth_corpus(n_pieces=1, seed=args.seed + 11)
        pretty_print_chorus(chords[0], pitches[0],
                            title="Synthesized chorus 1:")
        return

    print(f"blues-improvisation  seed={args.seed}  h1={args.h1}  h2={args.h2}  "
          f"epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"  env: {env_info()}")
    t0 = time.time()
    params, history, _, (chords, pitches) = train(
        seed=args.seed, h1=args.h1, h2=args.h2,
        n_pieces=args.n_pieces, epochs=args.epochs,
        batch_size=args.batch, lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        eval_every=args.eval_every,
        save_snapshots=False,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Two free-running generations:
    #   1) deterministic chord (argmax) + sampled pitch  → headline metric
    #   2) sampled both                                   → musical variety
    det_c, det_p = generate(params, n_steps=STEPS_PER_CHORUS,
                            seed=args.seed + 999,
                            temperature=args.temperature,
                            chord_temperature=0.0)
    smp_c, smp_p = generate(params, n_steps=STEPS_PER_CHORUS,
                            seed=args.seed + 999,
                            temperature=args.temperature)
    det_match = chord_progression_match(det_c)
    det_bar_match = bar_onset_chord_match(det_c)
    det_on_beat = on_beat_note_rate(det_p)
    det_chord_tone = chord_tone_rate(det_c, det_p)
    smp_match = chord_progression_match(smp_c)
    smp_bar_match = bar_onset_chord_match(smp_c)

    print(f"\n[final] elapsed {elapsed:.1f}s")
    print(f"  train chord-acc {history.chord_acc[-1]:.3f}  "
          f"pitch-acc {history.pitch_acc[-1]:.3f}")
    print(f"  deterministic gen: bar-onset chord match = {det_bar_match:.3f}  "
          f"step-level chord match = {det_match:.3f}")
    print(f"                     on-beat note rate = {det_on_beat:.3f}  "
          f"chord-tone rate = {det_chord_tone:.3f}")
    print(f"  sampled gen:       bar-onset chord match = {smp_bar_match:.3f}  "
          f"step-level chord match = {smp_match:.3f}")
    pretty_print_chorus(det_c, det_p,
                        title="\nFree-running chorus (argmax chord, sampled melody):")
    # for the saved JSON / downstream visuals we keep the deterministic version
    gen_c, gen_p = det_c, det_p
    chord_match = det_match
    on_beat = det_on_beat
    chord_tone = det_chord_tone
    bar_match = det_bar_match

    if args.save_history:
        out = {
            "args": vars(args),
            "env": env_info(),
            "history": history.to_dict(),
            "elapsed_sec": elapsed,
            "gen_c": gen_c.tolist(),
            "gen_p": gen_p.tolist(),
            "chord_match": chord_match,
            "bar_onset_chord_match": bar_match,
            "on_beat_note_rate": on_beat,
            "chord_tone_rate": chord_tone,
            "sampled_chord_match": smp_match,
            "sampled_bar_onset_chord_match": smp_bar_match,
            "sampled_gen_c": smp_c.tolist(),
            "sampled_gen_p": smp_p.tolist(),
        }
        with open(args.save_history, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.save_history}")


if __name__ == "__main__":
    main()
