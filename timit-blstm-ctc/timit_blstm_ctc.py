"""timit-blstm-ctc -- Graves & Schmidhuber, *Framewise Phoneme Classification
with Bidirectional LSTM and Other Neural Network Architectures*, Neural
Networks 18 (2005); Graves, Fernandez, Gomez, Schmidhuber, *Connectionist
Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent
Neural Networks*, ICML 2006.

The 2005/2006 Graves+Schmidhuber pair argues two things together:

  1. Bidirectional LSTM (BLSTM) beats unidirectional LSTM, BRNN, and
     time-windowed MLPs on TIMIT framewise phoneme classification, because
     phoneme identity is influenced by both past and future acoustic
     context.
  2. CTC removes the need for pre-segmented training data: the network
     emits a per-frame distribution over labels (plus a special "blank"),
     and the CTC forward-backward decoder marginalises over all frame-to-
     label alignments consistent with the target label sequence.

Per SPEC issue #1 (cybertronai/schmidhuber-problems), v1 stubs use
pure-numpy synthetic data instead of TIMIT itself (which was originally
v1.5-deferred for the external dataset). The synthetic phoneme corpus
captures the structural property the algorithm exploits: short, locally
characteristic acoustic units concatenated into variable-length sequences
*without* frame-level alignment labels. CTC + BLSTM must (a) learn the
spectral signature of each phoneme from frame features alone and (b)
discover the alignment to the unsegmented label sequence.

Synthetic phoneme corpus
------------------------

  K = 6 "phonemes" plus a CTC blank symbol (index 0).
  Each phoneme has a *characteristic* low-frequency content concentrated
  in 1-2 of `n_features = 8` mel-like bands. Token k's frame at time t is

      f_kj(t) = A_k * cos(2 pi w_kj t / W_kj + phi_kj) + noise

  averaged into the j-th band, where w_kj is sampled per (token, band)
  from a phoneme-specific distribution. Each phoneme realisation has
  variable length (5..15 frames), and consecutive phonemes are
  separated by 2..5 frames of low-amplitude background noise (silence).
  The full sequence has 3..8 phonemes -> total length T ~ 25..170 frames.

  Importantly the *training labels* are the phoneme sequence only -- the
  per-frame phoneme is hidden. CTC is what makes that work.

Architecture
------------

  - Bidirectional LSTM cell (Gers/Schmidhuber/Cummins 2000 forget gate).
    Two independent LSTMs run forward and backward over the sequence;
    their hidden states are concatenated at each time step.
  - Linear projection (2H -> K+1) -> softmax over the CTC alphabet
    (K phoneme labels + blank).
  - CTC forward-backward computed in log-space (no underflow), with
    closed-form gradient on the softmax pre-activation:
        dL/da_t,k = y_t,k - (1/P) * sum_{s: l'_s = k} alpha_t(s) beta_t(s)

  - A *unidirectional* LSTM baseline of the same hidden size is also
    trained, to confirm the "BLSTM beats forward-only" headline.

CLI
---

  python3 timit_blstm_ctc.py --seed 0                # train BLSTM (default)
  python3 timit_blstm_ctc.py --seed 0 --uni          # train uni-LSTM
  python3 timit_blstm_ctc.py --gradcheck             # numerical gradient check
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


# ============================================================================
# Synthetic phoneme corpus (pure numpy)
# ============================================================================

@dataclass
class CorpusConfig:
    n_phonemes: int = 6              # K phoneme classes (CTC alphabet = K+1)
    n_features: int = 8              # mel-like band count
    min_phonemes_per_seq: int = 3
    max_phonemes_per_seq: int = 8
    min_frames_per_phoneme: int = 4
    max_frames_per_phoneme: int = 10
    min_silence_frames: int = 2
    max_silence_frames: int = 5
    noise_std: float = 0.18          # higher SNR difficulty
    silence_amp: float = 0.05
    phoneme_amp: float = 1.0
    # Co-articulation: each phoneme shares an "onset" formant band with one
    # neighbour; the distinguishing formant only fully emerges in the second
    # half of the phoneme. This is what makes future context useful: at the
    # *start* of a phoneme, past + present alone can't tell some pairs apart.
    coarticulation: bool = True
    onset_share_bands: int = 1       # number of bands shared at onset
    onset_fraction: float = 0.45     # fraction of phoneme dominated by shared onset


def make_phoneme_signatures(cfg: CorpusConfig,
                            rng: np.random.RandomState) -> dict:
    """Pick fixed (seeded) spectral signatures per phoneme.

    Each phoneme k gets:
      - `late_centers[k]`: a length-F profile peaked at the phoneme's
        DISTINGUISHING formant band(s). This is what dominates the second
        half of the phoneme.
      - `early_centers[k]`: a length-F profile peaked at a SHARED onset
        formant band that's the same across the phoneme's "neighbour group"
        of size 2-3. This is what dominates the first ~`onset_fraction` of
        the phoneme. With this in place, the first few frames of a
        phoneme are ambiguous between members of the neighbour group, and
        a uni-directional LSTM has a harder time than a BLSTM.
      - `freqs[k][j]`, `phases[k][j]`: per-band oscillation parameters.
    """
    K = cfg.n_phonemes
    F = cfg.n_features
    late_centers = np.zeros((K, F))
    early_centers = np.zeros((K, F))
    # Distinct "late" formants (1-2 bands per phoneme).
    for k in range(K):
        n_formants = int(rng.choice([1, 2]))
        formant_bands = rng.choice(F, size=n_formants, replace=False)
        for b in formant_bands:
            late_centers[k, b] = 1.0
        late_centers[k] += 0.08
    # Shared "onset" formants. Group phonemes into clusters of 2 (or 3 if odd).
    perm = rng.permutation(K)
    cluster_starts = list(range(0, K, 2))
    for cs in cluster_starts:
        members = perm[cs: cs + 2] if cs + 2 <= K else perm[cs: K]
        share_bands = rng.choice(F, size=cfg.onset_share_bands, replace=False)
        for k in members:
            for b in share_bands:
                early_centers[k, b] = 1.0
            early_centers[k] += 0.08
    if not cfg.coarticulation:
        early_centers = late_centers.copy()
    freqs = rng.uniform(0.05, 0.40, size=(K, F))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(K, F))
    return dict(late_centers=late_centers, early_centers=early_centers,
                freqs=freqs, phases=phases)


def render_phoneme(k: int, n_frames: int, sig: dict, cfg: CorpusConfig,
                   rng: np.random.RandomState) -> np.ndarray:
    """Render one realisation of phoneme k for n_frames frames -> (n_frames, F).

    The frame's mean spectrum interpolates from `early_centers[k]` (the
    onset, possibly shared with a neighbour phoneme) to `late_centers[k]`
    (the distinguishing payload). The interpolation knee is at
    `cfg.onset_fraction * n_frames`.
    """
    F = cfg.n_features
    t = np.arange(n_frames).reshape(-1, 1)
    # Mixing weight from early -> late along the phoneme.
    knee = max(1, int(round(cfg.onset_fraction * n_frames)))
    w_early = np.zeros(n_frames)
    if knee > 0:
        # Smooth ramp from 1 -> 0 across [0, knee], hold at 0 afterward.
        w_early[:knee] = np.linspace(1.0, 0.0, knee)
    w_late = 1.0 - w_early
    base = (w_early[:, None] * sig["early_centers"][k][None, :]
            + w_late[:, None] * sig["late_centers"][k][None, :])  # (T, F)
    osc = 0.35 * np.cos(sig["freqs"][k][None, :] * t + sig["phases"][k][None, :])
    amp = base + osc
    # Envelope so phoneme starts and ends are softer than the middle.
    env_t = np.linspace(0, 1, n_frames)
    envelope = (0.6 + 0.4 * np.sin(np.pi * env_t)).reshape(-1, 1)
    frames = cfg.phoneme_amp * envelope * amp
    frames = frames + rng.randn(n_frames, F) * cfg.noise_std
    return frames


def render_silence(n_frames: int, cfg: CorpusConfig,
                   rng: np.random.RandomState) -> np.ndarray:
    F = cfg.n_features
    return cfg.silence_amp * rng.randn(n_frames, F).astype(np.float64)


def make_sequence(cfg: CorpusConfig, sig: dict,
                  rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Sample one (X, labels) sequence.

    X       : (T, F)   acoustic features
    labels  : (L,)     phoneme labels in {1, .., K}  (label 0 reserved for CTC blank)
    """
    L = rng.randint(cfg.min_phonemes_per_seq, cfg.max_phonemes_per_seq + 1)
    pieces = []
    labels = np.empty(L, dtype=np.int64)
    # Optional initial silence
    if rng.uniform() < 0.5:
        pieces.append(render_silence(
            rng.randint(cfg.min_silence_frames, cfg.max_silence_frames + 1),
            cfg, rng))
    for i in range(L):
        k = rng.randint(0, cfg.n_phonemes)  # 0..K-1
        labels[i] = k + 1  # shift: 0 reserved for blank
        n = rng.randint(cfg.min_frames_per_phoneme,
                        cfg.max_frames_per_phoneme + 1)
        pieces.append(render_phoneme(k, n, sig, cfg, rng))
        if i < L - 1:  # silence between
            n_sil = rng.randint(cfg.min_silence_frames,
                                cfg.max_silence_frames + 1)
            pieces.append(render_silence(n_sil, cfg, rng))
    if rng.uniform() < 0.5:
        pieces.append(render_silence(
            rng.randint(cfg.min_silence_frames, cfg.max_silence_frames + 1),
            cfg, rng))
    X = np.concatenate(pieces, axis=0)
    return X, labels


def make_batch(cfg: CorpusConfig, sig: dict, rng: np.random.RandomState,
               batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Pad to longest sequence in the batch.

    Returns:
        X      : (T_max, B, F)
        x_lens : (B,)
        labels : list of length-B int arrays  (labels in {1..K})
        l_lens : (B,)
    """
    seqs = [make_sequence(cfg, sig, rng) for _ in range(batch_size)]
    Xs = [s[0] for s in seqs]
    Ls = [s[1] for s in seqs]
    T_max = max(x.shape[0] for x in Xs)
    F = cfg.n_features
    X = np.zeros((T_max, batch_size, F))
    x_lens = np.zeros(batch_size, dtype=np.int64)
    for b, x in enumerate(Xs):
        X[: x.shape[0], b, :] = x
        x_lens[b] = x.shape[0]
    l_lens = np.array([len(l) for l in Ls], dtype=np.int64)
    return X, x_lens, Ls, l_lens


# ============================================================================
# Activations
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def dsigmoid_y(y):
    return y * (1.0 - y)


def dtanh_y(y):
    return 1.0 - y * y


def softmax_logits(a):
    """Stable softmax along the last axis."""
    m = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - m)
    return e / np.sum(e, axis=-1, keepdims=True)


# ============================================================================
# LSTM cell with manual forward + BPTT
# ============================================================================

@dataclass
class LSTMParams:
    Wx: np.ndarray
    Wh: np.ndarray
    b: np.ndarray
    name: str = "lstm"

    def keys(self):
        return ["Wx", "Wh", "b"]

    def get(self, k):
        return getattr(self, k)

    def set(self, k, v):
        setattr(self, k, v)


def init_lstm(input_dim: int, H: int, rng: np.random.RandomState,
              name: str = "lstm") -> LSTMParams:
    sx = 1.0 / math.sqrt(input_dim)
    sh = 1.0 / math.sqrt(H)
    Wx = rng.randn(input_dim, 4 * H) * sx * 0.5
    Wh = rng.randn(H, 4 * H) * sh * 0.5
    b = np.zeros(4 * H)
    # Forget-gate bias = 1.0  (Gers/Schmidhuber/Cummins 2000)
    b[H:2 * H] = 1.0
    return LSTMParams(Wx=Wx, Wh=Wh, b=b, name=name)


def lstm_forward(p: LSTMParams, X: np.ndarray, mask: np.ndarray):
    """Run LSTM left-to-right.

    X    : (T, B, D)
    mask : (T, B)       1 inside the sequence, 0 outside  (used to freeze state).

    Returns:
        H_out : (T, B, H)   per-time hidden state (with masked steps copied
                            from the previous valid step)
        cache : dict for backprop
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
    for t in range(T):
        z = X[t] @ p.Wx + h[t] @ p.Wh + p.b
        i_g[t] = sigmoid(z[:, 0:H])
        f_g[t] = sigmoid(z[:, H:2 * H])
        g_g[t] = np.tanh(z[:, 2 * H:3 * H])
        o_g[t] = sigmoid(z[:, 3 * H:4 * H])
        c_new = f_g[t] * c[t] + i_g[t] * g_g[t]
        tc_new = np.tanh(c_new)
        h_new = o_g[t] * tc_new
        m = mask[t][:, None]
        c[t + 1] = m * c_new + (1.0 - m) * c[t]
        h[t + 1] = m * h_new + (1.0 - m) * h[t]
        tc[t] = tc_new
    H_out = h[1:].copy()
    cache = dict(X=X, mask=mask, h=h, c=c, i=i_g, f=f_g, g=g_g, o=o_g, tc=tc)
    return H_out, cache


def lstm_backward(p: LSTMParams, cache: dict,
                  dH_out: np.ndarray) -> tuple[dict, np.ndarray]:
    """BPTT given dL/dh_t for every t. Returns grads + dL/dX."""
    X = cache["X"]
    mask = cache["mask"]
    h = cache["h"]
    c = cache["c"]
    i_g, f_g, g_g, o_g, tc = cache["i"], cache["f"], cache["g"], cache["o"], cache["tc"]
    T, B, D = X.shape
    H = p.Wh.shape[0]

    grads = {k: np.zeros_like(p.get(k)) for k in p.keys()}
    dX = np.zeros_like(X)
    dh_next = np.zeros((B, H))
    dc_next = np.zeros((B, H))

    for t in reversed(range(T)):
        m = mask[t][:, None]  # (B, 1)
        # h[t+1] = m * h_new + (1-m) * h[t]   => dh_new = m * dh, dh_t += (1-m) * dh
        dh_total = dH_out[t] + dh_next
        dh_new = m * dh_total
        dh_pass_through = (1.0 - m) * dh_total
        # c[t+1] similarly:
        dc_total_into_step = dc_next  # gradient onto c_{t+1}
        dc_new = m * dc_total_into_step
        dc_pass_through = (1.0 - m) * dc_total_into_step

        # h_new = o_t * tanh(c_new)
        do_t = dh_new * tc[t]
        dtc_new = dh_new * o_g[t]
        # c_new = f * c[t] + i * g
        dc_full = dc_new + dtc_new * dtanh_y(tc[t])
        df_t = dc_full * c[t]
        dc_t = dc_full * f_g[t] + dc_pass_through
        di_t = dc_full * g_g[t]
        dg_t = dc_full * i_g[t]
        # Pre-activations
        dz_i = di_t * dsigmoid_y(i_g[t])
        dz_f = df_t * dsigmoid_y(f_g[t])
        dz_g = dg_t * dtanh_y(g_g[t])
        dz_o = do_t * dsigmoid_y(o_g[t])
        dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)
        grads["Wx"] += X[t].T @ dz
        grads["Wh"] += h[t].T @ dz
        grads["b"] += dz.sum(axis=0)
        dX[t] = dz @ p.Wx.T
        dh_next = dz @ p.Wh.T + dh_pass_through
        dc_next = dc_t

    return grads, dX


# ============================================================================
# BLSTM = forward LSTM + backward LSTM (concatenated states)
# ============================================================================

def reverse_seq(X: np.ndarray, x_lens: np.ndarray) -> np.ndarray:
    """Reverse each sample's valid prefix; padding (after x_lens[b]) stays put."""
    T, B = X.shape[0], X.shape[1]
    out = X.copy()
    for b in range(B):
        n = x_lens[b]
        out[:n, b] = X[:n, b][::-1]
    return out


def reverse_seq_3d_back(X_rev_grad: np.ndarray, x_lens: np.ndarray) -> np.ndarray:
    """Inverse of reverse_seq: same operation (involution)."""
    return reverse_seq(X_rev_grad, x_lens)


# ============================================================================
# CTC loss (log-space forward-backward)
# ============================================================================

LOG_ZERO = -1e18


def logsumexp_pair(a, b):
    """Stable log(exp(a) + exp(b)). Works on scalars or arrays."""
    m = np.maximum(a, b)
    safe = np.where(np.isneginf(m), 0.0, m)
    return safe + np.log(np.exp(a - safe) + np.exp(b - safe))


def expand_label(labels: np.ndarray, blank: int = 0) -> np.ndarray:
    """l of length L -> l' of length 2L+1 with blanks interleaved."""
    L = len(labels)
    lp = np.full(2 * L + 1, blank, dtype=np.int64)
    lp[1::2] = labels
    return lp


def ctc_loss_and_grad(logp: np.ndarray, labels: np.ndarray,
                      x_len: int, blank: int = 0):
    """Log-space CTC for ONE sample.

    logp   : (T, K_full)  log-softmax over the full alphabet (blank + phonemes)
    labels : (L,)         label sequence in {1..K}
    x_len  : int          valid length of logp along T
    Returns:
        nll   : float        -log P(l | x)
        dL_da : (T, K_full)  gradient w.r.t. softmax PRE-activations
                             (zero outside x_len)
    """
    T_total, K_full = logp.shape
    T = x_len
    L = len(labels)
    if L == 0:
        # All-blank sequence. Probability = product of blank probs at each step.
        nll = -float(logp[:T, blank].sum())
        # Gradient: dL/da_tk = exp(logp_tk) - 1[k==blank]
        y = np.exp(logp)
        dL_da = np.zeros_like(logp)
        dL_da[:T] = y[:T]
        dL_da[:T, blank] -= 1.0
        return nll, dL_da

    if T < L:
        # Impossible alignment (need at least L emissions).
        # Return the per-frame negative log prob of the labels averaged --
        # this gives a non-zero gradient that still pushes labels up but
        # this case is filtered out at dataset generation.
        # Fall back: treat as if alignment doesn't exist; loss = +inf.
        # In practice we should never hit this branch with our generator.
        y = np.exp(logp)
        dL_da = np.zeros_like(logp)
        dL_da[:T] = y[:T]
        dL_da[:T, labels[0]] -= 1.0
        return 1e6, dL_da

    lp = expand_label(labels, blank=blank)  # (2L+1,)
    S = len(lp)

    # Pre-compute the "skip-2 allowed" mask for each s in [2, S):
    # alpha can absorb alpha_{s-2} only when lp[s] != blank and lp[s] != lp[s-2].
    # Equivalently: at odd s = 2i+1 with lp[s] != lp[s-2].
    allow3 = np.zeros(S, dtype=bool)
    for s in range(2, S):
        if lp[s] != blank and lp[s] != lp[s - 2]:
            allow3[s] = True

    # Per-time emission log-probs along the expanded label sequence.
    emit = logp[:, lp]  # (T, S)

    log_alpha = np.full((T, S), LOG_ZERO)
    log_alpha[0, 0] = logp[0, lp[0]]
    if S >= 2:
        log_alpha[0, 1] = logp[0, lp[1]]

    for t in range(1, T):
        prev = log_alpha[t - 1]                       # (S,)
        # Three predecessor sources, each shifted into position s.
        a0 = prev                                     # contributes at s
        a1 = np.full(S, LOG_ZERO)
        a1[1:] = prev[:-1]                            # contributes at s>=1
        a2 = np.full(S, LOG_ZERO)
        a2[2:] = prev[:-2]                            # contributes at s>=2 if allow3
        a2 = np.where(allow3, a2, LOG_ZERO)
        # logsumexp of three values along an axis -- stable.
        stack = np.stack([a0, a1, a2], axis=0)        # (3, S)
        m = np.max(stack, axis=0)
        m_safe = np.where(np.isneginf(m), 0.0, m)
        log_alpha[t] = m_safe + np.log(
            np.sum(np.exp(stack - m_safe[None, :]), axis=0)) + emit[t]

    log_beta = np.full((T, S), LOG_ZERO)
    log_beta[T - 1, S - 1] = 0.0
    if S >= 2:
        log_beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = log_beta[t + 1] + emit[t + 1]           # (S,)
        b0 = nxt
        b1 = np.full(S, LOG_ZERO)
        b1[:-1] = nxt[1:]
        # For the s+2 jump, the *destination* lp[s+2] must satisfy the rule;
        # but allow3 was indexed by destination s, so the source mask is
        # "for each s, can we use nxt[s+2]?" -- yes iff allow3[s+2].
        b2 = np.full(S, LOG_ZERO)
        b2[:-2] = nxt[2:]
        # mask by allow3 at s+2:
        skip_mask = np.zeros(S, dtype=bool)
        skip_mask[:-2] = allow3[2:]
        b2 = np.where(skip_mask, b2, LOG_ZERO)
        stack = np.stack([b0, b1, b2], axis=0)
        m = np.max(stack, axis=0)
        m_safe = np.where(np.isneginf(m), 0.0, m)
        log_beta[t] = m_safe + np.log(
            np.sum(np.exp(stack - m_safe[None, :]), axis=0))

    # log P(l|x) from alpha at final step: sum of alpha_T(2L) and alpha_T(2L-1).
    final_a = log_alpha[T - 1, S - 1]
    if S >= 2:
        final_a = logsumexp_pair(final_a, log_alpha[T - 1, S - 2])
    log_P = final_a
    nll = -float(log_P)

    # Gradient on softmax pre-activations:
    # dL/da_tk = y_tk - (1/P) * sum_{s: l'_s=k} alpha_t(s) beta_t(s)
    #         = exp(logp_tk) - exp( logsumexp_{s: l'_s=k} (log alpha + log beta) - logP )
    log_ab = log_alpha + log_beta  # (T, S)
    # Per-(t, k) bucket logsumexp over s with l'_s == k. We stable-shift by
    # the per-time max (over S) and accumulate exp into bucket sums.
    m_t = np.max(log_ab, axis=1, keepdims=True)              # (T, 1)
    m_t_safe = np.where(np.isneginf(m_t), 0.0, m_t)
    weights = np.exp(log_ab - m_t_safe)                       # (T, S)
    bucket = np.zeros((T, K_full))
    np.add.at(bucket, (slice(None), lp), weights)             # per-class sum
    bucket = np.maximum(bucket, 1e-300)                       # avoid log 0
    log_gamma = np.log(bucket) + m_t_safe                     # restore shift
    # Where m_t was -inf, all buckets stay LOG_ZERO (zero occupancy).
    log_gamma = np.where(np.isneginf(m_t), LOG_ZERO, log_gamma)
    log_gamma -= log_P

    y = np.exp(logp)
    dL_da = np.zeros_like(logp)
    dL_da[:T] = y[:T] - np.exp(log_gamma)
    return nll, dL_da


# ============================================================================
# Greedy CTC decoder + edit distance for PER
# ============================================================================

def ctc_greedy_decode(logp: np.ndarray, x_len: int, blank: int = 0) -> np.ndarray:
    argmax = np.argmax(logp[:x_len], axis=1)
    out = []
    prev = -1
    for k in argmax:
        if int(k) != blank and int(k) != prev:
            out.append(int(k))
        prev = int(k)
    return np.array(out, dtype=np.int64)


def edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Levenshtein on 1-D int arrays (insert/delete/substitute = 1 each)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,
                           dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


# ============================================================================
# Adam optimizer
# ============================================================================

class Adam:
    def __init__(self, named_params, lr=3e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, clip=1.0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.clip = clip
        self.t = 0
        self.m = {n: np.zeros_like(v) for n, v in named_params.items()}
        self.v = {n: np.zeros_like(v) for n, v in named_params.items()}

    def step(self, named_params: dict, grads: dict):
        if self.clip is not None:
            total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
            if total > self.clip:
                scale = self.clip / (total + 1e-12)
                for k in grads:
                    grads[k] = grads[k] * scale
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for n in named_params:
            g = grads[n]
            self.m[n] = self.b1 * self.m[n] + (1.0 - self.b1) * g
            self.v[n] = self.b2 * self.v[n] + (1.0 - self.b2) * (g * g)
            mh = self.m[n] / bc1
            vh = self.v[n] / bc2
            named_params[n] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ============================================================================
# Full BLSTM-CTC model
# ============================================================================

@dataclass
class BLSTMCTCModel:
    """All parameters of the full BLSTM (or uni-LSTM) + linear projection."""
    fwd: LSTMParams
    bwd: Optional[LSTMParams]   # None for uni-LSTM
    Wy: np.ndarray              # (2H or H, K_full)
    by: np.ndarray              # (K_full,)
    bidirectional: bool

    def named_params(self) -> dict:
        d = {}
        for k in self.fwd.keys():
            d[f"fwd.{k}"] = self.fwd.get(k)
        if self.bidirectional:
            for k in self.bwd.keys():
                d[f"bwd.{k}"] = self.bwd.get(k)
        d["Wy"] = self.Wy
        d["by"] = self.by
        return d

    def set_named(self, name: str, value: np.ndarray):
        if name.startswith("fwd."):
            self.fwd.set(name[4:], value)
        elif name.startswith("bwd."):
            self.bwd.set(name[4:], value)
        elif name == "Wy":
            self.Wy = value
        elif name == "by":
            self.by = value
        else:
            raise KeyError(name)


def init_model(input_dim: int, H: int, K_full: int,
               bidirectional: bool, rng: np.random.RandomState) -> BLSTMCTCModel:
    fwd = init_lstm(input_dim, H, rng, name="fwd")
    bwd = init_lstm(input_dim, H, rng, name="bwd") if bidirectional else None
    proj_in = (2 * H) if bidirectional else H
    Wy = rng.randn(proj_in, K_full) * (1.0 / math.sqrt(proj_in))
    by = np.zeros(K_full)
    return BLSTMCTCModel(fwd=fwd, bwd=bwd, Wy=Wy, by=by, bidirectional=bidirectional)


def forward_model(m: BLSTMCTCModel, X: np.ndarray, x_lens: np.ndarray):
    """Returns log-softmax (T, B, K_full) and a cache dict."""
    T, B, D = X.shape
    mask = np.zeros((T, B))
    for b in range(B):
        mask[: x_lens[b], b] = 1.0
    Hf, cache_f = lstm_forward(m.fwd, X, mask)  # (T, B, H)
    if m.bidirectional:
        X_rev = reverse_seq(X, x_lens)
        mask_rev = reverse_seq(mask[:, :, None], x_lens).squeeze(-1)
        Hb_rev, cache_b = lstm_forward(m.bwd, X_rev, mask_rev)
        Hb = reverse_seq(Hb_rev, x_lens)  # back to forward time
        H_concat = np.concatenate([Hf, Hb], axis=2)  # (T, B, 2H)
    else:
        cache_b = None
        Hb = None
        H_concat = Hf
    logits = H_concat @ m.Wy + m.by  # (T, B, K_full)
    # log softmax
    mlog = np.max(logits, axis=2, keepdims=True)
    e = np.exp(logits - mlog)
    s = e.sum(axis=2, keepdims=True)
    log_y = (logits - mlog) - np.log(s)
    cache = dict(
        X=X, x_lens=x_lens, mask=mask,
        Hf=Hf, Hb=Hb, H_concat=H_concat,
        cache_f=cache_f, cache_b=cache_b,
        logits=logits, log_y=log_y,
    )
    return log_y, cache


def backward_model(m: BLSTMCTCModel, cache: dict,
                   dlogits: np.ndarray) -> dict:
    """dlogits: (T, B, K_full). Returns grads keyed by named_params()."""
    H_concat = cache["H_concat"]
    T, B, _ = dlogits.shape
    grads = {}
    grads["Wy"] = np.zeros_like(m.Wy)
    grads["by"] = np.zeros_like(m.by)
    for t in range(T):
        grads["Wy"] += H_concat[t].T @ dlogits[t]
        grads["by"] += dlogits[t].sum(axis=0)
    dH_concat = dlogits @ m.Wy.T  # (T, B, 2H or H)
    if m.bidirectional:
        H = m.fwd.Wh.shape[0]
        dHf = dH_concat[:, :, :H].copy()
        dHb = dH_concat[:, :, H:].copy()
        x_lens = cache["x_lens"]
        # backward LSTM was run on reversed input; reverse dHb to its time-axis.
        dHb_rev = reverse_seq(dHb, x_lens)
        gf, dXf = lstm_backward(m.fwd, cache["cache_f"], dHf)
        gb, dXb_rev = lstm_backward(m.bwd, cache["cache_b"], dHb_rev)
        # The backward LSTM operates on X_rev, so dXb_rev is wrt X_rev.
        # We do not need dX since X is the data.
        for k in m.fwd.keys():
            grads[f"fwd.{k}"] = gf[k]
        for k in m.bwd.keys():
            grads[f"bwd.{k}"] = gb[k]
    else:
        gf, _ = lstm_backward(m.fwd, cache["cache_f"], dH_concat)
        for k in m.fwd.keys():
            grads[f"fwd.{k}"] = gf[k]
    return grads


# ============================================================================
# Loss + grad on a batch
# ============================================================================

def batch_loss_grad(m: BLSTMCTCModel, X, x_lens, labels_list, l_lens,
                    blank: int = 0):
    log_y, cache = forward_model(m, X, x_lens)  # (T, B, K)
    T, B, K_full = log_y.shape
    # CTC sample by sample.
    dlogits = np.zeros_like(log_y)
    total_nll = 0.0
    n_total_labels = 0
    for b in range(B):
        nll, dL_da = ctc_loss_and_grad(log_y[:, b, :], labels_list[b],
                                       x_lens[b], blank=blank)
        dlogits[:, b, :] = dL_da
        total_nll += nll
        n_total_labels += int(l_lens[b])
    # Average gradient over batch (so dL/dW scales with mean nll, not sum).
    dlogits /= B
    grads = backward_model(m, cache, dlogits)
    mean_nll = total_nll / B
    return mean_nll, grads, log_y, cache


# ============================================================================
# Eval: per-sample edit distance / phoneme error rate
# ============================================================================

def evaluate(m: BLSTMCTCModel, cfg: CorpusConfig, sig: dict,
             rng: np.random.RandomState, n_samples: int = 64,
             batch_size: int = 16):
    total_edits = 0
    total_labels = 0
    n_correct = 0
    seqs_done = 0
    while seqs_done < n_samples:
        b = min(batch_size, n_samples - seqs_done)
        X, x_lens, labels_list, l_lens = make_batch(cfg, sig, rng, b)
        log_y, _ = forward_model(m, X, x_lens)
        for i in range(b):
            pred = ctc_greedy_decode(log_y[:, i, :], x_lens[i], blank=0)
            ed = edit_distance(pred, labels_list[i])
            total_edits += ed
            total_labels += len(labels_list[i])
            if ed == 0:
                n_correct += 1
        seqs_done += b
    per = total_edits / max(1, total_labels)
    seq_acc = n_correct / max(1, n_samples)
    return per, seq_acc


# ============================================================================
# Training loop
# ============================================================================

@dataclass
class TrainHistory:
    iters: list = field(default_factory=list)
    nll: list = field(default_factory=list)
    eval_per: list = field(default_factory=list)
    eval_seq_acc: list = field(default_factory=list)

    def to_dict(self):
        return {k: list(v) for k, v in self.__dict__.items()}


def make_named(model: BLSTMCTCModel) -> dict:
    return model.named_params()


def update_named(model: BLSTMCTCModel, named: dict):
    for k, v in named.items():
        model.set_named(k, v)


def train(model_kind: str, seed: int, n_iters: int, batch_size: int,
          hidden: int, lr: float, eval_every: int,
          cfg: CorpusConfig, verbose: bool = True,
          snapshot_every: Optional[int] = None) -> tuple:
    """Train BLSTM-CTC or uni-LSTM-CTC. Returns (model, history, snapshots, sig)."""
    rng = np.random.RandomState(seed)
    train_rng = np.random.RandomState(seed + 17)
    eval_rng = np.random.RandomState(seed + 1_000_003)
    init_rng = np.random.RandomState(seed + 7)
    sig = make_phoneme_signatures(cfg, rng)

    bidirectional = (model_kind == "blstm")
    K_full = cfg.n_phonemes + 1
    model = init_model(cfg.n_features, hidden, K_full, bidirectional, init_rng)

    named = model.named_params()
    opt = Adam(named, lr=lr, clip=1.0)
    history = TrainHistory()
    snapshots = []
    t0 = time.time()
    last_nll = float("nan")

    for it in range(1, n_iters + 1):
        X, x_lens, labels_list, l_lens = make_batch(
            cfg, sig, train_rng, batch_size)
        nll, grads, log_y, _ = batch_loss_grad(model, X, x_lens,
                                               labels_list, l_lens, blank=0)
        # Mutate named-params dict in place via Adam.
        named = model.named_params()
        opt.step(named, grads)
        update_named(model, named)
        last_nll = nll

        if it == 1 or it % eval_every == 0 or it == n_iters:
            per, sa = evaluate(model, cfg, sig, eval_rng,
                               n_samples=64, batch_size=16)
            history.iters.append(it)
            history.nll.append(last_nll)
            history.eval_per.append(per)
            history.eval_seq_acc.append(sa)
            if verbose:
                el = time.time() - t0
                print(f"  [{model_kind}] iter {it:5d}  nll {last_nll:7.3f}  "
                      f"PER {per:.3f}  seq_acc {sa:.3f}  ({el:.1f}s)")
            if snapshot_every is not None and (
                    it == 1 or it % snapshot_every == 0 or it == n_iters):
                # Snapshot for GIF: a fixed eval batch's log_y at this iter.
                snap_rng = np.random.RandomState(seed + 99)
                Xs, xls, lbls, lls = make_batch(cfg, sig, snap_rng, 4)
                snap_logy, _ = forward_model(model, Xs, xls)
                snapshots.append(dict(
                    iter=it,
                    nll=last_nll,
                    per=per,
                    seq_acc=sa,
                    Xs=Xs.copy(),
                    x_lens=xls.copy(),
                    labels=[l.copy() for l in lbls],
                    log_y=snap_logy.copy(),
                ))

    return model, history, snapshots, sig


# ============================================================================
# Numerical gradient check
# ============================================================================

def gradcheck(model_kind: str = "blstm", seed: int = 0,
              hidden: int = 6, eps: float = 1e-5, n_checks: int = 12):
    """Finite-difference gradient check on a tiny instance.

    Compares analytic CTC + BPTT gradients against numerical gradients
    on a 2-sample, short-sequence batch. Should give <1e-5 rel error.
    """
    rng = np.random.RandomState(seed)
    cfg = CorpusConfig(
        n_phonemes=3, n_features=4,
        min_phonemes_per_seq=2, max_phonemes_per_seq=3,
        min_frames_per_phoneme=3, max_frames_per_phoneme=5,
        min_silence_frames=1, max_silence_frames=2,
    )
    sig = make_phoneme_signatures(cfg, rng)
    init_rng = np.random.RandomState(seed + 7)
    K_full = cfg.n_phonemes + 1
    model = init_model(cfg.n_features, hidden, K_full,
                       bidirectional=(model_kind == "blstm"), rng=init_rng)
    data_rng = np.random.RandomState(seed + 13)
    X, x_lens, labels_list, l_lens = make_batch(cfg, sig, data_rng, 2)

    def loss_only():
        log_y, _ = forward_model(model, X, x_lens)
        total = 0.0
        for b in range(X.shape[1]):
            nll, _ = ctc_loss_and_grad(log_y[:, b, :], labels_list[b],
                                       x_lens[b], blank=0)
            total += nll
        return total / X.shape[1]

    # Analytic gradients via the same path used in training:
    nll, grads, _, _ = batch_loss_grad(model, X, x_lens, labels_list, l_lens, blank=0)

    rel_errs = []
    check_rng = np.random.RandomState(seed + 31)
    for name, W in model.named_params().items():
        flat = W.reshape(-1)
        idx = check_rng.choice(flat.size,
                               size=min(n_checks, flat.size),
                               replace=False)
        for i in idx:
            saved = flat[i]
            flat[i] = saved + eps
            lp = loss_only()
            flat[i] = saved - eps
            lm = loss_only()
            flat[i] = saved
            num = (lp - lm) / (2 * eps)
            an = grads[name].reshape(-1)[i]
            denom = max(1e-10, abs(num) + abs(an))
            rel = abs(num - an) / denom
            rel_errs.append((name, i, num, an, rel))
    max_rel = max(r[-1] for r in rel_errs)
    print(f"[{model_kind}] gradcheck: max relative error = {max_rel:.2e} "
          f"over {len(rel_errs)} samples")
    for r in rel_errs[:6]:
        print(f"   {r[0]:<10s} idx {r[1]:>4d}  num {r[2]:+.6f}  "
              f"an {r[3]:+.6f}  rel {r[4]:.2e}")
    return max_rel


# ============================================================================
# Reproducibility metadata
# ============================================================================

def env_info():
    import platform
    import sys
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--uni", action="store_true",
                    help="train uni-directional LSTM baseline instead of BLSTM")
    ap.add_argument("--gradcheck", action="store_true",
                    help="run numerical gradient check and exit")
    ap.add_argument("--save-history", type=str, default=None)
    ap.add_argument("--save-model", type=str, default=None)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="save eval-batch snapshot every N iters (0 disables)")
    ap.add_argument("--save-snapshots", type=str, default=None,
                    help="path to .npz to dump training snapshots (for GIF)")
    args = ap.parse_args()

    if args.gradcheck:
        for k in ("blstm", "uni"):
            gradcheck(model_kind="blstm" if k == "blstm" else "uni")
        return

    cfg = CorpusConfig()
    model_kind = "uni" if args.uni else "blstm"
    print(f"[{model_kind}] seed={args.seed} hidden={args.hidden} "
          f"batch={args.batch} iters={args.iters} lr={args.lr}")
    print(f"  env: {env_info()}")

    snap_every = args.snapshot_every if args.snapshot_every > 0 else None
    t0 = time.time()
    model, history, snapshots, sig = train(
        model_kind=model_kind,
        seed=args.seed,
        n_iters=args.iters,
        batch_size=args.batch,
        hidden=args.hidden,
        lr=args.lr,
        eval_every=args.eval_every,
        cfg=cfg,
        verbose=True,
        snapshot_every=snap_every,
    )
    elapsed = time.time() - t0
    final_per = history.eval_per[-1] if history.eval_per else float("nan")
    final_sa = history.eval_seq_acc[-1] if history.eval_seq_acc else float("nan")
    print(f"\n[{model_kind}] DONE  iters={args.iters}  "
          f"final PER={final_per:.3f}  seq_acc={final_sa:.3f}  "
          f"wallclock={elapsed:.1f}s")

    if args.save_history:
        with open(args.save_history, "w") as f:
            json.dump({
                "model_kind": model_kind,
                "seed": args.seed,
                "iters": args.iters,
                "hidden": args.hidden,
                "lr": args.lr,
                "batch": args.batch,
                "history": history.to_dict(),
                "env": env_info(),
                "wallclock_s": elapsed,
                "final_per": final_per,
                "final_seq_acc": final_sa,
            }, f, indent=2)
        print(f"  history -> {args.save_history}")

    if args.save_snapshots and snapshots:
        # Save snapshots in a compact .npz (one record per snapshot iter).
        out = {}
        out["iters"] = np.array([s["iter"] for s in snapshots])
        out["nlls"] = np.array([s["nll"] for s in snapshots])
        out["pers"] = np.array([s["per"] for s in snapshots])
        out["seq_accs"] = np.array([s["seq_acc"] for s in snapshots])
        # All snapshots use the same sample batch, save Xs + lengths once.
        out["Xs"] = snapshots[0]["Xs"]
        out["x_lens"] = snapshots[0]["x_lens"]
        for i, lbl in enumerate(snapshots[0]["labels"]):
            out[f"label_{i}"] = lbl
        out["log_y"] = np.stack([s["log_y"] for s in snapshots], axis=0)
        np.savez(args.save_snapshots, **out)
        print(f"  snapshots -> {args.save_snapshots}")


if __name__ == "__main__":
    main()
