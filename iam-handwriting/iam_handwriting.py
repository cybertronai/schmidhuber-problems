"""iam-handwriting -- Graves, Liwicki, Fernandez, Bertolami, Bunke, Schmidhuber,
*A Novel Connectionist System for Unconstrained Handwriting Recognition*,
IEEE TPAMI 31(5), 2009. (ICDAR 2009 winner.)

The paper trains a Bidirectional LSTM with a Connectionist Temporal
Classification (CTC) output layer on the IAM-OnDB online handwriting database
(5,364 train lines, 3,859 test) and the IAM-DB offline scanned database.
Decoding uses token-passing against a 20K-word dictionary plus a bigram LM.
Online word accuracy: 79.7% (vs HMM baseline 65.0%); offline 74.1% (vs 64.5%).

Per SPEC issue #1 (cybertronai/schmidhuber-problems), v1 stays pure-numpy and
laptop-runnable; the IAM datasets are external + heavyweight, so this stub
captures the *algorithmic* claim -- BLSTM + CTC reads variable-length
handwriting trajectories at low character error rate -- on a synthetic
pen-trajectory dataset generated in numpy.

Synthetic handwriting:
    - 10-character alphabet: c, o, l, i, t, n, m, a, e, u.
    - Each character is encoded as one or more stroke polylines in a unit
      bounding box. Polylines are resampled to a fixed arc-length and rendered
      as (dx, dy, pen_up) triplets (the same online feature representation
      used in the IAM-OnDB experiments, minus the 25-channel feature vector).
    - Words are concatenated with a per-character horizontal offset; an
      explicit pen-up between letters separates strokes. Per-stroke jitter
      and per-word affine slant are applied.
    - 30-word vocabulary built from the alphabet (cat, dog -- not in alphabet
      -- so e.g. cone, mint, lit, name, ant, moon, etc.). Train / test split.

Architecture (Graves et al. 2009 §III, scaled down):
    input   : (T, 3) trajectory (dx, dy, pen_up)
    BLSTM   : forward LSTM (hidden=H=48) + backward LSTM (hidden=H=48),
              concatenated to (T, 2H)
    output  : linear (2H -> alphabet+1) + log-softmax
              alphabet+1 includes the CTC blank class (label 0).

CTC forward-backward in log space (Graves, Fernandez, Gomez, Schmidhuber 2006).
Greedy collapse decoding. Character error rate via Levenshtein.

Both directions of LSTM and the CTC layer are hand-coded in numpy. Adam is
hand-coded; gradient clipping by global norm.

Determinism: --seed seeds numpy. Two runs with the same seed produce identical
training curves and final CER (verified -- see §Results in README).

CLI:
    python3 iam_handwriting.py --seed 0
    python3 iam_handwriting.py --seed 0 --quick           # smaller smoke test
    python3 iam_handwriting.py --seed 0 --save-json run.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------------------------------------------------
# Reproducibility metadata
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


def env_metadata() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "git": git_hash(),
    }


# ----------------------------------------------------------------------
# Synthetic handwriting alphabet
# ----------------------------------------------------------------------
# Each character is a list of strokes. Each stroke is a list of (x, y)
# control points in a [0, 1] x [0, 1] bounding box (y goes up).
# We resample each stroke to a fixed arc-length number of points later.

ALPHABET = ["c", "o", "l", "i", "t", "n", "m", "a", "e", "u"]
# Index 0 reserved for CTC blank in the output layer.
CHAR2ID: Dict[str, int] = {c: i + 1 for i, c in enumerate(ALPHABET)}
ID2CHAR: Dict[int, str] = {v: k for k, v in CHAR2ID.items()}
N_CLASSES = len(ALPHABET) + 1  # blank + 10 chars
BLANK = 0


def _arc(cx: float, cy: float, rx: float, ry: float,
         theta0: float, theta1: float, n: int) -> List[Tuple[float, float]]:
    """Sample n points along an elliptic arc."""
    thetas = np.linspace(theta0, theta1, n)
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in thetas]


def _line(p0: Tuple[float, float], p1: Tuple[float, float],
          n: int) -> List[Tuple[float, float]]:
    """Sample n points along a line segment (inclusive)."""
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    return list(zip(xs.tolist(), ys.tolist()))


def char_strokes(c: str) -> List[List[Tuple[float, float]]]:
    """Return list of strokes for character c.

    Coordinates roughly in [0, 1] x [0, 1] (y up). Stroke shapes are stylised
    "block-handwriting" with smooth curves; distinct enough that a BLSTM+CTC
    can solve it but compositional enough that it still has to decode the
    whole sequence.
    """
    n_arc = 12
    n_seg = 8
    if c == "c":
        # Open arc on the right
        return [_arc(0.5, 0.5, 0.42, 0.42, math.radians(35), math.radians(325), n_arc)]
    if c == "o":
        # Closed ellipse
        return [_arc(0.5, 0.5, 0.42, 0.45, 0, 2 * math.pi, n_arc + 2)]
    if c == "l":
        # Tall vertical line
        return [_line((0.5, 1.0), (0.5, 0.0), n_seg)]
    if c == "i":
        # Short vertical + dot above
        return [
            _line((0.5, 0.6), (0.5, 0.0), n_seg),
            _line((0.5, 0.92), (0.5, 0.85), 3),
        ]
    if c == "t":
        # Vertical + crossbar
        return [
            _line((0.5, 0.95), (0.5, 0.0), n_seg),
            _line((0.25, 0.65), (0.75, 0.65), 4),
        ]
    if c == "n":
        # Arch: up the left, over the top, down the right
        return [
            _line((0.15, 0.0), (0.15, 0.6), 4) +
            _arc(0.5, 0.6, 0.35, 0.30, math.pi, 0, n_arc) +
            _line((0.85, 0.6), (0.85, 0.0), 4)
        ]
    if c == "m":
        # Two arches
        return [
            _line((0.10, 0.0), (0.10, 0.6), 4) +
            _arc(0.30, 0.6, 0.20, 0.28, math.pi, 0, n_arc) +
            _line((0.50, 0.6), (0.50, 0.0), 3) +
            _line((0.50, 0.0), (0.50, 0.6), 3) +
            _arc(0.70, 0.6, 0.20, 0.28, math.pi, 0, n_arc) +
            _line((0.90, 0.6), (0.90, 0.0), 4)
        ]
    if c == "a":
        # Closed loop on left + vertical on right
        loop = _arc(0.40, 0.30, 0.30, 0.30, 0, 2 * math.pi, n_arc + 2)
        return [
            loop,
            _line((0.70, 0.6), (0.70, 0.0), n_seg),
        ]
    if c == "e":
        # Closed-ish loop with horizontal middle bar baked in (use 3/4 ellipse
        # then horizontal "tongue").
        arc1 = _arc(0.5, 0.5, 0.40, 0.42,
                    math.radians(0), math.radians(330), n_arc)
        return [
            _line((0.10, 0.5), (0.90, 0.5), 5) +
            arc1
        ]
    if c == "u":
        # U-shape: down the left, arc at the bottom, up the right.
        return [
            _line((0.15, 1.0), (0.15, 0.30), 4) +
            _arc(0.5, 0.30, 0.35, 0.22, math.pi, 2 * math.pi, n_arc) +
            _line((0.85, 0.30), (0.85, 1.0), 4)
        ]
    raise ValueError(f"unknown character {c!r}")


CHAR_WIDTH = 1.1   # horizontal advance per character (in unit-box units)
LETTER_GAP = 0.25  # extra horizontal gap between characters


# ----------------------------------------------------------------------
# Word -> (T, 3) trajectory
# ----------------------------------------------------------------------

def render_word(word: str,
                rng: np.random.Generator,
                jitter: float = 0.020,
                slant_max: float = 0.18,
                ) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Render `word` as a (T, 3) (dx, dy, pen_up) trajectory.

    Returns:
        traj (T, 3): per-step (dx, dy, pen_up) where pen_up = 1 marks the
            sample where a *new stroke* begins (Graves et al. online encoding,
            simplified).
        labels (list[int]): integer IDs of characters in `word` (1-indexed,
            0 is CTC blank).
        abs_xy (T, 2): absolute coordinates (for visualisation).
    """
    cur_x = 0.0
    abs_pts: List[Tuple[float, float]] = []
    pen_up_flags: List[int] = []  # 1 when this point is the first of a stroke
    # Per-word slant: x' = x + slant * y (mild italic).
    slant = float(rng.uniform(-slant_max, slant_max))
    for c in word:
        strokes = char_strokes(c)
        for s in strokes:
            for j, (xs, ys) in enumerate(s):
                # Apply slant + per-character offset.
                x_world = cur_x + xs + slant * ys
                y_world = ys
                # Per-point Gaussian jitter.
                x_world += float(rng.normal(0.0, jitter))
                y_world += float(rng.normal(0.0, jitter))
                abs_pts.append((x_world, y_world))
                pen_up_flags.append(1 if j == 0 else 0)
        cur_x += CHAR_WIDTH + LETTER_GAP

    abs_xy = np.array(abs_pts, dtype=np.float64)
    pen = np.array(pen_up_flags, dtype=np.float64)
    # Convert to (dx, dy) deltas. The first point's dx/dy is 0 (no previous).
    dxy = np.zeros_like(abs_xy)
    dxy[1:] = abs_xy[1:] - abs_xy[:-1]
    # When the pen lifts, we don't want a huge "jump" delta to be the input
    # cue (the paper's online encoding includes pen-up as an explicit binary
    # channel). Zero out the delta on pen-up samples.
    dxy[pen.astype(bool)] = 0.0
    traj = np.concatenate([dxy, pen.reshape(-1, 1)], axis=1)
    labels = [CHAR2ID[c] for c in word]
    return traj, labels, abs_xy


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
# Words built from the 10-character alphabet. Mix of 3, 4, 5-letter words.

VOCAB = [
    # 3-letter
    "ant", "ill", "tan", "lot", "mit", "nan", "tic", "tin", "out", "ate",
    "eat", "tea", "ice", "lit", "non", "nun", "men", "mat", "net", "moo",
    # 4-letter
    "moon", "loom", "noon", "name", "nice", "cone", "tone", "lane", "amen",
    "main", "mine", "lent", "tent", "team", "lime", "time", "mile", "tail",
    "lion", "into", "icon",
    # 5-letter
    "actin", "tonic", "milne", "linen", "matte", "atone",
]


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
# LSTM cell with forget gate (gate order: i, f, g, o)
# ----------------------------------------------------------------------

@dataclass
class LSTM:
    """Hand-coded LSTM cell with forget gate. BPTT in numpy.

    Gate order in the concatenated weight slab: i (input), f (forget),
    g (cell candidate, tanh), o (output). All sigmoids except g.
    Forget bias initialised to 1.0 (Gers/Schmidhuber/Cummins 2000).
    """
    in_dim: int
    H: int

    def init(self, rng: np.random.Generator):
        scale_x = 1.0 / math.sqrt(self.in_dim)
        scale_h = 1.0 / math.sqrt(self.H)
        self.Wx = rng.standard_normal((self.in_dim, 4 * self.H)) * scale_x * 0.5
        self.Wh = rng.standard_normal((self.H, 4 * self.H)) * scale_h * 0.5
        self.b = np.zeros(4 * self.H)
        self.b[self.H:2 * self.H] = 1.0  # forget bias
        # Adam state
        self._adam_init()

    def _adam_init(self):
        self.adam_m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.adam_t = 0

    def params(self) -> Dict[str, np.ndarray]:
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run forward over X (T, in_dim). Returns h (T, H) and cache."""
        T = X.shape[0]
        H = self.H
        h = np.zeros((T + 1, H))
        c = np.zeros((T + 1, H))
        i_g = np.zeros((T, H))
        f_g = np.zeros((T, H))
        g_g = np.zeros((T, H))
        o_g = np.zeros((T, H))
        tc = np.zeros((T, H))
        for t in range(T):
            z = X[t] @ self.Wx + h[t] @ self.Wh + self.b
            i_g[t] = sigmoid(z[0:H])
            f_g[t] = sigmoid(z[H:2 * H])
            g_g[t] = np.tanh(z[2 * H:3 * H])
            o_g[t] = sigmoid(z[3 * H:4 * H])
            c[t + 1] = f_g[t] * c[t] + i_g[t] * g_g[t]
            tc[t] = np.tanh(c[t + 1])
            h[t + 1] = o_g[t] * tc[t]
        cache = dict(X=X, h=h, c=c, i=i_g, f=f_g, g=g_g, o=o_g, tc=tc)
        return h[1:], cache

    def backward(self, cache: Dict, dh_seq: np.ndarray
                 ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Backprop given dL/dh for each timestep. Returns grads and dL/dX."""
        X = cache["X"]
        h = cache["h"]
        c = cache["c"]
        i_g = cache["i"]
        f_g = cache["f"]
        g_g = cache["g"]
        o_g = cache["o"]
        tc = cache["tc"]
        T = X.shape[0]
        H = self.H

        grads = {"Wx": np.zeros_like(self.Wx),
                 "Wh": np.zeros_like(self.Wh),
                 "b": np.zeros_like(self.b)}
        dX = np.zeros_like(X)

        dh_next = np.zeros(H)
        dc_next = np.zeros(H)

        for t in reversed(range(T)):
            dh = dh_seq[t] + dh_next
            do_t = dh * tc[t]
            dtc_t = dh * o_g[t]
            dc = dc_next + dtc_t * dtanh_from_y(tc[t])
            df_t = dc * c[t]
            dc_prev = dc * f_g[t]
            di_t = dc * g_g[t]
            dg_t = dc * i_g[t]
            dz_i = di_t * dsigmoid_from_y(i_g[t])
            dz_f = df_t * dsigmoid_from_y(f_g[t])
            dz_g = dg_t * dtanh_from_y(g_g[t])
            dz_o = do_t * dsigmoid_from_y(o_g[t])
            dz = np.concatenate([dz_i, dz_f, dz_g, dz_o])  # (4H,)
            grads["Wx"] += np.outer(X[t], dz)
            grads["Wh"] += np.outer(h[t], dz)
            grads["b"] += dz
            dX[t] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc_prev

        return grads, dX

    def adam_step(self, grads: Dict[str, np.ndarray], lr: float = 1e-3,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.adam_t += 1
        bc1 = 1.0 - beta1 ** self.adam_t
        bc2 = 1.0 - beta2 ** self.adam_t
        for k, p in self.params().items():
            g = grads[k]
            self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * g
            self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * (g ** 2)
            mhat = self.adam_m[k] / bc1
            vhat = self.adam_v[k] / bc2
            p -= lr * mhat / (np.sqrt(vhat) + eps)


# ----------------------------------------------------------------------
# BLSTM + linear output -> CTC
# ----------------------------------------------------------------------

@dataclass
class BLSTMCTC:
    in_dim: int
    H: int
    n_classes: int

    def init(self, rng: np.random.Generator):
        self.fwd = LSTM(in_dim=self.in_dim, H=self.H)
        self.bwd = LSTM(in_dim=self.in_dim, H=self.H)
        self.fwd.init(np.random.default_rng(int(rng.integers(0, 2**31))))
        self.bwd.init(np.random.default_rng(int(rng.integers(0, 2**31))))
        scale = 1.0 / math.sqrt(2 * self.H)
        self.W = rng.standard_normal((2 * self.H, self.n_classes)) * scale
        self.b = np.zeros(self.n_classes)
        # Adam state for output layer
        self.adam_m_W = np.zeros_like(self.W)
        self.adam_v_W = np.zeros_like(self.W)
        self.adam_m_b = np.zeros_like(self.b)
        self.adam_v_b = np.zeros_like(self.b)
        self.adam_t = 0

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run BLSTM + linear over X (T, in_dim). Returns log-softmax (T, K)
        and cache for backprop."""
        T = X.shape[0]
        h_fwd, cache_fwd = self.fwd.forward(X)
        # Reverse for backward LSTM, then reverse outputs back.
        X_rev = X[::-1]
        h_bwd_rev, cache_bwd = self.bwd.forward(X_rev)
        h_bwd = h_bwd_rev[::-1]
        H_cat = np.concatenate([h_fwd, h_bwd], axis=1)  # (T, 2H)
        logits = H_cat @ self.W + self.b  # (T, K)
        # Log-softmax along K
        m = logits.max(axis=1, keepdims=True)
        z = logits - m
        log_norm = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_probs = z - log_norm  # (T, K)
        cache = {
            "X": X, "cache_fwd": cache_fwd, "cache_bwd": cache_bwd,
            "h_fwd": h_fwd, "h_bwd": h_bwd, "H_cat": H_cat,
            "logits": logits, "log_probs": log_probs,
        }
        return log_probs, cache

    def backward(self, cache: Dict, dlogits: np.ndarray) -> Dict:
        """Backprop from dL/dlogits. Returns gradients for all params."""
        H_cat = cache["H_cat"]
        T = dlogits.shape[0]
        # Output layer gradients
        grads = {}
        grads["W"] = H_cat.T @ dlogits  # (2H, K)
        grads["b"] = dlogits.sum(axis=0)  # (K,)
        dH_cat = dlogits @ self.W.T  # (T, 2H)
        dh_fwd = dH_cat[:, :self.H]
        dh_bwd = dH_cat[:, self.H:]
        # Backward LSTM: input was reversed, so reverse dh_bwd before passing.
        grads_fwd, _ = self.fwd.backward(cache["cache_fwd"], dh_fwd)
        grads_bwd, _ = self.bwd.backward(cache["cache_bwd"], dh_bwd[::-1])
        grads["fwd"] = grads_fwd
        grads["bwd"] = grads_bwd
        return grads

    def step(self, grads: Dict, lr: float = 1e-3, clip: float = 5.0):
        # Global-norm gradient clip across all params.
        all_g = []
        for g in (grads["W"], grads["b"]):
            all_g.append(g)
        for k in ("Wx", "Wh", "b"):
            all_g.append(grads["fwd"][k])
            all_g.append(grads["bwd"][k])
        total = sum(float((g ** 2).sum()) for g in all_g)
        norm = math.sqrt(total)
        if norm > clip:
            scale = clip / (norm + 1e-12)
            grads = self._scale_grads(grads, scale)
        # Adam for output layer
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1 ** self.adam_t
        bc2 = 1.0 - beta2 ** self.adam_t
        self.adam_m_W = beta1 * self.adam_m_W + (1 - beta1) * grads["W"]
        self.adam_v_W = beta2 * self.adam_v_W + (1 - beta2) * (grads["W"] ** 2)
        mW = self.adam_m_W / bc1
        vW = self.adam_v_W / bc2
        self.W -= lr * mW / (np.sqrt(vW) + eps)
        self.adam_m_b = beta1 * self.adam_m_b + (1 - beta1) * grads["b"]
        self.adam_v_b = beta2 * self.adam_v_b + (1 - beta2) * (grads["b"] ** 2)
        mb = self.adam_m_b / bc1
        vb = self.adam_v_b / bc2
        self.b -= lr * mb / (np.sqrt(vb) + eps)
        # Adam for LSTMs
        self.fwd.adam_step(grads["fwd"], lr=lr)
        self.bwd.adam_step(grads["bwd"], lr=lr)

    @staticmethod
    def _scale_grads(grads: Dict, scale: float) -> Dict:
        out = {"W": grads["W"] * scale, "b": grads["b"] * scale,
               "fwd": {k: v * scale for k, v in grads["fwd"].items()},
               "bwd": {k: v * scale for k, v in grads["bwd"].items()}}
        return out


# ----------------------------------------------------------------------
# CTC forward-backward in log space
# ----------------------------------------------------------------------

NEG_INF = -1e30


def logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_loss_and_grad(log_probs: np.ndarray,
                      labels: List[int],
                      blank: int = BLANK
                      ) -> Tuple[float, np.ndarray]:
    """Standard CTC forward-backward (Graves et al. 2006), all in log space.

    log_probs: (T, K) log-softmax output.
    labels:    list of integer label IDs (must not contain blank).
    Returns (loss, dlogits) where dlogits = d loss / d logits, shape (T, K).
    The closed-form gradient is dL/dlogits = softmax - posteriors.
    """
    T, K = log_probs.shape
    if not labels:
        # Edge case: empty target. Loss = -sum log P(blank, t).
        loss = -float(log_probs[:, blank].sum())
        probs = np.exp(log_probs)
        post = np.zeros_like(probs)
        post[:, blank] = 1.0
        return loss, probs - post

    # Build extended label sequence with blanks: l' = (b, l1, b, l2, ..., b)
    L = len(labels)
    S = 2 * L + 1
    l_ext = [blank] * S
    for i, c in enumerate(labels):
        l_ext[2 * i + 1] = c

    if T < L:
        # Not enough timesteps -- standard CTC convention: return inf-ish loss.
        # This shouldn't happen in our setup given character widths, but guard.
        loss = float("inf")
        return loss, np.zeros_like(log_probs)

    # Forward variables in log space: alpha[t, s] = log P(label prefix
    # ending at extended-position s by time t).
    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    alpha[0, 0] = log_probs[0, blank]
    if S > 1:
        alpha[0, 1] = log_probs[0, l_ext[1]]
    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s]
            if s - 1 >= 0:
                a = logaddexp(a, alpha[t - 1, s - 1])
            # The "s-2" term is allowed only when l_ext[s] != blank and
            # l_ext[s] != l_ext[s-2] (skip-blank rule).
            if (s - 2 >= 0 and l_ext[s] != blank
                    and l_ext[s] != l_ext[s - 2]):
                a = logaddexp(a, alpha[t - 1, s - 2])
            alpha[t, s] = a + log_probs[t, l_ext[s]]

    # Total log-likelihood
    log_p = logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2]) if S > 1 \
        else alpha[T - 1, 0]
    loss = -log_p

    # Backward variables: beta[t, s] = log P(label suffix from extended-pos s,
    # starting at time t).
    beta = np.full((T, S), NEG_INF, dtype=np.float64)
    beta[T - 1, S - 1] = 0.0  # log P(emit blank at last frame | already at S-1)
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            b = beta[t + 1, s] + log_probs[t + 1, l_ext[s]]
            if s + 1 < S:
                b = logaddexp(b, beta[t + 1, s + 1] + log_probs[t + 1, l_ext[s + 1]])
            if (s + 2 < S and l_ext[s] != blank
                    and l_ext[s] != l_ext[s + 2]):
                b = logaddexp(b, beta[t + 1, s + 2] + log_probs[t + 1, l_ext[s + 2]])
            beta[t, s] = b

    # Posteriors per (t, k): gamma[t, k] = sum over s with l_ext[s] == k of
    # exp(alpha[t,s] + beta[t,s] - log_p).
    gamma = np.zeros_like(log_probs)
    for t in range(T):
        for s in range(S):
            v = alpha[t, s] + beta[t, s]
            if v == NEG_INF:
                continue
            gamma[t, l_ext[s]] += math.exp(v - log_p)

    # dL / d logits (softmax inputs) = softmax_probs - gamma.
    probs = np.exp(log_probs)
    dlogits = probs - gamma
    return loss, dlogits


# ----------------------------------------------------------------------
# CTC greedy decoding + character error rate
# ----------------------------------------------------------------------

def greedy_decode(log_probs: np.ndarray, blank: int = BLANK) -> List[int]:
    """Argmax per timestep -> collapse repeats -> remove blanks."""
    best = log_probs.argmax(axis=1).tolist()
    out: List[int] = []
    prev = -1
    for k in best:
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return out


def levenshtein(a: List[int], b: List[int]) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            if x == y:
                new[j] = dp[j - 1]
            else:
                new[j] = 1 + min(dp[j], new[j - 1], dp[j - 1])
        dp = new
    return dp[-1]


def cer(pred: List[int], target: List[int]) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


# ----------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------

@dataclass
class Sample:
    word: str
    traj: np.ndarray       # (T, 3)
    labels: List[int]      # 1-indexed character IDs
    abs_xy: np.ndarray     # (T, 2) for visualisation


def make_dataset(words: List[str], rng: np.random.Generator,
                 jitter: float, slant_max: float) -> List[Sample]:
    out = []
    for w in words:
        traj, labels, abs_xy = render_word(w, rng, jitter=jitter,
                                           slant_max=slant_max)
        out.append(Sample(word=w, traj=traj, labels=labels, abs_xy=abs_xy))
    return out


def split_vocab(vocab: List[str], rng: np.random.Generator,
                test_frac: float = 0.2) -> Tuple[List[str], List[str]]:
    idx = rng.permutation(len(vocab))
    n_test = max(1, int(round(test_frac * len(vocab))))
    test_idx = set(idx[:n_test].tolist())
    train, test = [], []
    for i, w in enumerate(vocab):
        (test if i in test_idx else train).append(w)
    return train, test


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    H: int = 64
    epochs: int = 25
    lr: float = 5e-3
    jitter: float = 0.014
    slant_max: float = 0.15
    holdout_frac: float = 0.20  # fraction of vocabulary held out for the
                                # compositional-generalisation eval (new
                                # words, never seen during training).
    word_repeats_per_epoch: int = 6  # resample each in-vocab word this
                                     # many times per epoch
    eval_repeats: int = 8  # # of fresh renderings per word at eval
    grad_clip: float = 5.0


def evaluate(model: BLSTMCTC, samples: List[Sample]) -> Dict:
    cers = []
    losses = []
    word_hits = 0
    for s in samples:
        log_probs, _ = model.forward(s.traj)
        loss, _ = ctc_loss_and_grad(log_probs, s.labels)
        pred = greedy_decode(log_probs)
        cers.append(cer(pred, s.labels))
        losses.append(loss / max(1, len(s.labels)))
        if pred == s.labels:
            word_hits += 1
    return {
        "cer_mean": float(np.mean(cers)),
        "cer_std": float(np.std(cers)),
        "loss_mean": float(np.mean(losses)),
        "word_acc": word_hits / len(samples),
    }


def train(cfg: RunConfig, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    # Vocabulary split:
    #   in_vocab  -- model trains on these. Eval here mirrors the IAM
    #                benchmark setting: same vocabulary, *different rendering
    #                samples* (different jitter / slant). This is the headline
    #                CER number.
    #   ood_vocab -- words held out entirely from training. Eval here measures
    #                compositional generalisation (does the BLSTM+CTC actually
    #                learn the per-character mapping, or memorise full words?).
    in_vocab, ood_vocab = split_vocab(VOCAB, rng,
                                      test_frac=cfg.holdout_frac)

    # Build *fresh-rendering* eval sets using their own rng (deterministic,
    # but disjoint from training renderings).
    eval_rng = np.random.default_rng(cfg.seed + 1)
    test_samples: List[Sample] = []
    for _ in range(cfg.eval_repeats):
        test_samples.extend(make_dataset(in_vocab, eval_rng,
                                         jitter=cfg.jitter,
                                         slant_max=cfg.slant_max))
    ood_samples: List[Sample] = []
    for _ in range(cfg.eval_repeats):
        ood_samples.extend(make_dataset(ood_vocab, eval_rng,
                                        jitter=cfg.jitter,
                                        slant_max=cfg.slant_max))
    train_words = in_vocab
    test_words = in_vocab  # eval = same vocab, fresh renderings

    # Build model
    model_rng = np.random.default_rng(cfg.seed + 2)
    in_dim = 3
    model = BLSTMCTC(in_dim=in_dim, H=cfg.H, n_classes=N_CLASSES)
    model.init(model_rng)

    history = {
        "epoch": [], "train_loss": [],
        "test_cer": [], "test_loss": [], "test_word_acc": [],
        "ood_cer": [], "ood_word_acc": [],
        "wallclock_per_epoch": [],
    }

    t0 = time.time()
    train_rng = np.random.default_rng(cfg.seed + 3)
    for epoch in range(cfg.epochs):
        ep_start = time.time()
        # Shuffle and resample training words this epoch.
        order = train_rng.permutation(len(train_words))
        epoch_words = [train_words[i] for i in order
                       for _ in range(cfg.word_repeats_per_epoch)]
        train_rng_shuffle = np.random.default_rng(
            cfg.seed + 1000 + epoch
        )
        order2 = train_rng_shuffle.permutation(len(epoch_words))
        epoch_words = [epoch_words[i] for i in order2]
        ep_losses = []
        for w in epoch_words:
            traj, labels, _ = render_word(w, train_rng,
                                          jitter=cfg.jitter,
                                          slant_max=cfg.slant_max)
            log_probs, cache = model.forward(traj)
            loss, dlogits = ctc_loss_and_grad(log_probs, labels)
            if not math.isfinite(loss):
                continue
            grads = model.backward(cache, dlogits)
            model.step(grads, lr=cfg.lr, clip=cfg.grad_clip)
            ep_losses.append(loss / max(1, len(labels)))
        ep_train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        # Eval
        test_metrics = evaluate(model, test_samples)
        ood_metrics = evaluate(model, ood_samples)
        history["epoch"].append(epoch)
        history["train_loss"].append(ep_train_loss)
        history["test_cer"].append(test_metrics["cer_mean"])
        history["test_loss"].append(test_metrics["loss_mean"])
        history["test_word_acc"].append(test_metrics["word_acc"])
        history["ood_cer"].append(ood_metrics["cer_mean"])
        history["ood_word_acc"].append(ood_metrics["word_acc"])
        history["wallclock_per_epoch"].append(time.time() - ep_start)
        if verbose:
            print(f"epoch {epoch:2d}  "
                  f"train_loss={ep_train_loss:.3f}  "
                  f"test_cer={test_metrics['cer_mean']:.3f}  "
                  f"test_word_acc={test_metrics['word_acc']:.2f}  "
                  f"ood_cer={ood_metrics['cer_mean']:.3f}  "
                  f"ood_word_acc={ood_metrics['word_acc']:.2f}")

    wall = time.time() - t0

    # Per-test-word breakdown for §Results in README
    def per_word_breakdown(samples: List[Sample], words: List[str]) -> List[Dict]:
        rows = []
        for w in words:
            sub = [s for s in samples if s.word == w]
            if not sub:
                continue
            sub_cers = []
            sub_hits = 0
            for s in sub:
                log_probs, _ = model.forward(s.traj)
                pred = greedy_decode(log_probs)
                sub_cers.append(cer(pred, s.labels))
                if pred == s.labels:
                    sub_hits += 1
            rows.append({
                "word": w, "n": len(sub),
                "cer_mean": float(np.mean(sub_cers)),
                "word_acc": sub_hits / len(sub),
            })
        return rows

    per_word_test = per_word_breakdown(test_samples, in_vocab)
    per_word_ood = per_word_breakdown(ood_samples, ood_vocab)

    # CTC alignment trace for one example test word -- save log_probs and
    # decoded path for visualisation.
    align_word = in_vocab[0]
    align_sample = next(s for s in test_samples if s.word == align_word)
    align_log_probs, _ = model.forward(align_sample.traj)
    align_argmax_path = align_log_probs.argmax(axis=1).tolist()
    align_decoded = greedy_decode(align_log_probs)

    # Save another alignment for the GIF (longer word for visual interest)
    long_test = max(test_samples, key=lambda s: len(s.labels))
    long_log_probs, _ = model.forward(long_test.traj)
    long_argmax = long_log_probs.argmax(axis=1).tolist()
    long_decoded = greedy_decode(long_log_probs)

    # Final metrics
    final_test = evaluate(model, test_samples)
    final_ood = evaluate(model, ood_samples)

    summary = {
        "config": asdict(cfg),
        "alphabet": ALPHABET,
        "vocab": VOCAB,
        "in_vocab": in_vocab,
        "ood_vocab": ood_vocab,
        "history": history,
        "final_test": final_test,
        "final_ood": final_ood,
        "per_word_test": per_word_test,
        "per_word_ood": per_word_ood,
        "alignment": {
            "word": align_word,
            "labels": align_sample.labels,
            "label_chars": list(align_word),
            "abs_xy": align_sample.abs_xy.tolist(),
            "traj": align_sample.traj.tolist(),
            "log_probs": align_log_probs.tolist(),
            "argmax_path": align_argmax_path,
            "decoded": align_decoded,
            "decoded_chars": [ID2CHAR[i] for i in align_decoded],
        },
        "long_alignment": {
            "word": long_test.word,
            "labels": long_test.labels,
            "label_chars": list(long_test.word),
            "abs_xy": long_test.abs_xy.tolist(),
            "traj": long_test.traj.tolist(),
            "log_probs": long_log_probs.tolist(),
            "argmax_path": long_argmax,
            "decoded": long_decoded,
            "decoded_chars": [ID2CHAR[i] for i in long_decoded],
        },
        "wallclock_sec": wall,
        "env": env_metadata(),
    }
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Smaller / shorter run for smoke testing.")
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed)
    if args.quick:
        cfg.epochs = 4
        cfg.word_repeats_per_epoch = 2
        cfg.eval_repeats = 2
        cfg.H = 24
    if args.epochs is not None:
        cfg.epochs = args.epochs

    summary = train(cfg, verbose=not args.quiet)

    print()
    print("=== Final ===")
    print(f"in-vocab fresh-rendering CER = {summary['final_test']['cer_mean']:.3f} "
          f"(word acc {summary['final_test']['word_acc']:.2f})")
    print(f"out-of-vocab CER             = {summary['final_ood']['cer_mean']:.3f} "
          f"(word acc {summary['final_ood']['word_acc']:.2f})")
    print()
    print("Per-test-word breakdown (in-vocab, fresh renderings):")
    print(f"{'word':>10}  {'n':>3}  {'cer':>6}  {'word_acc':>8}")
    for r in summary["per_word_test"]:
        print(f"{r['word']:>10}  {r['n']:>3}  "
              f"{r['cer_mean']:>6.3f}  {r['word_acc']:>8.2f}")
    print()
    print("Per-test-word breakdown (held-out vocab, compositional):")
    print(f"{'word':>10}  {'n':>3}  {'cer':>6}  {'word_acc':>8}")
    for r in summary["per_word_ood"]:
        print(f"{r['word']:>10}  {r['n']:>3}  "
              f"{r['cer_mean']:>6.3f}  {r['word_acc']:>8.2f}")
    print()
    print(f"Wallclock: {summary['wallclock_sec']:.1f}s   git={git_hash()}")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
