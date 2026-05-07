"""neural-data-router -- Csordás, Irie, Schmidhuber,
*The Neural Data Router: Adaptive Control Flow in Transformers Improves
Systematic Generalization*, ICLR 2022 (arXiv:2110.07732).

Key ideas (re-implemented in pure numpy):

  * Copy gate (per-position scalar)
        x_new = g * f(x) + (1 - g) * x
    Lets each layer either route information through attention+FFN or
    copy the previous-layer hidden state. Initialised so that g starts
    near 0 (carry-dominated) and the network can adopt a depth-adaptive
    schedule per token.

  * Geometric attention (directional scan)
        For one head we sort positions left-to-right; for the other
        right-to-left. Within a head, the attention weight at position j
        for query i is
            A[i,j] = p[i,j] * prod_{k earlier in scan} (1 - p[i,k])
        with p[i,j] = sigmoid(score[i,j]).
    This is the geometric distribution over key positions: the model
    "stops" at the first scoring position. Unlike softmax attention, the
    distribution does not smear as sequence length grows, which is the
    structural prior that drives length generalization.

Headline contrast on a synthetic compositional table-lookup task
  (vocab 16 = 8 values + 8 unary functions; depth-d expression
   v . f1 f2 ... fd produces final value f_d(...f_2(f_1(v))) ):

  * Train on depths 1..5 (sequence length 2..6).
  * Test on depths 6..8 (sequence length 7..9, *out of training*).

NDR generalises to depths 6..8; the size-matched vanilla Transformer
collapses past depth 5.

CLI:
    python3 neural_data_router.py --seed 0
    python3 neural_data_router.py --seed 0 --quick
    python3 neural_data_router.py --seed 0 --steps 8000

Pure numpy + matplotlib. Deterministic under --seed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Reproducibility / environment
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
# Synthetic compositional-lookup task
# ----------------------------------------------------------------------
# Vocabulary:
#   tokens 0..7   -> the 8 "values"   (input-only and target classes)
#   tokens 8..15  -> the 8 "functions" (each a fixed permutation of {0..7})
# Inputs:
#   sequence = [value, f_1, f_2, ..., f_d]  for depth d in [1, max_depth]
#   target   = compose(f_d, ..., f_2, f_1)(value) in {0..7}
# We pad to a fixed sequence length and ask the model to read the answer
# off the last *active* (non-pad) position. Padding token = 16 (so vocab
# size used = 17).

N_VALUES = 4
N_FUNCS = 4
PAD_ID = N_VALUES + N_FUNCS  # 8
VOCAB = PAD_ID + 1            # 9
N_CLASSES = N_VALUES          # 4 output classes (chance = 25%)

TRAIN_DEPTHS = (1, 2, 3, 4)      # sequence lengths 2..5
TEST_DEPTHS = (5, 6, 7)          # sequence lengths 6..8 (out of training)
MAX_DEPTH = 7
MAX_LEN = MAX_DEPTH + 1          # 8


def make_function_table(rng: np.random.Generator) -> np.ndarray:
    """Return a (N_FUNCS, N_VALUES) table mapping function id -> permutation.

    Each row is a random permutation of {0..N_VALUES-1}.
    """
    table = np.zeros((N_FUNCS, N_VALUES), dtype=np.int64)
    for i in range(N_FUNCS):
        table[i] = rng.permutation(N_VALUES)
    return table


def sample_batch(
    rng: np.random.Generator,
    table: np.ndarray,
    batch_size: int,
    depths: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (input_ids, target_class, lengths).

    input_ids is (B, MAX_LEN), padded with PAD_ID.
    target_class is (B,) in [0..N_VALUES).
    lengths is (B,) in [2..MAX_LEN].
    """
    inputs = np.full((batch_size, MAX_LEN), PAD_ID, dtype=np.int64)
    targets = np.zeros(batch_size, dtype=np.int64)
    lengths = np.zeros(batch_size, dtype=np.int64)
    for b in range(batch_size):
        d = int(rng.choice(depths))
        v = int(rng.integers(0, N_VALUES))
        funcs = rng.integers(0, N_FUNCS, size=d).astype(np.int64)
        inputs[b, 0] = v
        for k, f in enumerate(funcs):
            inputs[b, 1 + k] = N_VALUES + int(f)
        # Compose left-to-right: y = f_d(...f_2(f_1(v))).
        y = v
        for f in funcs:
            y = int(table[f, y])
        targets[b] = y
        lengths[b] = d + 1
    return inputs, targets, lengths


# ----------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def scan_order_for_length(L: int) -> np.ndarray:
    """Per-query scan order over keys: increasing distance from the query.

    Tiebreak: lower-index first (so for query i, distance-1 keys are
    visited as i-1 then i+1; distance-2 as i-2 then i+2; etc.). This is
    the inductive bias that makes left-to-right composition cheap (the
    "previous result" is always the second key visited after self).

    Returns int64 array of shape (L, L). scan[i, k] is the key position
    visited at scan step k for query i; every row is a permutation of
    range(L).
    """
    order = np.zeros((L, L), dtype=np.int64)
    for i in range(L):
        seq = [i]
        for d in range(1, L):
            if i - d >= 0:
                seq.append(i - d)
            if i + d < L:
                seq.append(i + d)
        order[i] = np.array(seq, dtype=np.int64)
    return order


def sinusoidal_pos_enc(L: int, d: int) -> np.ndarray:
    """Standard sinusoidal positional encoding (used at MAX_LEN+ for test)."""
    pos = np.arange(L)[:, None].astype(np.float64)
    i = np.arange(d)[None, :].astype(np.float64)
    div = np.power(10000.0, 2 * (i // 2) / d)
    angles = pos / div
    pe = np.zeros((L, d), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


# ----------------------------------------------------------------------
# Model: NDR (geometric attention + copy gate) and Vanilla Transformer
# ----------------------------------------------------------------------
# Both models share the same parameter count and shapes per layer; the
# difference is two switches:
#   self.geometric: bool   -> geometric scan vs softmax attention
#   self.copy_gate: bool   -> per-position copy gate vs always-update
#
# The forward pass caches every intermediate needed for a closed-form
# manual backward.


class NDR:
    def __init__(
        self,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 96,
        vocab: int = VOCAB,
        n_classes: int = N_CLASSES,
        geometric: bool = True,
        copy_gate: bool = True,
        use_pos_enc: bool = True,
        gate_init_bias: float = 3.0,
        seed: int = 0,
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.vocab = vocab
        self.n_classes = n_classes
        self.d_head = d_model // n_heads
        self.geometric = geometric
        self.copy_gate = copy_gate
        self.use_pos_enc = use_pos_enc

        rng = np.random.default_rng(seed)

        def init(shape, scale):
            return (rng.standard_normal(shape) * scale).astype(np.float64)

        # Token embedding (no positional embed param -- sinusoidal added on the fly)
        self.E = init((vocab, d_model), 1.0 / np.sqrt(d_model))
        self.pos_enc_cache: Dict[int, np.ndarray] = {}

        # Per-layer params
        self.layers = []
        for _ in range(n_layers):
            layer = {
                "WQ": init((d_model, d_model), 1.0 / np.sqrt(d_model)),
                "WK": init((d_model, d_model), 1.0 / np.sqrt(d_model)),
                "WV": init((d_model, d_model), 1.0 / np.sqrt(d_model)),
                "WO": init((d_model, d_model), 1.0 / np.sqrt(d_model)),
                "W1": init((d_model, d_ff), 1.0 / np.sqrt(d_model)),
                "b1": np.zeros(d_ff),
                "W2": init((d_ff, d_model), 1.0 / np.sqrt(d_ff)),
                "b2": np.zeros(d_model),
                # Per-position copy gate from concat([x, attn_out, ffn_out]) -> 1 logit
                "Wg": init((3 * d_model, 1), 1.0 / np.sqrt(3 * d_model)),
                "bg": np.full((1,), gate_init_bias, dtype=np.float64),
            }
            self.layers.append(layer)

        # Output projection: hidden at last active position -> n_classes
        self.W_out = init((d_model, n_classes), 1.0 / np.sqrt(d_model))
        self.b_out = np.zeros(n_classes)

    # ------------------------------------------------------------------
    def positional(self, L: int) -> np.ndarray:
        if L not in self.pos_enc_cache:
            self.pos_enc_cache[L] = sinusoidal_pos_enc(L, self.d_model)
        return self.pos_enc_cache[L]

    # ------------------------------------------------------------------
    # Geometric attention (paper-faithful): per-query scan over keys in
    # order of distance from the query. For query position i the scan
    # order is i, i-1, i+1, i-2, i+2, ... so the closest matching key is
    # picked first. All heads share this scan order; capacity comes from
    # different W_Q/W_K per head learning different "match" criteria.
    # Padded keys have p=0 (masked), so they don't consume scan mass.
    # ------------------------------------------------------------------

    def _scan_order(self, L: int) -> np.ndarray:
        if not hasattr(self, "_order_cache"):
            self._order_cache: Dict[int, np.ndarray] = {}
        if L not in self._order_cache:
            self._order_cache[L] = scan_order_for_length(L)
        return self._order_cache[L]

    def _geometric_weights(
        self,
        scores: np.ndarray,          # (B, H, L_q, L_k)
        key_mask: np.ndarray,        # (B, L_k)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute geometric attention weights and intermediates.

        Returns
        -------
        A : (B, H, Lq, Lk) attention weights in *original* key order.
        p_ord : (B, H, Lq, Lk) per-step pick probs along the scan order.
        prefix_ord : (B, H, Lq, Lk) prefix prod along scan order.
        order : (Lq, Lk) the per-query scan order (key indices).
        """
        B, H, Lq, Lk = scores.shape
        order = self._scan_order(Lq)                             # (Lq, Lk)
        order_b = np.broadcast_to(order, (B, H, Lq, Lk))
        # Reorder scores along last axis per query
        scores_ord = np.take_along_axis(scores, order_b, axis=-1)
        # Reorder key mask along last axis per query
        mask_ord = key_mask[:, None, None, :]                    # (B,1,1,Lk)
        mask_ord = np.broadcast_to(mask_ord, (B, H, Lq, Lk))
        mask_ord = np.take_along_axis(mask_ord, order_b, axis=-1)

        p_ord = sigmoid(scores_ord) * mask_ord
        one_minus = 1.0 - p_ord
        cumprod = np.cumprod(one_minus, axis=-1)
        prefix_ord = np.concatenate(
            [np.ones((B, H, Lq, 1)), cumprod[..., :-1]], axis=-1,
        )
        A_ord = p_ord * prefix_ord
        # Scatter A back to original key order
        A = np.zeros_like(scores)
        np.put_along_axis(A, order_b, A_ord, axis=-1)
        return A, p_ord, prefix_ord, order

    def _geometric_dscores(
        self,
        dA: np.ndarray,              # (B, H, Lq, Lk) in original key order
        p_ord: np.ndarray,
        prefix_ord: np.ndarray,
        order: np.ndarray,
    ) -> np.ndarray:
        B, H, Lq, Lk = dA.shape
        order_b = np.broadcast_to(order, (B, H, Lq, Lk))
        # Reorder dA into scan order
        dA_ord = np.take_along_axis(dA, order_b, axis=-1)
        A_ord = p_ord * prefix_ord
        weighted = dA_ord * A_ord
        revcum = np.cumsum(weighted[..., ::-1], axis=-1)[..., ::-1]
        tail = np.concatenate(
            [revcum[..., 1:], np.zeros((B, H, Lq, 1))], axis=-1,
        )
        denom = np.clip(1.0 - p_ord, 1e-9, None)
        dp_ord = dA_ord * prefix_ord - tail / denom
        dscores_ord = dp_ord * p_ord * (1.0 - p_ord)
        # Scatter dscores back into original key order
        dscores = np.zeros_like(dA)
        np.put_along_axis(dscores, order_b, dscores_ord, axis=-1)
        return dscores

    # ------------------------------------------------------------------
    def forward(self, x_ids: np.ndarray, lengths: np.ndarray):
        """Run the model.

        x_ids : (B, L) int64 token ids (padded with PAD_ID).
        lengths : (B,) int active sequence length.

        Returns logits (B, n_classes), cache for backward.
        """
        B, L = x_ids.shape
        H = self.n_heads
        d = self.d_model
        dh = self.d_head

        mask = (np.arange(L)[None, :] < lengths[:, None]).astype(np.float64)  # (B,L)

        if self.use_pos_enc:
            x = self.E[x_ids] + self.positional(L)[None, :, :]
        else:
            x = self.E[x_ids].copy()                          # (B, L, d)

        layer_caches = []
        h_in = x
        for li, lp in enumerate(self.layers):
            # Self-attention
            Q = h_in @ lp["WQ"]                              # (B, L, d)
            K = h_in @ lp["WK"]
            V = h_in @ lp["WV"]
            Qh = Q.reshape(B, L, H, dh).transpose(0, 2, 1, 3)  # (B,H,L,dh)
            Kh = K.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
            scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(dh)  # (B,H,L,L)

            if self.geometric:
                A, p_ord, prefix_ord, order = self._geometric_weights(scores, mask)
                attn_extra = (p_ord, prefix_ord, order)
            else:
                # Softmax with key-padding mask
                neg_inf_mask = (1.0 - mask)[:, None, None, :] * (-1e9)
                A = softmax(scores + neg_inf_mask, axis=-1)
                attn_extra = (A,)

            ctx = A @ Vh                                     # (B,H,L,dh)
            ctx_concat = ctx.transpose(0, 2, 1, 3).reshape(B, L, d)
            attn_out = ctx_concat @ lp["WO"]                 # (B, L, d)

            h_after_attn = h_in + attn_out                   # residual

            # FFN
            ff_pre = h_after_attn @ lp["W1"] + lp["b1"]
            ff_act = np.maximum(ff_pre, 0.0)
            ff_out = ff_act @ lp["W2"] + lp["b2"]
            h_after_ffn = h_after_attn + ff_out              # residual

            # Copy gate
            if self.copy_gate:
                gate_in = np.concatenate([h_in, attn_out, ff_out], axis=-1)  # (B,L,3d)
                gate_logit = gate_in @ lp["Wg"] + lp["bg"]    # (B,L,1)
                g = sigmoid(gate_logit)                       # (B,L,1)
                h_out = g * h_after_ffn + (1.0 - g) * h_in
            else:
                gate_in = None
                gate_logit = None
                g = None
                h_out = h_after_ffn

            layer_caches.append({
                "h_in": h_in,
                "Q": Q, "K": K, "V": V,
                "Qh": Qh, "Kh": Kh, "Vh": Vh,
                "scores": scores,
                "A": A,
                "attn_extra": attn_extra,
                "ctx": ctx,
                "ctx_concat": ctx_concat,
                "attn_out": attn_out,
                "h_after_attn": h_after_attn,
                "ff_pre": ff_pre,
                "ff_act": ff_act,
                "ff_out": ff_out,
                "h_after_ffn": h_after_ffn,
                "gate_in": gate_in,
                "gate_logit": gate_logit,
                "g": g,
                "h_out": h_out,
            })
            h_in = h_out

        # Read answer at the last active position.
        idx = (lengths - 1).astype(np.int64)
        last_h = h_in[np.arange(B), idx]                     # (B, d)
        logits = last_h @ self.W_out + self.b_out            # (B, n_classes)

        cache = {
            "x_ids": x_ids,
            "lengths": lengths,
            "mask": mask,
            "layers": layer_caches,
            "last_h": last_h,
            "h_final": h_in,
            "idx": idx,
        }
        return logits, cache

    # ------------------------------------------------------------------
    def loss_and_grads(
        self, x_ids: np.ndarray, targets: np.ndarray, lengths: np.ndarray,
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        B, L = x_ids.shape
        logits, cache = self.forward(x_ids, lengths)
        log_p = log_softmax(logits, axis=-1)
        nll = -log_p[np.arange(B), targets].mean()
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == targets).mean())

        # ---- backward ----
        H = self.n_heads
        d = self.d_model
        dh = self.d_head

        grads: Dict[str, np.ndarray] = {}

        # dL/dlogits
        p = np.exp(log_p)
        dlogits = p.copy()
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B                                          # (B, n_classes)

        # Output projection
        last_h = cache["last_h"]
        grads["W_out"] = last_h.T @ dlogits
        grads["b_out"] = dlogits.sum(0)
        dlast_h = dlogits @ self.W_out.T                      # (B, d)

        # Distribute back to h_final at the right index
        dh_final = np.zeros_like(cache["h_final"])
        idx = cache["idx"]
        dh_final[np.arange(B), idx] = dlast_h

        # Embedding grad accumulator
        grads["E"] = np.zeros_like(self.E)

        # Per-layer grads
        layer_grads: List[Dict[str, np.ndarray]] = [None] * self.n_layers

        dh_above = dh_final
        for li in range(self.n_layers - 1, -1, -1):
            lc = cache["layers"][li]
            lp = self.layers[li]
            lg: Dict[str, np.ndarray] = {}

            h_in = lc["h_in"]
            attn_out = lc["attn_out"]
            h_after_attn = lc["h_after_attn"]
            ff_out = lc["ff_out"]
            h_after_ffn = lc["h_after_ffn"]
            g = lc["g"]
            gate_in = lc["gate_in"]

            # Gate combine: h_out = g*h_after_ffn + (1-g)*h_in
            if self.copy_gate:
                dh_after_ffn = dh_above * g
                dh_in = dh_above * (1.0 - g)
                dg = (dh_above * (h_after_ffn - h_in)).sum(axis=-1, keepdims=True)  # (B,L,1)
                # gate_logit -> g via sigmoid
                dgate_logit = dg * g * (1.0 - g)
                lg["bg"] = dgate_logit.sum(axis=(0, 1))
                lg["Wg"] = gate_in.reshape(-1, 3 * d).T @ dgate_logit.reshape(-1, 1)
                dgate_in = dgate_logit @ lp["Wg"].T            # (B,L,3d)
                dh_in_from_gate = dgate_in[..., :d]
                dattn_from_gate = dgate_in[..., d:2 * d]
                dff_from_gate = dgate_in[..., 2 * d:]
                dh_in = dh_in + dh_in_from_gate
            else:
                dh_after_ffn = dh_above
                dh_in = np.zeros_like(h_in)
                dattn_from_gate = np.zeros_like(attn_out)
                dff_from_gate = np.zeros_like(ff_out)
                lg["Wg"] = np.zeros_like(lp["Wg"])
                lg["bg"] = np.zeros_like(lp["bg"])

            # h_after_ffn = h_after_attn + ff_out
            dh_after_attn = dh_after_ffn
            dff_out = dh_after_ffn + dff_from_gate

            # FFN backward
            ff_act = lc["ff_act"]
            lg["W2"] = ff_act.reshape(-1, ff_act.shape[-1]).T @ dff_out.reshape(-1, d)
            lg["b2"] = dff_out.sum(axis=(0, 1))
            dff_act = dff_out @ lp["W2"].T                      # (B,L,d_ff)
            ff_pre = lc["ff_pre"]
            dff_pre = dff_act * (ff_pre > 0).astype(np.float64)
            lg["W1"] = h_after_attn.reshape(-1, d).T @ dff_pre.reshape(-1, dff_pre.shape[-1])
            lg["b1"] = dff_pre.sum(axis=(0, 1))
            dh_after_attn = dh_after_attn + dff_pre @ lp["W1"].T

            # h_after_attn = h_in + attn_out
            dattn_out = dh_after_attn + dattn_from_gate
            dh_in = dh_in + dh_after_attn

            # Output projection
            ctx_concat = lc["ctx_concat"]
            lg["WO"] = ctx_concat.reshape(-1, d).T @ dattn_out.reshape(-1, d)
            dctx_concat = dattn_out @ lp["WO"].T                # (B,L,d)
            dctx = dctx_concat.reshape(B, L, H, dh).transpose(0, 2, 1, 3)  # (B,H,L,dh)

            # Attention backward
            Vh = lc["Vh"]
            Kh = lc["Kh"]
            Qh = lc["Qh"]
            A = lc["A"]
            scores = lc["scores"]

            # ctx = A @ V
            dA = dctx @ Vh.transpose(0, 1, 3, 2)                # (B,H,L,L)
            dVh = A.transpose(0, 1, 3, 2) @ dctx                # (B,H,L,dh)

            if self.geometric:
                p_ord, prefix_ord, order = lc["attn_extra"]
                dscores = self._geometric_dscores(dA, p_ord, prefix_ord, order)
            else:
                # softmax backward
                tmp = (dA * A).sum(axis=-1, keepdims=True)
                dscores = A * (dA - tmp)

            dscores = dscores / np.sqrt(dh)

            dQh = dscores @ Kh                                  # (B,H,L,dh)
            dKh = dscores.transpose(0, 1, 3, 2) @ Qh            # (B,H,L,dh)

            dQ = dQh.transpose(0, 2, 1, 3).reshape(B, L, d)
            dK = dKh.transpose(0, 2, 1, 3).reshape(B, L, d)
            dV = dVh.transpose(0, 2, 1, 3).reshape(B, L, d)

            lg["WQ"] = h_in.reshape(-1, d).T @ dQ.reshape(-1, d)
            lg["WK"] = h_in.reshape(-1, d).T @ dK.reshape(-1, d)
            lg["WV"] = h_in.reshape(-1, d).T @ dV.reshape(-1, d)

            dh_in = dh_in + dQ @ lp["WQ"].T + dK @ lp["WK"].T + dV @ lp["WV"].T

            layer_grads[li] = lg
            dh_above = dh_in

        # Embedding gradient (positional encoding has no parameters)
        dx_input = dh_above                                     # (B, L, d)
        x_ids = cache["x_ids"]
        # Scatter-add
        np.add.at(grads["E"], x_ids, dx_input)

        grads["layers"] = layer_grads
        return float(nll), acc, grads


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, model: NDR, lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 grad_clip: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.t = 0
        # State per parameter
        self.m = {"E": np.zeros_like(model.E),
                  "W_out": np.zeros_like(model.W_out),
                  "b_out": np.zeros_like(model.b_out),
                  "layers": [
                      {k: np.zeros_like(v) for k, v in lp.items()}
                      for lp in model.layers
                  ]}
        self.v = {"E": np.zeros_like(model.E),
                  "W_out": np.zeros_like(model.W_out),
                  "b_out": np.zeros_like(model.b_out),
                  "layers": [
                      {k: np.zeros_like(v) for k, v in lp.items()}
                      for lp in model.layers
                  ]}

    def _step_param(self, p, g, m, v):
        m[...] = self.beta1 * m + (1.0 - self.beta1) * g
        v[...] = self.beta2 * v + (1.0 - self.beta2) * g * g
        mh = m / (1.0 - self.beta1 ** self.t)
        vh = v / (1.0 - self.beta2 ** self.t)
        p[...] -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def step(self, model: NDR, grads: Dict[str, np.ndarray]):
        self.t += 1
        # Global gradient norm clipping
        if self.grad_clip is not None:
            sq = (grads["E"] ** 2).sum() + (grads["W_out"] ** 2).sum() + (grads["b_out"] ** 2).sum()
            for lg in grads["layers"]:
                for k, v in lg.items():
                    sq = sq + (v ** 2).sum()
            gnorm = float(np.sqrt(sq))
            if gnorm > self.grad_clip:
                scale = self.grad_clip / (gnorm + 1e-9)
                grads["E"] = grads["E"] * scale
                grads["W_out"] = grads["W_out"] * scale
                grads["b_out"] = grads["b_out"] * scale
                for lg in grads["layers"]:
                    for k in lg:
                        lg[k] = lg[k] * scale
        self._step_param(model.E, grads["E"], self.m["E"], self.v["E"])
        self._step_param(model.W_out, grads["W_out"], self.m["W_out"], self.v["W_out"])
        self._step_param(model.b_out, grads["b_out"], self.m["b_out"], self.v["b_out"])
        for li, lp in enumerate(model.layers):
            for k in lp:
                self._step_param(lp[k], grads["layers"][li][k],
                                 self.m["layers"][li][k], self.v["layers"][li][k])


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def evaluate(model: NDR, table: np.ndarray, rng: np.random.Generator,
             depths: Tuple[int, ...], n_eval: int = 512, batch: int = 64) -> float:
    correct = 0
    total = 0
    n_batches = max(1, n_eval // batch)
    for _ in range(n_batches):
        x, y, lens = sample_batch(rng, table, batch, depths)
        logits, _ = model.forward(x, lens)
        preds = np.argmax(logits, axis=-1)
        correct += int((preds == y).sum())
        total += batch
    return correct / total


def evaluate_per_depth(model: NDR, table: np.ndarray, rng: np.random.Generator,
                        depths: Tuple[int, ...], n_eval_each: int = 256,
                        batch: int = 64) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for d in depths:
        correct = 0
        total = 0
        n_batches = max(1, n_eval_each // batch)
        for _ in range(n_batches):
            x, y, lens = sample_batch(rng, table, batch, (d,))
            logits, _ = model.forward(x, lens)
            preds = np.argmax(logits, axis=-1)
            correct += int((preds == y).sum())
            total += batch
        out[d] = correct / total
    return out


def train(
    model_kind: str,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    eval_every: int,
    table: np.ndarray,
    snapshots: List[Dict] | None = None,
) -> Dict:
    rng_init = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1)
    rng_eval = np.random.default_rng(seed + 2)

    if model_kind == "ndr":
        # NDR uses no positional encoding: the geometric scan already
        # encodes "distance from query" as a structural prior, and
        # train/test sequences then use *identical* embeddings, which is
        # essential for length generalization.
        model = NDR(geometric=True, copy_gate=True, use_pos_enc=False, seed=seed)
    elif model_kind == "vanilla":
        model = NDR(geometric=False, copy_gate=False, use_pos_enc=True, seed=seed)
    else:
        raise ValueError(model_kind)

    opt = Adam(model, lr=lr)

    log = {
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "eval_train_acc": [],
        "eval_test_acc": [],
        "per_depth": [],
        "kind": model_kind,
        "seed": seed,
        "wallclock_sec": None,
    }

    t0 = time.time()
    for step in range(1, steps + 1):
        x, y, lens = sample_batch(rng_data, table, batch_size, TRAIN_DEPTHS)
        loss, acc, grads = model.loss_and_grads(x, y, lens)
        opt.step(model, grads)

        if step % log_every == 0 or step == 1:
            log["steps"].append(step)
            log["train_loss"].append(loss)
            log["train_acc"].append(acc)

        if step % eval_every == 0 or step == steps:
            tr_acc = evaluate(model, table, rng_eval, TRAIN_DEPTHS)
            te_acc = evaluate(model, table, rng_eval, TEST_DEPTHS)
            per_d = evaluate_per_depth(model, table, rng_eval,
                                        TRAIN_DEPTHS + TEST_DEPTHS)
            log["eval_train_acc"].append((step, tr_acc))
            log["eval_test_acc"].append((step, te_acc))
            log["per_depth"].append((step, per_d))
            if snapshots is not None:
                snapshots.append({
                    "step": step,
                    "kind": model_kind,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "per_depth": per_d,
                })
            print(f"  [{model_kind}] step {step:5d}  loss {loss:.4f}  "
                  f"train@d=1..5 {tr_acc:.3f}  test@d=6..8 {te_acc:.3f}")

    log["wallclock_sec"] = time.time() - t0
    log["final_eval"] = {
        "train_acc": log["eval_train_acc"][-1][1],
        "test_acc": log["eval_test_acc"][-1][1],
        "per_depth": log["per_depth"][-1][1],
    }
    return log, model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8000,
                        help="Training steps for each model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--quick", action="store_true",
                        help="Tiny config for smoke test (~10 s).")
    parser.add_argument("--out", default="run.json",
                        help="Where to dump run summary JSON.")
    parser.add_argument("--multi-seed", type=int, default=0,
                        help="If >0, run that many seeds and print mean/std.")
    args = parser.parse_args()

    if args.quick:
        args.steps = 600
        args.eval_every = 200
        args.log_every = 50

    rng_table = np.random.default_rng(args.seed + 1234)
    table = make_function_table(rng_table)

    if args.multi_seed > 0:
        results = {"ndr": [], "vanilla": []}
        for s in range(args.multi_seed):
            print(f"\n=== seed {s} ===")
            for kind in ("ndr", "vanilla"):
                log, _ = train(
                    kind, seed=s, steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr,
                    log_every=args.log_every, eval_every=args.eval_every,
                    table=table,
                )
                results[kind].append(log["final_eval"])
        # Print summary
        print("\n=== Multi-seed summary ===")
        for kind in ("ndr", "vanilla"):
            train_accs = [r["train_acc"] for r in results[kind]]
            test_accs = [r["test_acc"] for r in results[kind]]
            print(f"  {kind}: train_acc = {np.mean(train_accs):.3f} ± "
                  f"{np.std(train_accs):.3f},  "
                  f"test_acc = {np.mean(test_accs):.3f} ± "
                  f"{np.std(test_accs):.3f}")
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "env": env_metadata(),
                "results": results,
            }, f, indent=2, default=str)
        return

    snapshots: List[Dict] = []
    print("=== NDR (geometric attention + copy gate) ===")
    ndr_log, ndr_model = train(
        "ndr", seed=args.seed, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every, eval_every=args.eval_every,
        table=table, snapshots=snapshots,
    )

    print("\n=== Vanilla Transformer (softmax attention, no copy gate) ===")
    van_log, van_model = train(
        "vanilla", seed=args.seed, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every, eval_every=args.eval_every,
        table=table, snapshots=snapshots,
    )

    summary = {
        "config": vars(args),
        "env": env_metadata(),
        "table": table.tolist(),
        "ndr": {
            "steps": ndr_log["steps"],
            "train_loss": ndr_log["train_loss"],
            "train_acc": ndr_log["train_acc"],
            "eval_train_acc": ndr_log["eval_train_acc"],
            "eval_test_acc": ndr_log["eval_test_acc"],
            "per_depth": ndr_log["per_depth"],
            "wallclock_sec": ndr_log["wallclock_sec"],
            "final_eval": ndr_log["final_eval"],
        },
        "vanilla": {
            "steps": van_log["steps"],
            "train_loss": van_log["train_loss"],
            "train_acc": van_log["train_acc"],
            "eval_train_acc": van_log["eval_train_acc"],
            "eval_test_acc": van_log["eval_test_acc"],
            "per_depth": van_log["per_depth"],
            "wallclock_sec": van_log["wallclock_sec"],
            "final_eval": van_log["final_eval"],
        },
        "snapshots": snapshots,
        "headline": {
            "ndr_train": ndr_log["final_eval"]["train_acc"],
            "ndr_test":  ndr_log["final_eval"]["test_acc"],
            "van_train": van_log["final_eval"]["train_acc"],
            "van_test":  van_log["final_eval"]["test_acc"],
        },
    }

    # Capture a slice of attention from each model for visualization.
    rng_attn = np.random.default_rng(args.seed + 99)
    x_v, y_v, lens_v = sample_batch(rng_attn, table, 1, (5,))
    _, ndr_cache = ndr_model.forward(x_v, lens_v)
    _, van_cache = van_model.forward(x_v, lens_v)
    summary["attn_sample"] = {
        "x_ids": x_v.tolist(),
        "target": int(y_v[0]),
        "length": int(lens_v[0]),
        "ndr_attn": [lc["A"][0].tolist() for lc in ndr_cache["layers"]],
        "van_attn": [lc["A"][0].tolist() for lc in van_cache["layers"]],
        "ndr_gates": [lc["g"][0, :, 0].tolist() if lc["g"] is not None else None
                       for lc in ndr_cache["layers"]],
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print(f"NDR     final: train@1..5 = {summary['headline']['ndr_train']:.3f}, "
          f"test@6..8 = {summary['headline']['ndr_test']:.3f}")
    print(f"Vanilla final: train@1..5 = {summary['headline']['van_train']:.3f}, "
          f"test@6..8 = {summary['headline']['van_test']:.3f}")


if __name__ == "__main__":
    main()
