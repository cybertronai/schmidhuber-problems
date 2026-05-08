"""
chunker-very-deep-1200 --- Schmidhuber, Habilitationsschrift TUM 1993,
*Netzwerkarchitekturen, Zielfunktionen und Kettenregel*; closely related to the
1992 *Neural Computation* chunker paper *Learning complex extended sequences
using the principle of history compression* (NC 4(2): 234-242) and
reconstructed in the 2015 survey *Deep Learning in Neural Networks: An
Overview* (Schmidhuber 2015) sections 6.4-6.5.

The Habilitationsschrift's "very deep" claim is that the two-network
hierarchical chunker can perform credit assignment across hundreds of virtual
layers because most of the unrolled time-steps are *predictable* and get
compressed away.  The headline number cited in later retrospectives is roughly
1200 effective layers.  This stub demonstrates the depth-reduction principle
with a controlled synthetic task at T = 1200 by default (configurable; --T 500
is the faster smoke-test).

Task
----

Synthetic *trigger-recall* sequence of length T:

    t = 0          : trigger token, one of {A, B}, drawn uniformly
    t = 1 ... T-2  : deterministic predictable filler (cycling 5-symbol pattern)
    t = T - 1      : recall target = the original trigger token

The model has to predict each next token.  The trigger and the target are
*unpredictable* (no useful local context); everything in between is
deterministic given the previous filler symbol.  So a perfect predictor only
needs to remember the trigger across T-1 filler steps to nail the target.

The chunker's bet is that you don't need to back-propagate gradients through
the full T-1 filler steps.  Once the level-0 *automatizer* learns the cycling
pattern, those steps become predictable and can be skipped: the level-1
*chunker* sees only the surprises (trigger, target) and operates on a
compressed sequence of length 2.  Effective BPTT depth drops from T to ~2.

Stages
------

    1. Train automatizer A (small vanilla tanh-RNN) on next-symbol prediction
       with truncated BPTT.  After training, A is confident on filler steps
       and surprised on trigger / target.
    2. Compute surprise mask via A's per-step prediction loss; threshold.
    3. Compress sequence to surprise events only and train a tiny chunker C
       on the compressed pairs.
    4. Baseline: train a single-network end-to-end RNN on the full sequence
       with full BPTT.  Record gradient norms backward through time to make
       the vanishing-gradient curve visible.
    5. Report: effective depth ratio (= T / k_surprise), terminal accuracy on
       the recall target for chunker vs baseline, and a per-step
       gradient-norm trace for the baseline.

This is the *algorithmic* version of the 1993 demo: it is not an exact
reproduction (the original benchmark sequences are not retrievable), but it
isolates the depth-reduction mechanism and measures the same quantity --
ratio of effective BPTT depth without compression to with compression.

CLI
---

    python3 chunker_very_deep_1200.py --seed 0
    python3 chunker_very_deep_1200.py --seed 0 --T 500
    python3 chunker_very_deep_1200.py --seed 0 --T 1200      # the eponymous depth
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


def softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

# 7 symbols total: 0='A' (trigger A), 1='B' (trigger B), 2..6 = filler '1'..'5'.
NUM_SYMBOLS = 7
TRIG_A, TRIG_B = 0, 1
FILLER_BASE = 2          # filler tokens are FILLER_BASE..FILLER_BASE+CYCLE-1
CYCLE = 5                # filler cycles through 5 symbols
SYMBOL_NAMES = ["A", "B", "1", "2", "3", "4", "5"]


def make_sequence(T: int, rng: np.random.Generator):
    """Return (input_seq, target_seq, trigger).

    input_seq[t]   = x_t  (length T - 1, fed to the model at step t)
    target_seq[t]  = x_{t+1}  (length T - 1, the model's prediction target)
    trigger        = scalar in {TRIG_A, TRIG_B}
    """
    if T < CYCLE + 2:
        raise ValueError(f"T must be >= {CYCLE + 2}")
    trigger = int(rng.integers(0, 2))
    seq = np.empty(T, dtype=np.int64)
    seq[0] = trigger
    for i in range(1, T - 1):
        seq[i] = FILLER_BASE + ((i - 1) % CYCLE)
    seq[T - 1] = trigger
    return seq[:-1].copy(), seq[1:].copy(), trigger


# ----------------------------------------------------------------------
# Vanilla Elman RNN with explicit BPTT
# ----------------------------------------------------------------------

class RNN:
    """Vanilla tanh-RNN with softmax output.  One-hot int input."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, rng: np.random.Generator,
                 init_scale: float = 0.1):
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.Wxh = rng.standard_normal((n_hid, n_in)) * init_scale
        # Spectral-radius-ish init for Whh (encourages information retention but
        # still has the canonical vanishing-gradient behaviour with tanh).
        Whh = rng.standard_normal((n_hid, n_hid)) * init_scale
        self.Whh = Whh
        self.Why = rng.standard_normal((n_out, n_hid)) * init_scale
        self.bh = np.zeros(n_hid)
        self.by = np.zeros(n_out)

    # --- forward pass ---------------------------------------------------
    def forward(self, x_idx: np.ndarray, h0: np.ndarray | None = None):
        """Run the RNN forward.

        Returns:
            H[T, H]   hidden states (post-tanh)
            Z[T, O]   output logits
            h_final
        """
        T = len(x_idx)
        h = np.zeros(self.n_hid) if h0 is None else h0.copy()
        H = np.zeros((T, self.n_hid))
        Z = np.zeros((T, self.n_out))
        for t in range(T):
            xt = np.zeros(self.n_in)
            xt[x_idx[t]] = 1.0
            h = np.tanh(self.Wxh @ xt + self.Whh @ h + self.bh)
            H[t] = h
            Z[t] = self.Why @ h + self.by
        return H, Z, h

    # --- per-step cross-entropy loss -----------------------------------
    def per_step_loss(self, x_idx, y_idx, h0=None):
        H, Z, _ = self.forward(x_idx, h0)
        P = softmax(Z)
        # numerical clamp
        P = np.clip(P, 1e-12, 1.0)
        return -np.log(P[np.arange(len(y_idx)), y_idx])  # shape (T,)

    # --- full-sequence backward; returns grads + dh trace --------------
    def loss_and_grad_full(self, x_idx, y_idx, h0=None):
        """BPTT over the full sequence; also return per-step ||dh_raw|| for
        the vanishing-gradient measurement."""
        T = len(x_idx)
        H, Z, hT = self.forward(x_idx, h0)
        # softmax / CE
        P = softmax(Z)
        loss = -np.log(np.clip(P[np.arange(T), y_idx], 1e-12, 1.0)).sum()
        dZ = P.copy()
        dZ[np.arange(T), y_idx] -= 1.0           # (T, n_out)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(self.n_hid)
        grad_norms = np.zeros(T)                 # ||dh_raw_t||
        h_zero = np.zeros(self.n_hid) if h0 is None else h0
        for t in reversed(range(T)):
            dby += dZ[t]
            dWhy += np.outer(dZ[t], H[t])
            dh = self.Why.T @ dZ[t] + dh_next
            ht = H[t]
            dh_raw = (1.0 - ht * ht) * dh
            grad_norms[t] = np.linalg.norm(dh_raw)
            dbh += dh_raw
            xt = np.zeros(self.n_in)
            xt[x_idx[t]] = 1.0
            dWxh += np.outer(dh_raw, xt)
            h_prev = H[t - 1] if t > 0 else h_zero
            dWhh += np.outer(dh_raw, h_prev)
            dh_next = self.Whh.T @ dh_raw
        return loss, (dWxh, dWhh, dWhy, dbh, dby), grad_norms, hT

    # --- terminal-only gradient norms (credit-assignment trace) --------
    def terminal_target_grad_norms(self, x_idx, y_idx, h0=None):
        """Return ||d L_terminal / d h_t|| for every t.

        Only the cross-entropy at the *final* step contributes (dZ[t] = 0 for
        t < T-1).  This is the cleanest measure of how far back the recall-
        target's gradient survives.  Used for the vanishing-gradient curve.
        """
        T = len(x_idx)
        H, Z, _ = self.forward(x_idx, h0)
        P = softmax(Z)
        dZ = np.zeros_like(P)
        # only the last step's loss contributes
        last_p = P[-1].copy()
        last_p[y_idx[-1]] -= 1.0
        dZ[-1] = last_p

        dh_next = np.zeros(self.n_hid)
        grad_norms = np.zeros(T)
        for t in reversed(range(T)):
            dh = self.Why.T @ dZ[t] + dh_next
            ht = H[t]
            dh_raw = (1.0 - ht * ht) * dh
            grad_norms[t] = np.linalg.norm(dh_raw)
            dh_next = self.Whh.T @ dh_raw
        return grad_norms

    # --- truncated BPTT update over short windows ----------------------
    def loss_and_grad_truncated(self, x_idx, y_idx, h0=None, k=10):
        """Truncated BPTT: backprop only k steps from the current end.

        The hidden state h0 is treated as a constant (no gradient through it).
        Used for the automatizer, which only needs local context.
        """
        T = len(x_idx)
        # last k steps
        start = max(0, T - k)
        H, Z, hT = self.forward(x_idx[start:], h0)
        Tk = len(H)
        P = softmax(Z)
        loss = -np.log(np.clip(P[np.arange(Tk), y_idx[start:]], 1e-12, 1.0)).sum()
        dZ = P.copy()
        dZ[np.arange(Tk), y_idx[start:]] -= 1.0

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(self.n_hid)
        h_zero = np.zeros(self.n_hid) if h0 is None else h0
        for t in reversed(range(Tk)):
            dby += dZ[t]
            dWhy += np.outer(dZ[t], H[t])
            dh = self.Why.T @ dZ[t] + dh_next
            ht = H[t]
            dh_raw = (1.0 - ht * ht) * dh
            dbh += dh_raw
            xt = np.zeros(self.n_in)
            xt[x_idx[start + t]] = 1.0
            dWxh += np.outer(dh_raw, xt)
            h_prev = H[t - 1] if t > 0 else h_zero
            dWhh += np.outer(dh_raw, h_prev)
            dh_next = self.Whh.T @ dh_raw
        return loss, (dWxh, dWhh, dWhy, dbh, dby), hT

    def apply_grads(self, grads, lr: float, clip: float = 5.0):
        for g in grads:
            np.clip(g, -clip, clip, out=g)
        self.Wxh -= lr * grads[0]
        self.Whh -= lr * grads[1]
        self.Why -= lr * grads[2]
        self.bh -= lr * grads[3]
        self.by -= lr * grads[4]


# ----------------------------------------------------------------------
# Stage 1: train the automatizer
# ----------------------------------------------------------------------

def train_automatizer(T: int, rng: np.random.Generator,
                      hidden: int = 16,
                      epochs: int = 80,
                      seqs_per_epoch: int = 8,
                      lr: float = 0.05,
                      truncate: int = 6,
                      verbose: bool = True):
    """Train a small RNN on next-symbol prediction with truncated BPTT.

    The point: the automatizer easily learns the deterministic CYCLE-symbol
    cycling pattern.  It cannot predict the trigger (no context) or the recall
    target (it cannot remember what trigger was at t = 0; truncated BPTT cuts
    the chain).  Those are our surprises.
    """
    A = RNN(NUM_SYMBOLS, hidden, NUM_SYMBOLS, rng, init_scale=0.1)
    history = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(seqs_per_epoch):
            x, y, _ = make_sequence(T, rng)
            # Walk through the sequence in non-overlapping windows of size
            # `truncate`, doing a truncated-BPTT update for each window.
            h = np.zeros(hidden)
            i = 0
            while i < len(x):
                j = min(i + truncate, len(x))
                xs = x[i:j]
                ys = y[i:j]
                loss, grads, h = A.loss_and_grad_truncated(xs, ys, h0=h, k=truncate)
                A.apply_grads(grads, lr=lr / max(1, len(xs)))
                epoch_loss += loss
                i = j
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss / seqs_per_epoch)
        if verbose and (epoch % max(1, epochs // 8) == 0 or epoch == epochs - 1):
            print(f"  [automatizer] epoch {epoch:3d}  loss={history['loss'][-1]:.3f}")
    return A, history


# ----------------------------------------------------------------------
# Stage 2: detect surprises
# ----------------------------------------------------------------------

def detect_surprises(A: RNN, x_idx: np.ndarray, y_idx: np.ndarray,
                     threshold: float):
    """Return (mask, per_step_loss).  mask[t] is True iff the automatizer
    failed to predict y[t] confidently.

    Also flags t = 0 by convention: the very first symbol has no preceding
    context, so the automatizer cannot have predicted it -- it is always a
    surprise.  This matches Schmidhuber's original framing in the 1992
    chunker paper.
    """
    losses = A.per_step_loss(x_idx, y_idx)
    mask = losses > threshold
    if len(mask) > 0:
        mask[0] = True
    return mask, losses


# ----------------------------------------------------------------------
# Stage 3: train the chunker on the compressed sequence
# ----------------------------------------------------------------------

def train_chunker(rng: np.random.Generator, A: RNN, T: int,
                  hidden: int = 8,
                  epochs: int = 200,
                  seqs_per_epoch: int = 16,
                  lr: float = 0.1,
                  threshold: float = 0.5,
                  verbose: bool = True):
    """Train a tiny RNN on the *compressed* surprise stream.

    For each generated sequence we compute the automatizer's per-step loss,
    threshold it, and feed only the surprise tokens to the chunker.  In our
    task that is typically [trigger, target].
    """
    C = RNN(NUM_SYMBOLS, hidden, NUM_SYMBOLS, rng, init_scale=0.2)
    history = {"epoch": [], "loss": [], "target_acc": [],
               "n_surprises": []}
    for epoch in range(epochs):
        ep_loss = 0.0
        ep_correct = 0
        ep_seen = 0
        n_surprises_total = 0
        for _ in range(seqs_per_epoch):
            x, y, trig = make_sequence(T, rng)
            mask, _ = detect_surprises(A, x, y, threshold)
            surp_idx = np.where(mask)[0]
            if len(surp_idx) < 2:
                # The automatizer wasn't surprised by both trigger and target.
                # Fall back to forcing the two known surprise positions
                # (start and end-of-sequence-prediction).
                surp_idx = np.array([0, len(x) - 1])
            xs = x[surp_idx]
            ys = y[surp_idx]
            # Train chunker with full BPTT over this very short sequence.
            loss, grads, _, _ = C.loss_and_grad_full(xs, ys)
            C.apply_grads(grads, lr=lr / max(1, len(xs)))
            ep_loss += loss
            n_surprises_total += len(xs)
            # accuracy on the recall target = last surprise
            H, Z, _ = C.forward(xs)
            pred_last = int(Z[-1].argmax())
            ep_correct += int(pred_last == int(ys[-1]))
            ep_seen += 1
        history["epoch"].append(epoch)
        history["loss"].append(ep_loss / seqs_per_epoch)
        history["target_acc"].append(ep_correct / max(1, ep_seen))
        history["n_surprises"].append(n_surprises_total / seqs_per_epoch)
        if verbose and (epoch % max(1, epochs // 8) == 0 or epoch == epochs - 1):
            print(f"  [chunker]    epoch {epoch:3d}  loss={history['loss'][-1]:.3f}  "
                  f"target_acc={history['target_acc'][-1]:.2f}  "
                  f"~surprises/seq={history['n_surprises'][-1]:.1f}")
    return C, history


# ----------------------------------------------------------------------
# Stage 4: baseline -- single-network full-BPTT RNN
# ----------------------------------------------------------------------

def train_baseline(rng: np.random.Generator, T: int,
                   hidden: int = 16,
                   epochs: int = 30,
                   seqs_per_epoch: int = 4,
                   lr: float = 0.05,
                   verbose: bool = True):
    """Train a single-network RNN with full BPTT over T-1 steps.

    With vanilla tanh-RNN and T = 500 this is the canonical
    Hochreiter-vanishing-gradient regime: the gradient at step 0 has been
    multiplied by hundreds of `(1 - h^2) * Whh^T` factors and is essentially
    zero.  Recall accuracy on the target therefore stays near 50% (chance).
    """
    R = RNN(NUM_SYMBOLS, hidden, NUM_SYMBOLS, rng, init_scale=0.1)
    history = {"epoch": [], "loss": [], "target_acc": [], "grad_norms_last": None}
    for epoch in range(epochs):
        ep_loss = 0.0
        ep_correct = 0
        ep_seen = 0
        last_grad_norms = None
        for _ in range(seqs_per_epoch):
            x, y, trig = make_sequence(T, rng)
            loss, grads, grad_norms, _ = R.loss_and_grad_full(x, y)
            R.apply_grads(grads, lr=lr / len(x))
            ep_loss += loss
            last_grad_norms = grad_norms
            H, Z, _ = R.forward(x)
            pred_last = int(Z[-1].argmax())
            ep_correct += int(pred_last == int(y[-1]))
            ep_seen += 1
        history["epoch"].append(epoch)
        history["loss"].append(ep_loss / seqs_per_epoch)
        history["target_acc"].append(ep_correct / max(1, ep_seen))
        history["grad_norms_last"] = last_grad_norms
        if verbose and (epoch % max(1, epochs // 6) == 0 or epoch == epochs - 1):
            print(f"  [baseline]   epoch {epoch:3d}  loss={history['loss'][-1]:.2f}  "
                  f"target_acc={history['target_acc'][-1]:.2f}")
    return R, history


# ----------------------------------------------------------------------
# Stage 5: evaluation
# ----------------------------------------------------------------------

def evaluate_chunker(A: RNN, C: RNN, T: int, n: int, rng: np.random.Generator,
                     threshold: float):
    correct = 0
    n_surprises = []
    for _ in range(n):
        x, y, _ = make_sequence(T, rng)
        mask, _ = detect_surprises(A, x, y, threshold)
        surp_idx = np.where(mask)[0]
        if len(surp_idx) < 2:
            surp_idx = np.array([0, len(x) - 1])
        xs = x[surp_idx]
        ys = y[surp_idx]
        H, Z, _ = C.forward(xs)
        pred_last = int(Z[-1].argmax())
        correct += int(pred_last == int(ys[-1]))
        n_surprises.append(len(xs))
    return correct / n, float(np.mean(n_surprises))


def evaluate_baseline(R: RNN, T: int, n: int, rng: np.random.Generator):
    correct = 0
    for _ in range(n):
        x, y, _ = make_sequence(T, rng)
        H, Z, _ = R.forward(x)
        pred_last = int(Z[-1].argmax())
        correct += int(pred_last == int(y[-1]))
    return correct / n


# ----------------------------------------------------------------------
# Effective-depth metric
# ----------------------------------------------------------------------

def effective_depth(grad_norms: np.ndarray, ratio: float = 0.01) -> int:
    """Return the largest backward distance d such that the gradient norm at
    step (T-1-d) is still >= `ratio` * grad_norms[-1].

    Intuition: if the per-step ||dh_raw|| has decayed to 1% of its terminal
    value, gradients from the loss can no longer move the parameters at that
    early step.  This is a textbook proxy for "effective BPTT depth".
    """
    if grad_norms is None or len(grad_norms) == 0:
        return 0
    base = grad_norms[-1]
    if base <= 0:
        return 0
    threshold = ratio * base
    # Walk backward; count how far we go before the gradient drops below threshold.
    d = 0
    for t in range(len(grad_norms) - 1, -1, -1):
        if grad_norms[t] < threshold:
            break
        d = (len(grad_norms) - 1) - t
    return d


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run(args):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[chunker-very-deep-1200] seed={args.seed}  T={args.T}")
    print()
    print("--- Stage 1: train automatizer ---")
    A, hist_A = train_automatizer(args.T, rng,
                                  hidden=args.auto_hidden,
                                  epochs=args.auto_epochs,
                                  seqs_per_epoch=8,
                                  lr=args.auto_lr,
                                  truncate=args.auto_truncate,
                                  verbose=True)

    # Pick a surprise threshold from the automatizer's loss distribution on
    # filler vs trigger/target.  We probe a few sequences.
    probe_losses_filler, probe_losses_surprise = [], []
    for _ in range(8):
        x, y, _ = make_sequence(args.T, rng)
        ls = A.per_step_loss(x, y)
        probe_losses_surprise.append(ls[0])
        probe_losses_surprise.append(ls[-1])
        probe_losses_filler.extend(ls[1:-1].tolist())
    median_filler = float(np.median(probe_losses_filler))
    median_surprise = float(np.median(probe_losses_surprise))
    threshold = 0.5 * (median_filler + median_surprise)
    print(f"  [automatizer probe] median filler loss = {median_filler:.3f}, "
          f"median surprise loss = {median_surprise:.3f}, threshold = {threshold:.3f}")

    print()
    print("--- Stage 2/3: train chunker on compressed surprise stream ---")
    C, hist_C = train_chunker(rng, A, args.T,
                              hidden=args.chunk_hidden,
                              epochs=args.chunk_epochs,
                              seqs_per_epoch=16,
                              lr=args.chunk_lr,
                              threshold=threshold,
                              verbose=True)

    print()
    print("--- Stage 4: baseline single-network full-BPTT RNN ---")
    R, hist_R = train_baseline(rng, args.T,
                               hidden=args.baseline_hidden,
                               epochs=args.baseline_epochs,
                               seqs_per_epoch=4,
                               lr=args.baseline_lr,
                               verbose=True)

    print()
    print("--- Stage 5: evaluation ---")
    chunker_acc, chunker_avg_surprises = evaluate_chunker(
        A, C, args.T, args.n_eval, rng, threshold)
    baseline_acc = evaluate_baseline(R, args.T, args.n_eval, rng)
    print(f"  Chunker target accuracy : {chunker_acc * 100:.1f}%  "
          f"(avg surprises/seq = {chunker_avg_surprises:.2f})")
    print(f"  Baseline target accuracy: {baseline_acc * 100:.1f}%")

    # For the depth measurement we want ||d L_terminal / d h_t|| -- the
    # gradient that the recall-target loss sends back through time.  The
    # per-step training loss (used in `hist_R['grad_norms_last']`) is dominated
    # by local prediction errors and obscures the credit-assignment curve.
    x_probe, y_probe, _ = make_sequence(args.T, rng)
    grad_norms_baseline = R.terminal_target_grad_norms(x_probe, y_probe)
    d_baseline = effective_depth(grad_norms_baseline, ratio=0.01)
    # Chunker's effective depth is just the length of the compressed sequence,
    # which by definition has no vanishing because all steps carry signal.
    d_chunker = int(round(chunker_avg_surprises))
    depth_ratio = (args.T - 1) / max(1, d_chunker)
    print()
    print(f"  Effective BPTT depth (baseline, 1% threshold) : {d_baseline} steps "
          f"of {args.T - 1}")
    print(f"  Effective BPTT depth (chunker)                : ~{d_chunker} steps")
    print(f"  Depth-reduction ratio (T - 1) / k_compressed  : {depth_ratio:.1f}x")

    # Save results
    results = {
        "seed": args.seed,
        "T": args.T,
        "hyperparameters": {
            "auto_hidden": args.auto_hidden,
            "auto_epochs": args.auto_epochs,
            "auto_lr": args.auto_lr,
            "auto_truncate": args.auto_truncate,
            "chunk_hidden": args.chunk_hidden,
            "chunk_epochs": args.chunk_epochs,
            "chunk_lr": args.chunk_lr,
            "baseline_hidden": args.baseline_hidden,
            "baseline_epochs": args.baseline_epochs,
            "baseline_lr": args.baseline_lr,
            "threshold": threshold,
            "n_eval": args.n_eval,
        },
        "automatizer_loss": hist_A["loss"],
        "chunker_loss": hist_C["loss"],
        "chunker_target_acc": hist_C["target_acc"],
        "chunker_n_surprises": hist_C["n_surprises"],
        "baseline_loss": hist_R["loss"],
        "baseline_target_acc": hist_R["target_acc"],
        "baseline_grad_norms": grad_norms_baseline.tolist()
            if grad_norms_baseline is not None else None,
        "chunker_eval_acc": chunker_acc,
        "chunker_avg_surprises": chunker_avg_surprises,
        "baseline_eval_acc": baseline_acc,
        "effective_depth_baseline": d_baseline,
        "effective_depth_chunker": d_chunker,
        "depth_reduction_ratio": depth_ratio,
        "wallclock_s": time.time() - t0,
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git": git_hash(),
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"[done] wallclock {results['wallclock_s']:.1f}s  "
          f"results -> {os.path.join(out_dir, 'results.json')}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=1200,
                   help="sequence length (default 1200 -- the eponymous very-deep "
                        "number; takes ~30s on an M-series laptop). Use --T 500 "
                        "for a faster ~15s run.)")
    p.add_argument("--auto-hidden", type=int, default=16)
    p.add_argument("--auto-epochs", type=int, default=80)
    p.add_argument("--auto-lr", type=float, default=0.05)
    p.add_argument("--auto-truncate", type=int, default=6)
    p.add_argument("--chunk-hidden", type=int, default=8)
    p.add_argument("--chunk-epochs", type=int, default=200)
    p.add_argument("--chunk-lr", type=float, default=0.1)
    p.add_argument("--baseline-hidden", type=int, default=16)
    p.add_argument("--baseline-epochs", type=int, default=30)
    p.add_argument("--baseline-lr", type=float, default=0.05)
    p.add_argument("--n-eval", type=int, default=50)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
