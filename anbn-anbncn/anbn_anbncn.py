"""anbn-anbncn — Gers & Schmidhuber 2001 (IEEE TNN 12(6)).

Two formal languages:

* **a^n b^n** is context-free: n a's followed by n b's. One counter suffices.
* **a^n b^n c^n** is context-sensitive: n a's, n b's, n c's. Two counters
  required.

The original paper is the first RNN result on a context-sensitive language.
A peephole LSTM (cells feed the gates with element-wise weighted
connections) trained on n in 1..10 generalises to much larger n at test
time. The peephole, introduced for precise-timing tasks in Gers,
Schraudolph & Schmidhuber 2002, makes the gate decisions sensitive to the
exact counter value held in the CEC, not just to the post-output-gate
hidden state.

This file holds the entire pipeline: per-language dataset, peephole-LSTM
forward and backward, online training loop, generalisation evaluator that
sweeps n=1..N_TEST, an analytic-vs-finite-difference gradient check, and a
CLI that reproduces the headline run. Pure numpy.

Encoding (both languages):

  vocabulary = {S, a, b, [c,] T}
  input at step t  = one-hot of the t-th symbol in the actual string
                     S a^n b^n [c^n] T
  target at step t = binary mask over vocabulary indicating which symbols
                     are *legal* next under the language given the prefix.

A test sequence is "accepted" if at every step the network's sigmoid
outputs, thresholded at 0.5, equal the target binary mask exactly. This is
the standard Reber-grammar criterion adapted to a^n b^n / a^n b^n c^n.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# anbn alphabet:    0=S 1=a 2=b 3=T
# anbncn alphabet:  0=S 1=a 2=b 3=c 4=T
ANBN_VOCAB = ("S", "a", "b", "T")
ANBNCN_VOCAB = ("S", "a", "b", "c", "T")


def _make_inputs_targets(symbols: list[int], legal_next: list[set[int]],
                         vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert (symbol stream, per-step legal-next-set) to one-hot pair."""
    T = len(symbols) - 1  # we predict the next symbol after each input
    inputs = np.zeros((T, vocab_size), dtype=np.float64)
    targets = np.zeros((T, vocab_size), dtype=np.float64)
    for t in range(T):
        inputs[t, symbols[t]] = 1.0
        for j in legal_next[t]:
            targets[t, j] = 1.0
    return inputs, targets


def make_anbn(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs, targets) for the unique a^n b^n string with start/end markers."""
    assert n >= 1
    S, a, b, T = 0, 1, 2, 3
    symbols = [S] + [a] * n + [b] * n + [T]
    legal: list[set[int]] = []
    # Compute legal-next set at every step except the last (we don't predict past T).
    for t in range(len(symbols) - 1):
        cur = symbols[t]
        if cur == S:
            legal.append({a})
        elif cur == a:
            # Could continue with another a or switch to b.
            legal.append({a, b})
        elif cur == b:
            # Position within the b-block: count b's seen so far including this one.
            j = sum(1 for s in symbols[: t + 1] if s == b)
            legal.append({b} if j < n else {T})
        else:
            raise AssertionError("unreachable")
    return _make_inputs_targets(symbols, legal, vocab_size=4)


def make_anbncn(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs, targets) for the unique a^n b^n c^n string."""
    assert n >= 1
    S, a, b, c, T = 0, 1, 2, 3, 4
    symbols = [S] + [a] * n + [b] * n + [c] * n + [T]
    legal: list[set[int]] = []
    for t in range(len(symbols) - 1):
        cur = symbols[t]
        if cur == S:
            legal.append({a})
        elif cur == a:
            legal.append({a, b})
        elif cur == b:
            j = sum(1 for s in symbols[: t + 1] if s == b)
            legal.append({b} if j < n else {c})
        elif cur == c:
            j = sum(1 for s in symbols[: t + 1] if s == c)
            legal.append({c} if j < n else {T})
        else:
            raise AssertionError("unreachable")
    return _make_inputs_targets(symbols, legal, vocab_size=5)


def make_sample(rng: np.random.Generator, lang: str, n_max: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(rng.integers(1, n_max + 1))
    if lang == "anbn":
        inp, tgt = make_anbn(n)
    elif lang == "anbncn":
        inp, tgt = make_anbncn(n)
    else:
        raise ValueError(lang)
    return inp, tgt, n


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


def bce(y: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-12
    return float(-(target * np.log(y + eps) + (1.0 - target) * np.log(1.0 - y + eps)).sum())


# ---------------------------------------------------------------------------
# Peephole LSTM (Gers & Schmidhuber 2001 / Gers, Schraudolph & Schmidhuber 2002)
# ---------------------------------------------------------------------------

@dataclass
class LSTMParams:
    Wi: np.ndarray   # (V+H, H)  input gate weights from [x; h_{t-1}]
    bi: np.ndarray   # (H,)
    pi: np.ndarray   # (H,)      peephole from c_{t-1} into input gate
    Wf: np.ndarray   # (V+H, H)
    bf: np.ndarray
    pf: np.ndarray   # (H,)      peephole from c_{t-1} into forget gate
    Wg: np.ndarray   # (V+H, H)  cell input
    bg: np.ndarray
    Wo: np.ndarray   # (V+H, H)
    bo: np.ndarray
    po: np.ndarray   # (H,)      peephole from c_t   into output gate
    Wy: np.ndarray   # (H, V)
    by: np.ndarray   # (V,)


_PARAM_KEYS = ("Wi", "bi", "pi", "Wf", "bf", "pf", "Wg", "bg", "Wo", "bo", "po", "Wy", "by")


def init_lstm(rng: np.random.Generator, *, vocab: int, hidden: int) -> LSTMParams:
    s = 0.1
    H = hidden
    V = vocab
    return LSTMParams(
        Wi=rng.standard_normal((V + H, H)) * s,
        bi=np.full(H, -1.0),    # bias input gate slightly closed at start
        pi=rng.standard_normal(H) * s,
        Wf=rng.standard_normal((V + H, H)) * s,
        bf=np.full(H,  1.0),    # bias forget gate slightly open (remember by default)
        pf=rng.standard_normal(H) * s,
        Wg=rng.standard_normal((V + H, H)) * s,
        bg=np.zeros(H),
        Wo=rng.standard_normal((V + H, H)) * s,
        bo=np.zeros(H),
        po=rng.standard_normal(H) * s,
        Wy=rng.standard_normal((H, V)) * s,
        by=np.zeros(V),
    )


def lstm_forward(p: LSTMParams, X: np.ndarray) -> dict:
    """Forward one sequence (batch=1).

    X: (T, V) one-hot input stream. Returns cache for backward, including
    per-step gate activations, cells, hiddens, output sigmoids.
    """
    T, V = X.shape
    H = p.bi.shape[0]
    h = np.zeros(H)
    c = np.zeros(H)
    i_seq = np.empty((T, H))
    f_seq = np.empty((T, H))
    g_seq = np.empty((T, H))
    o_seq = np.empty((T, H))
    c_seq = np.empty((T, H))
    h_seq = np.empty((T, H))
    tanh_c_seq = np.empty((T, H))
    z_seq = np.empty((T, V + H))
    y_seq = np.empty((T, V))

    for t in range(T):
        z = np.concatenate([X[t], h])
        i = sigmoid(z @ p.Wi + p.pi * c + p.bi)
        f = sigmoid(z @ p.Wf + p.pf * c + p.bf)
        g = np.tanh(z @ p.Wg + p.bg)
        c = f * c + i * g
        o = sigmoid(z @ p.Wo + p.po * c + p.bo)
        tanh_c = np.tanh(c)
        h = o * tanh_c
        y = sigmoid(h @ p.Wy + p.by)

        i_seq[t] = i; f_seq[t] = f; g_seq[t] = g; o_seq[t] = o
        c_seq[t] = c; h_seq[t] = h; tanh_c_seq[t] = tanh_c
        z_seq[t] = z; y_seq[t] = y

    return dict(X=X, i=i_seq, f=f_seq, g=g_seq, o=o_seq,
                c=c_seq, h=h_seq, tanh_c=tanh_c_seq, z=z_seq, y=y_seq)


def lstm_backward(p: LSTMParams, cache: dict, target: np.ndarray) -> tuple[float, dict]:
    """Compute total per-step BCE loss and gradients."""
    X = cache["X"]
    i_seq = cache["i"]; f_seq = cache["f"]; g_seq = cache["g"]; o_seq = cache["o"]
    c_seq = cache["c"]; h_seq = cache["h"]; tanh_c_seq = cache["tanh_c"]
    z_seq = cache["z"]; y_seq = cache["y"]
    T, V = X.shape
    H = p.bi.shape[0]

    loss = sum(bce(y_seq[t], target[t]) for t in range(T))

    # Initialise weight grads.
    dWi = np.zeros_like(p.Wi); dbi = np.zeros_like(p.bi); dpi = np.zeros_like(p.pi)
    dWf = np.zeros_like(p.Wf); dbf = np.zeros_like(p.bf); dpf = np.zeros_like(p.pf)
    dWg = np.zeros_like(p.Wg); dbg = np.zeros_like(p.bg)
    dWo = np.zeros_like(p.Wo); dbo = np.zeros_like(p.bo); dpo = np.zeros_like(p.po)
    dWy = np.zeros_like(p.Wy); dby = np.zeros_like(p.by)

    dh_next = np.zeros(H)
    dc_next = np.zeros(H)

    for t in reversed(range(T)):
        i_t = i_seq[t]; f_t = f_seq[t]; g_t = g_seq[t]; o_t = o_seq[t]
        c_t = c_seq[t]; tanh_c = tanh_c_seq[t]; h_t = h_seq[t]
        z_t = z_seq[t]; y_t = y_seq[t]
        c_prev = c_seq[t - 1] if t > 0 else np.zeros(H)

        # Per-step output gradient via sigmoid + BCE: dlogits = y - target
        dlogits = y_t - target[t]
        dWy += np.outer(h_t, dlogits)
        dby += dlogits
        dh = dh_next + dlogits @ p.Wy.T

        # h_t = o_t * tanh(c_t)
        do = dh * tanh_c
        dtanh = dh * o_t

        # c_t enters tanh_c_t and the output gate via the o-peephole p_o.
        dc = dc_next + dtanh * (1.0 - tanh_c ** 2)

        # o_t = σ(W_o z + p_o ⊙ c_t + b_o)
        do_pre = do * o_t * (1.0 - o_t)
        dc += do_pre * p.po  # output-gate peephole back-prop into c_t

        # c_t = f_t * c_{t-1} + i_t * g_t
        df = dc * c_prev
        di = dc * g_t
        dg = dc * i_t
        dc_prev = dc * f_t

        # gate pre-activations
        di_pre = di * i_t * (1.0 - i_t)
        df_pre = df * f_t * (1.0 - f_t)
        dg_pre = dg * (1.0 - g_t ** 2)

        # input/forget peepholes from c_{t-1}
        dc_prev = dc_prev + di_pre * p.pi + df_pre * p.pf

        # weight grads
        dWi += np.outer(z_t, di_pre)
        dbi += di_pre
        dpi += di_pre * c_prev
        dWf += np.outer(z_t, df_pre)
        dbf += df_pre
        dpf += df_pre * c_prev
        dWg += np.outer(z_t, dg_pre)
        dbg += dg_pre
        dWo += np.outer(z_t, do_pre)
        dbo += do_pre
        dpo += do_pre * c_t

        # gradient on z = [x; h_{t-1}]
        dz = di_pre @ p.Wi.T + df_pre @ p.Wf.T + dg_pre @ p.Wg.T + do_pre @ p.Wo.T
        dh_prev = dz[V:]

        dh_next = dh_prev
        dc_next = dc_prev

    grads = dict(Wi=dWi, bi=dbi, pi=dpi, Wf=dWf, bf=dbf, pf=dpf,
                 Wg=dWg, bg=dbg, Wo=dWo, bo=dbo, po=dpo, Wy=dWy, by=dby)
    return loss, grads


# ---------------------------------------------------------------------------
# Adam + glue
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def lstm_to_dict(p: LSTMParams) -> dict:
    return {k: getattr(p, k) for k in _PARAM_KEYS}


def dict_to_lstm(d: dict) -> LSTMParams:
    return LSTMParams(**{k: d[k] for k in _PARAM_KEYS})


def grad_clip(grads: dict, max_norm: float = 1.0) -> float:
    total = 0.0
    for g in grads.values():
        total += float((g * g).sum())
    norm = float(np.sqrt(total))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for g in grads.values():
            g *= scale
    return norm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def accept_sequence(p: LSTMParams, inputs: np.ndarray, targets: np.ndarray,
                    threshold: float = 0.5) -> bool:
    """A sequence is accepted iff at every step the thresholded predictions
    match the binary legal-next-set exactly."""
    cache = lstm_forward(p, inputs)
    pred = (cache["y"] >= threshold).astype(np.float64)
    return bool((pred == targets).all())


def eval_generalisation(p: LSTMParams, lang: str, n_test: int) -> dict:
    """Run the network on n=1..n_test (one sequence per n). Return dict with
    accepted set, max accepted run starting from n=1, per-n accept array."""
    accepted = []
    per_n: list[bool] = []
    make = make_anbn if lang == "anbn" else make_anbncn
    for n in range(1, n_test + 1):
        inp, tgt = make(n)
        ok = accept_sequence(p, inp, tgt)
        per_n.append(ok)
        if ok:
            accepted.append(n)
    # max contiguous run starting from n=1
    max_run = 0
    for ok in per_n:
        if ok:
            max_run += 1
        else:
            break
    return dict(accepted=accepted, per_n=per_n, max_run=max_run)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(*, lang: str, hidden: int, n_train_max: int, n_test: int,
          n_steps: int, lr: float, seed: int, log_every: int = 200,
          early_stop_target: int | None = None,
          history: list | None = None) -> tuple[LSTMParams, dict]:
    """Online BPTT training. One sequence per Adam step.

    Returns trained params and a stats dict containing the loss history,
    generalisation history (every log_every steps), and the final eval.
    """
    rng = np.random.default_rng(seed)
    vocab = 4 if lang == "anbn" else 5
    p = init_lstm(rng, vocab=vocab, hidden=hidden)
    pdict = lstm_to_dict(p)
    opt = Adam(lr=lr)
    loss_hist = []
    gen_hist = []  # list of (step, max_run)
    t0 = time.time()
    final_loss = float("nan")

    for step in range(1, n_steps + 1):
        inp, tgt, n = make_sample(rng, lang, n_train_max)
        cache = lstm_forward(dict_to_lstm(pdict), inp)
        loss, grads = lstm_backward(dict_to_lstm(pdict), cache, tgt)
        norm = grad_clip(grads, max_norm=1.0)
        opt.step(pdict, grads)
        loss_hist.append(loss / max(1, inp.shape[0]))  # per-step BCE for plotting
        final_loss = loss

        if step % log_every == 0 or step == n_steps:
            ev = eval_generalisation(dict_to_lstm(pdict), lang, n_test)
            gen_hist.append((step, ev["max_run"]))
            if history is not None:
                history.append(dict(step=step, params=lstm_to_dict(dict_to_lstm(pdict)),
                                    max_run=ev["max_run"]))
            if early_stop_target is not None and ev["max_run"] >= early_stop_target:
                break

    p_final = dict_to_lstm(pdict)
    final_eval = eval_generalisation(p_final, lang, n_test)
    stats = dict(
        lang=lang,
        loss_hist=loss_hist,
        gen_hist=gen_hist,
        final_eval=final_eval,
        wallclock=time.time() - t0,
        steps_run=step,
        final_loss=final_loss,
    )
    return p_final, stats


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def gradient_check(seed: int = 0) -> float:
    """Finite-difference gradient check on a^n b^n with n=2. Returns max relative error."""
    rng = np.random.default_rng(seed)
    p = init_lstm(rng, vocab=4, hidden=3)
    pdict = lstm_to_dict(p)
    inp, tgt = make_anbn(2)

    cache = lstm_forward(dict_to_lstm(pdict), inp)
    _, grads = lstm_backward(dict_to_lstm(pdict), cache, tgt)

    eps = 1e-5
    max_rel = 0.0
    rng2 = np.random.default_rng(seed + 1)
    for k, g in grads.items():
        flat = pdict[k].reshape(-1)
        if flat.size == 0:
            continue
        # check up to 5 random entries per param block
        idxs = rng2.choice(flat.size, size=min(5, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            l_plus, _ = lstm_backward(dict_to_lstm(pdict),
                                      lstm_forward(dict_to_lstm(pdict), inp), tgt)
            flat[idx] = orig - eps
            l_minus, _ = lstm_backward(dict_to_lstm(pdict),
                                       lstm_forward(dict_to_lstm(pdict), inp), tgt)
            flat[idx] = orig
            num = (l_plus - l_minus) / (2.0 * eps)
            ana = float(g.reshape(-1)[idx])
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
    return max_rel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def env_record() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        commit=commit,
    )


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("anbn", "anbncn", "both"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-anbn", type=int, default=4000)
    ap.add_argument("--steps-anbncn", type=int, default=8000)
    ap.add_argument("--hidden-anbn", type=int, default=2)
    ap.add_argument("--hidden-anbncn", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--gradcheck", action="store_true",
                    help="run finite-difference gradient check and exit")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.gradcheck:
        rel = gradient_check(seed=args.seed)
        print(f"max relative gradient error = {rel:.2e}")
        return

    np.random.seed(args.seed)
    runs = ("anbn", "anbncn") if args.lang == "both" else (args.lang,)
    out: dict = {"env": env_record(), "args": vars(args), "runs": {}}
    for lang in runs:
        hidden = args.hidden_anbn if lang == "anbn" else args.hidden_anbncn
        steps = args.steps_anbn if lang == "anbn" else args.steps_anbncn
        early = 2 * args.n_train  # stop early when generalisation crosses 2k
        if lang == "anbncn":
            early = None  # anbncn is harder; let it run the full budget
        print(f"\n=== {lang}  hidden={hidden}  steps={steps}  n_train≤{args.n_train}  n_test≤{args.n_test} ===")
        p_final, stats = train(
            lang=lang, hidden=hidden, n_train_max=args.n_train, n_test=args.n_test,
            n_steps=steps, lr=args.lr, seed=args.seed,
            log_every=200, early_stop_target=early,
        )
        ev = stats["final_eval"]
        print(f"  final loss/step={stats['loss_hist'][-1]:.4f}"
              f"  max contiguous accept run from n=1: {ev['max_run']}"
              f"  total accepted (over 1..{args.n_test}): {len(ev['accepted'])}"
              f"  wallclock={stats['wallclock']:.1f}s")
        out["runs"][lang] = dict(
            hidden=hidden, steps=stats["steps_run"], wallclock=stats["wallclock"],
            final_loss_per_step=stats["loss_hist"][-1],
            max_run=ev["max_run"], n_accepted=len(ev["accepted"]),
            accepted=ev["accepted"],
        )

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    _cli()
