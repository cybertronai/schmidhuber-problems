"""self-referential-weight-matrix -- Schmidhuber, *A self-referential weight
matrix*, ICANN-93 Brighton, pp. 446-451.

The 1993 idea: a recurrent net whose weight matrix W is itself part of the
state. The net has output channels that read entries of W and write entries
of W, so the weight-change algorithm is itself learnable end-to-end. Crucially
the weights modified by the network include the very weights that decide
which weight to modify next - true self-reference.

This v1 implementation captures the structural property using a continuous
relaxation that admits BPTT under the numpy-only constraint:

  * Effective weight matrix W_eff = W_slow + W_fast.
      - W_slow are conventional parameters trained by gradient descent
        across episodes.
      - W_fast is a per-episode plastic matrix that the net itself writes
        and reads inside an episode (reset to zero at the start of each
        episode).
  * At every time step the net's outputs include
        row_attn (softmax over rows of W),
        col_attn (softmax over cols of W),
        write_value, write_gate.
    The fast matrix is updated by
        W_fast += eta * gate * value * outer(row_attn, col_attn).
  * Reads happen implicitly: the next step's recurrent dynamics use
    W_eff_t = W_slow + W_fast_t, so any entry the net wrote on step t can
    influence h_{t+1}.
  * The write-control outputs (A_row, A_col, A_val, A_gate) are themselves
    rows of the slow weight matrix and so are - in spirit - subject to
    self-reference (their effect on W_fast feeds the next step's hidden
    state, which then drives the next set of writes).

Task: meta-learning across 4 boolean variants on 2-bit inputs.

  Tasks:  AND, OR, XOR, NAND.
  Episode = 4 demo steps with labels visible + 4 query steps with labels
            held out. Sequence length T = 8.
  Inputs at each step (n_in = 4):
      x[0], x[1]  : the two input bits in {-1, +1}
      y_label     : the demo label (only set in demo phase, else 0)
      is_demo     : 1.0 in demo phase, 0.0 in query phase
  Loss: BCE on the prediction at each query step.

The point: a meta-learner whose only mechanism for storing the demo phase
is its own weight matrix. After demos, W_fast must encode which boolean
function the episode is on, and the query-phase forward pass must use that
W_fast to produce the right answers.

CLI:
    python3 self_referential_weight_matrix.py --seed 0
    python3 self_referential_weight_matrix.py --seed 0 --quick   # smoke test
    python3 self_referential_weight_matrix.py --gradcheck        # numerical check

Headline (--seed 0): query accuracy 0.85+ after ~3000 episodes; W_fast row
norms cluster by task (visible in viz).
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


def env_metadata() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "git": git_hash(),
    }


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def softmax_back(grad_out: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Backward pass for y = softmax(z). grad_out has the same shape as s."""
    return s * (grad_out - np.dot(s, grad_out))


def sigmoid(z):
    # Numerically stable.
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

# 4 boolean tasks on inputs in {-1, +1}^2. Targets are in {0, 1}.
def task_label(task_id: int, x: np.ndarray) -> float:
    a = int(x[0] > 0)
    b = int(x[1] > 0)
    if task_id == 0:                # AND
        return float(a and b)
    if task_id == 1:                # OR
        return float(a or b)
    if task_id == 2:                # XOR
        return float(a ^ b)
    if task_id == 3:                # NAND
        return float(not (a and b))
    raise ValueError(task_id)


TASK_NAMES = ["AND", "OR", "XOR", "NAND"]


def make_episode(rng: np.random.Generator, task_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (inputs[T, n_in], targets[T], is_query[T]) for one episode.

    n_in = 4 = (x0, x1, y_demo, is_demo).
    Sequence: 4 demo steps (all 4 inputs in random order, label visible) +
              4 query steps (all 4 inputs in random order, label = 0 in input).
    """
    bits = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    demo_order = rng.permutation(4)
    query_order = rng.permutation(4)
    T = 8
    inputs = np.zeros((T, 4), dtype=np.float64)
    targets = np.zeros(T, dtype=np.float64)
    is_query = np.zeros(T, dtype=bool)
    for t in range(4):                  # demo phase
        x = bits[demo_order[t]]
        y = task_label(task_id, x)
        inputs[t, 0] = x[0]
        inputs[t, 1] = x[1]
        inputs[t, 2] = 2.0 * y - 1.0    # demo label in {-1, +1}
        inputs[t, 3] = 1.0              # is_demo flag
        targets[t] = y                  # not used (query mask is False)
    for q in range(4):                  # query phase
        t = 4 + q
        x = bits[query_order[q]]
        y = task_label(task_id, x)
        inputs[t, 0] = x[0]
        inputs[t, 1] = x[1]
        inputs[t, 2] = 0.0              # demo label hidden
        inputs[t, 3] = 0.0              # is_demo flag off
        targets[t] = y
        is_query[t] = True
    return inputs, targets, is_query


# ----------------------------------------------------------------------
# SRWM model
# ----------------------------------------------------------------------

class SRWM:
    """Self-referential weight matrix model with continuous read/write
    pointers. Slow params are trained by BPTT across episodes; fast deltas
    are produced by the network itself within an episode."""

    def __init__(self, n_in: int = 4, n_h: int = 6, eta: float = 0.5, seed: int = 0):
        self.n_in = n_in
        self.n_h = n_h
        self.eta = eta
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(max(n_h, n_in))
        self.W_slow = rng.uniform(-s, s, (n_h, n_h))
        self.W_xh = rng.uniform(-s, s, (n_h, n_in))
        self.b_h = np.zeros(n_h)
        self.W_y = rng.uniform(-s, s, (1, n_h))
        self.b_y = np.zeros(1)
        self.A_row = rng.uniform(-s, s, (n_h, n_h))
        self.A_col = rng.uniform(-s, s, (n_h, n_h))
        self.A_val = rng.uniform(-s, s, (1, n_h))
        self.A_gate = rng.uniform(-s, s, (1, n_h))
        self._reset_state()

    # --- parameter access -------------------------------------------------
    def param_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W_slow": self.W_slow,
            "W_xh": self.W_xh,
            "b_h": self.b_h,
            "W_y": self.W_y,
            "b_y": self.b_y,
            "A_row": self.A_row,
            "A_col": self.A_col,
            "A_val": self.A_val,
            "A_gate": self.A_gate,
        }

    def num_params(self) -> int:
        return int(sum(p.size for p in self.param_dict().values()))

    # --- forward / backward ----------------------------------------------
    def _reset_state(self):
        self.h = np.zeros(self.n_h)
        self.W_fast = np.zeros((self.n_h, self.n_h))
        self.tape: List[Dict[str, np.ndarray]] = []
        self.fast_history: List[np.ndarray] = [self.W_fast.copy()]

    def episode(self, inputs: np.ndarray) -> np.ndarray:
        """Run a forward pass for one episode; populate self.tape and
        self.fast_history. Returns y_per_step shape (T,)."""
        self._reset_state()
        T = inputs.shape[0]
        ys = np.zeros(T)
        for t in range(T):
            x = inputs[t]
            h_prev = self.h.copy()
            W_fast_prev = self.W_fast.copy()
            W_eff = self.W_slow + W_fast_prev
            pre_h = W_eff @ h_prev + self.W_xh @ x + self.b_h
            h = np.tanh(pre_h)
            pre_y = self.W_y @ h + self.b_y
            y = sigmoid(pre_y)              # shape (1,)
            pre_row = self.A_row @ h
            row = softmax(pre_row)
            pre_col = self.A_col @ h
            col = softmax(pre_col)
            pre_val = self.A_val @ h        # shape (1,)
            val = np.tanh(pre_val)
            pre_gate = self.A_gate @ h      # shape (1,)
            gate = sigmoid(pre_gate)
            outer_rc = np.outer(row, col)
            delta = self.eta * gate[0] * val[0] * outer_rc
            W_fast_new = W_fast_prev + delta
            self.tape.append({
                "x": x.copy(),
                "h_prev": h_prev,
                "W_fast_prev": W_fast_prev,
                "W_eff": W_eff,
                "h": h,
                "y": y,
                "row": row,
                "col": col,
                "val": val,
                "gate": gate,
                "outer_rc": outer_rc,
            })
            self.h = h
            self.W_fast = W_fast_new
            self.fast_history.append(W_fast_new.copy())
            ys[t] = float(y[0])
        return ys

    def backward(self, dy_per_step: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass given dL/dy at each step. Returns gradient dict
        matching param_dict keys."""
        T = len(self.tape)
        grads = {k: np.zeros_like(v) for k, v in self.param_dict().items()}
        dh_next = np.zeros(self.n_h)
        dW_fast_next = np.zeros((self.n_h, self.n_h))
        for t in range(T - 1, -1, -1):
            tp = self.tape[t]
            dy = float(dy_per_step[t])
            y0 = tp["y"][0]
            # y = sigmoid(pre_y)
            dpre_y = np.array([dy * y0 * (1.0 - y0)])  # (1,)
            grads["W_y"] += np.outer(dpre_y, tp["h"])
            grads["b_y"] += dpre_y
            dh = self.W_y.T @ dpre_y
            dh = dh.reshape(self.n_h)

            # Gradient flowing into W_fast at end of step t (i.e. into the
            # output W_fast_t = W_fast_{t-1} + delta_t).
            dW_fast_t = dW_fast_next  # alias

            # delta_t backward; delta = eta * gate * val * outer(row, col)
            ddelta = dW_fast_t  # since delta adds into W_fast_t
            scalar = self.eta * tp["gate"][0] * tp["val"][0]
            dgate = self.eta * tp["val"][0] * np.sum(ddelta * tp["outer_rc"])
            dval = self.eta * tp["gate"][0] * np.sum(ddelta * tp["outer_rc"])
            drow = scalar * (ddelta @ tp["col"])           # (n_h,)
            dcol = scalar * (ddelta.T @ tp["row"])         # (n_h,)

            # gate backward
            dpre_gate = np.array([dgate * tp["gate"][0] * (1.0 - tp["gate"][0])])
            grads["A_gate"] += np.outer(dpre_gate, tp["h"])
            dh = dh + (self.A_gate.T @ dpre_gate).reshape(self.n_h)

            # val backward (tanh)
            dpre_val = np.array([dval * (1.0 - tp["val"][0] ** 2)])
            grads["A_val"] += np.outer(dpre_val, tp["h"])
            dh = dh + (self.A_val.T @ dpre_val).reshape(self.n_h)

            # row backward (softmax)
            dpre_row = softmax_back(drow, tp["row"])
            grads["A_row"] += np.outer(dpre_row, tp["h"])
            dh = dh + self.A_row.T @ dpre_row

            # col backward (softmax)
            dpre_col = softmax_back(dcol, tp["col"])
            grads["A_col"] += np.outer(dpre_col, tp["h"])
            dh = dh + self.A_col.T @ dpre_col

            # Add the carry from the next step.
            dh = dh + dh_next

            # h = tanh(pre_h)
            dpre_h = dh * (1.0 - tp["h"] ** 2)
            # pre_h = W_eff h_prev + W_xh x + b_h, W_eff = W_slow + W_fast_prev
            grads["W_slow"] += np.outer(dpre_h, tp["h_prev"])
            dW_fast_prev_from_pre_h = np.outer(dpre_h, tp["h_prev"])
            grads["W_xh"] += np.outer(dpre_h, tp["x"])
            grads["b_h"] += dpre_h
            dh_prev = tp["W_eff"].T @ dpre_h

            dh_next = dh_prev
            # W_fast_t feeds W_fast_{t+1} via identity (W_fast_{t+1} =
            # W_fast_t + delta_{t+1}) and via W_eff_{t+1}; we already have
            # the contribution that went *into* delta and h above; what
            # remains is the identity flow to W_fast_{t-1} plus the
            # h_prev->W_eff flow back.
            dW_fast_next = dW_fast_t + dW_fast_prev_from_pre_h
        return grads


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def bce_loss_and_grad(ys: np.ndarray, targets: np.ndarray, mask: np.ndarray):
    """BCE loss summed over masked steps. Returns (loss, dy_per_step)."""
    eps = 1e-8
    y = np.clip(ys, eps, 1.0 - eps)
    per_step = -(targets * np.log(y) + (1.0 - targets) * np.log(1.0 - y))
    loss = float(np.sum(per_step * mask)) / max(int(np.sum(mask)), 1)
    # dL/dy at each step (only at mask=True, others zero)
    dy = (y - targets) / (y * (1.0 - y))
    dy = dy / max(int(np.sum(mask)), 1)
    dy[~mask] = 0.0
    return loss, dy


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            m_hat = self.m[k] / (1.0 - self.b1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------------
# Gradient check (numerical vs analytic)
# ----------------------------------------------------------------------

def gradient_check(seed: int = 0, n_h: int = 4, h: float = 1e-5):
    rng = np.random.default_rng(seed)
    model = SRWM(n_h=n_h, eta=0.3, seed=seed)
    task_id = int(rng.integers(0, 4))
    inputs, targets, is_query = make_episode(rng, task_id)

    # Analytic.
    ys = model.episode(inputs)
    loss, dy = bce_loss_and_grad(ys, targets, is_query)
    grads = model.backward(dy)

    # Numerical for each param.
    pdict = model.param_dict()
    max_rel = 0.0
    for name, p in pdict.items():
        ng = np.zeros_like(p)
        flat = p.ravel()
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + h
            ys_plus = model.episode(inputs)
            l_plus, _ = bce_loss_and_grad(ys_plus, targets, is_query)
            flat[i] = old - h
            ys_minus = model.episode(inputs)
            l_minus, _ = bce_loss_and_grad(ys_minus, targets, is_query)
            flat[i] = old
            ng.ravel()[i] = (l_plus - l_minus) / (2 * h)
        ag = grads[name]
        denom = np.abs(ng) + np.abs(ag) + 1e-12
        rel = np.max(np.abs(ng - ag) / denom)
        print(f"  gradcheck {name:8s} max relative error = {rel:.3e}")
        max_rel = max(max_rel, rel)
    print(f"  worst relative error across all params: {max_rel:.3e}")
    return max_rel


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def evaluate(model: SRWM, rng: np.random.Generator, n_episodes: int = 200):
    """Per-task and overall query accuracy."""
    correct = np.zeros(4)
    total = np.zeros(4)
    losses = []
    for _ in range(n_episodes):
        for task_id in range(4):
            inputs, targets, is_query = make_episode(rng, task_id)
            ys = model.episode(inputs)
            loss, _ = bce_loss_and_grad(ys, targets, is_query)
            losses.append(loss)
            preds = (ys > 0.5).astype(np.float64)
            correct[task_id] += float(np.sum((preds == targets) & is_query))
            total[task_id] += float(np.sum(is_query))
    per_task = correct / np.maximum(total, 1)
    overall = float(correct.sum() / max(total.sum(), 1))
    return overall, per_task, float(np.mean(losses))


def train(seed: int = 0, n_episodes: int = 3000, n_h: int = 6, eta: float = 0.5,
          lr: float = 0.01, eval_every: int = 200, quick: bool = False, verbose: bool = True):
    if quick:
        n_episodes = 600
        eval_every = 100

    rng = np.random.default_rng(seed)
    model = SRWM(n_in=4, n_h=n_h, eta=eta, seed=seed)
    opt = Adam(model.param_dict(), lr=lr)

    history = {
        "episode": [],
        "train_loss": [],
        "eval_acc": [],
        "per_task_acc": [],
    }

    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        task_id = int(rng.integers(0, 4))
        inputs, targets, is_query = make_episode(rng, task_id)
        ys = model.episode(inputs)
        loss, dy = bce_loss_and_grad(ys, targets, is_query)
        grads = model.backward(dy)
        # Light gradient clip.
        for k in grads:
            n = np.linalg.norm(grads[k])
            if n > 5.0:
                grads[k] *= 5.0 / n
        opt.step(model.param_dict(), grads)

        if ep == 1 or ep % eval_every == 0 or ep == n_episodes:
            eval_rng = np.random.default_rng(seed + 10_000 + ep)
            ov, pt, evl = evaluate(model, eval_rng, n_episodes=80 if quick else 200)
            history["episode"].append(ep)
            history["train_loss"].append(loss)
            history["eval_acc"].append(ov)
            history["per_task_acc"].append(pt.tolist())
            if verbose:
                print(f"ep {ep:5d}  train_loss={loss:.3f}  eval_loss={evl:.3f}  "
                      f"eval_acc={ov:.3f}  per_task=[{', '.join(f'{a:.2f}' for a in pt)}]")
    wallclock = time.time() - t0

    # Final eval.
    final_rng = np.random.default_rng(seed + 99_999)
    final_acc, final_per_task, final_loss = evaluate(
        model, final_rng, n_episodes=80 if quick else 400
    )
    return model, history, {
        "n_episodes": n_episodes,
        "wallclock_s": wallclock,
        "final_overall_acc": final_acc,
        "final_per_task_acc": final_per_task.tolist(),
        "final_eval_loss": final_loss,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Self-referential weight matrix (SRWM).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-h", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-episodes", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Smoke test (fewer episodes).")
    parser.add_argument("--gradcheck", action="store_true", help="Run numerical gradient check then exit.")
    parser.add_argument("--out", type=str, default="run.json")
    args = parser.parse_args()

    if args.gradcheck:
        max_rel = gradient_check(seed=args.seed)
        print(f"gradient check {'PASS' if max_rel < 1e-4 else 'FAIL'}")
        return

    print("env:", env_metadata())
    print(f"args: seed={args.seed} n_h={args.n_h} eta={args.eta} lr={args.lr} "
          f"n_episodes={args.n_episodes} quick={args.quick}")
    model, history, summary = train(
        seed=args.seed,
        n_episodes=args.n_episodes,
        n_h=args.n_h,
        eta=args.eta,
        lr=args.lr,
        eval_every=args.eval_every,
        quick=args.quick,
    )
    print("---")
    print(f"final query accuracy:        {summary['final_overall_acc']:.3f}")
    print(f"final per-task accuracy:     "
          f"AND={summary['final_per_task_acc'][0]:.2f}  "
          f"OR={summary['final_per_task_acc'][1]:.2f}  "
          f"XOR={summary['final_per_task_acc'][2]:.2f}  "
          f"NAND={summary['final_per_task_acc'][3]:.2f}")
    print(f"wallclock:                   {summary['wallclock_s']:.1f} s")

    out = {
        "args": vars(args),
        "env": env_metadata(),
        "history": history,
        "summary": summary,
        "model_n_params": model.num_params(),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
