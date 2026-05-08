"""
flip-flop --- Schmidhuber, *Making the world differentiable*, FKI-126-90 / IJCNN
1990 (vol. 2, pp. 253-258).

A controller C has to act like a flip-flop latch: output 1 whenever event B
fires after the last A, and 0 otherwise (until the next A resets the latch).
There is no labeled target -- the only feedback is a scalar pain signal that
the environment emits each step.  The 1990 paper trains C *through* a
differentiable world-model M:

    1.  M is trained to predict the next pain from (observation, action).
    2.  C is trained to minimize predicted future pain by back-propagating
        through (frozen) M into C's recurrent weights.

This file implements that two-network setup with pure numpy:

    o  Controller C: vanilla Elman RNN, 5 inputs (A, B, X, bias, pain) plus
       its own previous output, sigmoid scalar output.
    o  World-model M: vanilla Elman RNN, takes the same observation plus C's
       action, sigmoid scalar pain prediction.

Both nets are trained by truncated BPTT through one episode at a time (the
1990 paper used RTRL; BPTT is equivalent for fixed-length episodes and is
much cheaper to write).

CLI
---

    python3 flip_flop.py --seed 0
    python3 flip_flop.py --seed 0 --regime parallel

Single seed, sequential regime:  ~30s on an M-series laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import platform
import subprocess

import numpy as np


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


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


# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------

class Controller:
    """Vanilla Elman RNN -> sigmoid scalar.

    Inputs  : 5-d observation (A, B, X, bias, pain) plus previous output.
    Hidden  : tanh.
    Output  : sigmoid scalar in (0, 1) -- the "probabilistic real-valued
              output unit" of the 1990 paper.
    """

    def __init__(self, n_in: int = 5, n_hidden: int = 16,
                 init_scale: float = 0.5, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.n_in = n_in
        self.n_hidden = n_hidden
        n_x = n_in + 1
        self.W_xh = rng.standard_normal((n_hidden, n_x)) * (init_scale / np.sqrt(n_x))
        self.W_hh = rng.standard_normal((n_hidden, n_hidden)) * (init_scale / np.sqrt(n_hidden))
        self.b_h = np.zeros(n_hidden)
        self.W_ho = rng.standard_normal((1, n_hidden)) * (init_scale / np.sqrt(n_hidden))
        self.b_o = np.zeros(1)

    def params(self):
        return [self.W_xh, self.W_hh, self.b_h, self.W_ho, self.b_o]

    def param_names(self):
        return ["W_xh", "W_hh", "b_h", "W_ho", "b_o"]


class WorldModel:
    """Vanilla Elman RNN -> sigmoid scalar pain prediction."""

    def __init__(self, n_in: int = 5, n_hidden: int = 16,
                 init_scale: float = 0.5, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(1)
        self.n_in = n_in
        self.n_hidden = n_hidden
        n_x = n_in + 1
        self.W_xh = rng.standard_normal((n_hidden, n_x)) * (init_scale / np.sqrt(n_x))
        self.W_hh = rng.standard_normal((n_hidden, n_hidden)) * (init_scale / np.sqrt(n_hidden))
        self.b_h = np.zeros(n_hidden)
        self.W_hp = rng.standard_normal((1, n_hidden)) * (init_scale / np.sqrt(n_hidden))
        self.b_p = np.zeros(1)

    def params(self):
        return [self.W_xh, self.W_hh, self.b_h, self.W_hp, self.b_p]

    def param_names(self):
        return ["W_xh", "W_hh", "b_h", "W_hp", "b_p"]


# ----------------------------------------------------------------------
# Episode generation
# ----------------------------------------------------------------------

def make_episode(T: int, rng: np.random.Generator,
                 p_A: float = 0.10, p_B: float = 0.15, p_X: float = 0.25):
    """Generate one flip-flop episode of length T.

    Each step exactly one of {A, B, X, nothing} fires (or none).

    Latch semantics (Schmidhuber 1990):
      o  state starts at 0
      o  A resets the latch to 0
      o  B sets the latch to 1 (only counts when state == 0; redundant Bs
         are effectively no-ops)
      o  X is an irrelevant distractor

    Returns:
      events  : (T, 3) float, columns [A, B, X]
      desired : (T,)  float, the target output for that step
    """
    events = np.zeros((T, 3), dtype=np.float64)
    desired = np.zeros(T, dtype=np.float64)
    state = 0
    for t in range(T):
        u = rng.random()
        if u < p_A:
            events[t, 0] = 1.0
            state = 0
        elif u < p_A + p_B:
            events[t, 1] = 1.0
            state = 1
        elif u < p_A + p_B + p_X:
            events[t, 2] = 1.0
        desired[t] = float(state)
    return events, desired


# ----------------------------------------------------------------------
# Forward passes
# ----------------------------------------------------------------------

def rollout_random(events: np.ndarray, desired: np.ndarray,
                   rng: np.random.Generator):
    """Generate a uniform-random-action rollout.

    Used to keep M's training data covering the full action range.  Without
    this, once C's policy concentrates on a handful of actions M only ever
    sees a slice of the (action, pain) joint and the gradient through M for
    novel actions becomes unreliable.  See README §Deviations.
    """
    T = events.shape[0]
    obs = np.zeros((T, 5))
    obs[:, :3] = events
    obs[:, 3] = 1.0  # bias

    y_seq = np.zeros(T)
    pain_seq = np.zeros(T)
    pain_prev = 0.0
    for t in range(T):
        obs[t, 4] = pain_prev
        y = float(rng.random())
        p = (y - desired[t]) ** 2
        y_seq[t] = y
        pain_seq[t] = p
        pain_prev = p
    return {"obs": obs, "y": y_seq, "pain": pain_seq}


def rollout_controller(C: Controller, events: np.ndarray, desired: np.ndarray):
    """Run C through the episode.  Computes pain with the env formula
    pain_t = (y_t - desired_t)^2.  Pain is fed back into the next obs.

    All activation tensors are stored so BPTT can be done later.
    """
    T = events.shape[0]
    obs = np.zeros((T, 5))
    obs[:, :3] = events
    obs[:, 3] = 1.0  # bias

    x_seq = np.zeros((T, C.n_in + 1))
    h_seq = np.zeros((T + 1, C.n_hidden))
    z_seq = np.zeros((T, C.n_hidden))
    o_seq = np.zeros(T)
    y_seq = np.zeros(T)
    pain_seq = np.zeros(T)

    h = h_seq[0]
    y_prev = 0.5
    pain_prev = 0.0

    for t in range(T):
        obs[t, 4] = pain_prev
        x = np.concatenate([obs[t], [y_prev]])
        z = C.W_xh @ x + C.W_hh @ h + C.b_h
        h_new = np.tanh(z)
        o = float((C.W_ho @ h_new + C.b_o)[0])
        y = float(sigmoid(np.array(o)))
        p = (y - desired[t]) ** 2

        x_seq[t] = x
        z_seq[t] = z
        h_seq[t + 1] = h_new
        o_seq[t] = o
        y_seq[t] = y
        pain_seq[t] = p

        h = h_new
        y_prev = y
        pain_prev = p

    return {
        "obs": obs,
        "x_C": x_seq,
        "z_C": z_seq,
        "h_C": h_seq,
        "o_C": o_seq,
        "y": y_seq,
        "pain": pain_seq,
    }


def forward_world_model(M: WorldModel, obs: np.ndarray, y: np.ndarray):
    """Run M through the episode given fixed obs and action sequence."""
    T = obs.shape[0]
    x_seq = np.zeros((T, M.n_in + 1))
    h_seq = np.zeros((T + 1, M.n_hidden))
    z_seq = np.zeros((T, M.n_hidden))
    pre_seq = np.zeros(T)
    pred_seq = np.zeros(T)

    h = h_seq[0]
    for t in range(T):
        x = np.concatenate([obs[t], [y[t]]])
        z = M.W_xh @ x + M.W_hh @ h + M.b_h
        h_new = np.tanh(z)
        pre = float((M.W_hp @ h_new + M.b_p)[0])
        pred = float(sigmoid(np.array(pre)))

        x_seq[t] = x
        z_seq[t] = z
        h_seq[t + 1] = h_new
        pre_seq[t] = pre
        pred_seq[t] = pred

        h = h_new

    return {
        "x_M": x_seq,
        "z_M": z_seq,
        "h_M": h_seq,
        "pre_M": pre_seq,
        "pred_pain": pred_seq,
    }


# ----------------------------------------------------------------------
# BPTT --- world-model update
# ----------------------------------------------------------------------

def backward_world_model(M: WorldModel, traj_M: dict, true_pain: np.ndarray):
    """Backprop M's MSE loss vs. observed pain.  Returns gradient dict."""
    T = len(true_pain)
    pred = traj_M["pred_pain"]
    err = pred - true_pain  # d/dpred of 0.5 * (pred - target)^2

    g_W_xh = np.zeros_like(M.W_xh)
    g_W_hh = np.zeros_like(M.W_hh)
    g_b_h = np.zeros_like(M.b_h)
    g_W_hp = np.zeros_like(M.W_hp)
    g_b_p = np.zeros_like(M.b_p)

    dh_next = np.zeros(M.n_hidden)
    for t in reversed(range(T)):
        # d Loss / d pred[t] = err[t]; d pred / d pre = pred * (1 - pred)
        d_pre = err[t] * pred[t] * (1.0 - pred[t])
        h_new = traj_M["h_M"][t + 1]

        g_W_hp += d_pre * h_new[None, :]
        g_b_p += d_pre

        dh_new = M.W_hp.flatten() * d_pre + dh_next
        dz = dh_new * (1.0 - h_new ** 2)

        g_W_xh += np.outer(dz, traj_M["x_M"][t])
        g_W_hh += np.outer(dz, traj_M["h_M"][t])
        g_b_h += dz

        dh_next = M.W_hh.T @ dz

    return {"W_xh": g_W_xh, "W_hh": g_W_hh, "b_h": g_b_h,
            "W_hp": g_W_hp, "b_p": g_b_p}


# ----------------------------------------------------------------------
# BPTT --- controller update through frozen M
# ----------------------------------------------------------------------

def backward_controller_via_M(C: Controller, M: WorldModel,
                               traj_C: dict, traj_M: dict,
                               truncate_M: bool = True):
    """Backprop sum_t pred_pain[t] through frozen M into C.

    Returns gradient dict for C only.  M is treated as a fixed simulator.
    The pain values fed back into obs are treated as observed constants
    (no gradient through the env -- that would be cheating, exposing the
    desired-output target to C through dpain/dy).

    With ``truncate_M=True`` (the default and the paper's "type A" recipe in
    section 6 of FKI-126-90), the action gradient at step t is computed only
    through the *local* M-jacobian d pred_pain[t] / d y[t], not through M's
    recurrent connections back through history.  Schmidhuber 1990 found that
    truncating M-side BPTT to a single step kept the C update from oscillating
    when M's long-horizon predictions were imperfect; we observe the same.
    With ``truncate_M=False``, full BPTT through M is used, which destabilizes
    training in our hands.
    """
    T = len(traj_C["y"])

    # ---- pass 1: backprop through M to get d/dy[t] for every t ----
    dy_from_M = np.zeros(T)
    dh_M_next = np.zeros(M.n_hidden)
    pred = traj_M["pred_pain"]
    for t in reversed(range(T)):
        # Loss = sum_t pred[t]; d/dpred[t] = 1
        d_pre = pred[t] * (1.0 - pred[t])
        h_new = traj_M["h_M"][t + 1]

        if truncate_M:
            dh_new = M.W_hp.flatten() * d_pre  # no future contribution
        else:
            dh_new = M.W_hp.flatten() * d_pre + dh_M_next
        dz = dh_new * (1.0 - h_new ** 2)

        # x_M[t] = [obs[t]; y[t]] -- the last entry is the action
        dx_M = M.W_xh.T @ dz
        dy_from_M[t] = dx_M[-1]

        if not truncate_M:
            dh_M_next = M.W_hh.T @ dz

    # ---- pass 2: backprop through C using dy_from_M ----
    g_W_xh = np.zeros_like(C.W_xh)
    g_W_hh = np.zeros_like(C.W_hh)
    g_b_h = np.zeros_like(C.b_h)
    g_W_ho = np.zeros_like(C.W_ho)
    g_b_o = np.zeros_like(C.b_o)

    dh_C_next = np.zeros(C.n_hidden)
    dy_carry = 0.0  # gradient flowing into y[t] from being y_prev at step t+1

    for t in reversed(range(T)):
        d_y = dy_from_M[t] + dy_carry
        y_t = traj_C["y"][t]
        # y = sigmoid(o); d/do = y * (1-y) * d_y
        d_o = y_t * (1.0 - y_t) * d_y

        h_new = traj_C["h_C"][t + 1]
        g_W_ho += d_o * h_new[None, :]
        g_b_o += d_o

        dh_new = C.W_ho.flatten() * d_o + dh_C_next
        dz = dh_new * (1.0 - h_new ** 2)

        g_W_xh += np.outer(dz, traj_C["x_C"][t])
        g_W_hh += np.outer(dz, traj_C["h_C"][t])
        g_b_h += dz

        # x_C[t] = [obs[t]; y_prev]; last entry is the previous action
        dx_C = C.W_xh.T @ dz
        dy_carry = dx_C[-1]

        dh_C_next = C.W_hh.T @ dz

    return {"W_xh": g_W_xh, "W_hh": g_W_hh, "b_h": g_b_h,
            "W_ho": g_W_ho, "b_o": g_b_o}


# ----------------------------------------------------------------------
# Optimizer (RMSProp-with-momentum, hand-rolled)
# ----------------------------------------------------------------------

class Adam:
    """Tiny Adam optimizer, one instance per network."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads_list, clip: float = 1.0):
        self.t += 1
        # global-norm gradient clipping
        if clip is not None and clip > 0:
            total = float(sum(float(np.sum(g * g)) for g in grads_list)) ** 0.5
            scale = 1.0 if total <= clip else clip / (total + 1e-12)
        else:
            scale = 1.0
        for p, g, m, v in zip(self.params, grads_list, self.m, self.v):
            g = g * scale
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * (g * g)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(seed: int = 0,
          regime: str = "sequential",
          n_steps: int = 3000,
          T: int = 30,
          n_hidden: int = 16,
          batch: int | None = None,
          lr_M: float = 5e-3,
          lr_C: float = 5e-3,
          M_warmup: int = 500,
          M_inner: int = 1,
          init_scale: float = 0.5,
          snapshot_every: int = 0,
          snapshot_callback=None,
          verbose: bool = True):
    """Train (C, M) by alternating M-updates and C-updates.

    Per outer step:
      1.  Roll out a uniform-random action policy on a fresh episode and
          train M's pain-prediction on that data.  This keeps M well-
          calibrated on the *whole* action distribution; if M only ever
          sees C's (highly concentrated) actions, the dpain/dy gradient
          for off-policy actions becomes unreliable and C oscillates.
      2.  After M_warmup steps: roll out C deterministically on a fresh
          episode, forward (now-updated) M to get predicted pain, and
          backprop sum predicted pain through M into C.

    `regime`:
      o  "sequential" : 1 episode per outer step (paper: 1 long stream).
      o  "parallel"   : `batch` episodes processed independently, gradients
         averaged per outer step (paper: `parallel' -- many episodes at once).
    """
    if batch is None:
        batch = 1 if regime == "sequential" else 16

    seed_seq = np.random.SeedSequence(seed)
    rng_C, rng_M, rng_data = (np.random.default_rng(s)
                              for s in seed_seq.spawn(3))

    C = Controller(n_hidden=n_hidden, init_scale=init_scale, rng=rng_C)
    M = WorldModel(n_hidden=n_hidden, init_scale=init_scale, rng=rng_M)

    opt_M = Adam(M.params(), lr=lr_M)
    opt_C = Adam(C.params(), lr=lr_C)

    history = {"step": [], "pain_mean": [], "M_loss": [], "accuracy": [],
               "regime": regime}

    # snapshot the initial state too
    if snapshot_callback is not None:
        snapshot_callback(-1, C, M, history, rng_data)

    for step in range(n_steps):
        # ---- step 1: M update on random-policy rollouts (always) ----
        grads_M_acc = None
        M_loss_acc = 0.0
        for b in range(batch):
            events, desired = make_episode(T, rng_data)
            roll_R = rollout_random(events, desired, rng_data)
            traj_M = forward_world_model(M, roll_R["obs"], roll_R["y"])
            M_loss_acc += float(0.5 * np.mean((traj_M["pred_pain"] - roll_R["pain"]) ** 2))
            grads_M = backward_world_model(M, traj_M, roll_R["pain"])
            if grads_M_acc is None:
                grads_M_acc = {k: g.copy() for k, g in grads_M.items()}
            else:
                for k in grads_M_acc:
                    grads_M_acc[k] += grads_M[k]
        for k in grads_M_acc:
            grads_M_acc[k] /= batch
        for _ in range(M_inner):
            opt_M.step([grads_M_acc[n] for n in M.param_names()])
        M_loss = M_loss_acc / batch

        # ---- step 2: C update on C's own deterministic rollouts ----
        grads_C_acc = None
        pain_acc = 0.0
        acc_acc = 0.0
        for b in range(batch):
            events, desired = make_episode(T, rng_data)
            traj_C = rollout_controller(C, events, desired)
            pain_acc += float(np.mean(traj_C["pain"]))
            preds_bin = (traj_C["y"] > 0.5).astype(np.float64)
            acc_acc += float(np.mean(preds_bin == desired))

            if step >= M_warmup:
                traj_M_C = forward_world_model(M, traj_C["obs"], traj_C["y"])
                grads_C = backward_controller_via_M(C, M, traj_C, traj_M_C)
                if grads_C_acc is None:
                    grads_C_acc = {k: g.copy() for k, g in grads_C.items()}
                else:
                    for k in grads_C_acc:
                        grads_C_acc[k] += grads_C[k]

        if grads_C_acc is not None:
            for k in grads_C_acc:
                grads_C_acc[k] /= batch
            opt_C.step([grads_C_acc[n] for n in C.param_names()])

        pain_mean = pain_acc / batch
        accuracy = acc_acc / batch

        history["step"].append(step)
        history["pain_mean"].append(pain_mean)
        history["M_loss"].append(M_loss)
        history["accuracy"].append(accuracy)

        if verbose and (step % max(n_steps // 30, 1) == 0 or step == n_steps - 1):
            tag = "M-only" if step < M_warmup else "C+M  "
            print(f"step {step:5d}  [{tag}]  pain={pain_mean:.4f}  "
                  f"M_loss={M_loss:.4f}  acc={accuracy*100:5.1f}%")

        if snapshot_callback is not None and snapshot_every > 0 \
                and (step % snapshot_every == 0 or step == n_steps - 1):
            snapshot_callback(step, C, M, history, rng_data)

    return C, M, history


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(C: Controller, n_episodes: int = 30, T: int = 60,
             seed: int = 12345):
    """Run controller on fresh episodes; report mean accuracy."""
    rng = np.random.default_rng(seed)
    accs = []
    pains = []
    for _ in range(n_episodes):
        events, desired = make_episode(T, rng)
        traj = rollout_controller(C, events, desired)
        preds = (traj["y"] > 0.5).astype(np.float64)
        accs.append(float(np.mean(preds == desired)))
        pains.append(float(np.mean(traj["pain"])))
    return {"mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_pain": float(np.mean(pains)),
            "n_solved_above_0p9": int(sum(a > 0.9 for a in accs)),
            "n_episodes": n_episodes}


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", choices=["sequential", "parallel"],
                   default="sequential")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr-M", type=float, default=1e-2)
    p.add_argument("--lr-C", type=float, default=5e-3)
    p.add_argument("--M-warmup", type=int, default=500)
    p.add_argument("--M-inner", type=int, default=1)
    p.add_argument("--init-scale", type=float, default=0.5)
    p.add_argument("--save", type=str, default="",
                   help="Optional JSON path to dump results to.")
    args = p.parse_args()

    print(f"# flip-flop  (seed={args.seed}  regime={args.regime})")
    for k, v in env_info().items():
        print(f"#   {k}: {v}")

    t0 = time.time()
    C, M, history = train(
        seed=args.seed,
        regime=args.regime,
        n_steps=args.steps,
        T=args.T,
        n_hidden=args.hidden,
        batch=args.batch,
        lr_M=args.lr_M,
        lr_C=args.lr_C,
        M_warmup=args.M_warmup,
        M_inner=args.M_inner,
        init_scale=args.init_scale,
    )
    train_time = time.time() - t0

    final = evaluate(C, n_episodes=30, T=60, seed=12345)
    print(f"\nFinal accuracy on 30 fresh test episodes "
          f"(T=60, seed=12345): {final['mean_acc']*100:.1f}% "
          f"+/- {final['std_acc']*100:.1f}%")
    print(f"   solved (acc > 0.9): {final['n_solved_above_0p9']}/{final['n_episodes']}")
    print(f"   mean residual pain: {final['mean_pain']:.4f}")
    print(f"   train time: {train_time:.1f}s")

    if args.save:
        out = {
            "args": vars(args),
            "env": env_info(),
            "history": history,
            "final": final,
            "train_time_s": train_time,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"   wrote {args.save}")

    return C, M, history, final


if __name__ == "__main__":
    main()
