"""
pole-balance-non-markov - Schmidhuber 1990, "Making the world differentiable"
(FKI-126-90).

Cart-pole balancing where the controller observes only positions (x, theta);
velocities (x_dot, theta_dot) are hidden. A recurrent controller C must
infer velocities from the position history. C is trained by backpropagating
through a recurrent world-model M that predicts next-step positions.

Pipeline:
  Phase 1 - train M on random rollouts. M = TanhRNN(in=3, hid=32, out=2)
            mapping (x, theta, u) and recurrent state -> next (x, theta).
  Phase 2 - freeze M. Train C = TanhRNN(in=2, hid=16, out=1) by unrolling
            C+M for T_unroll steps from a random initial position. Loss is
            sum_t (theta_norm^2 + lambda * x_norm^2). BPTT propagates through
            the C-M graph; only C is updated.
  Eval    - deploy C in the real cart-pole env. Success = balance for
            >= 1000 steps (Schmidhuber's threshold).

Pure numpy. No torch / no gym. CLI: python3 pole_balance_non_markov.py --seed N.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Cart-pole environment - standard equations (Sutton 1984, Florian 2007)
# ----------------------------------------------------------------------

GRAVITY = 9.8
M_CART  = 1.0
M_POLE  = 0.1
L_HALF  = 0.5          # half-length of the pole
M_TOTAL = M_CART + M_POLE
DT      = 0.02
FORCE   = 10.0
THETA_LIMIT = 12.0 * np.pi / 180.0     # 0.2094 rad
X_LIMIT     = 2.4


def cart_pole_step(state: np.ndarray, force: float) -> np.ndarray:
    """Euler step of cart-pole dynamics. state = (x, x_dot, theta, theta_dot)."""
    x, x_dot, theta, theta_dot = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    temp = (force + M_POLE * L_HALF * theta_dot * theta_dot * sin_t) / M_TOTAL
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        L_HALF * (4.0 / 3.0 - M_POLE * cos_t * cos_t / M_TOTAL)
    )
    x_acc = temp - M_POLE * L_HALF * theta_acc * cos_t / M_TOTAL
    return np.array([
        x + DT * x_dot,
        x_dot + DT * x_acc,
        theta + DT * theta_dot,
        theta_dot + DT * theta_acc,
    ])


def init_state(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.05, 0.05, size=4)


def is_failed(state: np.ndarray) -> bool:
    return abs(state[0]) > X_LIMIT or abs(state[2]) > THETA_LIMIT


# ----------------------------------------------------------------------
# Tanh RNN with hand-coded BPTT (used for M alone in phase 1)
# ----------------------------------------------------------------------

class TanhRNN:
    """RNN: h_t = tanh(W_h h_{t-1} + W_x x_t + b);  y_t = V h_t + c."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 rng: np.random.Generator, scale: float = 0.2):
        self.in_dim, self.hid_dim, self.out_dim = in_dim, hid_dim, out_dim
        self.W_h = rng.standard_normal((hid_dim, hid_dim)).astype(np.float64) * scale
        self.W_x = rng.standard_normal((hid_dim, in_dim)).astype(np.float64) * scale
        self.b   = np.zeros(hid_dim, dtype=np.float64)
        self.V   = rng.standard_normal((out_dim, hid_dim)).astype(np.float64) * scale
        self.c   = np.zeros(out_dim, dtype=np.float64)
        # Adam state
        self._m = {k: np.zeros_like(getattr(self, k)) for k in
                   ("W_h", "W_x", "b", "V", "c")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in
                   ("W_h", "W_x", "b", "V", "c")}
        self._t = 0

    # ---------- forward / backward over a sequence -------------------

    def forward(self, x_seq: np.ndarray, h0: Optional[np.ndarray] = None):
        T = x_seq.shape[0]
        h = np.zeros(self.hid_dim) if h0 is None else h0.copy()
        h_seq = np.zeros((T, self.hid_dim))
        y_seq = np.zeros((T, self.out_dim))
        for t in range(T):
            pre = self.W_h @ h + self.W_x @ x_seq[t] + self.b
            h = np.tanh(pre)
            h_seq[t] = h
            y_seq[t] = self.V @ h + self.c
        return h_seq, y_seq

    def backward(self, x_seq: np.ndarray, h_seq: np.ndarray,
                 dy_seq: np.ndarray, h0: Optional[np.ndarray] = None):
        T = x_seq.shape[0]
        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db   = np.zeros_like(self.b)
        dV   = np.zeros_like(self.V)
        dc   = np.zeros_like(self.c)
        dx_seq = np.zeros_like(x_seq)
        dh = np.zeros(self.hid_dim)
        h_init = np.zeros(self.hid_dim) if h0 is None else h0
        for t in range(T - 1, -1, -1):
            dV += np.outer(dy_seq[t], h_seq[t])
            dc += dy_seq[t]
            dh = dh + self.V.T @ dy_seq[t]
            dpre = dh * (1.0 - h_seq[t] ** 2)
            h_prev = h_seq[t - 1] if t > 0 else h_init
            dW_h += np.outer(dpre, h_prev)
            dW_x += np.outer(dpre, x_seq[t])
            db   += dpre
            dx_seq[t] = self.W_x.T @ dpre
            dh = self.W_h.T @ dpre
        return {"W_h": dW_h, "W_x": dW_x, "b": db, "V": dV, "c": dc}, dh, dx_seq

    # ---------- optimizer --------------------------------------------

    def step_adam(self, grads: dict, lr: float = 1e-3, clip: float = 5.0,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        # Global-norm gradient clipping
        norm_sq = sum((g * g).sum() for g in grads.values())
        norm = float(np.sqrt(norm_sq)) + 1e-12
        scale = min(1.0, clip / norm)
        self._t += 1
        for k, g in grads.items():
            g = g * scale
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            mh = self._m[k] / (1 - beta1 ** self._t)
            vh = self._v[k] / (1 - beta2 ** self._t)
            getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + eps)


# ----------------------------------------------------------------------
# Normalization (positions -> O(1) inputs for the RNNs)
# ----------------------------------------------------------------------

# Use limits as the natural scale; positions in [-1, 1] near nominal range.
def normalize_pos(pos: np.ndarray) -> np.ndarray:
    return np.array([pos[0] / X_LIMIT, pos[1] / THETA_LIMIT])


def denormalize_pos(pos_n: np.ndarray) -> np.ndarray:
    return np.array([pos_n[0] * X_LIMIT, pos_n[1] * THETA_LIMIT])


# ----------------------------------------------------------------------
# Phase 1 - train M on random rollouts
# ----------------------------------------------------------------------

def collect_random_rollout(rng: np.random.Generator, T_max: int = 100):
    """Random-action episode in the real env; ends on failure or T_max."""
    state = init_state(rng)
    in_seq, target_seq = [], []
    for _ in range(T_max):
        u = rng.uniform(-1.0, 1.0)              # normalized action
        force = u * FORCE
        next_state = cart_pole_step(state, force)
        pos_n = normalize_pos(np.array([state[0], state[2]]))
        nxt_n = normalize_pos(np.array([next_state[0], next_state[2]]))
        in_seq.append([pos_n[0], pos_n[1], u])
        target_seq.append([nxt_n[0], nxt_n[1]])
        if is_failed(next_state):
            break
        state = next_state
    return np.array(in_seq), np.array(target_seq)


def collect_controller_rollout(C: TanhRNN, rng: np.random.Generator,
                               T_max: int = 200, action_noise: float = 0.1):
    """Episode under controller C in the real env, with action noise for exploration."""
    state = init_state(rng)
    h_C = np.zeros(C.hid_dim)
    in_seq, target_seq = [], []
    for _ in range(T_max):
        pos_n = normalize_pos(np.array([state[0], state[2]]))
        pre_C = C.W_h @ h_C + C.W_x @ pos_n + C.b
        h_C = np.tanh(pre_C)
        u_pre = C.V @ h_C + C.c
        u = float(np.tanh(u_pre[0]))
        # noisy exploration around controller's action
        u_noisy = float(np.clip(u + rng.normal(0.0, action_noise), -1.0, 1.0))
        force = u_noisy * FORCE
        next_state = cart_pole_step(state, force)
        nxt_n = normalize_pos(np.array([next_state[0], next_state[2]]))
        in_seq.append([pos_n[0], pos_n[1], u_noisy])
        target_seq.append([nxt_n[0], nxt_n[1]])
        if is_failed(next_state):
            break
        state = next_state
    return np.array(in_seq), np.array(target_seq)


def train_world_model(M: TanhRNN, rng: np.random.Generator,
                      n_episodes: int = 400, lr: float = 5e-3,
                      T_max: int = 100, log_every: int = 50,
                      controller: Optional[TanhRNN] = None,
                      action_noise: float = 0.1,
                      label: str = "phase1", verbose: bool = True):
    """Train M on rollouts. If `controller` is given, half the episodes use C
    (with noise) so M sees C's state distribution; the other half stays random
    for coverage."""
    losses = []
    for ep in range(n_episodes):
        if controller is not None and ep % 2 == 0:
            in_seq, target_seq = collect_controller_rollout(
                controller, rng, T_max=T_max, action_noise=action_noise)
        else:
            in_seq, target_seq = collect_random_rollout(rng, T_max=T_max)
        T = in_seq.shape[0]
        if T < 2:
            losses.append(losses[-1] if losses else 0.0)
            continue
        h_seq, y_seq = M.forward(in_seq)
        diff = y_seq - target_seq
        loss = float((diff * diff).mean())
        dy = 2.0 * diff / (T * M.out_dim)
        grads, _, _ = M.backward(in_seq, h_seq, dy)
        M.step_adam(grads, lr=lr)
        losses.append(loss)
        if verbose and (ep % log_every == 0 or ep == n_episodes - 1):
            print(f"  [{label}] ep {ep + 1:4d}/{n_episodes}  "
                  f"T={T:3d}  loss={loss:.5f}")
    return losses


def evaluate_world_model(M: TanhRNN, rng: np.random.Generator,
                         n_episodes: int = 30, T_max: int = 100):
    """Mean MSE on held-out rollouts (positions only)."""
    errs = []
    for _ in range(n_episodes):
        in_seq, target_seq = collect_random_rollout(rng, T_max=T_max)
        if in_seq.shape[0] < 2:
            continue
        _, y_seq = M.forward(in_seq)
        errs.append(float(((y_seq - target_seq) ** 2).mean()))
    return float(np.mean(errs)) if errs else float("nan")


# ----------------------------------------------------------------------
# Phase 2 - train controller C by BPTT through frozen M
# ----------------------------------------------------------------------

def unroll_and_loss(C: TanhRNN, M: TanhRNN, init_pos_n: np.ndarray,
                    T_unroll: int, lam_x: float = 0.1):
    """Forward unroll of C+M for T_unroll steps. Returns cached activations."""
    pos = init_pos_n.copy()                     # normalized (x, theta)
    h_C = np.zeros(C.hid_dim)
    h_M = np.zeros(M.hid_dim)

    cache = {
        "in_C":     np.zeros((T_unroll, 2)),
        "h_C":      np.zeros((T_unroll, C.hid_dim)),
        "u_pre":    np.zeros((T_unroll, 1)),
        "u":        np.zeros((T_unroll, 1)),
        "in_M":     np.zeros((T_unroll, 3)),
        "h_M":      np.zeros((T_unroll, M.hid_dim)),
        "pos_pred": np.zeros((T_unroll, 2)),
    }
    total = 0.0
    for t in range(T_unroll):
        in_C = pos.copy()
        pre_C = C.W_h @ h_C + C.W_x @ in_C + C.b
        h_C = np.tanh(pre_C)
        u_pre = C.V @ h_C + C.c
        u = np.tanh(u_pre)                      # action in [-1, 1]
        in_M = np.concatenate([pos, u])
        pre_M = M.W_h @ h_M + M.W_x @ in_M + M.b
        h_M = np.tanh(pre_M)
        pos_next = M.V @ h_M + M.c
        cost = pos_next[1] ** 2 + lam_x * pos_next[0] ** 2
        total += cost
        cache["in_C"][t]     = in_C
        cache["h_C"][t]      = h_C
        cache["u_pre"][t]    = u_pre
        cache["u"][t]        = u
        cache["in_M"][t]     = in_M
        cache["h_M"][t]      = h_M
        cache["pos_pred"][t] = pos_next
        pos = pos_next
    return float(total), cache


def backprop_through_CM(C: TanhRNN, M: TanhRNN, cache: dict,
                        lam_x: float = 0.1):
    """Backward pass through the unrolled C+M graph. Updates only dC."""
    T = cache["pos_pred"].shape[0]
    dC = {k: np.zeros_like(getattr(C, k)) for k in
          ("W_h", "W_x", "b", "V", "c")}

    dh_C_next = np.zeros(C.hid_dim)
    dh_M_next = np.zeros(M.hid_dim)
    dpos_next = np.zeros(2)                 # gradient on pos at start of next step

    for t in range(T - 1, -1, -1):
        pos_pred = cache["pos_pred"][t]
        # dL / d(pos_pred): direct cost + downstream feedback
        dpos_pred = np.array([
            2.0 * lam_x * pos_pred[0],
            2.0 * pos_pred[1],
        ]) + dpos_next

        # pos_pred = M.V @ h_M + M.c
        h_M_t = cache["h_M"][t]
        dh_M = M.V.T @ dpos_pred + dh_M_next
        # h_M = tanh(pre_M)
        dpre_M = dh_M * (1.0 - h_M_t * h_M_t)
        # pre_M = M.W_h h_M_prev + M.W_x in_M + M.b
        in_M = cache["in_M"][t]
        din_M = M.W_x.T @ dpre_M
        dh_M_next = M.W_h.T @ dpre_M

        dpos_in_from_M = din_M[:2]
        du = din_M[2:3]

        # u = tanh(u_pre)
        u_t = cache["u"][t]
        du_pre = du * (1.0 - u_t * u_t)

        # u_pre = C.V @ h_C + C.c
        h_C_t = cache["h_C"][t]
        dC["V"] += np.outer(du_pre, h_C_t)
        dC["c"] += du_pre
        dh_C = C.V.T @ du_pre + dh_C_next
        dpre_C = dh_C * (1.0 - h_C_t * h_C_t)
        h_C_prev = cache["h_C"][t - 1] if t > 0 else np.zeros(C.hid_dim)
        in_C = cache["in_C"][t]
        dC["W_h"] += np.outer(dpre_C, h_C_prev)
        dC["W_x"] += np.outer(dpre_C, in_C)
        dC["b"]   += dpre_C
        din_C = C.W_x.T @ dpre_C
        dh_C_next = C.W_h.T @ dpre_C

        # gradient on pos at start of step t (= pos_pred at step t-1)
        dpos_next = dpos_in_from_M + din_C

    return dC


def eval_controller(C: TanhRNN, rng: np.random.Generator,
                    n_episodes: int = 10, T_max: int = 1000):
    """Real-env evaluation. Returns list of balance times (capped at T_max)."""
    times = []
    for _ in range(n_episodes):
        state = init_state(rng)
        h_C = np.zeros(C.hid_dim)
        steps = 0
        for t in range(T_max):
            pos_n = normalize_pos(np.array([state[0], state[2]]))
            pre_C = C.W_h @ h_C + C.W_x @ pos_n + C.b
            h_C = np.tanh(pre_C)
            u_pre = C.V @ h_C + C.c
            u = float(np.tanh(u_pre[0]))
            state = cart_pole_step(state, u * FORCE)
            steps += 1
            if is_failed(state):
                break
        times.append(steps)
    return times


def train_controller(C: TanhRNN, M: TanhRNN, rng: np.random.Generator,
                     n_iters: int = 600, T_unroll: int = 30,
                     lr: float = 5e-3, lam_x: float = 0.1,
                     init_pos_scale: float = 0.05,
                     batch_size: int = 4,
                     eval_every: int = 25, eval_eps: int = 5,
                     eval_T: int = 1000, verbose: bool = True):
    history = {"iter": [], "cost": [], "eval_iter": [], "balance": [],
               "balance_max": []}
    for it in range(n_iters):
        # batch of random initial positions (velocities unobserved -> implicit 0)
        batch_grads = {k: np.zeros_like(getattr(C, k)) for k in
                       ("W_h", "W_x", "b", "V", "c")}
        batch_cost = 0.0
        for _ in range(batch_size):
            pos0 = rng.uniform(-init_pos_scale, init_pos_scale, size=2)
            pos0_n = normalize_pos(pos0)
            cost, cache = unroll_and_loss(C, M, pos0_n, T_unroll, lam_x=lam_x)
            dC = backprop_through_CM(C, M, cache, lam_x=lam_x)
            for k in dC:
                batch_grads[k] += dC[k]
            batch_cost += cost
        for k in batch_grads:
            batch_grads[k] /= (batch_size * T_unroll)
        C.step_adam(batch_grads, lr=lr)
        history["iter"].append(it + 1)
        history["cost"].append(batch_cost / (batch_size * T_unroll))

        if (it + 1) % eval_every == 0 or it == n_iters - 1:
            times = eval_controller(C, rng, n_episodes=eval_eps, T_max=eval_T)
            mean_t = float(np.mean(times))
            max_t = int(np.max(times))
            history["eval_iter"].append(it + 1)
            history["balance"].append(mean_t)
            history["balance_max"].append(max_t)
            if verbose:
                print(f"  [phase2] iter {it + 1:4d}/{n_iters}  "
                      f"cost/step={cost / T_unroll:.4f}  "
                      f"eval mean={mean_t:6.1f}  max={max_t:4d} "
                      f"({eval_eps} eps cap {eval_T})")
    return history


# ----------------------------------------------------------------------
# End-to-end run
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    # iterative cycles after the initial phase 1 + phase 2 pair.
    # Each cycle re-trains M on the current C's rollouts (with noise) and then
    # re-trains C on the refreshed M.
    n_cycles: int = 3
    # phase 1 (initial M training on random rollouts)
    M_hidden: int = 32
    M_episodes: int = 600
    M_lr: float = 5e-3
    M_T_max: int = 150
    # phase 1.5 (per-cycle M refresh)
    M_refresh_episodes: int = 200
    M_refresh_lr: float = 2e-3
    # phase 2 (controller training)
    C_hidden: int = 16
    C_iters: int = 400
    C_T_unroll: int = 50
    C_lr: float = 5e-3
    C_lam_x: float = 0.1
    C_init_scale: float = 0.05
    C_batch_size: int = 4
    action_noise: float = 0.1
    eval_every: int = 50
    eval_eps: int = 5
    eval_T: int = 1000
    final_eval_eps: int = 30


def run(cfg: RunConfig, verbose: bool = True):
    rng = np.random.default_rng(cfg.seed)
    # seed sequence to give phase 1 / phase 2 / eval independent streams
    seeds = np.random.SeedSequence(cfg.seed).spawn(4)
    rng_init   = np.random.default_rng(seeds[0])
    rng_phase1 = np.random.default_rng(seeds[1])
    rng_phase2 = np.random.default_rng(seeds[2])
    rng_eval   = np.random.default_rng(seeds[3])

    M = TanhRNN(in_dim=3, hid_dim=cfg.M_hidden, out_dim=2,
                rng=rng_init, scale=0.3)

    if verbose:
        print(f"=== Phase 1: train world-model M (hidden={cfg.M_hidden}, "
              f"episodes={cfg.M_episodes}) ===")
    t0 = time.time()
    p1_losses = train_world_model(M, rng_phase1,
                                  n_episodes=cfg.M_episodes,
                                  lr=cfg.M_lr, T_max=cfg.M_T_max,
                                  verbose=verbose)
    held_out_mse = evaluate_world_model(M, rng_eval, n_episodes=30,
                                        T_max=cfg.M_T_max)
    t_phase1 = time.time() - t0
    if verbose:
        print(f"  Phase 1 done in {t_phase1:.1f}s.  held-out M MSE = "
              f"{held_out_mse:.5f}")

    C = TanhRNN(in_dim=2, hid_dim=cfg.C_hidden, out_dim=1,
                rng=rng_init, scale=0.3)

    p2_hist_all = []
    cycle_eval = []   # (cycle, mean_balance, max_balance, n_solved)
    t_phase2_total = 0.0
    refresh_losses_all = []
    for cycle in range(cfg.n_cycles):
        if verbose:
            print(f"\n=== Cycle {cycle + 1}/{cfg.n_cycles}: train controller C "
                  f"(iters={cfg.C_iters}, T_unroll={cfg.C_T_unroll}) ===")
        t0 = time.time()
        # offset history's iter axis so plots concatenate cleanly
        offset = sum(len(h["iter"]) for h in p2_hist_all)
        p2_hist = train_controller(C, M, rng_phase2,
                                   n_iters=cfg.C_iters,
                                   T_unroll=cfg.C_T_unroll,
                                   lr=cfg.C_lr, lam_x=cfg.C_lam_x,
                                   init_pos_scale=cfg.C_init_scale,
                                   batch_size=cfg.C_batch_size,
                                   eval_every=cfg.eval_every,
                                   eval_eps=cfg.eval_eps,
                                   eval_T=cfg.eval_T,
                                   verbose=verbose)
        # shift iter axes for concatenation
        p2_hist["iter"] = [i + offset for i in p2_hist["iter"]]
        p2_hist["eval_iter"] = [i + offset for i in p2_hist["eval_iter"]]
        p2_hist_all.append(p2_hist)
        t_phase2_total += time.time() - t0

        # mid-cycle eval and (if not the last cycle) refresh M
        times_now = eval_controller(C, rng_eval, n_episodes=10,
                                    T_max=cfg.eval_T)
        cycle_eval.append({
            "cycle": cycle + 1,
            "mean": float(np.mean(times_now)),
            "max": int(np.max(times_now)),
            "solved": int(sum(t >= cfg.eval_T for t in times_now)),
        })
        if verbose:
            print(f"  Cycle {cycle + 1} eval: mean={cycle_eval[-1]['mean']:.1f} "
                  f"max={cycle_eval[-1]['max']} "
                  f"solved={cycle_eval[-1]['solved']}/10")

        if cycle < cfg.n_cycles - 1 and cfg.M_refresh_episodes > 0:
            if verbose:
                print(f"  Refreshing M on C's rollouts "
                      f"({cfg.M_refresh_episodes} eps, noise={cfg.action_noise}) ...")
            t0 = time.time()
            r_losses = train_world_model(
                M, rng_phase1, n_episodes=cfg.M_refresh_episodes,
                lr=cfg.M_refresh_lr, T_max=cfg.M_T_max,
                controller=C, action_noise=cfg.action_noise,
                label=f"refresh{cycle + 1}", verbose=verbose,
                log_every=max(cfg.M_refresh_episodes // 5, 1))
            refresh_losses_all.append(r_losses)
            held_out_mse = evaluate_world_model(M, rng_eval, n_episodes=30,
                                                T_max=cfg.M_T_max)
            t_phase1 += time.time() - t0
            if verbose:
                print(f"  M refresh done. Held-out MSE = {held_out_mse:.5f}")

    p2_hist = {
        "iter":      [i for h in p2_hist_all for i in h["iter"]],
        "cost":      [c for h in p2_hist_all for c in h["cost"]],
        "eval_iter": [i for h in p2_hist_all for i in h["eval_iter"]],
        "balance":   [b for h in p2_hist_all for b in h["balance"]],
        "balance_max": [b for h in p2_hist_all for b in h["balance_max"]],
    }
    t_phase2 = t_phase2_total
    if verbose:
        print(f"\nTotal phase-2 time: {t_phase2:.1f}s.")

    if verbose:
        print(f"\n=== Final evaluation ({cfg.final_eval_eps} eps, "
              f"cap {cfg.eval_T}) ===")
    final_times = eval_controller(C, rng_eval,
                                  n_episodes=cfg.final_eval_eps,
                                  T_max=cfg.eval_T)
    n_solved = sum(1 for t in final_times if t >= cfg.eval_T)
    if verbose:
        print(f"  balance time mean={np.mean(final_times):.1f}  "
              f"median={int(np.median(final_times))}  "
              f"max={max(final_times)}  "
              f">={cfg.eval_T}: {n_solved}/{cfg.final_eval_eps}")

    return {
        "M": M, "C": C,
        "phase1_losses": p1_losses,
        "phase1_held_out_mse": held_out_mse,
        "phase2_history": p2_hist,
        "cycle_eval": cycle_eval,
        "refresh_losses": refresh_losses_all,
        "final_times": final_times,
        "final_solved": n_solved,
        "t_phase1": t_phase1,
        "t_phase2": t_phase2,
    }


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    defaults = RunConfig()
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--M-hidden", type=int, default=defaults.M_hidden)
    p.add_argument("--M-episodes", type=int, default=defaults.M_episodes)
    p.add_argument("--C-hidden", type=int, default=defaults.C_hidden)
    p.add_argument("--C-iters", type=int, default=defaults.C_iters)
    p.add_argument("--cycles", type=int, default=defaults.n_cycles)
    p.add_argument("--T-unroll", type=int, default=defaults.C_T_unroll)
    p.add_argument("--batch", type=int, default=defaults.C_batch_size)
    p.add_argument("--lam-x", type=float, default=defaults.C_lam_x)
    p.add_argument("--eval-T", type=int, default=defaults.eval_T)
    p.add_argument("--final-eps", type=int, default=defaults.final_eval_eps)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--save-json", type=str, default=None,
                   help="Path to dump final summary as JSON.")
    args = p.parse_args()

    cfg = RunConfig(
        seed=args.seed,
        n_cycles=args.cycles,
        M_hidden=args.M_hidden,
        M_episodes=args.M_episodes,
        C_hidden=args.C_hidden,
        C_iters=args.C_iters,
        C_T_unroll=args.T_unroll,
        C_batch_size=args.batch,
        C_lam_x=args.lam_x,
        eval_T=args.eval_T,
        final_eval_eps=args.final_eps,
    )
    t0 = time.time()
    res = run(cfg, verbose=not args.quiet)
    t_total = time.time() - t0

    summary = {
        "seed": cfg.seed,
        "config": cfg.__dict__,
        "env": env_info(),
        "phase1_held_out_mse": res["phase1_held_out_mse"],
        "phase1_seconds": res["t_phase1"],
        "phase2_seconds": res["t_phase2"],
        "wallclock_seconds": t_total,
        "final_balance_times": res["final_times"],
        "final_balance_mean": float(np.mean(res["final_times"])),
        "final_balance_median": int(np.median(res["final_times"])),
        "final_balance_max": int(np.max(res["final_times"])),
        "n_solved_1000": int(res["final_solved"]),
        "n_eval_episodes": cfg.final_eval_eps,
    }

    print(f"\nWallclock: {t_total:.1f}s "
          f"(phase1 {res['t_phase1']:.1f}s + phase2 {res['t_phase2']:.1f}s)")
    print(f"Held-out M MSE (positions, normalized): "
          f"{res['phase1_held_out_mse']:.5f}")
    print(f"Final balance: mean={summary['final_balance_mean']:.1f} "
          f"median={summary['final_balance_median']} "
          f"max={summary['final_balance_max']} "
          f">= {cfg.eval_T}: {summary['n_solved_1000']}/{cfg.final_eval_eps}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Wrote summary -> {args.save_json}")

    return summary


if __name__ == "__main__":
    main()
