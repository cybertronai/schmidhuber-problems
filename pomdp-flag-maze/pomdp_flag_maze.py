"""
pomdp-flag-maze - Schmidhuber 1991, "Reinforcement learning in Markovian and
non-Markovian environments", NIPS-3 (1991), pp. 500-506.

A 2-D T-maze flag task. The agent observes only its local cell context (4 wall
booleans) and a 1-bit indicator that is non-zero ONLY at the start cell. The
flag is in one of two terminal cells (top-end or bottom-end of a T-junction);
which one is signalled by the indicator at t=0. After leaving the start cell,
the indicator is no longer visible -- a memoryless agent cannot disambiguate
the two flag positions at the T-junction.

The Schmidhuber 1991 architecture: a recurrent world-model M and a recurrent
controller C, jointly trained by back-propagating predicted-cost gradients
through the unrolled C+M graph. M predicts (next_obs, next_reward) from
(obs, action_probs, M-hidden). C maps (obs, C-hidden) to action probabilities.
After phase 1 trains M on random rollouts, phase 2 freezes M and updates C
to maximize sum of M-imagined rewards over a T-step unroll. The recurrent
state of C learns to latch the indicator across the corridor.

Pure numpy. No torch / no gym. CLI: python3 pomdp_flag_maze.py --seed N.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# T-maze environment (POMDP flag task)
# ----------------------------------------------------------------------
#
# Layout (W = wall, . = corridor, S = start, F = candidate flag, T = T-junction):
#
#     col:   0 1 2 3 4
#  row 0:    . . . . F     <- top flag    (chosen if indicator = +1)
#  row 1:    W W W W .
#  row 2:    S . . . T     <- corridor row, agent moves here
#  row 3:    W W W W .
#  row 4:    . . . . F     <- bottom flag (chosen if indicator = -1)
#
# Walkable cells (always open):
#   - S = (2, 0)
#   - corridor cells (2, 1..3)
#   - T-junction (2, 4)
#   - vertical column at col 4: (0, 4), (1, 4), (3, 4), (4, 4)
#
# Walls are everything else. The 2-D obs is one of:
#   (N_wall, S_wall, W_wall, E_wall, indicator).
#   The indicator is +/- 1 at the start cell only at t=0; 0 everywhere else
#   and at all later time-steps. A memoryless agent therefore cannot tell the
#   two flag positions apart at the T-junction.

ACTIONS = np.array([(-1, 0), (0, +1), (+1, 0), (0, -1)], dtype=np.int32)
# 0 = N (row-1), 1 = E (col+1), 2 = S (row+1), 3 = W (col-1)

ROWS, COLS = 5, 5

START_RC = (2, 0)
TJUNC_RC = (2, 4)
TOP_FLAG_RC = (0, 4)
BOT_FLAG_RC = (4, 4)
CORRIDOR_RC = [(2, c) for c in range(5)]    # row 2, all cols
VERTICAL_RC = [(0, 4), (1, 4), (3, 4), (4, 4)]

WALKABLE = set(CORRIDOR_RC + VERTICAL_RC)

T_MAX_DEFAULT = 20      # 4 corridor + 2 vertical + slack
N_E_OPTIMAL = 4         # E moves needed from S to T-junction


def is_walkable(r: int, c: int) -> bool:
    return (r, c) in WALKABLE


def local_obs(r: int, c: int) -> np.ndarray:
    """4 wall booleans (N, S, W, E). 1 = wall in that direction."""
    return np.array([
        0.0 if is_walkable(r - 1, c) else 1.0,    # N
        0.0 if is_walkable(r + 1, c) else 1.0,    # S
        0.0 if is_walkable(r, c - 1) else 1.0,    # W
        0.0 if is_walkable(r, c + 1) else 1.0,    # E
    ], dtype=np.float64)


class TMazeEnv:
    """Stateful T-maze POMDP. Indicator visible only at t=0.

    Observation: 5-D float vector = (wall_N, wall_S, wall_W, wall_E, indicator).
    Action: int 0..3 (N, E, S, W).
    Reward: +1 on reaching the correct flag; 0 elsewhere; -0.05 step penalty
      to encourage shorter paths and give a dense gradient signal.
    Episode ends on reaching either flag, or after T_max steps.
    """
    OBS_DIM = 5
    ACT_DIM = 4
    STEP_PENALTY = 0.05
    FLAG_REWARD = 2.0
    WRONG_FLAG_REWARD = -2.0

    def __init__(self, t_max: int = T_MAX_DEFAULT):
        self.t_max = t_max
        self.r = 0
        self.c = 0
        self.indicator = 0.0
        self.t = 0
        self.done = True

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.r, self.c = START_RC
        self.indicator = 1.0 if rng.random() < 0.5 else -1.0
        self.t = 0
        self.done = False
        obs = local_obs(self.r, self.c)
        return np.concatenate([obs, [self.indicator]])

    def reset_to(self, indicator: float) -> np.ndarray:
        """Deterministic reset for a chosen indicator (used in eval)."""
        self.r, self.c = START_RC
        self.indicator = float(indicator)
        self.t = 0
        self.done = False
        obs = local_obs(self.r, self.c)
        return np.concatenate([obs, [self.indicator]])

    def step(self, action: int):
        assert not self.done, "step() called on terminated episode -- reset first"
        dr, dc = ACTIONS[action]
        nr, nc = self.r + dr, self.c + dc
        if is_walkable(nr, nc):
            self.r, self.c = nr, nc
        # else: bump into wall, stay put

        self.t += 1
        reward = -self.STEP_PENALTY

        # check terminal flag cell
        on_top = (self.r, self.c) == TOP_FLAG_RC
        on_bot = (self.r, self.c) == BOT_FLAG_RC
        flag_correct = (
            (on_top and self.indicator > 0.0) or
            (on_bot and self.indicator < 0.0)
        )
        flag_wrong = (on_top and self.indicator < 0.0) or \
                     (on_bot and self.indicator > 0.0)
        if flag_correct:
            reward += self.FLAG_REWARD
            self.done = True
        elif flag_wrong:
            # ran to the wrong flag -- explicit negative reinforcement
            reward += self.WRONG_FLAG_REWARD
            self.done = True

        if self.t >= self.t_max:
            self.done = True

        # post-start steps see indicator = 0
        obs = local_obs(self.r, self.c)
        return np.concatenate([obs, [0.0]]), reward, self.done


# ----------------------------------------------------------------------
# TanhRNN with hand-coded BPTT (used for both M and C)
# ----------------------------------------------------------------------

class TanhRNN:
    """Vanilla RNN: h_t = tanh(W_h h_{t-1} + W_x x_t + b);  y_t = V h_t + c.

    Hand-coded forward / backward / Adam to keep the dependency surface to
    just numpy.
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 rng: np.random.Generator, scale: float = 0.2,
                 identity_recurrence: float = 0.0):
        """
        identity_recurrence in [0, 1]: blend the recurrence init toward the
        identity. With identity_recurrence=0.9, W_h = 0.9 * I + 0.1 * random.
        Encourages the hidden state to persist over time, which is critical
        for latching small bits of info (e.g., the indicator in this task)
        across many steps with a vanilla tanh RNN (Le et al. 2015,
        "A Simple Way to Initialize Recurrent Networks of Rectified Linear
        Units").
        """
        self.in_dim, self.hid_dim, self.out_dim = in_dim, hid_dim, out_dim
        rand_h = rng.standard_normal((hid_dim, hid_dim)).astype(np.float64) * scale
        eye = np.eye(hid_dim, dtype=np.float64)
        self.W_h = identity_recurrence * eye + (1.0 - identity_recurrence) * rand_h
        self.W_x = rng.standard_normal((hid_dim, in_dim)).astype(np.float64) * scale
        self.b = np.zeros(hid_dim, dtype=np.float64)
        self.V = rng.standard_normal((out_dim, hid_dim)).astype(np.float64) * scale
        self.c = np.zeros(out_dim, dtype=np.float64)
        # Adam state
        self._m = {k: np.zeros_like(getattr(self, k)) for k in
                   ("W_h", "W_x", "b", "V", "c")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in
                   ("W_h", "W_x", "b", "V", "c")}
        self._t = 0

    # ---- forward / backward over a sequence (used to train M) -----------

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
        db = np.zeros_like(self.b)
        dV = np.zeros_like(self.V)
        dc = np.zeros_like(self.c)
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
            db += dpre
            dh = self.W_h.T @ dpre
        return {"W_h": dW_h, "W_x": dW_x, "b": db, "V": dV, "c": dc}

    # ---- single-step helpers (used in C+M unrolls) ----------------------

    def step_(self, x: np.ndarray, h_prev: np.ndarray):
        """One step. Returns (h_new, y, pre)."""
        pre = self.W_h @ h_prev + self.W_x @ x + self.b
        h = np.tanh(pre)
        y = self.V @ h + self.c
        return h, y, pre

    # ---- optimizer ------------------------------------------------------

    def step_adam(self, grads: dict, lr: float = 1e-3, clip: float = 5.0,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
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
# Numerically stable softmax + entropy
# ----------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    ez = np.exp(z)
    return ez / ez.sum()


def softmax_jacobian_apply(probs: np.ndarray, dprobs: np.ndarray) -> np.ndarray:
    """Returns dz where probs = softmax(z). dz_i = probs_i (dprobs_i - sum_j probs_j dprobs_j)."""
    s = float(np.dot(probs, dprobs))
    return probs * (dprobs - s)


# ----------------------------------------------------------------------
# Phase 1 - train M on random rollouts
# ----------------------------------------------------------------------
#
# M takes (obs[5], action_probs[4]) -> (next_obs[5], next_reward[1]).
# Random policy uses uniform action distribution; one-hot at execution.

ACT_ONEHOT = np.eye(4, dtype=np.float64)


def _onehot(a: int, n: int = 4) -> np.ndarray:
    v = np.zeros(n)
    v[a] = 1.0
    return v


def collect_random_rollout(env: TMazeEnv, rng: np.random.Generator,
                           soft: bool = False):
    """Random-action episode. M's action input is the ONE-HOT of the actually
    taken action (so M can correctly attribute next-state and reward to the
    discrete action). M ALSO receives the persistent indicator as a side input
    -- this is conceptually a "task descriptor" the world model gets for free
    (analogous to a goal embedding in goal-conditioned model-based RL). The
    POMDP burden is on C, which only sees the obs vector (indicator zeroed
    after t=0).

    Returns (in_seq, target_seq) for M training:
       in_seq[t]     = obs_t (5) || action_onehot (4) || indicator (1)  (10)
       target_seq[t] = next_obs (5) || reward (1)                       (6)
    """
    obs = env.reset(rng)
    indicator = env.indicator
    in_seq, target_seq = [], []
    while True:
        if soft:
            probs = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
            a = int(rng.choice(4, p=probs))
        else:
            a = int(rng.integers(4))
        a_in = _onehot(a)
        next_obs, reward, done = env.step(a)
        in_seq.append(np.concatenate([obs, a_in, [indicator]]))
        target_seq.append(np.concatenate([next_obs, [reward]]))
        obs = next_obs
        if done:
            break
    return np.array(in_seq), np.array(target_seq)


def collect_scripted_rollout(env: TMazeEnv, rng: np.random.Generator,
                             p_correct: float = 0.5):
    """Scripted exploration rollout: drive E to the T-junction, then choose
    N or S (with probability `p_correct` of choosing the indicator-correct
    direction). Same input layout as `collect_random_rollout` (M receives
    the indicator as a side input).
    """
    obs = env.reset(rng)
    indicator = env.indicator
    in_seq, target_seq = [], []
    while True:
        wall_E = obs[3]
        if wall_E < 0.5:
            probs = np.array([0.05, 0.85, 0.05, 0.05])
        else:
            if rng.random() < p_correct:
                go_N = indicator > 0.0
            else:
                go_N = indicator < 0.0
            if go_N:
                probs = np.array([0.85, 0.05, 0.05, 0.05])
            else:
                probs = np.array([0.05, 0.05, 0.85, 0.05])
        a = int(rng.choice(4, p=probs))
        a_in = _onehot(a)
        next_obs, reward, done = env.step(a)
        in_seq.append(np.concatenate([obs, a_in, [indicator]]))
        target_seq.append(np.concatenate([next_obs, [reward]]))
        obs = next_obs
        if done:
            break
    return np.array(in_seq), np.array(target_seq)


def collect_controller_rollout(C: TanhRNN, env: TMazeEnv,
                               rng: np.random.Generator,
                               action_noise: float = 0.3):
    """Roll out current C in the real env (with action noise for exploration)
    so M can be re-trained on C's actual visitation distribution. This is the
    'system identification' refresh of the iterative model-controller cycle
    (Schmidhuber 1990, Ha & Schmidhuber 2018)."""
    obs = env.reset(rng)
    indicator = env.indicator
    h_C = np.zeros(C.hid_dim)
    in_seq, target_seq = [], []
    while True:
        h_C, a_logit, _ = C.step_(obs, h_C)
        probs = softmax(a_logit)
        # mix with uniform noise
        probs = (1.0 - action_noise) * probs + action_noise * (1.0 / 4.0)
        a = int(rng.choice(4, p=probs))
        a_in = _onehot(a)
        next_obs, reward, done = env.step(a)
        in_seq.append(np.concatenate([obs, a_in, [indicator]]))
        target_seq.append(np.concatenate([next_obs, [reward]]))
        obs = next_obs
        if done:
            break
    return np.array(in_seq), np.array(target_seq)


# Per-output-dim weights in M's training loss. obs dims weight=1.0; reward
# weight=5.0 because the reward channel has rare-spike structure (+/-2 at
# terminal flag steps, ~-0.05 elsewhere) and would be drowned out by the
# 5 obs dims under uniform MSE weighting.
M_LOSS_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.0])


def train_world_model(M: TanhRNN, env: TMazeEnv, rng: np.random.Generator,
                      n_episodes: int = 2000, lr: float = 5e-3,
                      scripted_frac: float = 0.5,
                      controller: Optional[TanhRNN] = None,
                      controller_frac: float = 0.0,
                      action_noise: float = 0.3,
                      log_every: int = 200, label: str = "phase1",
                      verbose: bool = True):
    """Train M on a mix of random / scripted / controller-driven episodes.

    - `scripted_frac` of episodes follow the deterministic E-then-NS policy
      (essential: random walk almost never reaches the flag in 20 steps, so
      M would otherwise never see the +/-2 reward signal).
    - `controller_frac` of episodes use the supplied recurrent `controller`
      (with action noise) so M can be refreshed on C's real visitation
      distribution.
    - Reward channel is loss-weighted by `M_LOSS_WEIGHTS[5]` because it is
      the most informationally important output.
    """
    losses = []
    losses_r = []   # per-step reward MSE only (diagnostic)
    weights = M_LOSS_WEIGHTS
    for ep in range(n_episodes):
        u = rng.random()
        if controller is not None and u < controller_frac:
            in_seq, target_seq = collect_controller_rollout(
                controller, env, rng, action_noise=action_noise)
        elif u < controller_frac + scripted_frac:
            in_seq, target_seq = collect_scripted_rollout(env, rng,
                                                          p_correct=0.5)
        else:
            in_seq, target_seq = collect_random_rollout(env, rng,
                                                        soft=(ep % 2 == 0))
        T = in_seq.shape[0]
        if T < 1:
            losses.append(losses[-1] if losses else 0.0)
            continue
        h_seq, y_seq = M.forward(in_seq)
        diff = y_seq - target_seq
        # weighted MSE: weight per output dim
        wdiff2 = (diff * diff) * weights[None, :]
        loss = float(wdiff2.mean())
        loss_r = float(((diff[:, 5]) ** 2).mean())
        # gradient: d/dy (w * diff^2) = 2 w diff
        dy = 2.0 * diff * weights[None, :] / (T * M.out_dim)
        grads = M.backward(in_seq, h_seq, dy)
        M.step_adam(grads, lr=lr)
        losses.append(loss)
        losses_r.append(loss_r)
        if verbose and (ep % log_every == 0 or ep == n_episodes - 1):
            print(f"  [{label}] ep {ep + 1:5d}/{n_episodes}  T={T:2d}  "
                  f"loss={loss:.5f}  reward_mse={loss_r:.5f}")
    return losses


def evaluate_world_model(M: TanhRNN, env: TMazeEnv,
                         rng: np.random.Generator, n_episodes: int = 50):
    """Mean MSE on held-out random rollouts."""
    errs = []
    for _ in range(n_episodes):
        in_seq, target_seq = collect_random_rollout(env, rng, soft=False)
        if in_seq.shape[0] < 1:
            continue
        _, y_seq = M.forward(in_seq)
        errs.append(float(((y_seq - target_seq) ** 2).mean()))
    return float(np.mean(errs)) if errs else float("nan")


# ----------------------------------------------------------------------
# Phase 2 - train C through frozen M (BPTT on imagined rollouts)
# ----------------------------------------------------------------------
#
# Forward: at each unroll step t,
#     C_in_t        = obs_t (5)              [ initial obs from real reset ]
#     pre_C_t       = W_hC h_C_{t-1} + W_xC C_in_t + b_C
#     h_C_t         = tanh(pre_C_t)
#     a_logit_t     = V_C h_C_t + c_C        (4-dim)
#     a_probs_t     = softmax(a_logit_t)
#     M_in_t        = obs_t || a_probs_t     (9-dim)
#     pre_M_t       = W_hM h_M_{t-1} + W_xM M_in_t + b_M
#     h_M_t         = tanh(pre_M_t)
#     out_M_t       = V_M h_M_t + c_M        (6-dim: next_obs[5] || reward[1])
#     obs_{t+1}     = out_M_t[:5]
#     r_pred_t      = out_M_t[5]
# Reward to maximize:
#     R = sum_t gamma^t * r_pred_t   - lam_ent * sum_t H[a_probs_t]?
# (entropy regularizer is OPTIONAL; we use a small one only for the first
# few iterations to encourage exploration.)
# Loss = -R  (we descend on -R to maximize R).

def unroll_CM(C: TanhRNN, M: TanhRNN, init_obs: np.ndarray, indicator: float,
              T_unroll: int, gamma: float = 0.95, ent_coef: float = 0.0,
              rng: Optional[np.random.Generator] = None,
              straight_through: bool = True):
    """Forward unroll C+M for T_unroll steps. Returns (objective, R, cache).

    M's input includes the persistent indicator (provided externally as a
    "task descriptor"). C's input is just `obs`, with the indicator dim of
    the imagined obs structurally zeroed for t>=1 -- forcing C to latch the
    indicator into its own hidden state.

    `straight_through=True` (default): we sample a one-hot action from a_probs
    and feed THAT to M (forward), but in backprop we pretend M's input was
    a_probs (straight-through estimator, Bengio et al. 2013). This avoids the
    softmax-saturation problem where M sees a near-deterministic input and
    gradient on the off-action probs vanishes -- which empirically prevents
    C from latching the indicator. With one-hot input M sees clear discrete
    actions; gradient still flows to a_probs via the ST trick.
    """
    if rng is None:
        rng = np.random.default_rng()
    obs = init_obs.copy()
    h_C = np.zeros(C.hid_dim)
    h_M = np.zeros(M.hid_dim)

    cache = {
        "in_C": np.zeros((T_unroll, C.in_dim)),
        "h_C":  np.zeros((T_unroll, C.hid_dim)),
        "a_logit": np.zeros((T_unroll, C.out_dim)),
        "a_probs": np.zeros((T_unroll, C.out_dim)),
        "a_sample": np.zeros((T_unroll, C.out_dim)),  # one-hot of sampled a
        "in_M": np.zeros((T_unroll, M.in_dim)),
        "h_M":  np.zeros((T_unroll, M.hid_dim)),
        "out_M": np.zeros((T_unroll, M.out_dim)),
        "obs":  np.zeros((T_unroll, 5)),
    }
    R = 0.0
    ent_total = 0.0
    for t in range(T_unroll):
        cache["obs"][t]  = obs
        cache["in_C"][t] = obs
        h_C, a_logit, _ = C.step_(obs, h_C)
        a_probs = softmax(a_logit)
        cache["h_C"][t]    = h_C
        cache["a_logit"][t] = a_logit
        cache["a_probs"][t] = a_probs

        # Straight-through: forward uses one-hot of a sampled action; backward
        # treats the gradient on this slot as gradient on a_probs.
        if straight_through:
            a_idx = int(rng.choice(C.out_dim, p=a_probs))
            a_sample = np.zeros(C.out_dim)
            a_sample[a_idx] = 1.0
            a_input_M = a_sample
        else:
            a_input_M = a_probs
            a_sample = a_probs.copy()
        cache["a_sample"][t] = a_sample

        in_M = np.concatenate([obs, a_input_M, [indicator]])
        cache["in_M"][t] = in_M
        h_M, out_M, _ = M.step_(in_M, h_M)
        cache["h_M"][t]   = h_M
        cache["out_M"][t] = out_M

        next_obs = out_M[:5].copy()
        next_obs[4] = 0.0     # structural: indicator dim of obs is 0 for t>=1
        r_pred = float(out_M[5])
        R += (gamma ** t) * r_pred
        H = -float(np.sum(a_probs * np.log(a_probs + 1e-12)))
        ent_total += H
        obs = next_obs

    objective = R + ent_coef * ent_total
    return objective, R, cache


def backprop_CM(C: TanhRNN, M: TanhRNN, cache: dict,
                gamma: float = 0.95, ent_coef: float = 0.0):
    """Backward pass through unrolled C+M. Returns dC. M is frozen.

    Loss = - sum_t [ gamma^t r_pred_t + ent_coef * H(a_probs_t) ]
    dL/d(out_M_t[5])   = -gamma^t
    dL/d(a_probs_t[i]) (entropy contribution) = ent_coef * (log p_i + 1) gradient on -H?

    For maximizing entropy, we add +ent_coef*H to the objective; loss is -obj.
    dH/dp_i = -log p_i - 1.
    Combined gradient on a_probs from entropy term in loss: -ent_coef * (-log p - 1) = ent_coef * (log p + 1).
    """
    T = cache["out_M"].shape[0]
    dC = {k: np.zeros_like(getattr(C, k)) for k in
          ("W_h", "W_x", "b", "V", "c")}

    dh_C_next = np.zeros(C.hid_dim)
    dh_M_next = np.zeros(M.hid_dim)
    dobs_from_next = np.zeros(5)        # gradient on obs at start of next step

    for t in range(T - 1, -1, -1):
        # 1. gradient on out_M (M's output: 5 obs + 1 reward)
        dout_M = np.zeros(M.out_dim)
        dout_M[5] = -(gamma ** t)        # d(-R)/d(r_pred_t)
        # downstream: obs_{t+1} = out_M[:5], with [4] structurally zeroed in
        # the forward pass. So no gradient flows into out_M[4].
        dout_M[:5] += dobs_from_next
        dout_M[4] = 0.0                  # indicator dim is constant 0 for t>=1

        # 2. gradient back through M output linear layer
        h_M_t = cache["h_M"][t]
        # out_M = V_M h_M + c_M  -> dh_M = V_M^T dout_M
        dh_M = M.V.T @ dout_M + dh_M_next
        dpre_M = dh_M * (1.0 - h_M_t * h_M_t)
        # in_M = [obs_t, a_probs_t]
        din_M = M.W_x.T @ dpre_M
        dh_M_next = M.W_h.T @ dpre_M
        dobs_from_M = din_M[:5]
        da_probs_from_M = din_M[5:5 + C.out_dim]
        # din_M[5+C.out_dim:] is gradient on the indicator side-channel,
        # which is held constant -- discarded.

        # 3. add entropy contribution to da_probs
        a_probs = cache["a_probs"][t]
        if ent_coef != 0.0:
            # dL/da_probs_i (from -ent_coef * H term) = ent_coef * (log p_i + 1)
            da_probs_from_ent = ent_coef * (np.log(a_probs + 1e-12) + 1.0)
        else:
            da_probs_from_ent = 0.0
        da_probs = da_probs_from_M + da_probs_from_ent

        # 4. softmax backprop: a_probs = softmax(a_logit)
        da_logit = softmax_jacobian_apply(a_probs, da_probs)

        # 5. through C output linear
        h_C_t = cache["h_C"][t]
        dC["V"] += np.outer(da_logit, h_C_t)
        dC["c"] += da_logit
        dh_C = C.V.T @ da_logit + dh_C_next
        dpre_C = dh_C * (1.0 - h_C_t * h_C_t)
        h_C_prev = cache["h_C"][t - 1] if t > 0 else np.zeros(C.hid_dim)
        in_C = cache["in_C"][t]
        dC["W_h"] += np.outer(dpre_C, h_C_prev)
        dC["W_x"] += np.outer(dpre_C, in_C)
        dC["b"]   += dpre_C
        din_C = C.W_x.T @ dpre_C
        dh_C_next = C.W_h.T @ dpre_C

        # 6. accumulate gradient on obs at start of step t (= obs_{t} input)
        # obs_t feeds into both C (in_C) and M (first 5 of in_M).
        # Plus, obs at start of step t = predicted obs from step t-1 (M out).
        dobs_from_next = dobs_from_M + din_C
        # NOTE: at t=0, dobs_from_next is the grad on the real init obs (discarded).

    return dC


def train_controller(C: TanhRNN, M: TanhRNN, env: TMazeEnv,
                     rng: np.random.Generator,
                     n_iters: int = 1500, T_unroll: int = 14,
                     batch_size: int = 8, lr: float = 5e-3,
                     gamma: float = 0.95,
                     ent_coef_start: float = 0.05, ent_coef_end: float = 0.0,
                     ent_anneal_iters: int = 800,
                     eval_every: int = 100, eval_eps: int = 40,
                     verbose: bool = True):
    history = {"iter": [], "objective": [], "R": [],
               "eval_iter": [], "success": [], "mean_steps": []}
    eval_rng = np.random.default_rng(rng.integers(0, 2**31))

    for it in range(n_iters):
        # entropy schedule (linear anneal to ent_coef_end)
        if ent_anneal_iters <= 0:
            ent_c = ent_coef_end
        else:
            frac = min(1.0, it / float(ent_anneal_iters))
            ent_c = ent_coef_start + (ent_coef_end - ent_coef_start) * frac

        batch_grads = {k: np.zeros_like(getattr(C, k)) for k in
                       ("W_h", "W_x", "b", "V", "c")}
        batch_obj = 0.0
        batch_R = 0.0
        for _ in range(batch_size):
            init_obs = env.reset(rng)            # random indicator
            indicator = env.indicator
            obj, R_, cache = unroll_CM(C, M, init_obs, indicator, T_unroll,
                                       gamma=gamma, ent_coef=ent_c, rng=rng)
            dC = backprop_CM(C, M, cache, gamma=gamma, ent_coef=ent_c)
            for k in dC:
                batch_grads[k] += dC[k]
            batch_obj += obj
            batch_R += R_
        for k in batch_grads:
            batch_grads[k] /= float(batch_size)
        C.step_adam(batch_grads, lr=lr)
        history["iter"].append(it + 1)
        history["objective"].append(batch_obj / batch_size)
        history["R"].append(batch_R / batch_size)

        if (it + 1) % eval_every == 0 or it == n_iters - 1:
            succ, mean_steps = eval_controller(C, env, eval_rng, n_episodes=eval_eps)
            history["eval_iter"].append(it + 1)
            history["success"].append(succ)
            history["mean_steps"].append(mean_steps)
            if verbose:
                print(f"  [phase2] iter {it + 1:4d}/{n_iters}  "
                      f"obj/ep={batch_obj / batch_size:+6.3f}  "
                      f"R/ep={batch_R / batch_size:+6.3f}  "
                      f"ent={ent_c:.3f}  "
                      f"eval succ={succ:.2f}  "
                      f"mean_steps={mean_steps:.1f}")
    return history


# ----------------------------------------------------------------------
# Real-environment evaluation
# ----------------------------------------------------------------------

def eval_controller(C: TanhRNN, env: TMazeEnv, rng: np.random.Generator,
                    n_episodes: int = 100, greedy: bool = True):
    """Deploy C in real env. Return (success_rate, mean_steps_to_flag).

    Steps reported only for successful episodes (failures count as t_max).
    """
    successes = 0
    steps_list = []
    for ep in range(n_episodes):
        # half +1, half -1 to balance evaluation
        indicator = 1.0 if (ep % 2 == 0) else -1.0
        obs = env.reset_to(indicator)
        h_C = np.zeros(C.hid_dim)
        steps = 0
        ep_reward = 0.0
        while True:
            h_C, a_logit, _ = C.step_(obs, h_C)
            a_probs = softmax(a_logit)
            if greedy:
                a = int(np.argmax(a_probs))
            else:
                a = int(rng.choice(4, p=a_probs))
            obs, r, done = env.step(a)
            ep_reward += r
            steps += 1
            if done:
                break
        # success if final reward jump matches FLAG_REWARD signal
        # (env terminates either on flag or on T_max; reward bonus only on correct flag)
        if ep_reward > 0.0:
            successes += 1
        steps_list.append(steps)
    return successes / n_episodes, float(np.mean(steps_list))


# ----------------------------------------------------------------------
# Feed-forward baseline (no recurrence) -- expected to fail (~50%)
# ----------------------------------------------------------------------
#
# Same training loop but C has W_h = 0 (no memory); we simply drop the
# recurrent contribution by setting the W_h matrix to zero each step.
# Cleaner: a different controller class. We re-use TanhRNN with W_h zeroed
# and never updated.

def zero_recurrence(C: TanhRNN):
    C.W_h[...] = 0.0
    if "W_h" in C._m:
        C._m["W_h"][...] = 0.0
        C._v["W_h"][...] = 0.0


def train_feedforward_baseline(env: TMazeEnv, M: TanhRNN, rng: np.random.Generator,
                               n_iters: int = 1500, T_unroll: int = 14,
                               batch_size: int = 8, lr: float = 5e-3,
                               gamma: float = 0.95, ent_coef_start: float = 0.05,
                               ent_coef_end: float = 0.0,
                               ent_anneal_iters: int = 800,
                               C_hidden: int = 20, verbose: bool = True):
    """Same recipe as recurrent C but the W_h is held at zero throughout
    training -- the controller is effectively a single-hidden-layer MLP
    consuming only the current observation."""
    C = TanhRNN(in_dim=env.OBS_DIM, hid_dim=C_hidden, out_dim=env.ACT_DIM,
                rng=rng, scale=0.5)
    zero_recurrence(C)
    history = {"iter": [], "objective": [], "R": [],
               "eval_iter": [], "success": [], "mean_steps": []}
    eval_rng = np.random.default_rng(rng.integers(0, 2**31))
    for it in range(n_iters):
        if ent_anneal_iters <= 0:
            ent_c = ent_coef_end
        else:
            frac = min(1.0, it / float(ent_anneal_iters))
            ent_c = ent_coef_start + (ent_coef_end - ent_coef_start) * frac
        batch_grads = {k: np.zeros_like(getattr(C, k)) for k in
                       ("W_h", "W_x", "b", "V", "c")}
        batch_obj = 0.0
        batch_R = 0.0
        for _ in range(batch_size):
            init_obs = env.reset(rng)
            indicator = env.indicator
            obj, R_, cache = unroll_CM(C, M, init_obs, indicator, T_unroll,
                                       gamma=gamma, ent_coef=ent_c, rng=rng)
            dC = backprop_CM(C, M, cache, gamma=gamma, ent_coef=ent_c)
            for k in dC:
                batch_grads[k] += dC[k]
            batch_obj += obj
            batch_R += R_
        for k in batch_grads:
            batch_grads[k] /= float(batch_size)
        # zero out W_h gradient -- no recurrence allowed
        batch_grads["W_h"][...] = 0.0
        C.step_adam(batch_grads, lr=lr)
        zero_recurrence(C)              # keep W_h identically zero
        history["iter"].append(it + 1)
        history["objective"].append(batch_obj / batch_size)
        history["R"].append(batch_R / batch_size)
        if (it + 1) % 250 == 0 or it == n_iters - 1:
            succ, mean_steps = eval_controller(C, env, eval_rng, n_episodes=40)
            history["eval_iter"].append(it + 1)
            history["success"].append(succ)
            history["mean_steps"].append(mean_steps)
            if verbose:
                print(f"  [baseline FF] iter {it + 1}/{n_iters}  "
                      f"R={batch_R / batch_size:+6.3f}  "
                      f"eval succ={succ:.2f}  steps={mean_steps:.1f}")
    return C, history


def eval_random(env: TMazeEnv, rng: np.random.Generator, n_episodes: int = 200):
    successes = 0
    steps_list = []
    for ep in range(n_episodes):
        indicator = 1.0 if (ep % 2 == 0) else -1.0
        env.reset_to(indicator)
        ep_reward = 0.0
        steps = 0
        while True:
            a = int(rng.integers(4))
            _, r, done = env.step(a)
            ep_reward += r
            steps += 1
            if done:
                break
        if ep_reward > 0.0:
            successes += 1
        steps_list.append(steps)
    return successes / n_episodes, float(np.mean(steps_list))


# ----------------------------------------------------------------------
# End-to-end run
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    # phase 1 (initial M training)
    M_hidden: int = 40
    M_episodes: int = 4000
    M_lr: float = 5e-3
    # iterative phase-1.5 / phase-2 cycles. After each phase-2 block, refresh
    # M on rollouts from the current C (with action noise) so M's predictions
    # stay grounded in C's actual visitation distribution.
    n_cycles: int = 4
    M_refresh_episodes: int = 1500
    M_refresh_lr: float = 2e-3
    M_refresh_controller_frac: float = 0.5
    M_refresh_scripted_frac: float = 0.25
    refresh_action_noise: float = 0.3
    # phase 2 (C training)
    C_hidden: int = 24
    C_iters: int = 800          # per cycle
    C_T_unroll: int = 10
    C_lr: float = 2e-3
    C_batch_size: int = 12
    gamma: float = 0.95
    ent_coef_start: float = 0.20
    ent_coef_end: float = 0.05
    ent_anneal_iters: int = 1500
    # eval
    eval_every: int = 100
    eval_eps: int = 40
    final_eval_eps: int = 200
    # env
    t_max: int = 20
    # baselines
    run_baselines: bool = True
    baseline_iters: int = 1500


def run(cfg: RunConfig, verbose: bool = True):
    seeds = np.random.SeedSequence(cfg.seed).spawn(5)
    rng_init = np.random.default_rng(seeds[0])
    rng_phase1 = np.random.default_rng(seeds[1])
    rng_phase2 = np.random.default_rng(seeds[2])
    rng_eval = np.random.default_rng(seeds[3])
    rng_baseline = np.random.default_rng(seeds[4])

    env = TMazeEnv(t_max=cfg.t_max)

    # ---- phase 1: initial M training (random + scripted) ------------------
    if verbose:
        print(f"=== Phase 1: initial M training (hidden={cfg.M_hidden}, "
              f"episodes={cfg.M_episodes}) ===")
    M = TanhRNN(in_dim=10, hid_dim=cfg.M_hidden, out_dim=6,
                rng=rng_init, scale=0.3, identity_recurrence=0.9)
    t0 = time.time()
    p1_losses = train_world_model(M, env, rng_phase1,
                                  n_episodes=cfg.M_episodes,
                                  lr=cfg.M_lr, verbose=verbose)
    held_out_mse = evaluate_world_model(M, env, rng_eval, n_episodes=100)
    t_phase1 = time.time() - t0
    if verbose:
        print(f"  Phase 1 done in {t_phase1:.2f}s.  held-out M MSE = "
              f"{held_out_mse:.5f}")

    # ---- iterative cycles: train C through M, then refresh M -------------
    C = TanhRNN(in_dim=env.OBS_DIM, hid_dim=cfg.C_hidden, out_dim=env.ACT_DIM,
                rng=rng_init, scale=0.5, identity_recurrence=0.9)
    # Track best-checkpoint C across cycles -- iterative refresh occasionally
    # destabilizes a cycle's policy (a refreshed M can land C in a new local
    # optimum that is worse than the previous cycle). We keep the parameters
    # of the highest-eval C and return that one at the end.
    def snapshot_C(c: TanhRNN) -> dict:
        return {k: getattr(c, k).copy() for k in ("W_h", "W_x", "b", "V", "c")}

    def restore_C(c: TanhRNN, snap: dict) -> None:
        for k, v in snap.items():
            getattr(c, k)[...] = v

    best_succ = -1.0
    best_C_snap = snapshot_C(C)
    p2_hist_all = []
    cycle_eval = []
    refresh_losses_all = []
    t_phase2 = 0.0
    for cycle in range(cfg.n_cycles):
        if verbose:
            print(f"\n=== Cycle {cycle + 1}/{cfg.n_cycles}: train C "
                  f"(iters={cfg.C_iters}, T_unroll={cfg.C_T_unroll}) ===")
        t0 = time.time()
        offset = sum(len(h["iter"]) for h in p2_hist_all)
        p2_hist = train_controller(C, M, env, rng_phase2,
                                   n_iters=cfg.C_iters,
                                   T_unroll=cfg.C_T_unroll,
                                   batch_size=cfg.C_batch_size,
                                   lr=cfg.C_lr, gamma=cfg.gamma,
                                   ent_coef_start=cfg.ent_coef_start,
                                   ent_coef_end=cfg.ent_coef_end,
                                   ent_anneal_iters=cfg.ent_anneal_iters,
                                   eval_every=cfg.eval_every,
                                   eval_eps=cfg.eval_eps,
                                   verbose=verbose)
        p2_hist["iter"] = [i + offset for i in p2_hist["iter"]]
        p2_hist["eval_iter"] = [i + offset for i in p2_hist["eval_iter"]]
        p2_hist_all.append(p2_hist)
        t_phase2 += time.time() - t0

        succ_now, steps_now = eval_controller(C, env, rng_eval,
                                              n_episodes=cfg.final_eval_eps)
        cycle_eval.append({
            "cycle": cycle + 1,
            "success": float(succ_now),
            "mean_steps": float(steps_now),
        })
        if verbose:
            print(f"  Cycle {cycle + 1} eval: success={succ_now:.3f}  "
                  f"mean_steps={steps_now:.2f}")
        # snapshot if best
        if succ_now > best_succ:
            best_succ = succ_now
            best_C_snap = snapshot_C(C)
            if verbose:
                print(f"    (new best, snapshotting C; succ={best_succ:.3f})")

        if cycle < cfg.n_cycles - 1 and cfg.M_refresh_episodes > 0:
            if verbose:
                print(f"  Refresh M on C-driven rollouts "
                      f"({cfg.M_refresh_episodes} eps, noise={cfg.refresh_action_noise}) ...")
            t_r0 = time.time()
            r_losses = train_world_model(
                M, env, rng_phase1,
                n_episodes=cfg.M_refresh_episodes,
                lr=cfg.M_refresh_lr,
                scripted_frac=cfg.M_refresh_scripted_frac,
                controller=C,
                controller_frac=cfg.M_refresh_controller_frac,
                action_noise=cfg.refresh_action_noise,
                label=f"refresh{cycle + 1}", verbose=verbose,
                log_every=max(cfg.M_refresh_episodes // 5, 1))
            refresh_losses_all.append(r_losses)
            t_phase1 += time.time() - t_r0
            held_out_mse = evaluate_world_model(M, env, rng_eval,
                                                n_episodes=100)
            if verbose:
                print(f"  Refresh done. Held-out MSE = {held_out_mse:.5f}")

    # concatenate phase-2 histories across cycles
    p2_hist = {
        "iter":      [i for h in p2_hist_all for i in h["iter"]],
        "objective": [c for h in p2_hist_all for c in h["objective"]],
        "R":         [r for h in p2_hist_all for r in h["R"]],
        "eval_iter": [i for h in p2_hist_all for i in h["eval_iter"]],
        "success":   [s for h in p2_hist_all for s in h["success"]],
        "mean_steps": [s for h in p2_hist_all for s in h["mean_steps"]],
    }
    if verbose:
        print(f"\n  Total phase-2 wallclock: {t_phase2:.2f}s")

    # Restore the best-eval C snapshot for the final evaluation.
    restore_C(C, best_C_snap)
    if verbose:
        print(f"  Restored best C snapshot (cycle-eval succ={best_succ:.3f}).")

    # ---- final real-env eval ---------------------------------------------
    if verbose:
        print(f"\n=== Final eval (recurrent C, {cfg.final_eval_eps} eps, greedy) ===")
    succ, mean_steps = eval_controller(C, env, rng_eval,
                                       n_episodes=cfg.final_eval_eps)
    if verbose:
        print(f"  Recurrent C: success = {succ:.3f}  ({int(succ * cfg.final_eval_eps)}/"
              f"{cfg.final_eval_eps})  mean_steps = {mean_steps:.2f}")

    # ---- baselines --------------------------------------------------------
    rand_succ, rand_steps = eval_random(env, rng_eval, n_episodes=cfg.final_eval_eps)
    if verbose:
        print(f"  Random walk: success = {rand_succ:.3f}  mean_steps = {rand_steps:.2f}")

    ff_C, ff_hist = None, None
    ff_succ, ff_steps = None, None
    if cfg.run_baselines:
        if verbose:
            print(f"\n=== Baseline: feed-forward C (W_h=0), {cfg.baseline_iters} iters ===")
        t0 = time.time()
        ff_C, ff_hist = train_feedforward_baseline(
            env, M, rng_baseline, n_iters=cfg.baseline_iters,
            T_unroll=cfg.C_T_unroll, batch_size=cfg.C_batch_size,
            lr=cfg.C_lr, gamma=cfg.gamma,
            ent_coef_start=cfg.ent_coef_start, ent_coef_end=cfg.ent_coef_end,
            ent_anneal_iters=cfg.ent_anneal_iters,
            C_hidden=cfg.C_hidden, verbose=verbose,
        )
        t_baseline = time.time() - t0
        ff_succ, ff_steps = eval_controller(ff_C, env, rng_eval,
                                            n_episodes=cfg.final_eval_eps)
        if verbose:
            print(f"  Feed-forward C trained in {t_baseline:.2f}s.  "
                  f"success = {ff_succ:.3f}  mean_steps = {ff_steps:.2f}")

    return {
        "M": M, "C": C,
        "phase1_losses": p1_losses,
        "phase1_held_out_mse": held_out_mse,
        "phase2_history": p2_hist,
        "cycle_eval": cycle_eval,
        "refresh_losses": refresh_losses_all,
        "t_phase1": t_phase1,
        "t_phase2": t_phase2,
        "final_success": succ,
        "final_mean_steps": mean_steps,
        "random_success": rand_succ,
        "random_mean_steps": rand_steps,
        "ff_C": ff_C,
        "ff_history": ff_hist,
        "ff_success": ff_succ,
        "ff_mean_steps": ff_steps,
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
    p.add_argument("--T-unroll", type=int, default=defaults.C_T_unroll)
    p.add_argument("--batch", type=int, default=defaults.C_batch_size)
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--final-eps", type=int, default=defaults.final_eval_eps)
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip the feed-forward baseline run.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--save-json", type=str, default=None,
                   help="Path to dump final summary as JSON.")
    args = p.parse_args()

    cfg = RunConfig(
        seed=args.seed,
        M_hidden=args.M_hidden,
        M_episodes=args.M_episodes,
        C_hidden=args.C_hidden,
        C_iters=args.C_iters,
        C_T_unroll=args.T_unroll,
        C_batch_size=args.batch,
        gamma=args.gamma,
        final_eval_eps=args.final_eps,
        run_baselines=not args.no_baseline,
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
        "final_success": res["final_success"],
        "final_mean_steps": res["final_mean_steps"],
        "random_success": res["random_success"],
        "random_mean_steps": res["random_mean_steps"],
        "ff_success": res["ff_success"],
        "ff_mean_steps": res["ff_mean_steps"],
    }

    print(f"\nWallclock: {t_total:.2f}s "
          f"(phase1 {res['t_phase1']:.2f}s + phase2 {res['t_phase2']:.2f}s)")
    print(f"Held-out M MSE: {res['phase1_held_out_mse']:.5f}")
    print(f"Recurrent C    : success = {res['final_success']:.3f}  "
          f"mean steps = {res['final_mean_steps']:.2f}")
    print(f"Random walk    : success = {res['random_success']:.3f}  "
          f"mean steps = {res['random_mean_steps']:.2f}")
    if res["ff_success"] is not None:
        print(f"Feed-forward C : success = {res['ff_success']:.3f}  "
              f"mean steps = {res['ff_mean_steps']:.2f}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Wrote summary -> {args.save_json}")

    return summary


if __name__ == "__main__":
    main()
