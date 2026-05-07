"""
subgoal-obstacle-avoidance — Schmidhuber 1991, "Learning to generate sub-goals
for action sequences", ICANN-91, pp. 967-972.

The original ICANN-91 PDF is sometimes hard to retrieve; the algorithmic
recipe here is reconstructed from §6.10 of Schmidhuber's 2015 *Deep Learning
in Neural Networks: An Overview* and from the 2020 retrospective *Deep
Learning: Our Miraculous Year 1990-1991* (sub-goal generation paragraph).

Problem
-------
A point agent must travel from a fixed start (1, 1) to a fixed goal (9, 9)
inside a 10x10 continuous arena cluttered with N=3 circular obstacles
(radius 0.8). One obstacle is anchored near the diagonal so the
direct-line baseline is non-trivial.

Architecture
------------
Two networks, the canonical hierarchical RL decomposition:

  C_high (sub-goal generator):
      input  = [start (2), goal (2), obstacles (3*N=9)]   -> 13 dims
      hidden = 64 -> 64 (tanh)
      output = K=2 sub-goals * 2 coords                   -> 4 dims
      coords are clamped to the arena via sigmoid * ARENA_SIZE

  C_low  (low-level policy):
      input  = [rel_target (2), nearest_obstacle_rel (2),
                nearest_obstacle_dist (1)]                -> 5 dims
      hidden = 32 (tanh)
      output = action ∈ [-STEP_MAX, STEP_MAX]^2  via STEP_MAX * tanh

Algorithm (matches Schmidhuber 1991 recipe: gradient flows from total
cost back through a differentiable model into the sub-goal generator;
low-level policy is trained separately to imitate a known controller)

Phase 1 -- low-level policy.
    Train C_low by supervised regression on a known potential-field
    controller (attractive force toward target plus exponential repulsive
    force from each obstacle). This plays the role of "first the agent
    learns to act"; in the original paper it is trained jointly with the
    high-level network, here we decouple for clarity.

Phase 2 -- sub-goal generator (the differentiable-model step).
    A closed-form world-model M predicts the cost of any waypoint sequence
    as a sum over piecewise-linear leg costs:
        cost(a, b) = ||b - a|| + lam * (1/T) * sum_t sum_o exp(-d(t,o)^2 / 2 sigma^2)
    where d(t, o) is the distance from the t-fraction point on the segment
    a -> b to obstacle o. This is differentiable in a and b in closed form.
    C_high is trained to minimize the total cost
        J = sum_legs cost(a_leg, b_leg)
    by backpropagating dJ/d(sub_goal) through M into the network. M itself
    needs no parameters -- the obstacle geometry *is* the world model.

Eval
----
Closed-loop rollout: from start, the agent uses C_low to chase sub-goal 1
until close (or T_leg steps), then sub-goal 2, then the goal. Headline:
success rate (reach goal radius without collision), mean path length, vs
a no-subgoal baseline (C_low aimed directly at the goal).

Run
---
    python3 subgoal_obstacle_avoidance.py --seed 0
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

ARENA_SIZE = 10.0
START = np.array([1.0, 1.0], dtype=np.float32)
GOAL = np.array([9.0, 9.0], dtype=np.float32)
N_OBSTACLES = 3
OBS_RADIUS = 0.8

STEP_MAX = 0.4
T_MAX_STEPS = 80
GOAL_RADIUS = 0.4

# Cost surrogate
T_SAMPLES = 32                  # samples per leg in line-integral cost
SIGMA = OBS_RADIUS + 0.35       # Gaussian width on obstacle penalty (~1.15)
LAMBDA_OBS = 25.0               # weight on obstacle penalty in surrogate
N_SUBGOALS = 2                  # K in the paper

# Potential-field LL policy
# Khatib-style 1/r barrier: force grows as the agent enters the buffer zone
# of half-width REPULSION_SCALE around each obstacle surface.
REPULSION_BETA = 0.18
REPULSION_SCALE = 0.9
SAFETY_MARGIN = 0.05            # min clearance the rollout enforces post-step


# ----------------------------------------------------------------------
# Arena generation
# ----------------------------------------------------------------------

def sample_arena(rng: np.random.Generator,
                 force_blocking: bool = True) -> np.ndarray:
    """Sample N obstacles. If force_blocking, the first one is anchored near
    the start-goal diagonal so the direct line is blocked.
    Returns (N, 3) array of [cx, cy, radius]."""
    obstacles: List[np.ndarray] = []
    if force_blocking:
        for _ in range(50):
            t = float(rng.uniform(0.30, 0.70))
            base = (1.0 - t) * START + t * GOAL
            jitter = rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
            c = base + jitter
            if (np.linalg.norm(c - START) > OBS_RADIUS + 0.7 and
                np.linalg.norm(c - GOAL) > OBS_RADIUS + 0.7):
                obstacles.append(np.array([c[0], c[1], OBS_RADIUS], dtype=np.float32))
                break
    while len(obstacles) < N_OBSTACLES:
        c = rng.uniform(1.5, ARENA_SIZE - 1.5, size=2).astype(np.float32)
        if np.linalg.norm(c - START) < OBS_RADIUS + 0.7:
            continue
        if np.linalg.norm(c - GOAL) < OBS_RADIUS + 0.7:
            continue
        if any(np.linalg.norm(c - o[:2]) < OBS_RADIUS * 2.0 + 0.4 for o in obstacles):
            continue
        obstacles.append(np.array([c[0], c[1], OBS_RADIUS], dtype=np.float32))
    return np.stack(obstacles)


def featurize_state(start: np.ndarray, goal: np.ndarray,
                    obstacles: np.ndarray) -> np.ndarray:
    """Concatenate start (2), goal (2), and obstacle params (N*3)."""
    return np.concatenate([start, goal, obstacles.flatten()]).astype(np.float32)


# ----------------------------------------------------------------------
# Manual-MLP primitives
# ----------------------------------------------------------------------

def init_mlp(sizes: List[int], rng: np.random.Generator,
             scale: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    W, b = [], []
    for i in range(len(sizes) - 1):
        s = scale / np.sqrt(sizes[i])
        W.append((s * rng.standard_normal((sizes[i], sizes[i + 1]))).astype(np.float32))
        b.append(np.zeros(sizes[i + 1], dtype=np.float32))
    return W, b


def mlp_forward(W: List[np.ndarray], b: List[np.ndarray],
                x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Forward through tanh hidden layers, linear output. Returns (out, acts)."""
    acts = [x]
    h = x
    for i in range(len(W) - 1):
        h = np.tanh(h @ W[i] + b[i])
        acts.append(h)
    out = h @ W[-1] + b[-1]
    acts.append(out)
    return out, acts


def mlp_backward(W: List[np.ndarray], acts: List[np.ndarray],
                 dout: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Reverse-mode through the tanh-MLP. dout has shape of acts[-1]."""
    dW: List[np.ndarray] = [None] * len(W)
    db: List[np.ndarray] = [None] * len(W)
    dh = dout
    dW[-1] = acts[-2].T @ dh
    db[-1] = dh.sum(axis=0)
    dh = dh @ W[-1].T
    dx = None
    for i in range(len(W) - 2, -1, -1):
        dh = dh * (1.0 - acts[i + 1] ** 2)
        dW[i] = acts[i].T @ dh
        db[i] = dh.sum(axis=0)
        if i > 0:
            dh = dh @ W[i].T
        else:
            dx = dh @ W[0].T
    return dW, db, dx


def adam_init(W: List[np.ndarray], b: List[np.ndarray]) -> Dict[str, list]:
    return {
        "mW": [np.zeros_like(w) for w in W],
        "vW": [np.zeros_like(w) for w in W],
        "mb": [np.zeros_like(bb) for bb in b],
        "vb": [np.zeros_like(bb) for bb in b],
        "t": 0,
    }


def adam_step(W, b, dW, db, state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
              clip: float = None):
    state["t"] += 1
    t = state["t"]
    if clip is not None:
        # global norm clip
        total = 0.0
        for g in dW + db:
            total += float((g * g).sum())
        norm = np.sqrt(total) + 1e-12
        if norm > clip:
            f = clip / norm
            for i in range(len(dW)):
                dW[i] = dW[i] * f
            for i in range(len(db)):
                db[i] = db[i] * f
    for i in range(len(W)):
        state["mW"][i] = beta1 * state["mW"][i] + (1.0 - beta1) * dW[i]
        state["vW"][i] = beta2 * state["vW"][i] + (1.0 - beta2) * (dW[i] ** 2)
        m_hat = state["mW"][i] / (1.0 - beta1 ** t)
        v_hat = state["vW"][i] / (1.0 - beta2 ** t)
        W[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        state["mb"][i] = beta1 * state["mb"][i] + (1.0 - beta1) * db[i]
        state["vb"][i] = beta2 * state["vb"][i] + (1.0 - beta2) * (db[i] ** 2)
        m_hat = state["mb"][i] / (1.0 - beta1 ** t)
        v_hat = state["vb"][i] / (1.0 - beta2 ** t)
        b[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ----------------------------------------------------------------------
# Differentiable cost surrogate (the world-model M)
# ----------------------------------------------------------------------

def leg_cost_with_grad(a: np.ndarray, b: np.ndarray,
                       obstacles: np.ndarray,
                       t_samples: int = T_SAMPLES,
                       sigma: float = SIGMA,
                       lam: float = LAMBDA_OBS
                       ) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """Cost of going a -> b in a straight line and the gradient w.r.t. a, b.

    cost = ||b - a||_2  +  lam * (1/T) * sum_t sum_o exp(-||p(t) - o||^2 / 2 sigma^2)
    where p(t) = (1-t) a + t b.

    Returns (cost, dcost/da, dcost/db, length, mean_penalty)."""
    diff = b - a
    length = float(np.linalg.norm(diff)) + 1e-8
    da_len = -diff / length
    db_len = diff / length

    ts = np.linspace(0.0, 1.0, t_samples).astype(np.float32)
    obs_xy = obstacles[:, :2]                                   # (N, 2)
    pts = a[None, :] + ts[:, None] * diff[None, :]              # (T, 2)
    rel = pts[:, None, :] - obs_xy[None, :, :]                  # (T, N, 2)
    d2 = (rel ** 2).sum(axis=-1)                                # (T, N)
    pen_per = np.exp(-d2 / (2.0 * sigma ** 2))                  # (T, N)
    mean_penalty = float(pen_per.sum() / t_samples)

    # d pen_per / d p(t) = pen_per * (-(p - o)/sigma^2)
    # d p(t) / d a = (1 - t) * I,   d p(t) / d b = t * I
    grad_pt = pen_per[..., None] * (-rel / (sigma ** 2))        # (T, N, 2)
    da_pen = (grad_pt * (1.0 - ts)[:, None, None]).sum(axis=(0, 1)) / t_samples
    db_pen = (grad_pt * ts[:, None, None]).sum(axis=(0, 1)) / t_samples

    cost = length + lam * mean_penalty
    da = (da_len + lam * da_pen).astype(np.float32)
    db = (db_len + lam * db_pen).astype(np.float32)
    return cost, da, db, length, mean_penalty


def total_cost_with_grad(sgs: np.ndarray, start: np.ndarray, goal: np.ndarray,
                         obstacles: np.ndarray
                         ) -> Tuple[float, np.ndarray, float, float]:
    """Total path cost through start -> sgs[0] -> ... -> sgs[-1] -> goal.

    sgs: (N_SUBGOALS, 2). Returns (total_cost, d_total/d_sgs, total_length,
    total_penalty)."""
    pts = [start] + [sgs[i] for i in range(sgs.shape[0])] + [goal]
    total = 0.0
    total_length = 0.0
    total_penalty = 0.0
    grads = [np.zeros(2, dtype=np.float32) for _ in range(len(pts))]
    for i in range(len(pts) - 1):
        c, da, db, ln, pen = leg_cost_with_grad(pts[i], pts[i + 1], obstacles)
        total += c
        total_length += ln
        total_penalty += pen
        grads[i] = grads[i] + da
        grads[i + 1] = grads[i + 1] + db
    sg_grads = np.stack(grads[1:-1])   # drop fixed start/goal slots
    return total, sg_grads, total_length, total_penalty


# ----------------------------------------------------------------------
# C_high  --  sub-goal generator
# ----------------------------------------------------------------------

@dataclass
class SubgoalGenerator:
    """state -> K sub-goals. tanh -> sigmoid scaled to arena coords."""
    W: list
    b: list

    INPUT_DIM = 4 + 3 * N_OBSTACLES        # 13 for N=3

    @classmethod
    def make(cls, rng: np.random.Generator, hidden: int = 64) -> "SubgoalGenerator":
        sizes = [cls.INPUT_DIM, hidden, hidden, 2 * N_SUBGOALS]
        W, b = init_mlp(sizes, rng)
        # Bias the output to map roughly to the diagonal (1..9, 1..9 -> sigmoid 0..1)
        # so initial sub-goals start somewhere reasonable.
        return cls(W, b)

    def forward(self, state_b: np.ndarray) -> Tuple[np.ndarray, list, np.ndarray]:
        out, cache = mlp_forward(self.W, self.b, state_b)        # (B, 2K)
        s = sigmoid(out)                                         # (B, 2K)
        sg = ARENA_SIZE * s
        return sg.reshape(-1, N_SUBGOALS, 2), cache, out

    def backward(self, cache: list, raw_out: np.ndarray,
                 dsg: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """dsg: (B, K, 2) gradient of loss w.r.t. sg coords."""
        B = dsg.shape[0]
        s = sigmoid(raw_out)
        # sg = ARENA_SIZE * s, so dout = dsg * ARENA_SIZE * s * (1 - s)
        dout = (dsg.reshape(B, 2 * N_SUBGOALS) * ARENA_SIZE
                * s * (1.0 - s))
        dW, db, _ = mlp_backward(self.W, cache, dout)
        return dW, db


# ----------------------------------------------------------------------
# C_low  --  low-level policy (potential-field analytic, plus learned imitator)
# ----------------------------------------------------------------------

def potential_field_action(pos: np.ndarray, target: np.ndarray,
                           obstacles: np.ndarray,
                           beta: float = REPULSION_BETA,
                           scale: float = REPULSION_SCALE) -> np.ndarray:
    """Attractive pull toward target plus a Khatib-style 1/r barrier that
    blows up as the agent's clearance to any obstacle surface goes to zero.
    Capped to STEP_MAX in 2-norm."""
    diff = target - pos
    d = float(np.linalg.norm(diff)) + 1e-8
    a_attract = (STEP_MAX * diff / d) if d > STEP_MAX else diff
    a_repel = np.zeros(2, dtype=np.float32)
    for obs in obstacles:
        rel = pos - obs[:2]
        r = float(np.linalg.norm(rel)) + 1e-8
        clear = r - obs[2]
        if clear < scale:
            mag = beta * (1.0 / max(clear, 0.05) - 1.0 / scale)
            a_repel = a_repel + (mag * rel / r).astype(np.float32)
    a = (a_attract + a_repel).astype(np.float32)
    n = float(np.linalg.norm(a)) + 1e-8
    if n > STEP_MAX:
        a = a * STEP_MAX / n
    return a


def project_out_of_obstacles(pos: np.ndarray,
                             obstacles: np.ndarray,
                             margin: float = SAFETY_MARGIN) -> np.ndarray:
    """If pos has crossed into any obstacle, push it just outside that
    obstacle's surface along the radial direction. Used as a defensive last
    line in the rollout so a single missed step doesn't produce a permanent
    "inside the disk" state. The trained policy is still scored on whether
    raw positions ever entered an obstacle."""
    p = pos.copy()
    for obs in obstacles:
        rel = p - obs[:2]
        r = float(np.linalg.norm(rel)) + 1e-8
        if r < obs[2] + margin:
            p = obs[:2] + rel * (obs[2] + margin) / r
    return p.astype(np.float32)


@dataclass
class LowLevelPolicy:
    """Obstacle-blind point-to-point steering. Input = rel_target (2);
    output = 2-d action via STEP_MAX * tanh. The LL deliberately has no
    obstacle awareness -- it walks straight at whatever target it is given.
    The intelligence lives in the SGG: sub-goals are placed so that each
    leg's straight line is clear of obstacles. This matches the Schmidhuber
    1991 decomposition where the cost-of-getting-from-a-to-b model carries
    the obstacle information and the controller is a simple navigator."""
    W: list
    b: list
    INPUT_DIM = 2

    @classmethod
    def make(cls, rng: np.random.Generator, hidden: int = 16) -> "LowLevelPolicy":
        W, b = init_mlp([cls.INPUT_DIM, hidden, 2], rng)
        return cls(W, b)

    @classmethod
    def features_batch(cls, pos_b: np.ndarray, target_b: np.ndarray,
                       obstacles_b: List[np.ndarray]) -> np.ndarray:
        return (target_b - pos_b).astype(np.float32)

    def __call__(self, pos: np.ndarray, target: np.ndarray,
                 obstacles: np.ndarray) -> np.ndarray:
        x = self.features_batch(pos[None], target[None], [obstacles])
        out, _ = mlp_forward(self.W, self.b, x)
        return STEP_MAX * np.tanh(out[0]).astype(np.float32)


def train_low_level_policy(LL: LowLevelPolicy,
                           n_samples: int,
                           epochs: int,
                           lr: float,
                           rng: np.random.Generator,
                           verbose: bool = True) -> List[float]:
    """Generate (pos, target) samples and regress the LL net onto a
    straight-line "head toward target at unit speed" action. Obstacles are
    not part of the LL input by design (see LowLevelPolicy docstring)."""
    Xs = np.empty((n_samples, LL.INPUT_DIM), dtype=np.float32)
    Ys = np.empty((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        pos = rng.uniform(0.0, ARENA_SIZE, size=2).astype(np.float32)
        target = rng.uniform(0.0, ARENA_SIZE, size=2).astype(np.float32)
        rel = target - pos
        d = float(np.linalg.norm(rel)) + 1e-8
        # straight-line action capped to STEP_MAX
        action = STEP_MAX * (rel / d) if d > STEP_MAX else rel.copy()
        Xs[i] = rel
        Ys[i] = action / STEP_MAX            # in [-1, 1]

    state = adam_init(LL.W, LL.b)
    losses: List[float] = []
    n = n_samples
    for ep in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for s_start in range(0, n, 256):
            b_idx = idx[s_start:s_start + 256]
            xb, yb = Xs[b_idx], Ys[b_idx]
            out, cache = mlp_forward(LL.W, LL.b, xb)
            pred = np.tanh(out)
            err = pred - yb
            loss = float(0.5 * (err ** 2).mean())
            B = xb.shape[0]
            dout = err * (1.0 - pred ** 2) / float(B)
            dW, db, _ = mlp_backward(LL.W, cache, dout)
            adam_step(LL.W, LL.b, dW, db, state, lr=lr)
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [LL ep {ep:3d}] mse={losses[-1]:.5f}")
    return losses


# ----------------------------------------------------------------------
# Closed-loop rollout (eval-time)
# ----------------------------------------------------------------------

def rollout(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray,
            waypoints: List[np.ndarray],
            ll_policy=None,
            t_max: int = T_MAX_STEPS,
            terminate_on_collision: bool = True) -> Dict:
    """Sequentially head to each waypoint, then to goal. The LL is
    obstacle-blind: any collision (point enters an obstacle disk) ends the
    episode immediately. Sub-goals are how the SGG steers the LL around
    obstacles."""
    pos = start.copy().astype(np.float32)
    traj = [pos.copy()]
    targets = list(waypoints) + [goal]
    leg_idx = 0
    leg_target = targets[leg_idx]
    leg_steps = 0
    leg_max = max(10, int(np.ceil(t_max / max(len(targets), 1))))
    leg_arrival_radius = max(GOAL_RADIUS, STEP_MAX * 1.5)

    collided = False
    total_steps = 0
    while total_steps < t_max:
        if ll_policy is not None:
            action = ll_policy(pos, leg_target, obstacles)
        else:
            # fallback: straight line capped to STEP_MAX
            rel = leg_target - pos
            d = float(np.linalg.norm(rel)) + 1e-8
            action = (STEP_MAX * rel / d) if d > STEP_MAX else rel.copy()
        n = float(np.linalg.norm(action)) + 1e-8
        if n > STEP_MAX:
            action = action * STEP_MAX / n
        new_pos = (pos + action).astype(np.float32)
        new_pos = np.clip(new_pos, 0.0, ARENA_SIZE).astype(np.float32)
        # collision test on the new position
        hit = False
        for obs in obstacles:
            if np.linalg.norm(new_pos - obs[:2]) < obs[2]:
                hit = True
                break
        pos = new_pos
        traj.append(pos.copy())
        total_steps += 1
        leg_steps += 1
        if hit:
            collided = True
            if terminate_on_collision:
                break
        # advance leg
        radius = GOAL_RADIUS if leg_idx == len(targets) - 1 else leg_arrival_radius
        if np.linalg.norm(pos - leg_target) < radius or leg_steps >= leg_max:
            if leg_idx == len(targets) - 1:
                break
            leg_idx += 1
            leg_target = targets[leg_idx]
            leg_steps = 0

    traj_arr = np.stack(traj)
    path_length = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))
    success = (np.linalg.norm(pos - goal) < GOAL_RADIUS) and not collided
    return {
        "success": bool(success),
        "collided": bool(collided),
        "reached_goal": bool(np.linalg.norm(pos - goal) < GOAL_RADIUS),
        "path_length": path_length,
        "final_dist_to_goal": float(np.linalg.norm(pos - goal)),
        "n_steps": int(total_steps),
        "trajectory": traj_arr,
    }


# ----------------------------------------------------------------------
# Phase 2: train the sub-goal generator through the differentiable model
# ----------------------------------------------------------------------

def train_subgoal_generator(SGG: SubgoalGenerator,
                            n_arenas_per_epoch: int,
                            epochs: int,
                            lr: float,
                            rng: np.random.Generator,
                            verbose: bool = True
                            ) -> Dict[str, list]:
    """Each epoch: sample fresh arenas, forward through SGG, compute total
    cost via the differentiable surrogate, backprop into SGG."""
    state = adam_init(SGG.W, SGG.b)
    history = {
        "epoch": [],
        "total_cost": [],
        "path_length": [],
        "obstacle_penalty": [],
        "grad_norm": [],
    }
    for ep in range(epochs):
        # fresh batch of arenas
        states = []
        obstacles_b = []
        for _ in range(n_arenas_per_epoch):
            obs = sample_arena(rng)
            states.append(featurize_state(START, GOAL, obs))
            obstacles_b.append(obs)
        states = np.stack(states)

        sgs, cache, raw_out = SGG.forward(states)               # (B, K, 2)

        total_cost = 0.0
        total_length = 0.0
        total_penalty = 0.0
        dsg_b = np.zeros_like(sgs)
        for i in range(n_arenas_per_epoch):
            c, dsg, ln, pen = total_cost_with_grad(sgs[i], START, GOAL, obstacles_b[i])
            total_cost += c
            total_length += ln
            total_penalty += pen
            dsg_b[i] = dsg
        total_cost /= n_arenas_per_epoch
        total_length /= n_arenas_per_epoch
        total_penalty /= n_arenas_per_epoch
        dsg_b /= n_arenas_per_epoch

        dW, db = SGG.backward(cache, raw_out, dsg_b)
        gnorm = float(np.sqrt(sum((g * g).sum() for g in dW + db)))
        adam_step(SGG.W, SGG.b, dW, db, state, lr=lr, clip=5.0)

        history["epoch"].append(ep)
        history["total_cost"].append(float(total_cost))
        history["path_length"].append(float(total_length))
        history["obstacle_penalty"].append(float(total_penalty))
        history["grad_norm"].append(gnorm)

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [SGG ep {ep:3d}]  cost={total_cost:.3f}  "
                  f"length={total_length:.3f}  pen={total_penalty:.3f}  "
                  f"|grad|={gnorm:.3f}")
    return history


# ----------------------------------------------------------------------
# Eval over a held-out batch of arenas
# ----------------------------------------------------------------------

def evaluate_policies(SGG: SubgoalGenerator, LL: LowLevelPolicy,
                      n_arenas: int, rng: np.random.Generator
                      ) -> Dict:
    """Roll out (a) trained SGG + LL, (b) direct LL (no sub-goals) on the same
    arenas and report success and path length."""
    sgg_rollouts = []
    direct_rollouts = []
    arenas = []
    sgs_record = []
    for _ in range(n_arenas):
        obs = sample_arena(rng)
        arenas.append(obs)

    states = np.stack([featurize_state(START, GOAL, o) for o in arenas])
    sgs_b, _, _ = SGG.forward(states)                           # (B, K, 2)

    for i in range(n_arenas):
        obs = arenas[i]
        wps = [sgs_b[i, k] for k in range(N_SUBGOALS)]
        sgs_record.append(np.stack(wps))
        sgg_rollouts.append(rollout(START, GOAL, obs, wps, ll_policy=LL))
        direct_rollouts.append(rollout(START, GOAL, obs, [], ll_policy=LL))

    def agg(rolls):
        succ = float(np.mean([r["success"] for r in rolls]))
        coll = float(np.mean([r["collided"] for r in rolls]))
        reached = float(np.mean([r["reached_goal"] for r in rolls]))
        plen = float(np.mean([r["path_length"] for r in rolls]))
        plen_succ = float(np.mean([r["path_length"] for r in rolls if r["success"]])) \
            if any(r["success"] for r in rolls) else float("nan")
        steps = float(np.mean([r["n_steps"] for r in rolls]))
        return {
            "success_rate": succ,
            "collision_rate": coll,
            "reach_goal_rate": reached,
            "mean_path_length": plen,
            "mean_path_length_success_only": plen_succ,
            "mean_steps": steps,
        }

    return {
        "sgg": agg(sgg_rollouts),
        "direct": agg(direct_rollouts),
        "arenas": arenas,
        "sgs": sgs_record,
        "sgg_rollouts": sgg_rollouts,
        "direct_rollouts": direct_rollouts,
    }


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def env_info(seed: int) -> Dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "seed": int(seed),
    }


def train_full(seed: int = 0,
               ll_samples: int = 4000,
               ll_epochs: int = 20,
               ll_lr: float = 3e-3,
               sgg_arenas_per_epoch: int = 128,
               sgg_epochs: int = 400,
               sgg_lr: float = 3e-3,
               eval_arenas: int = 200,
               sgg_hidden: int = 96,
               ll_hidden: int = 16,
               quiet: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    if not quiet:
        print(f"[seed {seed}] subgoal-obstacle-avoidance — Schmidhuber 1991")
        print(f"  arena {ARENA_SIZE}x{ARENA_SIZE}, "
              f"start={tuple(START.tolist())}, goal={tuple(GOAL.tolist())}, "
              f"N_obstacles={N_OBSTACLES}, K_subgoals={N_SUBGOALS}, "
              f"step_max={STEP_MAX}, T_max={T_MAX_STEPS}")

    t0 = time.time()
    if not quiet:
        print(f"\n[phase 1] LL imitation: {ll_samples} samples x {ll_epochs} epochs")
    LL = LowLevelPolicy.make(rng, hidden=ll_hidden)
    ll_losses = train_low_level_policy(LL, ll_samples, ll_epochs, ll_lr, rng,
                                       verbose=not quiet)
    t1 = time.time()

    if not quiet:
        print(f"\n[phase 2] SGG via differentiable cost model: "
              f"{sgg_arenas_per_epoch} arenas x {sgg_epochs} epochs")
    SGG = SubgoalGenerator.make(rng, hidden=sgg_hidden)
    sgg_history = train_subgoal_generator(
        SGG, sgg_arenas_per_epoch, sgg_epochs, sgg_lr, rng,
        verbose=not quiet,
    )
    t2 = time.time()

    if not quiet:
        print(f"\n[eval] {eval_arenas} fresh arenas")
    eval_metrics = evaluate_policies(SGG, LL, eval_arenas, rng)
    t3 = time.time()

    if not quiet:
        sgg = eval_metrics["sgg"]
        direct = eval_metrics["direct"]
        print(f"  SGG (sub-goals):   "
              f"success {sgg['success_rate']*100:5.1f}%  "
              f"collisions {sgg['collision_rate']*100:5.1f}%  "
              f"reach-goal {sgg['reach_goal_rate']*100:5.1f}%  "
              f"path_len {sgg['mean_path_length']:.2f}  "
              f"path_len_succ {sgg['mean_path_length_success_only']:.2f}")
        print(f"  Direct (no sub-goals): "
              f"success {direct['success_rate']*100:5.1f}%  "
              f"collisions {direct['collision_rate']*100:5.1f}%  "
              f"reach-goal {direct['reach_goal_rate']*100:5.1f}%  "
              f"path_len {direct['mean_path_length']:.2f}")

        print(f"\nTimes: LL {t1-t0:.1f}s, SGG {t2-t1:.1f}s, eval {t3-t2:.1f}s, "
              f"total {t3-t0:.1f}s")

    return {
        "config": {
            "seed": seed,
            "arena_size": ARENA_SIZE,
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "n_obstacles": N_OBSTACLES,
            "obs_radius": OBS_RADIUS,
            "n_subgoals": N_SUBGOALS,
            "step_max": STEP_MAX,
            "t_max": T_MAX_STEPS,
            "goal_radius": GOAL_RADIUS,
            "t_samples": T_SAMPLES,
            "sigma": SIGMA,
            "lambda_obs": LAMBDA_OBS,
            "ll_samples": ll_samples,
            "ll_epochs": ll_epochs,
            "ll_lr": ll_lr,
            "ll_hidden": ll_hidden,
            "sgg_arenas_per_epoch": sgg_arenas_per_epoch,
            "sgg_epochs": sgg_epochs,
            "sgg_lr": sgg_lr,
            "sgg_hidden": sgg_hidden,
            "eval_arenas": eval_arenas,
        },
        "env": env_info(seed),
        "ll_losses": ll_losses,
        "sgg_history": sgg_history,
        "eval": {k: v for k, v in eval_metrics.items()
                 if k in ("sgg", "direct")},
        "wallclock": {"LL": t1 - t0, "SGG": t2 - t1, "eval": t3 - t2,
                      "total": t3 - t0},
        # weights kept for downstream plotting
        "_LL_W": [w.copy() for w in LL.W],
        "_LL_b": [bb.copy() for bb in LL.b],
        "_SGG_W": [w.copy() for w in SGG.W],
        "_SGG_b": [bb.copy() for bb in SGG.b],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ll-epochs", type=int, default=20)
    p.add_argument("--ll-samples", type=int, default=4000)
    p.add_argument("--sgg-epochs", type=int, default=400)
    p.add_argument("--sgg-arenas", type=int, default=128)
    p.add_argument("--eval-arenas", type=int, default=200)
    p.add_argument("--out-json", type=str, default=None)
    args = p.parse_args()

    result = train_full(
        seed=args.seed,
        ll_samples=args.ll_samples,
        ll_epochs=args.ll_epochs,
        sgg_arenas_per_epoch=args.sgg_arenas,
        sgg_epochs=args.sgg_epochs,
        eval_arenas=args.eval_arenas,
    )
    if args.out_json:
        light = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(args.out_json, "w") as f:
            json.dump(light, f, indent=2)
        print(f"\nSummary written to {args.out_json}")


if __name__ == "__main__":
    main()
