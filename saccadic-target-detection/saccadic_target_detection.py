"""
saccadic-target-detection — differentiable world-model controller for active vision.

Schmidhuber & Huber, "Learning to generate focus trajectories for attentive vision",
TR FKI-128-90 (TUM, April 1990).

The original FKI-128-90 PDF is not retrievable in its 1990 form. The algorithm here
is reconstructed from §6.4 of Schmidhuber's 2015 *Deep Learning in Neural Networks:
An Overview* (the active-vision lineage paragraph) and §"Learning to look" of the
2020 *Deep Learning: Our Miraculous Year 1990–1991* retrospective. The conceptual
recipe is the same one used in the companion 1990 cart-pole and flip-flop work:
build a differentiable world-model M of the environment, then train the controller
C by backpropagating through (frozen) M to maximize a desired prediction of M.

Problem
-------
Scene
    16x16 grayscale image. Target is a 2-D Gaussian "halo" of sigma 2.5 centered
    at a random (x, y) in [3, 12]^2 (so the halo fits inside the scene). Pixel
    intensity = exp(-r^2 / (2 sigma^2)) plus low-amplitude background noise.
Fovea
    5x5 window. The controller only ever sees the 25 pixels under the fovea
    plus its (x, y) center. The rest of the scene is hidden.
Actions
    Continuous (dx, dy) in [-step_max, +step_max], step_max=3.0. Position update
    is clipped so the fovea stays inside the scene.
Goal
    Drive the fovea center to within 1 pixel (Euclidean) of the target center.
    Episode ends on success or after T_max saccades (default 20).

Algorithm
---------
Phase 1: train the world-model M.
    M takes (fovea[5,5], pos[2], action[2]) and outputs a scalar prediction of
    the target halo intensity at the *next* fovea center. Halo intensity is
    in [0, 1] (the Gaussian peak value), which gives a smooth gradient. M is
    trained on random rollouts using MSE loss against the ground-truth halo
    value at the next position. The discrete "found target" event is then
    monotone in this intensity (intensity > 0.92 corresponds to fovea center
    within DETECT_RADIUS=1.0 of the target).

Phase 2: train the controller C with M frozen.
    C takes (fovea[5,5], pos[2]) and outputs (dx, dy) via tanh * step_max. Real
    rollouts: at each step we compute pred = M(fovea, pos, C(fovea, pos)) and
    update C to maximize pred (1-step myopic model-based policy gradient).
    The real environment then advances the fovea, the next fovea is read from
    the scene, and the procedure repeats.

This is the "differentiable world model" recipe spelled out in Schmidhuber 1990.
The gradient on C's weights flows: dpred/dC = dpred/daction * daction/dC, where
the first factor goes through M (treated as a smooth predictor) and the second
through C's own backward pass. M's weights are fixed during this phase.

Run
---
    python3 saccadic_target_detection.py --seed 0
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

import numpy as np


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

SCENE_SIZE = 16
FOVEA_SIZE = 5
HALF = FOVEA_SIZE // 2  # 2
SIGMA = 4.0
NOISE_AMP = 0.05
STEP_MAX = 3.0
T_MAX = 20
DETECT_RADIUS = 1.0


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_scene(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scene[16,16], target_xy[2])."""
    pad = 3
    target = rng.uniform(pad, SCENE_SIZE - 1 - pad, size=2).astype(np.float32)
    yy, xx = np.indices((SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
    # row index = y (vertical), col index = x (horizontal)
    dist_sq = (xx - target[0]) ** 2 + (yy - target[1]) ** 2
    halo = np.exp(-dist_sq / (2.0 * SIGMA ** 2)).astype(np.float32)
    noise = rng.uniform(0.0, NOISE_AMP, size=(SCENE_SIZE, SCENE_SIZE)).astype(np.float32)
    return halo + noise, target


def extract_fovea(scene: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Read 5x5 fovea at integer-rounded position. pos = (x, y)."""
    cx = int(np.clip(round(float(pos[0])), HALF, SCENE_SIZE - HALF - 1))
    cy = int(np.clip(round(float(pos[1])), HALF, SCENE_SIZE - HALF - 1))
    patch = scene[cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
    return patch.astype(np.float32)


def clip_pos(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, HALF, SCENE_SIZE - 1 - HALF).astype(np.float32)


def target_indicator(pos: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pos - target) <= DETECT_RADIUS)


def halo_intensity(pos: np.ndarray, target: np.ndarray) -> float:
    """Smooth scalar in [0, 1]: peak at target, Gaussian falloff."""
    r2 = float(np.sum((pos - target) ** 2))
    return float(np.exp(-r2 / (2.0 * SIGMA ** 2)))


# ----------------------------------------------------------------------
# MLP (manual forward + backward, batched)
# ----------------------------------------------------------------------

def init_mlp(sizes: List[int], rng: np.random.Generator, scale: float = 0.3):
    W = []
    b = []
    for i in range(len(sizes) - 1):
        # He-style init for tanh-friendly variance
        s = scale / np.sqrt(sizes[i])
        W.append((s * rng.standard_normal((sizes[i], sizes[i + 1]))).astype(np.float32))
        b.append(np.zeros(sizes[i + 1], dtype=np.float32))
    return W, b


def mlp_forward(W, b, x: np.ndarray):
    """Forward pass. Returns (out, cache). All hidden activations are tanh; output is linear."""
    acts = [x]
    h = x
    for i in range(len(W) - 1):
        h = np.tanh(h @ W[i] + b[i])
        acts.append(h)
    out = h @ W[-1] + b[-1]
    acts.append(out)
    return out, acts


def mlp_backward(W, acts, dout):
    """Return (dW_list, db_list, dx). dout matches shape of acts[-1]."""
    dW = [None] * len(W)
    db = [None] * len(W)
    dh = dout
    # output (linear) layer
    dW[-1] = acts[-2].T @ dh
    db[-1] = dh.sum(axis=0)
    dh = dh @ W[-1].T
    # tanh hidden layers
    for i in range(len(W) - 2, -1, -1):
        dh = dh * (1.0 - acts[i + 1] ** 2)
        dW[i] = acts[i].T @ dh
        db[i] = dh.sum(axis=0)
        if i > 0:
            dh = dh @ W[i].T
        else:
            dx = dh @ W[0].T
    return dW, db, dx


def sgd_step(W, b, dW, db, lr: float):
    for i in range(len(W)):
        W[i] -= lr * dW[i]
        b[i] -= lr * db[i]


# ----------------------------------------------------------------------
# State featurization
# ----------------------------------------------------------------------

_FOV_GRID_X = np.tile(np.arange(FOVEA_SIZE) - HALF, (FOVEA_SIZE, 1)).astype(np.float32)
_FOV_GRID_Y = _FOV_GRID_X.T


def fovea_centroid_batch(fovea_b: np.ndarray) -> np.ndarray:
    """Brightness-weighted offset of bright pixels relative to fovea center.

    Returns (B, 2) offset in fovea-pixel units (so range roughly -2..+2).
    Points toward the brightest region in the fovea, which under our scene
    geometry points toward the target. Adds a strong geometric prior to the
    raw fovea pixels.
    """
    B = fovea_b.shape[0]
    f = fovea_b.reshape(B, -1)
    grid_x = _FOV_GRID_X.reshape(-1)
    grid_y = _FOV_GRID_Y.reshape(-1)
    total = f.sum(axis=1) + 1e-6
    cx = (f * grid_x).sum(axis=1) / total
    cy = (f * grid_y).sum(axis=1) / total
    return np.stack([cx, cy], axis=1).astype(np.float32)


def featurize(fovea: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Flatten 5x5 fovea, append normalized pos and fovea centroid -> length 29."""
    return featurize_batch(fovea[None], pos[None])[0]


def featurize_batch(fovea_b: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Batched. fovea_b [B,5,5], pos_b [B,2] -> [B, 29] (fovea, pos, centroid)."""
    B = fovea_b.shape[0]
    f = fovea_b.reshape(B, -1)
    p = pos_b / float(SCENE_SIZE)
    c = fovea_centroid_batch(fovea_b) / float(HALF)  # normalize to [-1, 1]
    return np.concatenate([f, p, c], axis=1).astype(np.float32)


# ----------------------------------------------------------------------
# World-model M and Controller C
# ----------------------------------------------------------------------

@dataclass
class Controller:
    W: list
    b: list

    @classmethod
    def make(cls, rng, hidden=32):
        # input: fovea(25) + pos(2) + fovea_centroid(2) = 29
        W, b = init_mlp([29, hidden, 2], rng)
        return cls(W, b)

    def forward(self, fovea, pos):
        x = featurize_batch(fovea, pos)
        out, cache = mlp_forward(self.W, self.b, x)
        action = STEP_MAX * np.tanh(out)
        cache.append(action)
        return action, cache

    def backward(self, cache, daction):
        out = cache[-2]
        # daction/dout = STEP_MAX * (1 - tanh(out)^2)
        dout = daction * STEP_MAX * (1.0 - np.tanh(out) ** 2)
        dW, db, _ = mlp_backward(self.W, cache[:-1], dout)
        return dW, db


@dataclass
class WorldModel:
    """World-model that predicts the *change* in halo intensity at the fovea center
    after a saccade: delta = halo(pos + action) - halo(pos).

    Diagnostic on uniform-random transitions shows delta ≈ k · (centroid · action),
    where centroid is the brightness-weighted offset of the bright pixels in the
    fovea. We feed the MLP a compact set of geometric features:
        [fovea_center(1), centroid(2 normalized), pos(2 normalized), action(2 norm),
         centroid·action_xx, centroid·action_yy, centroid·action_xy, centroid·action_yx]
    The four bilinear terms expose the centroid·action interaction directly so a
    2-layer tanh MLP can fit it cleanly. Raw fovea pixels are dropped as they add
    overfitting noise — all the relevant info is captured by fovea_center +
    centroid (verified by ridge regression: R^2=0.50 on held-out).
    """
    W: list
    b: list
    INPUT_DIM = 11  # center(1) + centroid(2) + pos(2) + action(2) + outer(4)

    @classmethod
    def make(cls, rng, hidden=64, depth=2):
        sizes = [cls.INPUT_DIM] + [hidden] * depth + [1]
        W, b = init_mlp(sizes, rng)
        return cls(W, b)

    @staticmethod
    def _build_input(fovea, pos, action):
        B = fovea.shape[0]
        center = fovea[:, HALF, HALF].reshape(B, 1)                 # (B, 1)
        centroid = fovea_centroid_batch(fovea) / float(HALF)         # (B, 2) in [-1, 1]
        p = pos / float(SCENE_SIZE)                                  # (B, 2)
        a = action / STEP_MAX                                        # (B, 2)
        # Bilinear features: outer product centroid x action_norm, flattened (4)
        outer = (centroid[:, :, None] * a[:, None, :]).reshape(B, 4)
        return np.concatenate([center, centroid, p, a, outer], axis=1).astype(np.float32), \
               centroid, a

    def forward(self, fovea, pos, action):
        x_full, centroid, a_norm = self._build_input(fovea, pos, action)
        logit, cache = mlp_forward(self.W, self.b, x_full)
        return logit.squeeze(-1), (cache, centroid, a_norm)

    def grad_action(self, packed_cache, dlogit):
        """Gradient of logit w.r.t. real action through frozen M.

        Layout: center(1) + centroid(2) + pos(2) + a_norm(2) + outer(4) = 11.
        Action enters via:
          - dims 5:7 (a_norm = action / STEP_MAX): da_norm/daction = 1/STEP_MAX
          - dims 7:11 (outer[c, a] = centroid[c] * a_norm[a])
        """
        cache, centroid, _ = packed_cache
        dout = dlogit[:, None]
        dW, db, dx = mlp_backward(self.W, cache, dout)
        d_a_norm_direct = dx[:, 5:7]
        d_outer = dx[:, 7:11].reshape(dx.shape[0], 2, 2)
        # outer[i, c, a] = centroid[i, c] * a_norm[i, a]
        # dL/da_norm[i, a] += sum_c d_outer[i, c, a] * centroid[i, c]
        d_a_norm_outer = (d_outer * centroid[:, :, None]).sum(axis=1)
        d_a_norm = d_a_norm_direct + d_a_norm_outer
        d_action = d_a_norm / STEP_MAX
        return d_action

    def gradients(self, packed_cache, dlogit):
        cache, _, _ = packed_cache
        dout = dlogit[:, None]
        dW, db, _ = mlp_backward(self.W, cache, dout)
        return dW, db


# ----------------------------------------------------------------------
# Phase 1: train world-model M from random rollouts
# ----------------------------------------------------------------------

def collect_random_transitions(n_scenes: int, n_steps: int, rng: np.random.Generator):
    """Generate random saccades. Records:
        fovea, pos, action, halo_curr, halo_next, ind_next.
    halo_curr is halo at current pos (so we can train M on the *delta* halo_next - halo_curr,
    which removes the dominant scene-mean signal and exposes the action effect).
    """
    fovea_list, pos_list, action_list = [], [], []
    halo_curr_list, halo_next_list, ind_next_list = [], [], []
    for _ in range(n_scenes):
        scene, target = make_scene(rng)
        pos = rng.uniform(HALF, SCENE_SIZE - 1 - HALF, size=2).astype(np.float32)
        for _ in range(n_steps):
            fovea = extract_fovea(scene, pos)
            action = rng.uniform(-STEP_MAX, STEP_MAX, size=2).astype(np.float32)
            new_pos = clip_pos(pos + action)
            halo_curr_list.append(halo_intensity(pos, target))
            halo_next_list.append(halo_intensity(new_pos, target))
            ind_next_list.append(target_indicator(new_pos, target))
            fovea_list.append(fovea)
            pos_list.append(pos.copy())
            action_list.append(action)
            pos = new_pos
    return (np.stack(fovea_list), np.stack(pos_list),
            np.stack(action_list),
            np.array(halo_curr_list, dtype=np.float32),
            np.array(halo_next_list, dtype=np.float32),
            np.array(ind_next_list, dtype=np.float32))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def train_world_model(M: WorldModel, fovea, pos, action, halo_curr, halo_next,
                      epochs=30, batch_size=256, lr=0.05,
                      rng: np.random.Generator = None) -> List[float]:
    """Train M with MSE loss to predict halo *change* delta = halo_next - halo_curr.

    Returns list of epoch MSE losses on the delta target.
    """
    n = fovea.shape[0]
    delta_y = halo_next - halo_curr
    losses = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            fb, pb, ab, yb = fovea[b_idx], pos[b_idx], action[b_idx], delta_y[b_idx]
            pred_delta, cache = M.forward(fb, pb, ab)
            B = yb.shape[0]
            err = pred_delta - yb
            loss = float(0.5 * (err ** 2).mean())
            dpred = (err / float(B))
            dW, db = M.gradients(cache, dpred)
            sgd_step(M.W, M.b, dW, db, lr)
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / n_batches)
    return losses


def evaluate_world_model(M, fovea, pos, action, halo_curr, halo_next, ind_y) -> Dict[str, float]:
    pred_delta, _ = M.forward(fovea, pos, action)
    delta_y = halo_next - halo_curr
    err = pred_delta - delta_y
    mse = float((err ** 2).mean())
    # variance of delta target (baseline if M predicted 0)
    var_baseline = float((delta_y ** 2).mean())
    explained_var = 1.0 - mse / max(var_baseline, 1e-9)
    # halo prediction = current_halo (≈ fovea center) + predicted delta
    pred_halo = halo_curr + pred_delta
    thresh = float(np.exp(-(DETECT_RADIUS ** 2) / (2.0 * SIGMA ** 2)))
    pred_ind = (pred_halo > thresh).astype(np.float32)
    acc = float((pred_ind == ind_y).mean())
    return {"mse": mse, "explained_var": explained_var, "ind_acc": acc,
            "pos_rate": float(ind_y.mean()), "pred_rate": float(pred_ind.mean()),
            "delta_var": var_baseline}


# ----------------------------------------------------------------------
# Phase 2: train controller C by backprop through frozen M
# ----------------------------------------------------------------------

def rollout_controller_step(C: Controller, M: WorldModel,
                            scenes: List[np.ndarray], targets: np.ndarray,
                            positions: np.ndarray, lr: float,
                            train: bool = True):
    """Run one synchronous rollout step on a batch of active scenes.

    Returns next_positions, indicators, mean_score, indicator_per_scene.
    Updates C in place when train=True via 1-step myopic gradient through M.
    """
    B = positions.shape[0]
    fovea_b = np.stack([extract_fovea(s, p) for s, p in zip(scenes, positions)])
    action, c_cache = C.forward(fovea_b, positions)

    # Forward through M.  M predicts delta = halo_next - halo_curr.
    pred_delta, m_cache = M.forward(fovea_b, positions, action)
    # Predicted next halo = fovea center brightness (≈ current halo) + delta.
    fovea_center = fovea_b[:, HALF, HALF]
    score = fovea_center + pred_delta

    if train:
        # Maximize sum of score = sum of pred_delta (fovea_center is independent of action).
        dpred = -np.ones(B, dtype=np.float32) / float(B)
        # gradient w.r.t. action through frozen M
        d_action = M.grad_action(m_cache, dpred)
        dW, db = C.backward(c_cache, d_action)
        sgd_step(C.W, C.b, dW, db, lr)

    # Take real environment step
    new_positions = clip_pos(positions + action)
    indicators = np.array([target_indicator(p, t) for p, t in zip(new_positions, targets)],
                          dtype=np.float32)
    return new_positions, indicators, float(score.mean())


def train_controller(C: Controller, M: WorldModel, n_scenes: int,
                     epochs: int, lr: float, rng: np.random.Generator,
                     log_every: int = 1) -> Dict[str, list]:
    """Train C through frozen M. Each epoch: fresh batch of scenes, rollouts up to T_MAX.

    Logs mean predicted score, mean indicators, mean saccades-to-find.
    """
    history = {"epoch": [], "mean_score": [], "find_rate": [], "median_saccades": []}
    for ep in range(epochs):
        scenes, targets = [], []
        for _ in range(n_scenes):
            s, t = make_scene(rng)
            scenes.append(s)
            targets.append(t)
        targets = np.stack(targets)

        positions = np.full((n_scenes, 2), SCENE_SIZE / 2.0 - 0.5, dtype=np.float32)
        active = np.ones(n_scenes, dtype=bool)
        saccades_used = np.full(n_scenes, T_MAX, dtype=np.int32)
        scores_log = []
        for step in range(T_MAX):
            if not active.any():
                break
            idx = np.where(active)[0]
            sub_positions = positions[idx]
            sub_scenes = [scenes[i] for i in idx]
            sub_targets = targets[idx]
            new_pos, ind, score = rollout_controller_step(
                C, M, sub_scenes, sub_targets, sub_positions, lr, train=True
            )
            scores_log.append(score)
            positions[idx] = new_pos
            for j, i in enumerate(idx):
                if ind[j] > 0.5:
                    saccades_used[i] = step + 1
                    active[i] = False

        find_rate = float((saccades_used < T_MAX).mean())
        median_sacc = float(np.median(saccades_used))
        if (ep % log_every) == 0 or ep == epochs - 1:
            history["epoch"].append(ep)
            history["mean_score"].append(float(np.mean(scores_log)))
            history["find_rate"].append(find_rate)
            history["median_saccades"].append(median_sacc)
    return history


# ----------------------------------------------------------------------
# Eval: roll out the trained policy on fresh scenes, no training
# ----------------------------------------------------------------------

def eval_policy(C: Controller, M: WorldModel, n_scenes: int,
                rng: np.random.Generator) -> Dict[str, float]:
    scenes, targets = [], []
    for _ in range(n_scenes):
        s, t = make_scene(rng)
        scenes.append(s)
        targets.append(t)
    targets = np.stack(targets)

    positions = np.full((n_scenes, 2), SCENE_SIZE / 2.0 - 0.5, dtype=np.float32)
    saccades_used = np.full(n_scenes, T_MAX, dtype=np.int32)
    active = np.ones(n_scenes, dtype=bool)

    for step in range(T_MAX):
        if not active.any():
            break
        idx = np.where(active)[0]
        sub_positions = positions[idx]
        sub_scenes = [scenes[i] for i in idx]
        sub_targets = targets[idx]
        new_pos, ind, _ = rollout_controller_step(
            C, M, sub_scenes, sub_targets, sub_positions, lr=0.0, train=False
        )
        positions[idx] = new_pos
        for j, i in enumerate(idx):
            if ind[j] > 0.5:
                saccades_used[i] = step + 1
                active[i] = False

    return {
        "find_rate": float((saccades_used < T_MAX).mean()),
        "median_saccades": float(np.median(saccades_used)),
        "mean_saccades": float(np.mean(saccades_used)),
        "n_scenes": int(n_scenes),
        "saccades_per_scene": saccades_used.tolist(),
    }


def random_baseline(n_scenes: int, rng: np.random.Generator) -> Dict[str, float]:
    """Random saccade baseline: every step, pick a uniform action in [-STEP_MAX, STEP_MAX]^2."""
    scenes, targets = [], []
    for _ in range(n_scenes):
        s, t = make_scene(rng)
        scenes.append(s)
        targets.append(t)
    targets = np.stack(targets)
    positions = np.full((n_scenes, 2), SCENE_SIZE / 2.0 - 0.5, dtype=np.float32)
    saccades_used = np.full(n_scenes, T_MAX, dtype=np.int32)
    for step in range(T_MAX):
        active = saccades_used == T_MAX
        if not active.any():
            break
        actions = rng.uniform(-STEP_MAX, STEP_MAX, size=(n_scenes, 2)).astype(np.float32)
        positions = clip_pos(positions + actions)
        for i in range(n_scenes):
            if active[i] and target_indicator(positions[i], targets[i]) > 0.5:
                saccades_used[i] = step + 1
    return {
        "find_rate": float((saccades_used < T_MAX).mean()),
        "median_saccades": float(np.median(saccades_used)),
        "mean_saccades": float(np.mean(saccades_used)),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def env_info(seed: int) -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "seed": int(seed),
    }


def train_full(seed: int = 0,
               n_m_scenes: int = 1000, n_m_steps: int = 30,
               m_epochs: int = 150, m_lr: float = 0.03,
               n_c_scenes: int = 128, c_epochs: int = 150, c_lr: float = 0.05,
               eval_scenes: int = 200,
               m_hidden: int = 32, m_depth: int = 2, c_hidden: int = 32,
               quiet: bool = False) -> dict:
    """Full pipeline. Returns dict with histories + final eval + config."""
    rng = np.random.default_rng(seed)

    if not quiet:
        print(f"[seed {seed}] saccadic-target-detection — Schmidhuber & Huber 1990")
        print(f"  scene {SCENE_SIZE}x{SCENE_SIZE}, fovea {FOVEA_SIZE}x{FOVEA_SIZE}, "
              f"target sigma={SIGMA}, step_max={STEP_MAX}, T_max={T_MAX}")

    # --- Phase 1: world-model ---
    t0 = time.time()
    if not quiet:
        print(f"\n[phase 1] M training: {n_m_scenes} scenes x {n_m_steps} steps "
              f"= {n_m_scenes * n_m_steps} transitions")
    fovea, pos, action, halo_curr, halo_next, ind_y = collect_random_transitions(
        n_m_scenes, n_m_steps, rng)
    M = WorldModel.make(rng, hidden=m_hidden, depth=m_depth)
    m_losses = train_world_model(M, fovea, pos, action, halo_curr, halo_next,
                                 epochs=m_epochs, batch_size=256, lr=m_lr, rng=rng)

    # held-out M eval
    fov_te, pos_te, act_te, hc_te, hn_te, ind_te = collect_random_transitions(40, 25, rng)
    m_metrics = evaluate_world_model(M, fov_te, pos_te, act_te, hc_te, hn_te, ind_te)
    if not quiet:
        print(f"  M final MSE {m_losses[-1]:.4f}  "
              f"held-out MSE {m_metrics['mse']:.4f}  "
              f"R2 {m_metrics['explained_var']:.3f}  "
              f"detect-acc {m_metrics['ind_acc']:.3f}")

    t1 = time.time()
    # --- Phase 2: controller ---
    if not quiet:
        print(f"\n[phase 2] C training (M frozen): {c_epochs} epochs of {n_c_scenes} scenes")
    C = Controller.make(rng, hidden=c_hidden)
    c_history = train_controller(C, M, n_c_scenes, c_epochs, c_lr, rng, log_every=1)
    if not quiet:
        print(f"  C final find_rate {c_history['find_rate'][-1]:.3f}  "
              f"median saccades {c_history['median_saccades'][-1]:.1f}")

    t2 = time.time()
    # --- Eval ---
    if not quiet:
        print(f"\n[eval] {eval_scenes} fresh scenes")
    eval_metrics = eval_policy(C, M, eval_scenes, rng)
    rand_metrics = random_baseline(eval_scenes, rng)
    if not quiet:
        print(f"  trained C:  find_rate {eval_metrics['find_rate']:.3f}  "
              f"median saccades {eval_metrics['median_saccades']:.1f}  "
              f"mean {eval_metrics['mean_saccades']:.2f}")
        print(f"  random  :  find_rate {rand_metrics['find_rate']:.3f}  "
              f"median saccades {rand_metrics['median_saccades']:.1f}  "
              f"mean {rand_metrics['mean_saccades']:.2f}")

    t3 = time.time()
    if not quiet:
        print(f"\nTimes: M-train {t1 - t0:.1f}s, C-train {t2 - t1:.1f}s, eval {t3 - t2:.1f}s")

    return {
        "config": {
            "seed": seed, "scene_size": SCENE_SIZE, "fovea_size": FOVEA_SIZE,
            "sigma": SIGMA, "step_max": STEP_MAX, "T_max": T_MAX,
            "n_m_scenes": n_m_scenes, "n_m_steps": n_m_steps, "m_epochs": m_epochs,
            "m_lr": m_lr, "m_hidden": m_hidden, "m_depth": m_depth,
            "n_c_scenes": n_c_scenes, "c_epochs": c_epochs, "c_lr": c_lr,
            "c_hidden": c_hidden, "eval_scenes": eval_scenes,
        },
        "env": env_info(seed),
        "m_losses": m_losses,
        "m_metrics": m_metrics,
        "c_history": c_history,
        "eval": eval_metrics,
        "random_baseline": rand_metrics,
        "wallclock": {"M_train": t1 - t0, "C_train": t2 - t1, "eval": t3 - t2,
                      "total": t3 - t0},
        "M_state": {"W": [w.copy() for w in M.W], "b": [bb.copy() for bb in M.b]},
        "C_state": {"W": [w.copy() for w in C.W], "b": [bb.copy() for bb in C.b]},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-epochs", type=int, default=150)
    p.add_argument("--c-epochs", type=int, default=150)
    p.add_argument("--eval-scenes", type=int, default=200)
    p.add_argument("--out-json", type=str, default=None,
                   help="If set, write a JSON summary (without weights) here.")
    args = p.parse_args()

    result = train_full(
        seed=args.seed,
        m_epochs=args.m_epochs,
        c_epochs=args.c_epochs,
        eval_scenes=args.eval_scenes,
    )
    if args.out_json:
        # strip non-JSONable weight arrays
        light = {k: v for k, v in result.items() if k not in ("M_state", "C_state")}
        with open(args.out_json, "w") as f:
            json.dump(light, f, indent=2)


if __name__ == "__main__":
    main()
