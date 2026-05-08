"""world-models-vizdoom-dream -- Ha & Schmidhuber, *Recurrent World Models
Facilitate Policy Evolution*, NeurIPS 2018 (arXiv:1809.01999).

The headline of the paper's "DoomRNN dream" experiment is that a controller
trained ENTIRELY inside the learned world model M (its hallucinated rollouts,
no real-env steps) transfers zero-shot to the actual environment. Per SPEC
issue #1 (cybertronai/schmidhuber-problems), v1.5-deferred RL stubs that need
heavyweight env installs (VizDoom here) are finished under the synthetic-data
rule: a hand-rolled numpy mini-env replaces the simulator, and we keep the
algorithmic structure -- V (encoder), M (recurrent world model), C (controller
trained ONLY in M's dreams), zero-shot transfer to the real env.

The mini-env is a `DodgingEnv`: a small 2-D gridworld where monsters at the
top spawn fireballs that fall toward the agent. The agent moves at the bottom
row (left / stay / right) and survives one step at a time until a fireball
reaches its column. Reward = +1 per step survived. Direct numpy analog of
DoomTakeCover.

Pipeline (paper §A "iterative training procedure", reduced):

    1. collect REAL trajectories from a random policy
    2. train V: numpy MLP autoencoder on grid observations -> latent z
    3. train M: numpy LSTM on (z_t, a_t) -> (z_{t+1}, r_{t+1}, done_{t+1})
    4. train C: tiny linear controller, parameters optimised by ES (numpy
       analog of CMA-ES) with rollouts INSIDE the dream of M only -- never
       querying the real env during the inner loop
    5. evaluate C in the real env (zero-shot transfer); report headline

The headline picture is real-env vs dream-env survival curves side-by-side
across C's ES iterations. A "direct-trained" baseline C (same ES, real-env
rollouts in the inner loop) is included for reference -- the claim is that
the dream-trained C matches or approaches it.

CLI:
    python3 world_models_vizdoom_dream.py --seed 0
    python3 world_models_vizdoom_dream.py --seed 0 --quick
    python3 world_models_vizdoom_dream.py --seed 0 --save-json out/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

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
# Dodging-fireballs gridworld -- numpy analog of DoomTakeCover
# ----------------------------------------------------------------------

@dataclass
class DodgingEnv:
    """Small 2-D gridworld. Agent at bottom row dodges falling fireballs.

    Grid is `H` rows x `W` cols. Agent occupies the bottom row at column
    `agent_x`. At each step:
      1. agent moves left / stay / right (clipped at boundaries)
      2. existing fireballs descend by one row
      3. with probability `spawn_prob` (cap `max_fireballs` simultaneous),
         a new fireball spawns at row 0 in a random column
      4. terminal condition: any fireball lands on the agent's cell

    Reward = +1 per surviving step (including the terminal step in which
    `t == max_steps`). Done when a fireball collides with the agent or
    when `t == max_steps`.

    Observation = flattened HxW grid with three channels:
      channel 0: agent position (1 at agent cell)
      channel 1: fireball positions (1 at each fireball cell)
      channel 2: column-aggregated fireball "danger" -- the row of the
                 nearest fireball above each column, scaled to [0, 1].
                 This compresses the imminent threat into a fixed-size
                 vector even when fireballs are sparse, so the small V
                 has an easy time encoding "what to dodge".
    """

    W: int = 8
    H: int = 6
    max_fireballs: int = 4
    spawn_prob: float = 0.35
    max_steps: int = 120

    def __post_init__(self):
        self._rng: np.random.Generator | None = None
        self.agent_x = self.W // 2
        self.fireballs: List[Tuple[int, int]] = []
        self.t = 0
        self.done = False

    @property
    def obs_dim(self) -> int:
        return 3 * self.H * self.W

    def seed(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def reset(self) -> np.ndarray:
        if self._rng is None:
            self._rng = np.random.default_rng(0)
        self.agent_x = self.W // 2
        self.fireballs = []
        self.t = 0
        self.done = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        grid = np.zeros((3, self.H, self.W), dtype=np.float32)
        grid[0, self.H - 1, self.agent_x] = 1.0
        # Fireballs and per-column nearest-fireball row
        nearest_row = np.full(self.W, -1, dtype=np.int32)
        for x, y in self.fireballs:
            if 0 <= y < self.H and 0 <= x < self.W:
                grid[1, y, x] = 1.0
                if nearest_row[x] < 0 or y > nearest_row[x]:
                    # the *closest* fireball is the one with the largest y
                    # (lowest on screen, since y=0 is top)
                    nearest_row[x] = y
        for x in range(self.W):
            if nearest_row[x] >= 0:
                # broadcast: scale to [0, 1] where 1 = at agent's row
                grid[2, :, x] = (nearest_row[x] + 1) / self.H
        return grid.flatten()

    def step(self, a: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("step() after done")
        # 1. agent move
        if a == 0:
            self.agent_x = max(0, self.agent_x - 1)
        elif a == 2:
            self.agent_x = min(self.W - 1, self.agent_x + 1)
        # 2. fireballs descend
        new_fb = []
        for x, y in self.fireballs:
            ny = y + 1
            if ny < self.H:
                new_fb.append((x, ny))
        # 3. spawn
        if len(new_fb) < self.max_fireballs:
            if self._rng.random() < self.spawn_prob:
                fx = int(self._rng.integers(0, self.W))
                new_fb.append((fx, 0))
        self.fireballs = new_fb
        # 4. collision
        hit = any(
            (x == self.agent_x and y == self.H - 1)
            for x, y in self.fireballs
        )
        self.t += 1
        if hit:
            self.done = True
            return self._obs(), 1.0, True
        if self.t >= self.max_steps:
            self.done = True
            return self._obs(), 1.0, True
        return self._obs(), 1.0, False


# ----------------------------------------------------------------------
# V: numpy MLP autoencoder, obs -> z -> obs
# ----------------------------------------------------------------------

@dataclass
class V_Autoencoder:
    """48->32->z=8->32->48 MLP autoencoder, hand-coded forward + backward."""

    in_dim: int
    z_dim: int
    hidden: int
    rng: np.random.Generator

    def __post_init__(self):
        s1 = np.sqrt(1.0 / self.in_dim)
        s2 = np.sqrt(1.0 / self.hidden)
        s3 = np.sqrt(1.0 / self.z_dim)
        s4 = np.sqrt(1.0 / self.hidden)
        self.W1 = self.rng.normal(0, s1, (self.in_dim, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0, s2, (self.hidden, self.z_dim))
        self.b2 = np.zeros(self.z_dim)
        self.W3 = self.rng.normal(0, s3, (self.z_dim, self.hidden))
        self.b3 = np.zeros(self.hidden)
        self.W4 = self.rng.normal(0, s4, (self.hidden, self.in_dim))
        self.b4 = np.zeros(self.in_dim)
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3, "W4": self.W4, "b4": self.b4}

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(x @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        h2 = np.tanh(z @ self.W3 + self.b3)
        return h2 @ self.W4 + self.b4

    def forward(self, x: np.ndarray):
        h1 = np.tanh(x @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        h2 = np.tanh(z @ self.W3 + self.b3)
        x_hat = h2 @ self.W4 + self.b4
        cache = {"x": x, "h1": h1, "z": z, "h2": h2}
        return x_hat, cache

    def mse_grad(self, x_hat, x, cache):
        B = x.shape[0]
        d_x_hat = (x_hat - x) / B  # MSE grad wrt logits
        loss = float(((x_hat - x) ** 2).mean())
        h1, h2, z = cache["h1"], cache["h2"], cache["z"]
        x_in = cache["x"]
        dW4 = h2.T @ d_x_hat
        db4 = d_x_hat.sum(0)
        dh2 = d_x_hat @ self.W4.T
        dz_via_h2 = dh2 * (1.0 - h2 ** 2)
        dW3 = z.T @ dz_via_h2
        db3 = dz_via_h2.sum(0)
        dz = dz_via_h2 @ self.W3.T
        dW2 = h1.T @ dz
        db2 = dz.sum(0)
        dh1 = dz @ self.W2.T
        dz1 = dh1 * (1.0 - h1 ** 2)
        dW1 = x_in.T @ dz1
        db1 = dz1.sum(0)
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                      "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                  clip=5.0):
        total = 0.0
        for g in grads.values():
            total += float((g ** 2).sum())
        norm = np.sqrt(total)
        if norm > clip:
            scale = clip / (norm + 1e-12)
            for k in grads:
                grads[k] = grads[k] * scale
        self.t_step += 1
        bc1 = 1.0 - beta1 ** self.t_step
        bc2 = 1.0 - beta2 ** self.t_step
        for k, p in self.params().items():
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ----------------------------------------------------------------------
# M: small numpy LSTM (the "DoomRNN" world model).
# Predicts (z_{t+1}, r_{t+1}, done_logit_{t+1}) from (z_t, a_t).
# ----------------------------------------------------------------------

@dataclass
class M_LSTM:
    """Single-layer LSTM. Hand-coded forward + BPTT for training."""

    in_dim: int   # z_dim + n_actions (one-hot)
    z_dim: int
    hidden: int
    rng: np.random.Generator

    def __post_init__(self):
        s = np.sqrt(1.0 / (self.in_dim + self.hidden))
        # combined input for [i, f, g, o] gates
        self.W = self.rng.normal(0, s, (self.in_dim + self.hidden, 4 * self.hidden))
        self.b = np.zeros(4 * self.hidden)
        # output heads
        s_h = np.sqrt(1.0 / self.hidden)
        self.W_z = self.rng.normal(0, s_h, (self.hidden, self.z_dim))
        self.b_z = np.zeros(self.z_dim)
        self.W_r = self.rng.normal(0, s_h, (self.hidden, 1))
        self.b_r = np.zeros(1)
        self.W_d = self.rng.normal(0, s_h, (self.hidden, 1))
        self.b_d = np.zeros(1)
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self):
        return {
            "W": self.W, "b": self.b,
            "W_z": self.W_z, "b_z": self.b_z,
            "W_r": self.W_r, "b_r": self.b_r,
            "W_d": self.W_d, "b_d": self.b_d,
        }

    @staticmethod
    def _sigmoid(x):
        # numerically stable sigmoid
        out = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                       np.exp(x) / (1.0 + np.exp(x)))
        return out

    def lstm_step(self, x_t, h_prev, c_prev):
        """Single LSTM cell step (no output heads)."""
        cat = np.concatenate([x_t, h_prev], axis=-1)
        gates = cat @ self.W + self.b
        H = self.hidden
        i_g = self._sigmoid(gates[..., 0:H])
        f_g = self._sigmoid(gates[..., H:2*H])
        g_g = np.tanh(gates[..., 2*H:3*H])
        o_g = self._sigmoid(gates[..., 3*H:4*H])
        c = f_g * c_prev + i_g * g_g
        h = o_g * np.tanh(c)
        return h, c, (i_g, f_g, g_g, o_g, cat)

    def predict_step(self, h):
        """Read out (z_pred, r_pred, done_logit) from hidden h."""
        z_pred = h @ self.W_z + self.b_z
        r_pred = (h @ self.W_r + self.b_r).reshape(-1)
        d_logit = (h @ self.W_d + self.b_d).reshape(-1)
        return z_pred, r_pred, d_logit

    def init_state(self, batch=1):
        return (np.zeros((batch, self.hidden)),
                np.zeros((batch, self.hidden)))

    # -------- BPTT training over one length-T sequence per episode --------

    def train_step(self, X_seq, A_seq, Z_target_seq, R_target_seq,
                   D_target_seq, mask_seq, lr=1e-3):
        """Train M on a batch of length-T sequences.

        Shapes:
          X_seq         (T, B, z_dim)        z_t inputs
          A_seq         (T, B, n_actions)    one-hot action a_t
          Z_target_seq  (T, B, z_dim)        target z_{t+1}
          R_target_seq  (T, B)               target r_{t+1}
          D_target_seq  (T, B)               target done_{t+1} (0/1)
          mask_seq      (T, B)               1 where transition is valid

        Returns scalar loss; applies one Adam update.
        """
        T, B, _ = X_seq.shape
        H = self.hidden
        h, c = self.init_state(batch=B)
        caches = []
        z_preds = np.zeros((T, B, self.z_dim))
        r_preds = np.zeros((T, B))
        d_logits = np.zeros((T, B))
        for t in range(T):
            x_t = np.concatenate([X_seq[t], A_seq[t]], axis=-1)
            h_new, c_new, gate_cache = self.lstm_step(x_t, h, c)
            z_p = h_new @ self.W_z + self.b_z
            r_p = (h_new @ self.W_r + self.b_r).reshape(B)
            d_l = (h_new @ self.W_d + self.b_d).reshape(B)
            z_preds[t] = z_p
            r_preds[t] = r_p
            d_logits[t] = d_l
            caches.append((x_t, h, c, h_new, c_new, gate_cache))
            h, c = h_new, c_new

        # Losses
        m_b = mask_seq[..., None]
        n_valid = max(float(mask_seq.sum()), 1.0)
        z_err = (z_preds - Z_target_seq) * m_b
        loss_z = float((z_err ** 2).sum() / (n_valid * self.z_dim))
        r_err = (r_preds - R_target_seq) * mask_seq
        loss_r = float((r_err ** 2).sum() / n_valid)
        d_prob = self._sigmoid(d_logits)
        eps = 1e-7
        bce = -(D_target_seq * np.log(d_prob + eps) +
                (1 - D_target_seq) * np.log(1 - d_prob + eps)) * mask_seq
        loss_d = float(bce.sum() / n_valid)
        loss = loss_z + loss_r + loss_d

        # Backprop
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        # output-head grads
        d_z_pred = 2.0 * z_err / (n_valid * self.z_dim)        # (T,B,z)
        d_r_pred = 2.0 * r_err / n_valid                       # (T,B)
        d_d_logit = (d_prob - D_target_seq) * mask_seq / n_valid

        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            x_t, h_prev, c_prev, h_new, c_new, gate_cache = caches[t]
            i_g, f_g, g_g, o_g, cat = gate_cache
            # output heads
            grads["W_z"] += h_new.T @ d_z_pred[t]
            grads["b_z"] += d_z_pred[t].sum(0)
            grads["W_r"] += h_new.T @ d_r_pred[t][:, None]
            grads["b_r"] += d_r_pred[t].sum(0, keepdims=True)
            grads["W_d"] += h_new.T @ d_d_logit[t][:, None]
            grads["b_d"] += d_d_logit[t].sum(0, keepdims=True)
            dh = (d_z_pred[t] @ self.W_z.T
                  + d_r_pred[t][:, None] @ self.W_r.T
                  + d_d_logit[t][:, None] @ self.W_d.T)
            dh = dh + dh_next
            # h = o * tanh(c)
            tanh_c = np.tanh(c_new)
            do = dh * tanh_c
            dc = dh * o_g * (1 - tanh_c ** 2) + dc_next
            di = dc * g_g
            df = dc * c_prev
            dg = dc * i_g
            d_c_prev = dc * f_g
            # gate activations
            d_i_pre = di * i_g * (1 - i_g)
            d_f_pre = df * f_g * (1 - f_g)
            d_g_pre = dg * (1 - g_g ** 2)
            d_o_pre = do * o_g * (1 - o_g)
            d_gates = np.concatenate([d_i_pre, d_f_pre, d_g_pre, d_o_pre],
                                     axis=-1)
            grads["W"] += cat.T @ d_gates
            grads["b"] += d_gates.sum(0)
            d_cat = d_gates @ self.W.T
            d_h_prev = d_cat[:, self.in_dim:]
            dh_next = d_h_prev
            dc_next = d_c_prev

        # global-norm clip
        total = 0.0
        for g in grads.values():
            total += float((g ** 2).sum())
        norm = np.sqrt(total)
        clip = 5.0
        if norm > clip:
            scale = clip / (norm + 1e-12)
            for k in grads:
                grads[k] = grads[k] * scale

        # Adam
        self.t_step += 1
        beta1, beta2, eps_a = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1 ** self.t_step
        bc2 = 1.0 - beta2 ** self.t_step
        for k, p in self.params().items():
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            p -= lr * m_hat / (np.sqrt(v_hat) + eps_a)

        return loss, loss_z, loss_r, loss_d


# ----------------------------------------------------------------------
# C: tiny linear controller (Ha & Schmidhuber 2018 §A.6 -- C is just a
# linear layer over [z, h]).
# ----------------------------------------------------------------------

@dataclass
class C_Controller:
    """Either a pure-linear policy (paper-faithful, c_hidden = 0) or a small
    1-hidden-layer tanh MLP (c_hidden > 0). Parameters serialise to a flat
    vector for ES."""
    z_dim: int
    h_dim: int
    n_actions: int
    c_hidden: int = 0

    def __post_init__(self):
        in_dim = self.z_dim + self.h_dim
        if self.c_hidden > 0:
            self.W1 = np.zeros((in_dim, self.c_hidden))
            self.b1 = np.zeros(self.c_hidden)
            self.W2 = np.zeros((self.c_hidden, self.n_actions))
            self.b2 = np.zeros(self.n_actions)
            self._shapes = [self.W1.shape, self.b1.shape,
                            self.W2.shape, self.b2.shape]
            self._sizes = [int(np.prod(s)) for s in self._shapes]
        else:
            self.W = np.zeros((in_dim, self.n_actions))
            self.b = np.zeros(self.n_actions)

    @property
    def n_params(self) -> int:
        if self.c_hidden > 0:
            return sum(self._sizes)
        return self.W.size + self.b.size

    def get_flat(self) -> np.ndarray:
        if self.c_hidden > 0:
            return np.concatenate([
                self.W1.flatten(), self.b1, self.W2.flatten(), self.b2,
            ])
        return np.concatenate([self.W.flatten(), self.b])

    def set_flat(self, theta: np.ndarray) -> None:
        if self.c_hidden > 0:
            offsets = np.cumsum([0] + self._sizes)
            chunks = [theta[offsets[i]:offsets[i+1]].copy()
                      for i in range(len(self._sizes))]
            self.W1 = chunks[0].reshape(self._shapes[0])
            self.b1 = chunks[1]
            self.W2 = chunks[2].reshape(self._shapes[2])
            self.b2 = chunks[3]
        else:
            n_W = self.W.size
            self.W = theta[:n_W].reshape(self.W.shape).copy()
            self.b = theta[n_W:].copy()

    def act(self, z: np.ndarray, h: np.ndarray, greedy: bool = True,
            rng: np.random.Generator | None = None) -> int:
        x = np.concatenate([z.flatten(), h.flatten()])
        if self.c_hidden > 0:
            h1 = np.tanh(x @ self.W1 + self.b1)
            logits = h1 @ self.W2 + self.b2
        else:
            logits = x @ self.W + self.b
        if greedy:
            return int(np.argmax(logits))
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        if rng is None:
            return int(np.argmax(p))
        return int(rng.choice(self.n_actions, p=p))


# ----------------------------------------------------------------------
# Rollout helpers
# ----------------------------------------------------------------------

def random_episode(env: DodgingEnv, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random-policy episode. Returns (obs[T+1, obs_dim], a[T], r[T], done[T])."""
    obs_list = [env.reset()]
    a_list, r_list, d_list = [], [], []
    while not env.done:
        a = int(rng.integers(0, 3))
        o, r, d = env.step(a)
        obs_list.append(o)
        a_list.append(a)
        r_list.append(r)
        d_list.append(1.0 if d else 0.0)
    return (np.stack(obs_list), np.array(a_list, dtype=np.int64),
            np.array(r_list, dtype=np.float32),
            np.array(d_list, dtype=np.float32))


def real_rollout(env: DodgingEnv, V: V_Autoencoder, M: M_LSTM,
                 C: C_Controller, rng: np.random.Generator,
                 n_actions: int = 3) -> Tuple[float, int, List[Dict]]:
    """Run C in the real env. M is run alongside to maintain h_t; V encodes
    each obs into z. Returns (total_reward, length, frames-for-gif).
    """
    obs = env.reset()
    z = V.encode(obs[None])
    h, c = M.init_state(batch=1)
    total = 0.0
    frames: List[Dict] = []
    while not env.done:
        a = C.act(z[0], h[0], greedy=True, rng=rng)
        # save snapshot for animation BEFORE stepping
        frames.append({
            "agent_x": env.agent_x,
            "fireballs": list(env.fireballs),
            "t": env.t,
            "action": a,
        })
        o2, r, done = env.step(a)
        a_oh = np.zeros(n_actions); a_oh[a] = 1.0
        x_t = np.concatenate([z[0], a_oh])[None]
        h, c, _ = M.lstm_step(x_t, h, c)
        z = V.encode(o2[None])
        total += r
    frames.append({
        "agent_x": env.agent_x,
        "fireballs": list(env.fireballs),
        "t": env.t,
        "action": -1,
    })
    return total, env.t, frames


def dream_rollout(M: M_LSTM, C: C_Controller, z0: np.ndarray,
                  max_steps: int, n_actions: int = 3,
                  rng: np.random.Generator | None = None,
                  z_noise: float = 0.0,
                  done_threshold: float = 0.5,
                  ) -> Tuple[float, int]:
    """Run C entirely INSIDE M's hallucination (no real env). Returns
    (total_predicted_reward, length_until_predicted_done).

    `z_noise` controls the Ha & Schmidhuber 2018 "temperature" trick:
    Gaussian noise added to z_pred each step prevents C from exploiting
    deterministic idiosyncrasies of M's dream and improves real-env
    transfer. Paper used temperature=1.15 on the MDN-RNN's mixture
    sampling; we approximate with additive Gaussian on z_pred.
    """
    h, c = M.init_state(batch=1)
    z = z0.reshape(1, -1).copy()
    total = 0.0
    for t in range(max_steps):
        a = C.act(z[0], h[0], greedy=True, rng=rng)
        a_oh = np.zeros(n_actions); a_oh[a] = 1.0
        x_t = np.concatenate([z[0], a_oh])[None]
        h, c, _ = M.lstm_step(x_t, h, c)
        z_pred, r_pred, d_logit = M.predict_step(h)
        total += float(r_pred[0])
        d_prob = M._sigmoid(d_logit[0])
        z = z_pred
        if z_noise > 0.0 and rng is not None:
            z = z + rng.normal(0, z_noise, size=z.shape)
        if d_prob > done_threshold:
            return total, t + 1
    return total, max_steps


# ----------------------------------------------------------------------
# ES (numpy analog of CMA-ES; fixed-sigma OpenAI-ES)
# ----------------------------------------------------------------------

def es_step(theta: np.ndarray, scores_fn, rng: np.random.Generator,
            pop: int = 16, sigma: float = 0.2, lr: float = 0.05
            ) -> Tuple[np.ndarray, np.ndarray]:
    """One OpenAI-ES update. scores_fn maps an array of perturbed thetas
    (pop, n_params) to a vector of scalar scores (pop,)."""
    eps = rng.normal(0, 1, size=(pop, theta.size))
    perturbed = theta[None, :] + sigma * eps
    R = scores_fn(perturbed)
    R_centred = R - R.mean()
    if R.std() > 1e-8:
        R_centred = R_centred / (R.std() + 1e-8)
    grad = (eps.T @ R_centred) / (pop * sigma)
    theta_new = theta + lr * grad
    return theta_new, R


# ----------------------------------------------------------------------
# Train pipeline
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    # Env. Compact dodging task -- 5-wide, 5-tall, at most one fireball in
    # the air at a time, spawned from a random column at row 0 every step
    # the field is empty. Random policy survives ~6-10 steps before a hit;
    # an "always dodge to the side opposite the falling fireball" policy
    # can survive indefinitely. Max steps caps the survival reward so we
    # actually see the gap between random / dream / real.
    W: int = 5
    H: int = 5
    max_fireballs: int = 1
    spawn_prob: float = 1.0
    max_steps: int = 60
    # V
    z_dim: int = 8
    v_hidden: int = 32
    v_train_steps: int = 800
    v_lr: float = 2e-3
    v_batch: int = 64
    # M
    m_hidden: int = 16
    m_train_steps: int = 2500
    m_lr: float = 3e-3
    m_seq_len: int = 20
    m_batch: int = 16
    # Real-data buffer for V/M. Also adds a second pass of action-aware
    # episodes (rolled out by a *partially* trained C-dream) so M sees the
    # transitions the controller actually visits.
    n_random_episodes: int = 200
    n_extra_iters: int = 0     # how many extra "collect-with-C, retrain-M,
                               # retrain-C" iterations after the initial run.
                               # set to 0 -- the small env's transitions
                               # are simple enough that random-policy data
                               # already covers the relevant state space.
    c_hidden: int = 16         # if >0, C is a 1-hidden-layer MLP instead of
                               # the paper's pure linear policy. The paper's
                               # full pipeline pairs a tiny linear C with a
                               # very expressive V (CNN-VAE) and M
                               # (MDN-RNN); to compensate for our weaker V/M,
                               # we let C have a single tanh hidden layer.
    # C
    n_actions: int = 3
    es_iters: int = 100
    es_pop: int = 24
    es_sigma: float = 0.15
    es_lr: float = 0.10
    dream_max_steps: int = 40    # cap dream rollouts close to the
                                 # *distribution* of M's training data
                                 # (random policy's mean ep length ~22).
                                 # Letting the dream run far longer just
                                 # accumulates compounding-error and gives
                                 # C an unreliable training signal.
    es_z0_samples: int = 3     # average over this many initial z's per
                               # generation to reduce ES gradient variance
    dream_z_noise: float = 0.15  # paper's "temperature" trick: Gaussian
                                 # noise on z_pred prevents C from exploiting
                                 # M's deterministic idiosyncrasies; this
                                 # robustness term is what makes the dream-
                                 # trained C transfer to the real env.
                                 # Paper used temperature=1.15 on the
                                 # MDN-RNN's mixture sampling; we tune the
                                 # additive Gaussian sigma to give a similar
                                 # effect (mild blur of M's predictions).
    dream_done_threshold: float = 0.4   # slightly aggressive done threshold
                                        # so the dream doesn't let
                                        # near-collisions slip through
    # Eval
    eval_every: int = 5
    eval_episodes: int = 5
    # Whether to also train a "direct in real env" baseline C for comparison
    train_baseline: bool = True
    baseline_es_iters: int = 60


def collect_random_data(env: DodgingEnv, cfg: RunConfig,
                        rng: np.random.Generator):
    obs_seqs, act_seqs, rew_seqs, done_seqs = [], [], [], []
    for _ in range(cfg.n_random_episodes):
        env.seed(rng)
        o, a, r, d = random_episode(env, rng)
        obs_seqs.append(o)
        act_seqs.append(a)
        rew_seqs.append(r)
        done_seqs.append(d)
    return obs_seqs, act_seqs, rew_seqs, done_seqs


def train_V(V: V_Autoencoder, obs_seqs: List[np.ndarray], cfg: RunConfig,
            rng: np.random.Generator):
    """Plain MSE autoencoder on random-policy obs."""
    all_obs = np.concatenate(obs_seqs, axis=0).astype(np.float32)
    losses = []
    for step in range(cfg.v_train_steps):
        idx = rng.integers(0, all_obs.shape[0], size=cfg.v_batch)
        x = all_obs[idx]
        x_hat, cache = V.forward(x)
        loss, grads = V.mse_grad(x_hat, x, cache)
        V.adam_step(grads, lr=cfg.v_lr)
        losses.append(loss)
    return losses


def encode_seqs(V: V_Autoencoder, obs_seqs):
    return [V.encode(o.astype(np.float32)) for o in obs_seqs]


def make_seq_batch(z_seqs, act_seqs, rew_seqs, done_seqs,
                   cfg: RunConfig, rng: np.random.Generator):
    """Sample a batch of length-T windows from the episode buffer.

    Each window is constructed so that target z_{t+1}, r_{t+1}, done_{t+1}
    are aligned with input (z_t, a_t).
    """
    T = cfg.m_seq_len
    B = cfg.m_batch
    z_dim = cfg.z_dim
    n_act = cfg.n_actions
    X = np.zeros((T, B, z_dim))
    A = np.zeros((T, B, n_act))
    Zt = np.zeros((T, B, z_dim))
    Rt = np.zeros((T, B))
    Dt = np.zeros((T, B))
    M_mask = np.zeros((T, B))
    for b in range(B):
        # pick an episode with at least 2 transitions
        # (so we have at least one (z_t, z_{t+1}) pair)
        for _ in range(20):
            ep = int(rng.integers(0, len(z_seqs)))
            if z_seqs[ep].shape[0] >= 2:
                break
        z_e = z_seqs[ep]    # (L+1, z_dim)
        a_e = act_seqs[ep]  # (L,)
        r_e = rew_seqs[ep]
        d_e = done_seqs[ep]
        L = a_e.shape[0]
        if L == 0:
            continue
        if L >= T:
            t0 = int(rng.integers(0, L - T + 1))
            tT = t0 + T
            valid = T
        else:
            t0 = 0
            tT = L
            valid = L
        for k in range(valid):
            X[k, b] = z_e[t0 + k]
            A[k, b, a_e[t0 + k]] = 1.0
            Zt[k, b] = z_e[t0 + k + 1]
            Rt[k, b] = r_e[t0 + k]
            Dt[k, b] = d_e[t0 + k]
            M_mask[k, b] = 1.0
    return X, A, Zt, Rt, Dt, M_mask


def train_M(M: M_LSTM, z_seqs, act_seqs, rew_seqs, done_seqs,
            cfg: RunConfig, rng: np.random.Generator):
    losses = []
    for step in range(cfg.m_train_steps):
        X, A, Zt, Rt, Dt, Mask = make_seq_batch(
            z_seqs, act_seqs, rew_seqs, done_seqs, cfg, rng)
        l, lz, lr, ld = M.train_step(X, A, Zt, Rt, Dt, Mask, lr=cfg.m_lr)
        losses.append({"total": l, "z": lz, "r": lr, "done": ld})
    return losses


def evaluate_in_real_env(V, M, C, env: DodgingEnv, cfg: RunConfig,
                         rng: np.random.Generator) -> Tuple[float, float]:
    rets, lens = [], []
    for _ in range(cfg.eval_episodes):
        env.seed(rng)
        ret, length, _ = real_rollout(env, V, M, C, rng,
                                      n_actions=cfg.n_actions)
        rets.append(ret); lens.append(length)
    return float(np.mean(rets)), float(np.mean(lens))


def evaluate_in_dream(M, C, z0_pool: np.ndarray, cfg: RunConfig,
                      rng: np.random.Generator) -> Tuple[float, float]:
    rets, lens = [], []
    for _ in range(cfg.eval_episodes):
        z0 = z0_pool[rng.integers(0, z0_pool.shape[0])]
        ret, length = dream_rollout(M, C, z0, cfg.dream_max_steps,
                                    n_actions=cfg.n_actions, rng=rng)
        rets.append(ret); lens.append(length)
    return float(np.mean(rets)), float(np.mean(lens))


def train_C_in_dreams(V, M, env: DodgingEnv, cfg: RunConfig,
                      z0_pool: np.ndarray, rng: np.random.Generator):
    C = C_Controller(z_dim=cfg.z_dim, h_dim=cfg.m_hidden,
                     n_actions=cfg.n_actions,
                     c_hidden=cfg.c_hidden)
    theta = C.get_flat()
    n_params = theta.size

    history = {
        "iter": [], "dream_ret": [], "dream_len": [],
        "real_iter": [], "real_ret": [], "real_len": [],
        "es_pop_mean": [], "es_pop_max": [],
    }

    es_rng = np.random.default_rng(cfg.seed + 1000)

    def scores_fn(perturbed):
        scores = np.zeros(perturbed.shape[0])
        # Average over a few fixed z0's per generation so the ES gradient
        # is over the same evaluation distribution for all candidates --
        # within a generation each candidate sees the same set of z0's,
        # so comparing them is fair, but variance across generations is
        # reduced versus a single z0.
        z0_idx = es_rng.integers(0, z0_pool.shape[0],
                                 size=max(1, cfg.es_z0_samples))
        for k in range(perturbed.shape[0]):
            C.set_flat(perturbed[k])
            tot = 0.0
            for j in z0_idx:
                ret, _ = dream_rollout(
                    M, C, z0_pool[j], cfg.dream_max_steps,
                    n_actions=cfg.n_actions, rng=es_rng,
                    z_noise=cfg.dream_z_noise,
                    done_threshold=cfg.dream_done_threshold,
                )
                tot += ret
            scores[k] = tot / len(z0_idx)
        return scores

    for it in range(cfg.es_iters):
        theta, R = es_step(theta, scores_fn, es_rng,
                           pop=cfg.es_pop, sigma=cfg.es_sigma, lr=cfg.es_lr)
        C.set_flat(theta)
        # eval in dream
        d_ret, d_len = evaluate_in_dream(M, C, z0_pool, cfg, es_rng)
        history["iter"].append(it)
        history["dream_ret"].append(d_ret)
        history["dream_len"].append(d_len)
        history["es_pop_mean"].append(float(R.mean()))
        history["es_pop_max"].append(float(R.max()))
        # eval in real env (zero-shot transfer check)
        if it % cfg.eval_every == 0 or it == cfg.es_iters - 1:
            r_ret, r_len = evaluate_in_real_env(V, M, C, env, cfg, es_rng)
            history["real_iter"].append(it)
            history["real_ret"].append(r_ret)
            history["real_len"].append(r_len)
    return C, history, theta


def train_C_directly_in_real(V, M, env: DodgingEnv, cfg: RunConfig,
                             rng: np.random.Generator):
    """Baseline: train the same controller architecture but with ES rollouts
    in the REAL env (no dream)."""
    C = C_Controller(z_dim=cfg.z_dim, h_dim=cfg.m_hidden,
                     n_actions=cfg.n_actions,
                     c_hidden=cfg.c_hidden)
    theta = C.get_flat()

    history = {
        "iter": [], "real_ret": [], "real_len": [],
        "es_pop_mean": [], "es_pop_max": [],
    }

    es_rng = np.random.default_rng(cfg.seed + 2000)

    def scores_fn(perturbed):
        scores = np.zeros(perturbed.shape[0])
        for k in range(perturbed.shape[0]):
            C.set_flat(perturbed[k])
            env.seed(es_rng)
            ret, _, _ = real_rollout(env, V, M, C, es_rng,
                                     n_actions=cfg.n_actions)
            scores[k] = ret
        return scores

    for it in range(cfg.baseline_es_iters):
        theta, R = es_step(theta, scores_fn, es_rng,
                           pop=cfg.es_pop, sigma=cfg.es_sigma, lr=cfg.es_lr)
        C.set_flat(theta)
        # eval in real env
        r_ret, r_len = evaluate_in_real_env(V, M, C, env, cfg, es_rng)
        history["iter"].append(it)
        history["real_ret"].append(r_ret)
        history["real_len"].append(r_len)
        history["es_pop_mean"].append(float(R.mean()))
        history["es_pop_max"].append(float(R.max()))
    return C, history, theta


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------

def collect_with_policy(env: DodgingEnv, V, M, C, n_eps: int,
                        rng: np.random.Generator, n_actions: int = 3
                        ) -> Tuple[List[np.ndarray], List[np.ndarray],
                                   List[np.ndarray], List[np.ndarray]]:
    """Roll the trained C in the real env n_eps times and return
    (obs_seqs, act_seqs, rew_seqs, done_seqs) just like collect_random_data.
    Mixes some action noise so transitions cover more states.
    """
    obs_seqs, act_seqs, rew_seqs, done_seqs = [], [], [], []
    for _ in range(n_eps):
        env.seed(rng)
        obs = env.reset()
        z = V.encode(obs[None])
        h, c = M.init_state(batch=1)
        os_, as_, rs_, ds_ = [obs], [], [], []
        while not env.done:
            if rng.random() < 0.2:
                a = int(rng.integers(0, n_actions))   # 20% action noise
            else:
                a = C.act(z[0], h[0], greedy=True, rng=rng)
            o2, r, done = env.step(a)
            a_oh = np.zeros(n_actions); a_oh[a] = 1.0
            x_t = np.concatenate([z[0], a_oh])[None]
            h, c, _ = M.lstm_step(x_t, h, c)
            z = V.encode(o2[None])
            os_.append(o2); as_.append(a); rs_.append(r)
            ds_.append(1.0 if done else 0.0)
        obs_seqs.append(np.stack(os_))
        act_seqs.append(np.array(as_, dtype=np.int64))
        rew_seqs.append(np.array(rs_, dtype=np.float32))
        done_seqs.append(np.array(ds_, dtype=np.float32))
    return obs_seqs, act_seqs, rew_seqs, done_seqs


def train(cfg: RunConfig, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    env = DodgingEnv(W=cfg.W, H=cfg.H,
                     max_fireballs=cfg.max_fireballs,
                     spawn_prob=cfg.spawn_prob,
                     max_steps=cfg.max_steps)
    obs_dim = env.obs_dim

    t0 = time.time()
    if verbose:
        print("[1/6] collecting random-policy episodes...")
    obs_seqs, act_seqs, rew_seqs, done_seqs = collect_random_data(
        env, cfg, rng)
    n_transitions = sum(a.shape[0] for a in act_seqs)
    if verbose:
        print(f"      {len(obs_seqs)} eps, {n_transitions} transitions, "
              f"mean_len={n_transitions/len(obs_seqs):.1f}")

    # Random-policy survival baseline -- the "C is constant" floor.
    rand_lens = [a.shape[0] for a in act_seqs]
    rand_baseline_mean = float(np.mean(rand_lens))
    rand_baseline_std = float(np.std(rand_lens))

    if verbose:
        print(f"[2/6] training V (autoencoder) for {cfg.v_train_steps} steps...")
    V = V_Autoencoder(in_dim=obs_dim, z_dim=cfg.z_dim,
                      hidden=cfg.v_hidden,
                      rng=np.random.default_rng(cfg.seed + 11))
    v_losses = train_V(V, obs_seqs, cfg, rng)
    if verbose:
        print(f"      V loss start={v_losses[0]:.5f}  end={v_losses[-1]:.5f}")

    if verbose:
        print("[3/6] encoding obs to z and training M (LSTM world model)...")
    z_seqs = encode_seqs(V, obs_seqs)
    M = M_LSTM(in_dim=cfg.z_dim + cfg.n_actions, z_dim=cfg.z_dim,
               hidden=cfg.m_hidden,
               rng=np.random.default_rng(cfg.seed + 21))
    m_losses = train_M(M, z_seqs, act_seqs, rew_seqs, done_seqs, cfg, rng)
    if verbose:
        l0 = m_losses[0]; lf = m_losses[-1]
        print(f"      M loss start: total={l0['total']:.4f} "
              f"(z={l0['z']:.4f} r={l0['r']:.4f} d={l0['done']:.4f})")
        print(f"      M loss end  : total={lf['total']:.4f} "
              f"(z={lf['z']:.4f} r={lf['r']:.4f} d={lf['done']:.4f})")

    # Pool of initial z's for dream rollouts (encode every ep's first obs)
    z0_pool = np.stack([z[0] for z in z_seqs])

    if verbose:
        print(f"[4/6] training C in dreams of M for {cfg.es_iters} ES iters "
              f"(round 1)...")
    C_dream, dream_hist, theta_dream = train_C_in_dreams(
        V, M, env, cfg, z0_pool, rng)

    # Iterative refinement (Ha & Schmidhuber 2018, §A): collect data with the
    # current dream-trained C, retrain M on the union of random + on-policy
    # transitions, then retrain C in the new dream. The on-policy data fixes
    # M's distribution-shift problem (random-policy trajectories don't cover
    # the "agent stays alive 50+ steps" regime that C operates in).
    if cfg.n_extra_iters > 0:
        for round_i in range(1, cfg.n_extra_iters + 1):
            if verbose:
                print(f"[5/6] iterative round {round_i}: "
                      f"collecting on-policy data with current C_dream...")
            extra_obs, extra_act, extra_rew, extra_done = collect_with_policy(
                env, V, M, C_dream,
                n_eps=max(20, cfg.n_random_episodes // 4),
                rng=rng, n_actions=cfg.n_actions,
            )
            extra_n = sum(a.shape[0] for a in extra_act)
            extra_mean = (extra_n / max(1, len(extra_act)))
            if verbose:
                print(f"      collected {len(extra_obs)} eps, {extra_n} "
                      f"transitions, mean_len={extra_mean:.1f}")
            obs_seqs += extra_obs
            act_seqs += extra_act
            rew_seqs += extra_rew
            done_seqs += extra_done
            # Re-encode (V stays fixed since recon is already low) and retrain M.
            z_seqs = encode_seqs(V, obs_seqs)
            if verbose:
                print(f"      retraining M for {cfg.m_train_steps} more "
                      f"steps on combined data...")
            extra_m_losses = train_M(M, z_seqs, act_seqs, rew_seqs,
                                     done_seqs, cfg, rng)
            m_losses += extra_m_losses
            if verbose:
                lf = m_losses[-1]
                print(f"      M loss end  : total={lf['total']:.4f} "
                      f"(z={lf['z']:.4f} r={lf['r']:.4f} d={lf['done']:.4f})")
            z0_pool = np.stack([z[0] for z in z_seqs])
            if verbose:
                print(f"      retraining C in updated dream "
                      f"({cfg.es_iters} ES iters)...")
            C_dream, dream_hist_new, theta_dream = train_C_in_dreams(
                V, M, env, cfg, z0_pool, rng)
            # Append iter histories with offset so the figure stays one curve
            offset = (dream_hist["iter"][-1] + 1) if dream_hist["iter"] else 0
            for k, v in dream_hist_new.items():
                if k in ("iter", "real_iter"):
                    dream_hist[k] += [x + offset for x in v]
                else:
                    dream_hist[k] += v

    if cfg.train_baseline:
        if verbose:
            print(f"[6/6] training baseline C in REAL env for "
                  f"{cfg.baseline_es_iters} ES iters...")
        C_real, real_hist, theta_real = train_C_directly_in_real(
            V, M, env, cfg, rng)
    else:
        C_real, real_hist, theta_real = None, None, None

    # Final evaluations: more episodes for confidence
    final_rng = np.random.default_rng(cfg.seed + 9999)
    n_final_eval = max(50, cfg.eval_episodes * 4)

    # Dream-trained C in real env
    rets_dream, lens_dream = [], []
    for _ in range(n_final_eval):
        env.seed(final_rng)
        ret, length, _ = real_rollout(env, V, M, C_dream, final_rng,
                                      n_actions=cfg.n_actions)
        rets_dream.append(ret); lens_dream.append(length)

    rets_real, lens_real = [], []
    if C_real is not None:
        for _ in range(n_final_eval):
            env.seed(final_rng)
            ret, length, _ = real_rollout(env, V, M, C_real, final_rng,
                                          n_actions=cfg.n_actions)
            rets_real.append(ret); lens_real.append(length)

    rets_random, lens_random = [], []
    for _ in range(n_final_eval):
        env.seed(final_rng)
        o, a, r, d = random_episode(env, final_rng)
        rets_random.append(float(r.sum())); lens_random.append(int(a.shape[0]))

    wall = time.time() - t0
    if verbose:
        print()
        print(f"=== Final eval (n={n_final_eval} eps each) ===")
        print(f"  random policy : mean_len={np.mean(lens_random):.2f} "
              f"+/- {np.std(lens_random):.2f}")
        print(f"  C_dream       : mean_len={np.mean(lens_dream):.2f} "
              f"+/- {np.std(lens_dream):.2f}   "
              f"(trained INSIDE M's dream only)")
        if C_real is not None:
            print(f"  C_real (base) : mean_len={np.mean(lens_real):.2f} "
                  f"+/- {np.std(lens_real):.2f}   "
                  f"(trained ES in real env, reference)")
        print(f"  wallclock={wall:.1f}s   git={git_hash()}")

    summary = {
        "config": asdict(cfg),
        "env_metadata": env_metadata(),
        "data": {
            "n_random_episodes": len(obs_seqs),
            "n_random_transitions": n_transitions,
            "rand_baseline_len_mean": rand_baseline_mean,
            "rand_baseline_len_std": rand_baseline_std,
        },
        "v_losses": [float(x) for x in v_losses],
        "m_losses": [{k: float(v) for k, v in d.items()} for d in m_losses],
        "dream_history": dream_hist,
        "real_history": real_hist,
        "final_eval": {
            "n_episodes": n_final_eval,
            "C_dream_mean_len": float(np.mean(lens_dream)),
            "C_dream_std_len": float(np.std(lens_dream)),
            "C_dream_mean_ret": float(np.mean(rets_dream)),
            "C_real_mean_len": float(np.mean(lens_real)) if C_real else None,
            "C_real_std_len": float(np.std(lens_real)) if C_real else None,
            "C_real_mean_ret": float(np.mean(rets_real)) if C_real else None,
            "random_mean_len": float(np.mean(lens_random)),
            "random_std_len": float(np.std(lens_random)),
            "C_dream_lens": lens_dream,
            "C_real_lens": lens_real if C_real else [],
            "random_lens": lens_random,
        },
        "weights": {
            "theta_C_dream": theta_dream.tolist(),
            "theta_C_real": theta_real.tolist() if C_real else None,
        },
        "wallclock_sec": wall,
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
    p.add_argument("--save-json", type=str, default=None,
                   help="Path to dump full summary JSON.")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip the direct-trained C baseline.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed)
    if args.quick:
        cfg.n_random_episodes = 30
        cfg.v_train_steps = 200
        cfg.m_train_steps = 200
        cfg.es_iters = 20
        cfg.baseline_es_iters = 20
        cfg.es_pop = 8
        cfg.eval_episodes = 3
    if args.no_baseline:
        cfg.train_baseline = False

    summary = train(cfg, verbose=not args.quiet)

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                    exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.save_json}")


if __name__ == "__main__":
    main()
