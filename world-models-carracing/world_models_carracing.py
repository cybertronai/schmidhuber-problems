"""world-models-carracing -- Ha & Schmidhuber, *Recurrent World Models
Facilitate Policy Evolution*, NeurIPS 2018 (arXiv:1803.10122 / 1809.01999).

The paper trains three modules separately on OpenAI Gym CarRacing-v0:

    V  : Convolutional VAE        64x64x3 RGB obs  ->  z in R^32
    M  : MDN-LSTM world model     (z_t, a_t)       ->  next-z mixture
    C  : linear controller        (z, h_M) -> a    evolved with CMA-ES

then "the same network" rolls out the policy on the env. v1 of this catalog
forbids gym/PyBox2D installs (SPEC issue #1, RL-stub rule). This stub keeps
the V+M+C decomposition and the CMA-ES outer loop, but swaps CarRacing-v0 for
a hand-rolled numpy 2-D top-down racing track.

Numpy mini-env (substituting CarRacing-v0):
    * Track centerline = closed loop in R^2, generated from low-frequency
      sinusoids; rasterized once into a 200x200 binary track mask.
    * Car state = (x, y, theta, v); action = (steer, throttle), each in
      [-1, 1] after tanh.
    * Observation = 16x16 patch of the track mask, rotated to the car's
      forward frame (car always faces "up" in the patch).
    * Reward = forward arc-length progress along the centerline minus
      0.5 * (off-track distance). Episode ends if the car wanders more than
      2 * track-half-width from the centerline, or after t_max=120 steps.

Pipeline:
    1. Collect random-policy rollouts                      -> obs/action set
    2. Train V (linear AE 256->64->16->64->256)            -> z = V.encode(obs)
    3. Re-encode rollouts to z; train M (LSTM hidden=32)
       to predict z_{t+1} from (z_t, a_t)
    4. CMA-ES on linear controller W : R^48 -> R^2 (z|h_M -> action)
       with simplified rank-mu CMA-ES (isotropic sigma)

Architecture deviations from Ha & Schmidhuber (2018), all driven by the v1
"pure numpy, no torch, <5 min" constraint:

    paper                          | this stub
    ------------------------------ | --------------------------------
    Convolutional VAE              | 2-layer linear AE (no KL, no conv)
    MDN-LSTM (5 mixtures)          | deterministic LSTM (single mean)
    z dim 32, hidden 256           | z dim 16, hidden 32
    CarRacing-v0 (64x64x3, 3 act)  | numpy 2-D track (16x16x1, 2 act)
    CMA-ES popsize 64, gens 200    | simplified CMA-ES popsize 24, gens 30
    score >=900 over 100 trials    | mean fwd-progress > random baseline

These are documented again in README.md > Deviations.

CLI:
    python3 world_models_carracing.py --seed 0
    python3 world_models_carracing.py --seed 0 --quick
    python3 world_models_carracing.py --seed 0 --save-json out/run.json
"""
from __future__ import annotations

import argparse
import json
import math
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
# Numpy 2-D racing track (substitute for CarRacing-v0)
# ----------------------------------------------------------------------

@dataclass
class Track:
    """Closed loop track from low-frequency sinusoids.

    Centerline:
        r(s) = R + a1 * cos(4*pi*s + phi1) + a2 * cos(6*pi*s + phi2)
        x(s) = r(s) * cos(2*pi*s)
        y(s) = r(s) * sin(2*pi*s)
    half-width w defines the drivable corridor.

    The track is rasterized into a binary mask on a (grid_size x grid_size)
    array spanning world coords [-grid_extent, grid_extent]^2.
    """

    R: float = 8.0
    a1: float = 1.4
    phi1: float = 0.7
    a2: float = 0.6
    phi2: float = 1.9
    half_width: float = 1.4
    n_samples: int = 256
    grid_size: int = 200
    grid_extent: float = 12.0

    def __post_init__(self):
        s = np.linspace(0.0, 1.0, self.n_samples, endpoint=False)
        r = (self.R
             + self.a1 * np.cos(4.0 * np.pi * s + self.phi1)
             + self.a2 * np.cos(6.0 * np.pi * s + self.phi2))
        self.cx = r * np.cos(2.0 * np.pi * s)
        self.cy = r * np.sin(2.0 * np.pi * s)
        # tangent direction in world frame (used for spawn heading)
        dx = np.gradient(self.cx)
        dy = np.gradient(self.cy)
        n = np.hypot(dx, dy) + 1e-9
        self.tx = dx / n
        self.ty = dy / n
        # rasterize mask
        self._build_mask()

    def _build_mask(self):
        G = self.grid_size
        E = self.grid_extent
        xs = np.linspace(-E, E, G)
        ys = np.linspace(-E, E, G)
        X, Y = np.meshgrid(xs, ys, indexing="xy")  # X(row, col), Y(row, col)
        # squared distance to nearest centerline sample (chunked to save memory)
        flat_X = X.ravel()
        flat_Y = Y.ravel()
        n_pix = flat_X.shape[0]
        mask_flat = np.zeros(n_pix, dtype=np.float32)
        chunk = 4096
        cx = self.cx[None, :]
        cy = self.cy[None, :]
        thr = self.half_width ** 2
        for i in range(0, n_pix, chunk):
            xs_c = flat_X[i:i + chunk, None]
            ys_c = flat_Y[i:i + chunk, None]
            d2 = (xs_c - cx) ** 2 + (ys_c - cy) ** 2
            mask_flat[i:i + chunk] = (d2.min(axis=1) < thr).astype(np.float32)
        self.mask = mask_flat.reshape(G, G)
        self.world_xs = xs
        self.world_ys = ys

    # ------------------------------------------------------------------
    # closest-s queries (used for arc-progress reward)
    # ------------------------------------------------------------------

    def closest_s(self, x: float, y: float) -> Tuple[float, float]:
        """Return (s, dist) where s is normalized arclength of nearest
        centerline sample to (x, y) and dist is Euclidean distance."""
        d2 = (self.cx - x) ** 2 + (self.cy - y) ** 2
        i = int(np.argmin(d2))
        return float(i) / self.n_samples, float(np.sqrt(d2[i]))

    def signed_arc_delta(self, s_prev: float, s_now: float) -> float:
        """Signed shortest delta on the unit circle (loop)."""
        d = (s_now - s_prev) % 1.0
        if d > 0.5:
            d -= 1.0
        return d

    # ------------------------------------------------------------------
    # 16x16 rotated observation patch
    # ------------------------------------------------------------------

    def render_patch(self, x: float, y: float, theta: float,
                     patch_size: int = 16, patch_extent: float = 4.0,
                     ) -> np.ndarray:
        """Sample a patch_size x patch_size view of the track mask at the
        car's pose. The patch is rotated so the car's forward direction
        points UP (+y in patch frame).

        patch_extent is the half-extent of the patch in world units.
        """
        # build patch coord grid in car frame ([-pe, pe] x [-pe, pe])
        u = np.linspace(-patch_extent, patch_extent, patch_size)
        v = np.linspace(-patch_extent, patch_extent, patch_size)
        U, V = np.meshgrid(u, v, indexing="xy")
        # forward = +V axis in car frame; map to world frame
        # world_x = x + cos(theta) * (-V) - sin(theta) * U
        # world_y = y + sin(theta) * (-V) + cos(theta) * U
        # i.e. car y-axis (forward) aligns with theta direction in world
        c, s = np.cos(theta), np.sin(theta)
        wx = x + c * (-V) - s * U
        wy = y + s * (-V) + c * U
        # nearest-neighbor lookup
        E = self.grid_extent
        G = self.grid_size
        ix = np.clip(((wx + E) / (2 * E) * (G - 1)).astype(int), 0, G - 1)
        iy = np.clip(((wy + E) / (2 * E) * (G - 1)).astype(int), 0, G - 1)
        return self.mask[iy, ix].astype(np.float32)


@dataclass
class CarEnv:
    """Continuous-action 2-D car on a Track.

    state: (x, y, theta, v)
    action: (steer in [-1, 1], throttle in [-1, 1])

    Dynamics (kinematic bicycle, dt=1):
        v <- clip(v + 0.10 * throttle - 0.04 * v, 0, v_max)
        theta <- theta + 0.4 * steer * v
        x <- x + v * cos(theta)
        y <- y + v * sin(theta)

    Reward: signed forward arc-length progress along centerline minus
    0.5 * max(0, dist_to_centerline - track.half_width).
    Episode terminates if dist_to_centerline > 2*half_width or t > t_max.
    """

    track: Track
    v_max: float = 1.6
    t_max: int = 120
    margin_penalty: float = 0.5
    progress_scale: float = 30.0  # multiplier on arc-length delta (s in [0,1])

    def reset(self, rng: np.random.Generator | None = None
              ) -> np.ndarray:
        """Spawn at centerline sample 0, heading along forward tangent."""
        # Always start at the same centerline point (for determinism)
        i = 0
        self.x = float(self.track.cx[i])
        self.y = float(self.track.cy[i])
        self.theta = float(np.arctan2(self.track.ty[i], self.track.tx[i]))
        self.v = 0.0
        self.t = 0
        self.s_prev, _ = self.track.closest_s(self.x, self.y)
        self.done = False
        self.cum_reward = 0.0
        return self.obs()

    def obs(self) -> np.ndarray:
        return self.track.render_patch(self.x, self.y, self.theta)

    def state(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.theta, self.v)

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("step() after done")
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        # update speed and heading
        self.v = float(np.clip(self.v + 0.10 * throttle - 0.04 * self.v,
                               0.0, self.v_max))
        self.theta = self.theta + 0.4 * steer * self.v
        self.x = self.x + self.v * math.cos(self.theta)
        self.y = self.y + self.v * math.sin(self.theta)
        self.t += 1
        s_now, dist = self.track.closest_s(self.x, self.y)
        ds = self.track.signed_arc_delta(self.s_prev, s_now)
        progress = self.progress_scale * ds
        margin = max(0.0, dist - self.track.half_width)
        reward = progress - self.margin_penalty * margin
        self.cum_reward += reward
        self.s_prev = s_now
        off_track = dist > 2.0 * self.track.half_width
        timeout = self.t >= self.t_max
        self.done = off_track or timeout
        info = {"x": self.x, "y": self.y, "theta": self.theta, "v": self.v,
                "s": s_now, "dist": dist, "off_track": off_track,
                "ds": ds, "progress": progress, "margin": margin}
        return self.obs(), reward, self.done, info


# ----------------------------------------------------------------------
# V: linear autoencoder (substitute for the convolutional VAE)
# ----------------------------------------------------------------------

@dataclass
class V_AE:
    in_dim: int = 256
    h_dim: int = 64
    z_dim: int = 16
    rng: np.random.Generator = None

    def __post_init__(self):
        s1 = np.sqrt(1.0 / self.in_dim)
        s2 = np.sqrt(1.0 / self.h_dim)
        s3 = np.sqrt(1.0 / self.z_dim)
        s4 = np.sqrt(1.0 / self.h_dim)
        self.W1 = self.rng.normal(0, s1, (self.in_dim, self.h_dim))
        self.b1 = np.zeros(self.h_dim)
        self.W2 = self.rng.normal(0, s2, (self.h_dim, self.z_dim))
        self.b2 = np.zeros(self.z_dim)
        self.W3 = self.rng.normal(0, s3, (self.z_dim, self.h_dim))
        self.b3 = np.zeros(self.h_dim)
        self.W4 = self.rng.normal(0, s4, (self.h_dim, self.in_dim))
        self.b4 = np.zeros(self.in_dim)
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3,
                "W4": self.W4, "b4": self.b4}

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = np.tanh(z @ self.W3 + self.b3)
        return _sigmoid(h @ self.W4 + self.b4)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z = h1 @ self.W2 + self.b2
        z3 = z @ self.W3 + self.b3
        h3 = np.tanh(z3)
        logits = h3 @ self.W4 + self.b4
        x_hat = _sigmoid(logits)
        cache = {"x": x, "h1": h1, "z": z, "h3": h3,
                 "logits": logits, "x_hat": x_hat}
        return x_hat, cache

    def loss_and_grads(self, cache: Dict) -> Tuple[float, Dict]:
        x = cache["x"]
        x_hat = cache["x_hat"]
        # binary cross-entropy works well with sigmoid output and {0,1} mask
        # but obs is binary -> reduce to BCE
        eps = 1e-6
        loss = -np.mean(x * np.log(x_hat + eps)
                        + (1 - x) * np.log(1 - x_hat + eps))
        B = x.shape[0]
        # d_logits = (x_hat - x) / B  (BCE w/ sigmoid simplification)
        d_logits = (x_hat - x) / (B * x.shape[1])
        d_W4 = cache["h3"].T @ d_logits
        d_b4 = d_logits.sum(axis=0)
        d_h3 = d_logits @ self.W4.T
        d_z3 = d_h3 * (1 - cache["h3"] ** 2)
        d_W3 = cache["z"].T @ d_z3
        d_b3 = d_z3.sum(axis=0)
        d_z = d_z3 @ self.W3.T
        d_W2 = cache["h1"].T @ d_z
        d_b2 = d_z.sum(axis=0)
        d_h1 = d_z @ self.W2.T
        d_z1 = d_h1 * (1 - cache["h1"] ** 2)
        d_W1 = cache["x"].T @ d_z1
        d_b1 = d_z1.sum(axis=0)
        return loss, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
                      "W3": d_W3, "b3": d_b3, "W4": d_W4, "b4": d_b4}

    def adam_step(self, grads: Dict, lr: float = 1e-3,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.t_step += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mh = self.m[k] / (1 - b1 ** self.t_step)
            vh = self.v[k] / (1 - b2 ** self.t_step)
            getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ----------------------------------------------------------------------
# M: deterministic LSTM world model (MDN simplified to a single mean)
# ----------------------------------------------------------------------

@dataclass
class M_LSTM:
    in_dim: int = 18    # z(16) + a(2)
    hidden: int = 32
    out_dim: int = 16
    rng: np.random.Generator = None

    def __post_init__(self):
        sx = np.sqrt(1.0 / self.in_dim)
        sh = np.sqrt(1.0 / self.hidden)
        # gates: i, f, g, o packed as (4*H,)
        self.Wx = self.rng.normal(0, sx, (self.in_dim, 4 * self.hidden))
        self.Wh = self.rng.normal(0, sh, (self.hidden, 4 * self.hidden))
        self.bh = np.zeros(4 * self.hidden)
        # forget gate bias = 1 (Jozefowicz et al. 2015)
        self.bh[self.hidden:2 * self.hidden] = 1.0
        # output projection h -> next-z prediction
        self.Wy = self.rng.normal(0, sh, (self.hidden, self.out_dim))
        self.by = np.zeros(self.out_dim)
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self) -> Dict[str, np.ndarray]:
        return {"Wx": self.Wx, "Wh": self.Wh, "bh": self.bh,
                "Wy": self.Wy, "by": self.by}

    def init_state(self, batch: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        return (np.zeros((batch, self.hidden), dtype=np.float32),
                np.zeros((batch, self.hidden), dtype=np.float32))

    def step_one(self, x: np.ndarray, h: np.ndarray, c: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Single LSTM step. Returns (y_pred, h_new, c_new, cache)."""
        z = x @ self.Wx + h @ self.Wh + self.bh
        H = self.hidden
        i = _sigmoid(z[:, :H])
        f = _sigmoid(z[:, H:2 * H])
        g = np.tanh(z[:, 2 * H:3 * H])
        o = _sigmoid(z[:, 3 * H:4 * H])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        y = h_new @ self.Wy + self.by
        cache = {"x": x, "h_prev": h, "c_prev": c, "i": i, "f": f, "g": g,
                 "o": o, "c_new": c_new, "h_new": h_new, "tanh_c_new": np.tanh(c_new)}
        return y, h_new, c_new, cache

    def forward_seq(self, X: np.ndarray
                    ) -> Tuple[np.ndarray, List[Dict]]:
        """X: (B, T, in_dim). Returns Y: (B, T, out_dim) and per-step caches."""
        B, T, _ = X.shape
        h = np.zeros((B, self.hidden), dtype=np.float64)
        c = np.zeros((B, self.hidden), dtype=np.float64)
        outs = []
        caches = []
        for t in range(T):
            y, h, c, ca = self.step_one(X[:, t, :], h, c)
            outs.append(y)
            caches.append(ca)
        Y = np.stack(outs, axis=1)
        return Y, caches

    def loss_and_grads(self, caches: List[Dict], Y_pred: np.ndarray,
                       Y_true: np.ndarray) -> Tuple[float, Dict]:
        """MSE on next-z prediction. Backprop-through-time."""
        B, T, D = Y_pred.shape
        diff = (Y_pred - Y_true)  # (B, T, D)
        loss = float(np.mean(diff ** 2))
        # accumulate grads
        d_Wx = np.zeros_like(self.Wx)
        d_Wh = np.zeros_like(self.Wh)
        d_bh = np.zeros_like(self.bh)
        d_Wy = np.zeros_like(self.Wy)
        d_by = np.zeros_like(self.by)
        H = self.hidden
        d_h_next = np.zeros((B, H))
        d_c_next = np.zeros((B, H))
        # output gradient scale: dL/dy = 2*diff/(B*T*D)
        scale = 2.0 / (B * T * D)
        for t in reversed(range(T)):
            ca = caches[t]
            d_y = diff[:, t, :] * scale  # (B, D)
            d_Wy += ca["h_new"].T @ d_y
            d_by += d_y.sum(axis=0)
            d_h = d_y @ self.Wy.T + d_h_next
            # h_new = o * tanh(c_new)
            tanh_c = ca["tanh_c_new"]
            d_o = d_h * tanh_c
            d_c = d_h * ca["o"] * (1 - tanh_c ** 2) + d_c_next
            d_o_pre = d_o * ca["o"] * (1 - ca["o"])
            # c_new = f*c_prev + i*g
            d_f = d_c * ca["c_prev"]
            d_i = d_c * ca["g"]
            d_g = d_c * ca["i"]
            d_f_pre = d_f * ca["f"] * (1 - ca["f"])
            d_i_pre = d_i * ca["i"] * (1 - ca["i"])
            d_g_pre = d_g * (1 - ca["g"] ** 2)
            d_z = np.concatenate([d_i_pre, d_f_pre, d_g_pre, d_o_pre], axis=1)
            d_Wx += ca["x"].T @ d_z
            d_Wh += ca["h_prev"].T @ d_z
            d_bh += d_z.sum(axis=0)
            d_h_next = d_z @ self.Wh.T
            d_c_next = d_c * ca["f"]
        return loss, {"Wx": d_Wx, "Wh": d_Wh, "bh": d_bh,
                      "Wy": d_Wy, "by": d_by}

    def adam_step(self, grads: Dict, lr: float = 1e-3,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                  clip: float = 5.0):
        self.t_step += 1
        # global gradient clip
        gn = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if gn > clip:
            for k in grads:
                grads[k] = grads[k] * (clip / gn)
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mh = self.m[k] / (1 - b1 ** self.t_step)
            vh = self.v[k] / (1 - b2 ** self.t_step)
            getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + eps)


# ----------------------------------------------------------------------
# C: linear controller (z | h_M -> action), evolved with CMA-ES
# ----------------------------------------------------------------------

@dataclass
class C_Linear:
    z_dim: int
    h_dim: int  # M's hidden size
    out_dim: int = 2  # (steer, throttle)

    @property
    def in_dim(self) -> int:
        return self.z_dim + self.h_dim

    @property
    def n_params(self) -> int:
        return self.in_dim * self.out_dim + self.out_dim

    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W = theta[: self.in_dim * self.out_dim].reshape(
            self.in_dim, self.out_dim)
        b = theta[self.in_dim * self.out_dim:]
        return W, b

    def act(self, theta: np.ndarray, z: np.ndarray, h: np.ndarray
            ) -> np.ndarray:
        """Return action in [-1, 1]^2 via tanh."""
        W, b = self.unpack(theta)
        x = np.concatenate([z.ravel(), h.ravel()])
        return np.tanh(x @ W + b)


# ----------------------------------------------------------------------
# Rollout helpers
# ----------------------------------------------------------------------

def random_rollout(env: CarEnv, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-policy rollout. Returns (obs_seq, action_seq, rew_seq).

    Random actions are smoothed (low-pass) so the car actually moves rather
    than dithering -- otherwise no rollout escapes the spawn neighborhood and
    the AE/M see no diversity.
    """
    obs0 = env.reset()
    obs_list = [obs0]
    act_list = []
    rew_list = []
    a = np.zeros(2)
    while not env.done:
        a_target = rng.uniform(-1.0, 1.0, size=2)
        # bias throttle positive so the car moves
        a_target[1] = np.clip(a_target[1] + 0.5, -1.0, 1.0)
        a = 0.7 * a + 0.3 * a_target
        obs, r, done, info = env.step(a)
        obs_list.append(obs)
        act_list.append(a.copy())
        rew_list.append(r)
        if done:
            break
    return (np.stack(obs_list, axis=0),
            np.stack(act_list, axis=0) if act_list else np.zeros((0, 2)),
            np.array(rew_list, dtype=np.float64))


def policy_rollout(env: CarEnv, V: V_AE, M: M_LSTM, C: C_Linear,
                   theta: np.ndarray, return_states: bool = False
                   ) -> Tuple[float, Dict]:
    """Roll out the V+M+C policy. Returns total reward and trace info."""
    obs = env.reset()
    h, c = M.init_state(batch=1)
    cum = 0.0
    states = [(env.x, env.y, env.theta, env.v)]
    obs_seq = [obs]
    z_seq = []
    actions = []
    while not env.done:
        z = V.encode(obs.ravel().reshape(1, -1)).ravel()
        z_seq.append(z)
        a = C.act(theta, z, h[0])
        actions.append(a)
        obs, r, done, info = env.step(a)
        cum += r
        # advance world model
        x_in = np.concatenate([z, a])[None, :]
        _, h, c, _ = M.step_one(x_in, h, c)
        states.append((env.x, env.y, env.theta, env.v))
        obs_seq.append(obs)
        if done:
            break
    info_out = {"steps": env.t, "off_track": env.t < env.t_max,
                "final_s": env.s_prev}
    if return_states:
        info_out["states"] = states
        info_out["obs"] = obs_seq
        info_out["z"] = z_seq
        info_out["actions"] = actions
    return float(cum), info_out


# ----------------------------------------------------------------------
# Simplified rank-mu CMA-ES with isotropic step size
# ----------------------------------------------------------------------

def rank_mu_cma_es(eval_fn, n_params: int, popsize: int, n_gen: int,
                   sigma0: float, rng: np.random.Generator,
                   mu_init: np.ndarray | None = None,
                   verbose: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """Rank-mu (μ_w, λ)-ES with isotropic σ adaptation.

    We use the canonical weighted recombination of Hansen & Ostermeier (2001):

        weights_i = log(μ + 1) - log(i),    i = 1..μ   (then normalized)
        μ_eff = (Σ w_i)^2 / Σ w_i^2
        m <- m + σ Σ w_i * z_(i:λ)
        σ <- σ exp((c_σ / d_σ) * (||Σ w_i z_(i:λ)||/E[||N(0,I)||] - 1))

    No covariance update (no rank-1 path, no rank-μ on C). This trades
    quadratic convergence of true CMA-ES for a much smaller numpy footprint.
    Documented in §Deviations of README.md.
    """
    if mu_init is None:
        mu = np.zeros(n_params)
    else:
        mu = mu_init.copy()
    sigma = float(sigma0)
    n_elite = popsize // 2
    raw_w = np.log(n_elite + 1) - np.log(np.arange(1, n_elite + 1))
    w = raw_w / raw_w.sum()
    mu_eff = 1.0 / (w * w).sum()
    c_sigma = (mu_eff + 2) / (n_params + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_params + 1)) - 1) + c_sigma
    expected_norm = math.sqrt(n_params) * (1 - 1 / (4 * n_params)
                                           + 1 / (21 * n_params ** 2))
    p_sigma = np.zeros(n_params)
    history = []
    for gen in range(n_gen):
        z = rng.normal(0.0, 1.0, size=(popsize, n_params))
        cand = mu + sigma * z
        fit = np.array([eval_fn(c) for c in cand])
        order = np.argsort(-fit)  # descending (max fitness first)
        z_elite = z[order[:n_elite]]
        z_mean = w @ z_elite
        mu = mu + sigma * z_mean
        # isotropic CSA
        p_sigma = ((1 - c_sigma) * p_sigma
                   + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean)
        sigma = sigma * math.exp((c_sigma / d_sigma)
                                  * (np.linalg.norm(p_sigma) / expected_norm - 1))
        sigma = float(np.clip(sigma, 1e-4, 5.0))
        rec = {"gen": int(gen), "best": float(fit.max()),
               "mean": float(fit.mean()), "median": float(np.median(fit)),
               "sigma": float(sigma)}
        history.append(rec)
        if verbose:
            print(f"[CMA-ES] gen {gen:3d}  best={rec['best']:+.3f}  "
                  f"mean={rec['mean']:+.3f}  sigma={rec['sigma']:.3f}")
    return mu, history


# ----------------------------------------------------------------------
# Pipeline: collect rollouts -> train V -> train M -> CMA-ES on C
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    quick: bool = False
    # data collection
    n_random_episodes: int = 64
    # V (autoencoder)
    z_dim: int = 16
    v_hidden: int = 64
    v_epochs: int = 4
    v_batch: int = 64
    v_lr: float = 2e-3
    # M (LSTM)
    m_hidden: int = 32
    m_epochs: int = 4
    m_batch: int = 16
    m_lr: float = 5e-3
    m_seq_len: int = 30
    # C (CMA-ES)
    cma_popsize: int = 24
    cma_gens: int = 30
    cma_sigma0: float = 0.5
    cma_episodes_per_indiv: int = 1
    # eval
    n_eval_rollouts: int = 8


@dataclass
class RunResult:
    config: Dict
    env_meta: Dict
    track: Dict
    v_loss: List[float]
    m_loss: List[float]
    cma_history: List[Dict]
    random_baseline: Dict
    final_eval: Dict
    sample_recon: Dict
    elapsed_seconds: float


def collect_random_data(env: CarEnv, n_episodes: int,
                        rng: np.random.Generator
                        ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Returns (all_obs (N, 256), all_actions (N, 2), episode_lengths)."""
    obs_chunks = []
    act_chunks = []
    lens = []
    for _ in range(n_episodes):
        obs_seq, act_seq, _ = random_rollout(env, rng)
        # for AE: keep all obs
        obs_chunks.append(obs_seq[:-1].reshape(obs_seq.shape[0] - 1, -1))
        act_chunks.append(act_seq)
        lens.append(act_seq.shape[0])
    all_obs = np.concatenate(obs_chunks, axis=0)
    all_acts = np.concatenate(act_chunks, axis=0)
    return all_obs, all_acts, lens


def train_V(V: V_AE, all_obs: np.ndarray, cfg: RunConfig,
            rng: np.random.Generator) -> List[float]:
    """Train AE on all observed frames. Returns per-batch loss curve."""
    N = all_obs.shape[0]
    n_batches_per_epoch = max(1, N // cfg.v_batch)
    losses = []
    for ep in range(cfg.v_epochs):
        perm = rng.permutation(N)
        for b in range(n_batches_per_epoch):
            idx = perm[b * cfg.v_batch:(b + 1) * cfg.v_batch]
            x = all_obs[idx]
            _, cache = V.forward(x)
            loss, grads = V.loss_and_grads(cache)
            V.adam_step(grads, lr=cfg.v_lr)
            losses.append(loss)
    return losses


def train_M(M: M_LSTM, V: V_AE, env: CarEnv, n_episodes: int,
            cfg: RunConfig, rng: np.random.Generator
            ) -> List[float]:
    """Re-encode random rollouts to z; train LSTM on next-z prediction."""
    # collect z trajectories from fresh random rollouts
    Z_trajs = []
    A_trajs = []
    for _ in range(n_episodes):
        obs_seq, act_seq, _ = random_rollout(env, rng)
        if act_seq.shape[0] < 2:
            continue
        z_seq = V.encode(obs_seq.reshape(obs_seq.shape[0], -1))
        Z_trajs.append(z_seq)
        A_trajs.append(act_seq)
    # build (z_t, a_t) -> z_{t+1} samples padded into batches of seq_len
    inputs = []
    targets = []
    for z_seq, a_seq in zip(Z_trajs, A_trajs):
        T = a_seq.shape[0]
        if T < 2:
            continue
        x = np.concatenate([z_seq[:T], a_seq], axis=1)  # (T, 18)
        y = z_seq[1:T + 1]  # (T, 16)
        # split into chunks of length cfg.m_seq_len
        L = cfg.m_seq_len
        for i in range(0, T, L):
            xx = x[i:i + L]
            yy = y[i:i + L]
            if xx.shape[0] < 4:
                continue
            inputs.append(xx)
            targets.append(yy)
    # pad to common length within each batch
    losses = []
    n_batches_per_epoch = max(1, len(inputs) // cfg.m_batch)
    for ep in range(cfg.m_epochs):
        order = rng.permutation(len(inputs))
        for b in range(n_batches_per_epoch):
            batch_idx = order[b * cfg.m_batch:(b + 1) * cfg.m_batch]
            seqs_x = [inputs[i] for i in batch_idx]
            seqs_y = [targets[i] for i in batch_idx]
            if len(seqs_x) == 0:
                continue
            T_min = min(s.shape[0] for s in seqs_x)
            X = np.stack([s[:T_min] for s in seqs_x], axis=0)
            Y = np.stack([s[:T_min] for s in seqs_y], axis=0)
            Y_pred, caches = M.forward_seq(X)
            loss, grads = M.loss_and_grads(caches, Y_pred, Y)
            M.adam_step(grads, lr=cfg.m_lr)
            losses.append(loss)
    return losses


def evaluate_random_baseline(env: CarEnv, n: int, rng: np.random.Generator
                             ) -> Dict[str, float]:
    rs = []
    lens = []
    for _ in range(n):
        _, _, rew = random_rollout(env, rng)
        rs.append(float(rew.sum()))
        lens.append(int(rew.shape[0]))
    return {"mean_return": float(np.mean(rs)),
            "std_return": float(np.std(rs)),
            "mean_len": float(np.mean(lens)),
            "n": int(n)}


def evaluate_policy(env: CarEnv, V: V_AE, M: M_LSTM, C: C_Linear,
                    theta: np.ndarray, n: int) -> Dict[str, float]:
    rs = []
    lens = []
    finals = []
    for _ in range(n):
        cum, info = policy_rollout(env, V, M, C, theta)
        rs.append(cum)
        lens.append(info["steps"])
        finals.append(info["final_s"])
    return {"mean_return": float(np.mean(rs)),
            "std_return": float(np.std(rs)),
            "mean_len": float(np.mean(lens)),
            "mean_final_s": float(np.mean(finals)),
            "max_final_s": float(np.max(finals)),
            "n": int(n)}


def run(cfg: RunConfig, verbose: bool = True) -> RunResult:
    t0 = time.time()
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if cfg.quick:
        cfg.n_random_episodes = 12
        cfg.v_epochs = 2
        cfg.m_epochs = 2
        cfg.cma_popsize = 12
        cfg.cma_gens = 8
        cfg.n_eval_rollouts = 4

    # ---- env ----
    track = Track()
    env = CarEnv(track=track)

    if verbose:
        print(f"[init] track grid {track.grid_size}^2, "
              f"on-track ratio {float(track.mask.mean()):.3f}")

    # ---- random baseline ----
    rand = evaluate_random_baseline(env, cfg.n_eval_rollouts, rng)
    if verbose:
        print(f"[random baseline] return = "
              f"{rand['mean_return']:+.3f} +/- {rand['std_return']:.3f}, "
              f"len = {rand['mean_len']:.1f}")

    # ---- collect rollouts for V ----
    if verbose:
        print(f"[V] collecting {cfg.n_random_episodes} random rollouts ...")
    all_obs, all_acts, lens = collect_random_data(env, cfg.n_random_episodes,
                                                  rng)
    if verbose:
        print(f"[V] {all_obs.shape[0]} frames; "
              f"obs mean = {all_obs.mean():.3f}")

    # ---- train V ----
    V = V_AE(z_dim=cfg.z_dim, h_dim=cfg.v_hidden,
             rng=np.random.default_rng(cfg.seed + 1))
    if verbose:
        print(f"[V] training AE for {cfg.v_epochs} epochs ...")
    v_losses = train_V(V, all_obs, cfg, rng)
    if verbose:
        print(f"[V] final batch loss = {v_losses[-1]:.4f}")

    # ---- train M ----
    M = M_LSTM(in_dim=cfg.z_dim + 2, hidden=cfg.m_hidden, out_dim=cfg.z_dim,
               rng=np.random.default_rng(cfg.seed + 2))
    if verbose:
        print(f"[M] training LSTM for {cfg.m_epochs} epochs ...")
    m_losses = train_M(M, V, env, cfg.n_random_episodes, cfg, rng)
    if verbose:
        print(f"[M] final batch loss = {m_losses[-1]:.4f}")

    # ---- CMA-ES on C ----
    C = C_Linear(z_dim=cfg.z_dim, h_dim=cfg.m_hidden)

    def fitness(theta: np.ndarray) -> float:
        rs = []
        for k in range(cfg.cma_episodes_per_indiv):
            cum, _ = policy_rollout(env, V, M, C, theta)
            rs.append(cum)
        return float(np.mean(rs))

    if verbose:
        print(f"[C] CMA-ES popsize={cfg.cma_popsize} "
              f"gens={cfg.cma_gens} n_params={C.n_params}")
    cma_rng = np.random.default_rng(cfg.seed + 3)
    theta_star, cma_hist = rank_mu_cma_es(
        fitness, n_params=C.n_params,
        popsize=cfg.cma_popsize, n_gen=cfg.cma_gens,
        sigma0=cfg.cma_sigma0, rng=cma_rng, verbose=verbose,
    )

    # ---- final eval ----
    final = evaluate_policy(env, V, M, C, theta_star, cfg.n_eval_rollouts)
    if verbose:
        print(f"[final] policy return = "
              f"{final['mean_return']:+.3f} +/- {final['std_return']:.3f}, "
              f"final_s = {final['mean_final_s']:.3f}, "
              f"len = {final['mean_len']:.1f}")
        print(f"[final] vs random      = {rand['mean_return']:+.3f}")

    # ---- sample reconstruction (for viz) ----
    idx = rng.choice(all_obs.shape[0], size=8, replace=False)
    sample_in = all_obs[idx]
    sample_z = V.encode(sample_in)
    sample_out = V.decode(sample_z)

    # one trajectory for the gif
    cum_demo, demo_info = policy_rollout(env, V, M, C, theta_star,
                                         return_states=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[done] elapsed {elapsed:.1f}s")

    return RunResult(
        config=asdict(cfg),
        env_meta=env_metadata(),
        track={"R": track.R, "a1": track.a1, "phi1": track.phi1,
               "a2": track.a2, "phi2": track.phi2,
               "half_width": track.half_width,
               "grid_size": track.grid_size,
               "grid_extent": track.grid_extent,
               "n_samples": track.n_samples,
               "centerline_x": track.cx.tolist(),
               "centerline_y": track.cy.tolist(),
               "mask": track.mask.tolist()},
        v_loss=[float(x) for x in v_losses],
        m_loss=[float(x) for x in m_losses],
        cma_history=cma_hist,
        random_baseline=rand,
        final_eval=final,
        sample_recon={"input": sample_in.tolist(),
                       "z": sample_z.tolist(),
                       "output": sample_out.tolist()},
        elapsed_seconds=elapsed,
    ), {
        "theta_star": theta_star.tolist(),
        "demo_states": [list(s) for s in demo_info["states"]],
        "demo_obs": [o.tolist() for o in demo_info["obs"]],
        "demo_actions": [list(a) for a in demo_info["actions"]],
        "demo_z": [list(z) for z in demo_info["z"]],
        "demo_cum_reward": float(cum_demo),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="small popsize/gens, for smoke testing")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save-json", type=str, default=None)
    args = ap.parse_args()
    cfg = RunConfig(seed=args.seed, quick=args.quick)
    result, demo = run(cfg, verbose=not args.quiet)
    if args.save_json:
        out = asdict(result)
        out["demo"] = demo
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)),
                    exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f)
        if not args.quiet:
            print(f"[save] wrote {args.save_json}")


if __name__ == "__main__":
    main()
