"""relational-nem-bouncing-balls -- van Steenkiste, Chang, Greff, Schmidhuber,
*Relational Neural Expectation Maximization*, ICLR 2018 (arXiv:1802.10353).

The R-NEM paper extends N-EM (Greff et al. 2017) by adding a pairwise
interaction module to the per-slot M-step. On bouncing-balls, the relational
module captures collisions: a non-relational dynamics network sees only its
own slot state, so it can predict ballistic motion but not how a ball changes
direction when another ball hits it. The relational network exchanges
messages between every pair of slots, so it can.

Headline (this stub):
    Train both models on K=3 bouncing balls. Roll them forward 30 steps.
    Compare the per-step prediction MSE and the multi-step rollout error.
    Test extrapolation by running the same trained models on K=4 and K=5
    balls (the dynamics modules are slot-symmetric MLPs, so increasing K
    only adds more pairs to the message aggregation).

Architecture:
    Slot state s_k = (x, y, vx, vy)  in R^4 (oracle from physics simulator;
    see §Deviations -- we skip the N-EM segmentation E-step and ablate the
    M-step relational/non-relational dynamics directly).

    Non-relational dynamics:
        delta_k = MLP_dyn(s_k)                        -- 4 -> 64 -> 64 -> 4
        s_k(t+1) = s_k(t) + delta_k

    Relational dynamics (R-NEM core):
        m_{k<-j} = MLP_msg(s_k, s_j)        for all j != k          (pairwise)
        agg_k   = sum_{j != k} m_{k<-j}                              (sum-pool)
        delta_k = MLP_dyn(s_k, agg_k)                 -- 4+M -> 64 -> 64 -> 4
        s_k(t+1) = s_k(t) + delta_k

    Both are trained with single-step MSE on (s_k(t+1) - s_k(t)).
    Adam, ReLU MLPs, He init for hidden, scaled init for output.

Rollout extrapolation:
    The architecture is slot-permutation-equivariant (sum aggregation),
    so K can change at test time without retraining.

CLI:
    python3 relational_nem_bouncing_balls.py --seed 0
    python3 relational_nem_bouncing_balls.py --seed 0 --quick     # smaller, ~30s
    python3 relational_nem_bouncing_balls.py --seed 0 --epochs 80
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
# Environment metadata (reproducibility per .claude/rules)
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


# ======================================================================
# 1. Physics simulator: K balls bouncing in [0,1]^2 with elastic collisions
# ======================================================================

def _init_non_overlapping(K: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Place K disks in [radius, 1-radius]^2 with no pairwise overlap."""
    pos = np.zeros((K, 2))
    min_sep = 2.2 * radius
    for k in range(K):
        for _ in range(2000):
            p = rng.uniform(radius, 1.0 - radius, size=2)
            ok = True
            for j in range(k):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[k] = p
                break
        else:
            # fallback grid placement (rare, only K very large)
            ncols = int(np.ceil(np.sqrt(K)))
            row, col = k // ncols, k % ncols
            pos[k] = np.array([(col + 0.5) / ncols, (row + 0.5) / ncols])
    return pos


def simulate(K: int, T: int, dt: float, radius: float,
             rng: np.random.Generator,
             speed_min: float = 0.6, speed_max: float = 1.0) -> np.ndarray:
    """Simulate K balls of radius `radius` in [0,1]^2 for T time-steps.

    Returns: states of shape (T, K, 4) where last dim is (x, y, vx, vy).
    Wall collisions: reflect normal component, mirror penetration.
    Ball-ball collisions: equal-mass elastic (swap normal velocity components).
    """
    pos = _init_non_overlapping(K, radius, rng)
    angle = rng.uniform(0.0, 2 * np.pi, size=K)
    speed = rng.uniform(speed_min, speed_max, size=K)
    vel = np.stack([np.cos(angle) * speed, np.sin(angle) * speed], axis=-1)

    states = np.zeros((T, K, 4))
    for t in range(T):
        states[t, :, 0:2] = pos
        states[t, :, 2:4] = vel

        # Advance position
        pos = pos + vel * dt

        # Wall collisions
        for d in range(2):
            below = pos[:, d] < radius
            above = pos[:, d] > 1.0 - radius
            if below.any():
                pos[below, d] = 2.0 * radius - pos[below, d]
                vel[below, d] = np.abs(vel[below, d])
            if above.any():
                pos[above, d] = 2.0 * (1.0 - radius) - pos[above, d]
                vel[above, d] = -np.abs(vel[above, d])

        # Pairwise elastic collisions (equal mass)
        for i in range(K):
            for j in range(i + 1, K):
                d = pos[i] - pos[j]
                dist = float(np.linalg.norm(d))
                if dist < 2.0 * radius and dist > 1e-9:
                    n = d / dist
                    rel_v = vel[i] - vel[j]
                    rel_v_n = float(rel_v @ n)
                    if rel_v_n < 0.0:  # approaching each other
                        vel[i] = vel[i] - rel_v_n * n
                        vel[j] = vel[j] + rel_v_n * n
                        # un-overlap by half each, along the normal
                        overlap = 2.0 * radius - dist
                        pos[i] = pos[i] + n * (overlap / 2.0)
                        pos[j] = pos[j] - n * (overlap / 2.0)
    return states


def make_dataset(N: int, T: int, K: int, dt: float, radius: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Generate N independent trajectories. Returns (N, T, K, 4)."""
    out = np.zeros((N, T, K, 4))
    for i in range(N):
        out[i] = simulate(K, T, dt, radius, rng)
    return out


# ======================================================================
# 2. Renderer (used only for visualization; training uses oracle states)
# ======================================================================

def render_frame(state_k: np.ndarray, H: int, W: int, sigma: float) -> np.ndarray:
    """state_k: (K, 4). Returns (H, W) grayscale image with Gaussian blobs."""
    img = np.zeros((H, W))
    ys = np.linspace(0.0, 1.0, H)
    xs = np.linspace(0.0, 1.0, W)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    for k in range(state_k.shape[0]):
        cx = float(state_k[k, 0])
        cy = float(state_k[k, 1])
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        img += np.exp(-d2 / (2.0 * sigma ** 2))
    return np.clip(img, 0.0, 1.0)


# ======================================================================
# 3. MLP primitive (ReLU hidden, linear output)
# ======================================================================

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class MLP:
    def __init__(self, dims: List[int], rng: np.random.Generator,
                 out_scale: float = 0.1):
        self.dims = list(dims)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        L = len(dims) - 1
        for i in range(L):
            d_in, d_out = dims[i], dims[i + 1]
            if i < L - 1:
                # He init for hidden
                s = np.sqrt(2.0 / d_in)
            else:
                # Small init for output: predicts residuals/messages, want
                # them small at init so the trained signal dominates noise
                s = out_scale / np.sqrt(d_in)
            self.W.append(rng.normal(0.0, s, size=(d_out, d_in)))
            self.b.append(np.zeros(d_out))

    def num_params(self) -> int:
        return sum(W.size for W in self.W) + sum(b.size for b in self.b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        cache: List[np.ndarray] = [x]
        h = x
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i].T + self.b[i]
            if i < L - 1:
                h = _relu(z)
            else:
                h = z
            cache.append(h)
        return h, cache

    def backward(self, dh: np.ndarray, cache: List[np.ndarray]
                 ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        L = len(self.W)
        gW: List[np.ndarray] = [None] * L  # type: ignore
        gb: List[np.ndarray] = [None] * L  # type: ignore
        for i in reversed(range(L)):
            h_in = cache[i]
            h_out = cache[i + 1]
            if i < L - 1:
                dz = dh * (h_out > 0.0).astype(dh.dtype)
            else:
                dz = dh
            gW[i] = dz.T @ h_in
            gb[i] = dz.sum(axis=0)
            dh = dz @ self.W[i]
        return gW, gb, dh


class Adam:
    """Adam optimizer over a flat list of (W, b) parameter pairs."""

    def __init__(self, params: List[np.ndarray], lr: float = 3e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 clip: float = 5.0):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.clip = clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> float:
        # Global gradient clipping
        sqsum = 0.0
        for g in grads:
            sqsum += float((g * g).sum())
        gnorm = float(np.sqrt(sqsum))
        scale = 1.0 if gnorm < self.clip else (self.clip / (gnorm + 1e-12))
        self.t += 1
        for i, p in enumerate(self.params):
            g = grads[i] * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return gnorm


# ======================================================================
# 4. Dynamics models
# ======================================================================

class NonRelationalDynamics:
    """delta_k = MLP_dyn(s_k). Each slot updated in isolation."""

    NAME = "non-relational"

    def __init__(self, state_dim: int, hidden: int, rng: np.random.Generator):
        self.state_dim = state_dim
        self.hidden = hidden
        self.dyn = MLP([state_dim, hidden, hidden, state_dim], rng,
                       out_scale=0.05)

    def params(self) -> List[np.ndarray]:
        return self.dyn.W + self.dyn.b

    def forward(self, S: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """S: (B, K, D) -> delta: (B, K, D)."""
        B, K, D = S.shape
        flat = S.reshape(B * K, D)
        delta_flat, cache = self.dyn.forward(flat)
        delta = delta_flat.reshape(B, K, D)
        return delta, (cache, B, K, D)

    def backward(self, dDelta: np.ndarray, ctx: Tuple
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
        cache, B, K, D = ctx
        gW, gb, dS_flat = self.dyn.backward(dDelta.reshape(B * K, D), cache)
        dS = dS_flat.reshape(B, K, D)
        return gW + gb, dS


class RelationalDynamics:
    """R-NEM core: pairwise messages between slots, mean-aggregated, then per-slot dyn MLP.

    m_{k<-j} = MLP_msg(concat(s_k, s_j))      (B, K, K, msg_dim)
    agg_k    = mean_{j != k} m_{k<-j}         (B, K, msg_dim)   <- K-invariant
    delta_k  = MLP_dyn(concat(s_k, agg_k))    (B, K, D)

    Mean (rather than sum) keeps the aggregate scale fixed when K changes at
    test time -- the original R-NEM paper notes that the aggregator must be
    permutation-invariant; we additionally make it scale-invariant in K so
    extrapolation to more balls works without retraining.
    """

    NAME = "relational"

    def __init__(self, state_dim: int, hidden: int, msg_dim: int,
                 rng: np.random.Generator):
        self.state_dim = state_dim
        self.hidden = hidden
        self.msg_dim = msg_dim
        # Message net: (s_k, s_j) -> message
        self.msg = MLP([2 * state_dim, hidden, msg_dim], rng, out_scale=0.05)
        # Dynamics net: (s_k, agg_k) -> delta_k
        self.dyn = MLP([state_dim + msg_dim, hidden, hidden, state_dim], rng,
                       out_scale=0.05)

    def params(self) -> List[np.ndarray]:
        return self.msg.W + self.msg.b + self.dyn.W + self.dyn.b

    def forward(self, S: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        B, K, D = S.shape
        # Build all (k, j) pairs: pair[b, k, j] = (s_k, s_j)
        S_k = np.broadcast_to(S[:, :, None, :], (B, K, K, D))
        S_j = np.broadcast_to(S[:, None, :, :], (B, K, K, D))
        pair = np.concatenate([S_k, S_j], axis=-1)            # (B, K, K, 2D)
        mask = (1.0 - np.eye(K))[None, :, :, None]             # (1, K, K, 1) zero on j=k

        pair_flat = pair.reshape(B * K * K, 2 * D)
        msg_flat, msg_cache = self.msg.forward(pair_flat)
        msg = msg_flat.reshape(B, K, K, self.msg_dim) * mask   # zero diag
        agg_scale = 1.0 / max(K - 1, 1)
        agg = msg.sum(axis=2) * agg_scale                      # (B, K, msg_dim)

        feat = np.concatenate([S, agg], axis=-1)               # (B, K, D+msg_dim)
        feat_flat = feat.reshape(B * K, D + self.msg_dim)
        delta_flat, dyn_cache = self.dyn.forward(feat_flat)
        delta = delta_flat.reshape(B, K, D)

        ctx = (B, K, D, msg_cache, dyn_cache, mask, agg_scale)
        return delta, ctx

    def backward(self, dDelta: np.ndarray, ctx: Tuple
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
        B, K, D, msg_cache, dyn_cache, mask, agg_scale = ctx

        # back through dyn
        dflat = dDelta.reshape(B * K, D)
        gW_dyn, gb_dyn, dfeat_flat = self.dyn.backward(dflat, dyn_cache)
        dfeat = dfeat_flat.reshape(B, K, D + self.msg_dim)
        dS_from_dyn = dfeat[..., :D]                            # (B, K, D)
        dAgg = dfeat[..., D:]                                   # (B, K, msg_dim)

        # back through (sum * agg_scale): dMsg[b,k,j,m] = dAgg[b,k,m] * mask * agg_scale
        dMsg = (np.broadcast_to(dAgg[:, :, None, :],
                                (B, K, K, self.msg_dim))
                * mask * agg_scale)                              # (B,K,K,M)

        # back through msg MLP
        dmsg_flat = dMsg.reshape(B * K * K, self.msg_dim).copy()
        gW_msg, gb_msg, dpair_flat = self.msg.backward(dmsg_flat, msg_cache)
        dpair = dpair_flat.reshape(B, K, K, 2 * D)
        dS_k_from_msg = dpair[..., :D].sum(axis=2)              # sum over j  -> (B,K,D)
        dS_j_from_msg = dpair[..., D:].sum(axis=1)              # sum over k  -> (B,K,D) (j is axis 2; sum over axis 1)
        # Note: dS / dS[i] has contributions from being k (axis 1=i) and being j (axis 2=i)

        # Total dS is the sum of three pathways back to the input slot states:
        # the dyn-input slot, plus once-as-k and once-as-j inside the msg net.
        dS = dS_from_dyn + dS_k_from_msg + dS_j_from_msg

        return gW_msg + gb_msg + gW_dyn + gb_dyn, dS


# ======================================================================
# 5. Loss + training loop
# ======================================================================

def mse_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error over all elements. Returns (loss, dL/dpred)."""
    diff = pred - target
    n = float(diff.size)
    loss = float((diff * diff).sum() / n)
    dpred = 2.0 * diff / n
    return loss, dpred


def train_model(model, train_seq: np.ndarray, val_seq: np.ndarray,
                *, epochs: int, batch_size: int, lr: float, seed: int,
                label: str, t_bptt: int = 4,
                verbose: bool = True) -> Dict[str, object]:
    """Multi-step BPTT training (canonical R-NEM-style):

    Roll the model forward `t_bptt` steps, compare predicted state to ground
    truth at each step, sum MSE across steps, backprop through the chain.
    Random window-start within each sequence each batch.
    """
    rng = np.random.default_rng(seed + 7)
    N, T, K, D = train_seq.shape
    assert T >= t_bptt + 1, f"train_seq T={T} too short for t_bptt={t_bptt}"

    opt = Adam(model.params(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss_1step": [],
               "val_loss_bptt": [], "grad_norm": []}
    t0 = time.time()
    # Number of valid window starts per sequence
    n_starts = T - t_bptt
    # Total examples = N * n_starts. We sample with replacement-ish each epoch.
    examples_per_epoch = N * n_starts
    steps_per_epoch = max(1, examples_per_epoch // batch_size)

    for ep in range(epochs):
        ep_loss = 0.0
        ep_gn = 0.0
        for _ in range(steps_per_epoch):
            # Sample batch_size (sequence, start) pairs uniformly
            seq_idx = rng.integers(0, N, size=batch_size)
            t_idx = rng.integers(0, n_starts, size=batch_size)
            # Build the BPTT window: (t_bptt + 1, batch, K, D)
            window = np.zeros((t_bptt + 1, batch_size, K, D))
            for i in range(batch_size):
                window[:, i] = train_seq[seq_idx[i], t_idx[i]:t_idx[i] + t_bptt + 1]
            xb = window[0]
            ys_target = window[1:]                  # (t_bptt, B, K, D)

            # Forward roll
            preds: List[np.ndarray] = []
            ctxs: List[Tuple] = []
            s = xb.copy()
            for step in range(t_bptt):
                delta, ctx = model.forward(s)
                s = s + delta
                preds.append(s)
                ctxs.append(ctx)

            # Per-step losses (mean over elements per step, then mean over steps)
            step_loss_total = 0.0
            d_preds: List[np.ndarray] = []
            for step in range(t_bptt):
                diff = preds[step] - ys_target[step]
                n_el = float(diff.size)
                step_loss_total += float((diff * diff).sum() / n_el)
                d_preds.append(2.0 * diff / (n_el * t_bptt))
            loss = step_loss_total / t_bptt

            # Backward roll (BPTT). dS_after[step] is grad w.r.t. s after step.
            # s_{step+1} = s_step + delta(s_step). preds[step] = s_{step+1}.
            # Loss = (1/t_bptt) sum_{step} mse(preds[step], target[step])
            # dL/d(s_{step+1}) = d_preds[step] + dL/d(s_{step+2}) * d(s_{step+2})/d(s_{step+1})
            # where d(s_{step+2})/d(s_{step+1}) = I + d(delta_{step+1})/d(s_{step+1})
            # So dS_step is built bottom-up.
            #
            # We accumulate parameter gradients as numpy arrays in a list.
            params = model.params()
            grad_accum = [np.zeros_like(p) for p in params]
            ds_next = np.zeros_like(s)              # dL/d(s after final step) starts at 0
            for step in reversed(range(t_bptt)):
                # Gradient on the predicted state for this step:
                #   pred = s_after_step; loss term direct contribution is d_preds[step]
                #   carry from later step is ds_next (only for step < t_bptt-1; final step has 0 next)
                d_s_after = d_preds[step] + ds_next
                # delta = model(s_before); s_after = s_before + delta
                # so backward has d_delta = d_s_after, and d_s_before = d_s_after + (backward through delta)
                d_delta = d_s_after
                step_grads, ds_back = model.backward(d_delta, ctxs[step])
                # accumulate parameter grads
                for i_p, g in enumerate(step_grads):
                    grad_accum[i_p] = grad_accum[i_p] + g
                # propagate to previous step's output (carry pathway: identity)
                ds_next = d_s_after + ds_back

            gn = opt.step(grad_accum)
            ep_loss += loss
            ep_gn += gn

        ep_loss /= steps_per_epoch
        ep_gn /= steps_per_epoch

        # Validation: report both 1-step MSE and t_bptt-step rollout MSE
        va_1, va_bptt = _eval_loss_pair(model, val_seq, t_bptt=t_bptt,
                                        batch_size=512)
        history["epoch"].append(ep + 1)
        history["train_loss"].append(ep_loss)
        history["val_loss_1step"].append(va_1)
        history["val_loss_bptt"].append(va_bptt)
        history["grad_norm"].append(ep_gn)
        if verbose and ((ep + 1) % max(1, epochs // 10) == 0 or ep == 0):
            print(f"  [{label}] ep {ep+1:3d}/{epochs}  "
                  f"train {ep_loss:.4e}  "
                  f"val1 {va_1:.4e}  val{t_bptt} {va_bptt:.4e}  "
                  f"|g| {ep_gn:.2e}  t {time.time()-t0:.1f}s", flush=True)
    elapsed = time.time() - t0

    return {
        "label": label,
        "epochs": epochs,
        "batch_size": batch_size,
        "t_bptt": t_bptt,
        "lr": lr,
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss_1step": history["val_loss_1step"][-1],
        "final_val_loss_bptt": history["val_loss_bptt"][-1],
        "wallclock_sec": elapsed,
    }


def _eval_loss_pair(model, val_seq: np.ndarray, t_bptt: int,
                    batch_size: int = 512
                    ) -> Tuple[float, float]:
    """Return (1-step MSE, t_bptt-step rollout MSE) on validation set."""
    Nv, Tv, K, D = val_seq.shape
    # 1-step
    X1 = val_seq[:, :-1].reshape(Nv * (Tv - 1), K, D)
    Y1 = val_seq[:, 1:].reshape(Nv * (Tv - 1), K, D)
    n = X1.shape[0]
    total = 0.0
    count = 0
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        delta, _ = model.forward(X1[s:e])
        diff = delta - (Y1[s:e] - X1[s:e])
        total += float((diff * diff).sum())
        count += int(diff.size)
    one_step = total / count if count > 0 else 0.0

    # t_bptt-step rollout MSE
    n_starts = Tv - t_bptt
    if n_starts <= 0:
        return one_step, one_step
    total_b = 0.0
    count_b = 0
    for tstart in range(n_starts):
        s = val_seq[:, tstart].copy()
        for j in range(t_bptt):
            delta, _ = model.forward(s)
            s = s + delta
        target = val_seq[:, tstart + t_bptt]
        diff = s - target
        total_b += float((diff * diff).sum())
        count_b += int(diff.size)
    bptt_mse = total_b / count_b if count_b > 0 else 0.0
    return one_step, bptt_mse


# ======================================================================
# 6. Closed-loop rollout evaluation
# ======================================================================

def rollout(model, init_state: np.ndarray, n_steps: int) -> np.ndarray:
    """init_state: (B, K, D). Run model forward `n_steps` steps in closed loop.

    Returns predicted trajectory of shape (n_steps + 1, B, K, D), where
    index 0 is the input state.
    """
    B, K, D = init_state.shape
    out = np.zeros((n_steps + 1, B, K, D))
    out[0] = init_state
    s = init_state.copy()
    for t in range(n_steps):
        delta, _ = model.forward(s)
        s = s + delta
        out[t + 1] = s
    return out


def rollout_position_error(true_traj: np.ndarray, pred_traj: np.ndarray
                           ) -> np.ndarray:
    """Per-timestep mean Euclidean position error.

    Both arrays: (T, B, K, D=4). Returns (T,) of average pos error in box units.
    """
    T = true_traj.shape[0]
    err = np.zeros(T)
    for t in range(T):
        d = true_traj[t, ..., 0:2] - pred_traj[t, ..., 0:2]
        err[t] = float(np.sqrt((d * d).sum(axis=-1)).mean())
    return err


def rollout_velocity_error(true_traj: np.ndarray, pred_traj: np.ndarray
                           ) -> np.ndarray:
    """Per-timestep mean velocity-component error (where collisions matter).

    Both arrays: (T, B, K, D=4). Returns (T,) of average velocity RMSE.
    Collision events show up as velocity sign-flips; this metric is sensitive
    to whether the model handles them.
    """
    T = true_traj.shape[0]
    err = np.zeros(T)
    for t in range(T):
        d = true_traj[t, ..., 2:4] - pred_traj[t, ..., 2:4]
        err[t] = float(np.sqrt((d * d).sum(axis=-1)).mean())
    return err


# ======================================================================
# 7. Main
# ======================================================================

DEFAULT = dict(
    K_train=4,                # 4 balls in training (matches paper setup)
    extrapolate_K=[3, 5, 6],  # Test fewer (3), one more (5), and many more (6)
    radius=0.11,              # Larger radius -> more frequent collisions per sequence
    dt=0.05,
    T_train=25,
    T_eval=30,
    N_train=300,
    N_val=50,
    N_eval=50,
    hidden=64,
    msg_dim=8,
    epochs=60,
    batch_size=32,
    lr=3e-3,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULT["epochs"])
    p.add_argument("--batch", type=int, default=DEFAULT["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULT["lr"])
    p.add_argument("--hidden", type=int, default=DEFAULT["hidden"])
    p.add_argument("--msg-dim", type=int, default=DEFAULT["msg_dim"])
    p.add_argument("--n-train", type=int, default=DEFAULT["N_train"])
    p.add_argument("--n-val", type=int, default=DEFAULT["N_val"])
    p.add_argument("--n-eval", type=int, default=DEFAULT["N_eval"])
    p.add_argument("--t-train", type=int, default=DEFAULT["T_train"])
    p.add_argument("--t-eval", type=int, default=DEFAULT["T_eval"])
    p.add_argument("--k-train", type=int, default=DEFAULT["K_train"])
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: smaller epochs/dataset, ~30s.")
    p.add_argument("--out", type=str, default="run.json")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.epochs = 15
        args.n_train = 80
        args.n_val = 20
        args.n_eval = 20
        args.t_train = 15

    cfg = {
        "seed": args.seed,
        "K_train": args.k_train,
        "extrapolate_K": DEFAULT["extrapolate_K"],
        "radius": DEFAULT["radius"],
        "dt": DEFAULT["dt"],
        "T_train": args.t_train,
        "T_eval": args.t_eval,
        "N_train": args.n_train,
        "N_val": args.n_val,
        "N_eval": args.n_eval,
        "hidden": args.hidden,
        "msg_dim": args.msg_dim,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
    }
    print("relational-nem-bouncing-balls   cfg:", cfg, flush=True)
    print("env:", env_metadata(), flush=True)

    # ---- Datasets -----------------------------------------------------
    rng_data = np.random.default_rng(args.seed)
    print("\n[1/4] generating physics datasets ...", flush=True)
    t0 = time.time()
    train_seq = make_dataset(args.n_train, args.t_train, args.k_train,
                             DEFAULT["dt"], DEFAULT["radius"], rng_data)
    val_seq = make_dataset(args.n_val, args.t_train, args.k_train,
                           DEFAULT["dt"], DEFAULT["radius"], rng_data)
    eval_seq_train_K = make_dataset(args.n_eval, args.t_eval, args.k_train,
                                    DEFAULT["dt"], DEFAULT["radius"], rng_data)
    eval_seq_extrap = {}
    for K in DEFAULT["extrapolate_K"]:
        eval_seq_extrap[K] = make_dataset(args.n_eval, args.t_eval, K,
                                          DEFAULT["dt"], DEFAULT["radius"],
                                          rng_data)
    t_data = time.time() - t0
    print(f"    train {train_seq.shape}  val {val_seq.shape}  "
          f"eval-K{args.k_train} {eval_seq_train_K.shape}  "
          f"extrap {[v.shape for v in eval_seq_extrap.values()]}  "
          f"({t_data:.2f}s)", flush=True)

    # ---- Models -------------------------------------------------------
    rng_init_nr = np.random.default_rng(args.seed + 1)
    rng_init_r = np.random.default_rng(args.seed + 2)
    nr = NonRelationalDynamics(state_dim=4, hidden=args.hidden, rng=rng_init_nr)
    re = RelationalDynamics(state_dim=4, hidden=args.hidden,
                            msg_dim=args.msg_dim, rng=rng_init_r)
    n_params_nr = sum(p.size for p in nr.params())
    n_params_re = sum(p.size for p in re.params())
    print(f"    non-rel params: {n_params_nr}, relational params: {n_params_re}",
          flush=True)

    # ---- Train --------------------------------------------------------
    print("\n[2/4] training non-relational dynamics ...", flush=True)
    res_nr = train_model(nr, train_seq, val_seq,
                         epochs=args.epochs, batch_size=args.batch,
                         lr=args.lr, seed=args.seed, label="non-rel")
    print("\n[3/4] training relational dynamics ...", flush=True)
    res_re = train_model(re, train_seq, val_seq,
                         epochs=args.epochs, batch_size=args.batch,
                         lr=args.lr, seed=args.seed, label="rel")

    # ---- Rollout evaluation -------------------------------------------
    print("\n[4/4] rollout evaluation (closed loop) ...", flush=True)
    rollout_results: Dict[str, object] = {}
    for K_label, eval_seq in [(args.k_train, eval_seq_train_K)] + \
            [(K, eval_seq_extrap[K]) for K in DEFAULT["extrapolate_K"]]:
        T = eval_seq.shape[1]
        init = eval_seq[:, 0]                         # (N, K, 4)
        true_traj = np.transpose(eval_seq, (1, 0, 2, 3))  # (T, N, K, 4)
        pred_nr = rollout(nr, init, T - 1)
        pred_re = rollout(re, init, T - 1)
        err_nr = rollout_position_error(true_traj, pred_nr)
        err_re = rollout_position_error(true_traj, pred_re)
        verr_nr = rollout_velocity_error(true_traj, pred_nr)
        verr_re = rollout_velocity_error(true_traj, pred_re)
        rollout_results[f"K{K_label}"] = {
            "K": K_label,
            "T": T,
            "pos_err_non_relational": err_nr.tolist(),
            "pos_err_relational": err_re.tolist(),
            "vel_err_non_relational": verr_nr.tolist(),
            "vel_err_relational": verr_re.tolist(),
            "final_pos_err_non_relational": float(err_nr[-1]),
            "final_pos_err_relational": float(err_re[-1]),
            "mean_pos_err_non_relational": float(err_nr.mean()),
            "mean_pos_err_relational": float(err_re.mean()),
            "mean_vel_err_non_relational": float(verr_nr.mean()),
            "mean_vel_err_relational": float(verr_re.mean()),
        }
        print(f"    K={K_label} T={T}  "
              f"pos: non-rel {err_nr.mean():.4f} / rel {err_re.mean():.4f}  "
              f"vel: non-rel {verr_nr.mean():.4f} / rel {verr_re.mean():.4f}  "
              f"(vel rel/non-rel = {verr_re.mean()/(verr_nr.mean()+1e-12):.3f})",
              flush=True)

    # ---- Sample rollouts for visualization ----------------------------
    sample_init = eval_seq_train_K[:4, 0]          # 4 sequences
    T_samp = eval_seq_train_K.shape[1]
    sample_true = np.transpose(eval_seq_train_K[:4], (1, 0, 2, 3))
    sample_nr = rollout(nr, sample_init, T_samp - 1)
    sample_re = rollout(re, sample_init, T_samp - 1)

    out = {
        "config": cfg,
        "env": env_metadata(),
        "param_counts": {"non_relational": int(n_params_nr),
                         "relational": int(n_params_re)},
        "training": {"non_relational": res_nr, "relational": res_re},
        "rollout": rollout_results,
        "samples": {
            "true": sample_true.tolist(),
            "non_relational": sample_nr.tolist(),
            "relational": sample_re.tolist(),
        },
    }
    print("\n=== HEADLINE ===")
    print(f"  Mean rollout velocity-MSE over T={DEFAULT['T_eval']} steps")
    print(f"  (collision events show up as velocity sign flips; "
          f"this is the metric where relational helps most)")
    print(f"{'':4s}{'K':>4s}  {'non-rel':>12s}  {'relational':>12s}  {'rel/non-rel':>12s}")
    for K_label in [args.k_train] + DEFAULT["extrapolate_K"]:
        rk = rollout_results[f"K{K_label}"]
        nv = rk["mean_vel_err_non_relational"]
        rv = rk["mean_vel_err_relational"]
        ratio = rv / (nv + 1e-12)
        marker = "  <- rel wins" if ratio < 1.0 else "  (non-rel wins)"
        print(f"{'':4s}{K_label:>4d}  {nv:>12.4f}  {rv:>12.4f}  "
              f"{ratio:>12.3f}{marker}")
    print()
    print(f"  Mean rollout position-MSE over T={DEFAULT['T_eval']} steps "
          f"(dominated by ballistic drift; both models comparable):")
    print(f"{'':4s}{'K':>4s}  {'non-rel':>12s}  {'relational':>12s}")
    for K_label in [args.k_train] + DEFAULT["extrapolate_K"]:
        rk = rollout_results[f"K{K_label}"]
        np_ = rk["mean_pos_err_non_relational"]
        rp_ = rk["mean_pos_err_relational"]
        print(f"{'':4s}{K_label:>4d}  {np_:>12.4f}  {rp_:>12.4f}")

    if not args.no_save:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                args.out)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
