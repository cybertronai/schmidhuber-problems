"""neural-em-shapes -- Greff, van Steenkiste, Schmidhuber,
*Neural Expectation Maximization*, NIPS 2017 (arXiv:1708.03498).

K-slot Neural EM on synthetic static binary shapes (24x24 canvas, 3
shapes per image drawn from {square, disc, triangle}).

Each slot k carries a hidden state theta_k in R^H and explains each
pixel through a per-slot Bernoulli emission:

    mu_k     = sigmoid(W_dec @ theta_k + b_dec)        # (D,)
    log p(x_i | k) = x_i log mu_{k,i} + (1-x_i) log(1 - mu_{k,i})

EM is unrolled for T iterations.  At each iteration t = 0..T-1:

    E-step  gamma_{b,k,i}^t = softmax_k log_lik_{b,k,i}^t      (uniform prior)
    M-step  r_{b,k,:}^t = gamma_{b,k,:}^t * (x_b - mu_{b,k,:}^t)
            theta_{b,k,:}^{t+1} = tanh(W_x r_{b,k,:}^t + W_h theta_{b,k,:}^t + b_h)

The training objective is the per-iteration mixture negative
log-likelihood, summed across iterations:

    L = sum_t -mean_{b,i} logsumexp_k (log_lik_{b,k,i}^t - log K)

We backprop through every iteration with manual numpy chain rule, then
update via Adam.

Slot states theta^0 are initialised with small Gaussian noise per slot
per image; this is what breaks the K-slot symmetry so different slots
end up explaining different objects.

CLI
    python3 neural_em_shapes.py --seed 0
    python3 neural_em_shapes.py --seed 0 --quick
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


def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def softmax_k(log_x):
    # softmax along axis=1 (slot axis) for arrays of shape (B, K, D)
    m = log_x.max(axis=1, keepdims=True)
    z = np.exp(log_x - m)
    return z / z.sum(axis=1, keepdims=True)


def logsumexp_k(log_x):
    # logsumexp along axis=1
    m = log_x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(log_x - m).sum(axis=1, keepdims=True))).squeeze(1)


# ----------------------------------------------------------------------
# Synthetic shapes dataset (pure numpy, no external data)
# ----------------------------------------------------------------------

CANVAS = 24
SHAPE_SIZE_MIN = 2
SHAPE_SIZE_MAX = 4


def _draw_square(canvas, mask, label_id, cx, cy, s, rng):
    """Filled square of half-size s centred at (cy, cx)."""
    y0, y1 = max(0, cy - s), min(CANVAS, cy + s + 1)
    x0, x1 = max(0, cx - s), min(CANVAS, cx + s + 1)
    canvas[y0:y1, x0:x1] = 1.0
    mask[y0:y1, x0:x1] = label_id


def _draw_disc(canvas, mask, label_id, cx, cy, r, rng):
    """Filled disc of radius r."""
    yy, xx = np.ogrid[:CANVAS, :CANVAS]
    in_disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    canvas[in_disc] = 1.0
    mask[in_disc] = label_id


def _draw_triangle(canvas, mask, label_id, cx, cy, s, rng):
    """Filled upward-pointing isoceles triangle, base 2s, height 2s."""
    for dy in range(-s, s + 1):
        # y - cy = dy. Triangle goes from y=cy-s (apex) to y=cy+s (base).
        # half-width at row (cy+dy) is (dy + s) (linear ramp from 0 at apex to s at base)
        hw = dy + s
        if hw < 0:
            continue
        y = cy + dy
        if 0 <= y < CANVAS:
            x0 = max(0, cx - hw)
            x1 = min(CANVAS, cx + hw + 1)
            canvas[y, x0:x1] = 1.0
            mask[y, x0:x1] = label_id


SHAPE_DRAWERS = [_draw_square, _draw_disc, _draw_triangle]
SHAPE_NAMES = ["square", "disc", "triangle"]


def make_image(rng: np.random.Generator, n_shapes: int = 3,
               max_tries: int = 20):
    """Place n_shapes non-mostly-overlapping shapes on a CANVASxCANVAS grid.

    Returns (image, label_mask) where label_mask in {0..n_shapes} per pixel
    (0 = background, 1..n_shapes = which shape generated it).  When two
    shapes overlap at a pixel, the *later-drawn* shape's id wins (so the
    label mask is a hard segmentation).  Empty pixels are background.
    """
    img = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    mask = np.zeros((CANVAS, CANVAS), dtype=np.int64)
    placed = 0
    tries = 0
    while placed < n_shapes and tries < max_tries:
        tries += 1
        s = rng.integers(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX + 1)
        cx = rng.integers(s, CANVAS - s)
        cy = rng.integers(s, CANVAS - s)
        kind = rng.integers(0, 3)
        # Try to keep overlap small: discard if the new bbox is mostly inside
        # an existing shape (>60% overlap).
        bbox_y0, bbox_y1 = cy - s, cy + s + 1
        bbox_x0, bbox_x1 = cx - s, cx + s + 1
        existing = (mask[bbox_y0:bbox_y1, bbox_x0:bbox_x1] > 0).mean()
        if existing > 0.6:
            continue
        SHAPE_DRAWERS[kind](img, mask, placed + 1, cx, cy, s, rng)
        placed += 1
    return img, mask


def make_dataset(n: int, rng: np.random.Generator, n_shapes: int = 3):
    X = np.zeros((n, CANVAS * CANVAS), dtype=np.float64)
    M = np.zeros((n, CANVAS * CANVAS), dtype=np.int64)
    for i in range(n):
        img, mask = make_image(rng, n_shapes=n_shapes)
        X[i] = img.ravel()
        M[i] = mask.ravel()
    return X, M


# ----------------------------------------------------------------------
# Model: parameters + Adam state
# ----------------------------------------------------------------------

def init_params(rng: np.random.Generator, D: int, H: int, K: int):
    s_dec = 1.0 / np.sqrt(H)
    s_h = 1.0 / np.sqrt(H)
    s_x = 1.0 / np.sqrt(D)
    params = {
        # decoder: theta -> mu
        "W_dec": rng.uniform(-s_dec, s_dec, size=(D, H)),
        "b_dec": np.zeros(D),
        # M-step recurrence: theta_new = tanh(W_x r + W_h theta + b_h)
        "W_x": rng.uniform(-s_x, s_x, size=(H, D)),
        "W_h": rng.uniform(-s_h, s_h, size=(H, H)),
        "b_h": np.zeros(H),
        # Per-slot learnable initial state -- breaks K-slot symmetry.
        # theta_0[b, k, :] = theta_init[k, :] + noise.
        "theta_init": rng.normal(0.0, 0.5, size=(K, H)),
    }
    return params


def init_adam_state(params):
    return ({k: np.zeros_like(v) for k, v in params.items()},
            {k: np.zeros_like(v) for k, v in params.items()},
            [0])


def adam_step(params, grads, state, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, tlist = state
    tlist[0] += 1
    t = tlist[0]
    for k in params:
        g = grads[k]
        m[k] = b1 * m[k] + (1 - b1) * g
        v[k] = b2 * v[k] + (1 - b2) * g * g
        m_hat = m[k] / (1 - b1 ** t)
        v_hat = v[k] / (1 - b2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ----------------------------------------------------------------------
# Forward + backward through unrolled EM
# ----------------------------------------------------------------------

LOG_EPS = 1e-6


def forward(params, x, noise, T):
    """Run T EM iterations.  Returns:
        loss (scalar, sum over iterations of mean mixture NLL),
        history (list of dicts, one per iteration with mu, gamma, theta_in,
                 theta_out (or None at last), r, h, log_mix, log_lik_centered),
        gamma_T (final responsibility, shape (B, K, D)).

    x has shape (B, D) of {0,1}.  noise has shape (B, K, H).
    The actual initial slot state is theta_0 = params["theta_init"][None] + noise,
    so the per-slot learnable bias gets backproped.
    """
    W_dec = params["W_dec"]
    b_dec = params["b_dec"]
    W_x = params["W_x"]
    W_h = params["W_h"]
    b_h = params["b_h"]
    theta_init = params["theta_init"]  # (K, H)

    B, D = x.shape
    K = noise.shape[1]
    H = noise.shape[2]
    log_K = np.log(K)

    theta_t = theta_init[None, :, :] + noise  # (B, K, H)
    history = []
    total_loss = 0.0

    for t in range(T):
        # Decode
        # z_t = theta_t @ W_dec.T + b_dec  -> shape (B, K, D)
        z_t = np.einsum("bkh,dh->bkd", theta_t, W_dec) + b_dec
        mu_t = sigmoid(z_t)  # (B, K, D)
        mu_c = np.clip(mu_t, LOG_EPS, 1.0 - LOG_EPS)

        # log_lik_t[b,k,i] = log Bernoulli(x[b,i] | mu[b,k,i])
        x_b = x[:, None, :]  # (B, 1, D)
        log_lik = x_b * np.log(mu_c) + (1.0 - x_b) * np.log(1.0 - mu_c)

        # E-step responsibilities (uniform prior cancels in softmax)
        gamma = softmax_k(log_lik)  # (B, K, D)

        # Mixture log-likelihood loss for this iteration
        # L_t = -mean over (b,i) of logsumexp_k(log_lik - log K)
        # Use: log_mix = logsumexp(log_lik, k_axis) - log K
        log_mix = logsumexp_k(log_lik) - log_K  # (B, D)
        loss_t = -log_mix.mean()
        total_loss += loss_t

        # M-step (skip on last iteration -- theta_T is unused)
        if t < T - 1:
            r = gamma * (x_b - mu_t)  # (B, K, D)
            h = (
                np.einsum("bkd,hd->bkh", r, W_x)
                + np.einsum("bkh,gh->bkg", theta_t, W_h)
                + b_h
            )  # (B, K, H)
            theta_next = np.tanh(h)  # (B, K, H)
        else:
            r = None
            h = None
            theta_next = None

        history.append(
            dict(
                t=t,
                theta_in=theta_t,
                z=z_t,
                mu=mu_t,
                mu_c=mu_c,
                log_lik=log_lik,
                gamma=gamma,
                log_mix=log_mix,
                r=r,
                h=h,
                theta_out=theta_next,
                loss=loss_t,
            )
        )

        if t < T - 1:
            theta_t = theta_next

    return total_loss, history, history[-1]["gamma"]


def backward(params, x, history, T):
    """Compute parameter gradients given forward history."""
    W_dec = params["W_dec"]
    W_x = params["W_x"]
    W_h = params["W_h"]

    B, D = x.shape
    K = history[0]["gamma"].shape[1]
    H = history[0]["theta_in"].shape[2]
    inv_BD = 1.0 / (B * D)
    x_b = x[:, None, :]  # (B, 1, D)

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # Gradient flowing back into theta_t at iteration t (initially zero
    # because no iteration after T-1 contributes).
    d_theta_next = np.zeros((B, K, H))

    for t in reversed(range(T)):
        h = history[t]
        theta_t = h["theta_in"]
        z_t = h["z"]
        mu_t = h["mu"]
        mu_c = h["mu_c"]
        gamma = h["gamma"]

        # ---- Gradient through M-step (only for t < T-1) ----
        if t < T - 1:
            theta_out = h["theta_out"]
            r_t = h["r"]
            # d_h_t = d_theta_next * (1 - tanh^2)
            d_h_t = d_theta_next * (1.0 - theta_out * theta_out)
            # h = W_x r + W_h theta + b_h
            # dW_x[h_idx, d_idx] += sum_{b,k} d_h_t[b,k,h_idx] * r_t[b,k,d_idx]
            grads["W_x"] += np.einsum("bkh,bkd->hd", d_h_t, r_t)
            grads["W_h"] += np.einsum("bkh,bkg->hg", d_h_t, theta_t)
            grads["b_h"] += d_h_t.sum(axis=(0, 1))
            # gradient back into r and theta_t (via the M-step skip)
            d_r = np.einsum("bkh,hd->bkd", d_h_t, W_x)
            # h_new[b,k,g] = sum_h theta_t[b,k,h] * W_h[g, h]
            # => d theta_t[b,k,h] = sum_g d_h_t[b,k,g] * W_h[g, h]
            d_theta_t_via_M = np.einsum("bkg,gh->bkh", d_h_t, W_h)
            # r = gamma * (x - mu)
            d_gamma_via_r = d_r * (x_b - mu_t)
            d_mu_via_r = -d_r * gamma
        else:
            d_gamma_via_r = np.zeros_like(gamma)
            d_mu_via_r = np.zeros_like(mu_t)
            d_theta_t_via_M = np.zeros((B, K, H))

        # ---- Gradient back through gamma = softmax_k(log_lik) ----
        # d_log_lik (via gamma) = gamma * (d_gamma - sum_k(d_gamma * gamma))
        s = (d_gamma_via_r * gamma).sum(axis=1, keepdims=True)
        d_log_lik_via_gamma = gamma * (d_gamma_via_r - s)

        # ---- Direct gradient from loss L_t = -mean logsumexp_k(log_lik - log K) ----
        # dL_t / d log_lik[b,k,i] = -gamma[b,k,i] / (B*D)
        d_log_lik_direct = -gamma * inv_BD

        d_log_lik = d_log_lik_via_gamma + d_log_lik_direct

        # ---- log_lik = x log mu_c + (1-x) log (1-mu_c) ----
        # d log_lik / d mu_c = (x - mu_c) / (mu_c * (1 - mu_c))
        # We are not differentiating through clipping; treat mu_c as a
        # function of mu (identity outside the clip range).  Using mu_c in
        # the denominator is numerically safe.
        d_mu_via_loglik = d_log_lik * (x_b - mu_c) / (mu_c * (1.0 - mu_c))

        d_mu = d_mu_via_r + d_mu_via_loglik

        # ---- mu = sigmoid(z) ----
        d_z = d_mu * mu_t * (1.0 - mu_t)

        # ---- z = theta @ W_dec.T + b_dec, einsum 'bkh,dh->bkd' ----
        grads["W_dec"] += np.einsum("bkd,bkh->dh", d_z, theta_t)
        grads["b_dec"] += d_z.sum(axis=(0, 1))

        d_theta_t_via_dec = np.einsum("bkd,dh->bkh", d_z, W_dec)
        d_theta_next = d_theta_t_via_M + d_theta_t_via_dec

    # After the loop, d_theta_next holds dL/d theta_0.  Since
    # theta_0 = theta_init[None] + noise, sum across the batch dim gives
    # dL/d theta_init.
    grads["theta_init"] = d_theta_next.sum(axis=0)
    return grads


def gradient_check(seed: int = 0):
    """Numerical gradient check on a tiny instance.  Returns max relative
    error across all parameter entries sampled."""
    rng = np.random.default_rng(seed)
    D = 8
    H = 4
    K = 2
    T = 3
    B = 2
    params = init_params(rng, D, H, K)
    x = (rng.uniform(0, 1, size=(B, D)) > 0.5).astype(np.float64)
    noise = rng.normal(0, 0.1, size=(B, K, H))

    loss_an, hist, _ = forward(params, x, noise, T)
    grads = backward(params, x, hist, T)

    eps = 1e-5
    max_rel = 0.0
    rng2 = np.random.default_rng(seed + 1)
    for name, p in params.items():
        flat = p.reshape(-1)
        n = flat.size
        # sample 6 entries to check
        idxs = rng2.choice(n, size=min(6, n), replace=False)
        for idx in idxs:
            saved = flat[idx]
            flat[idx] = saved + eps
            l_plus, _, _ = forward(params, x, noise, T)
            flat[idx] = saved - eps
            l_minus, _, _ = forward(params, x, noise, T)
            flat[idx] = saved
            num = (l_plus - l_minus) / (2 * eps)
            ana = grads[name].reshape(-1)[idx]
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
    return max_rel


# ----------------------------------------------------------------------
# Evaluation: pixel NMI (normalized mutual information)
#   over foreground (x_i = 1) pixels only.
# ----------------------------------------------------------------------

def pixel_nmi(gamma: np.ndarray, mask: np.ndarray, x: np.ndarray):
    """gamma: (B, K, D) responsibilities;
    mask: (B, D) integer labels {0=bg, 1..n_shapes}.
    x: (B, D) of {0,1}.

    Slot-binding is a per-image phenomenon (slot 0 might be the square in
    image 1 and the triangle in image 2), so we compute NMI per image
    over foreground pixels and average across the batch.

    Returns (avg_nmi, n_fg_total).
    """
    B = gamma.shape[0]
    pred = gamma.argmax(axis=1)  # (B, D)
    nmi_sum = 0.0
    n_fg_total = 0
    images_counted = 0
    for b in range(B):
        fg = x[b].astype(bool)
        n_fg = int(fg.sum())
        if n_fg < 2:
            continue
        true = mask[b][fg]
        if np.unique(true).size < 2:
            continue  # no segmentation possible (only one shape labelled)
        nmi_sum += _nmi(true, pred[b][fg])
        n_fg_total += n_fg
        images_counted += 1
    if images_counted == 0:
        return 0.0, 0
    return nmi_sum / images_counted, n_fg_total


def _nmi(u: np.ndarray, v: np.ndarray):
    """Normalized Mutual Information (arithmetic-mean denominator)."""
    n = u.size
    if n == 0:
        return 0.0
    u_vals = np.unique(u)
    v_vals = np.unique(v)
    # Joint histogram
    joint = np.zeros((u_vals.size, v_vals.size))
    u_idx = {x: i for i, x in enumerate(u_vals)}
    v_idx = {x: i for i, x in enumerate(v_vals)}
    for a, b in zip(u, v):
        joint[u_idx[a.item()], v_idx[b.item()]] += 1
    p = joint / n
    pu = p.sum(axis=1, keepdims=True)
    pv = p.sum(axis=0, keepdims=True)
    eps = 1e-12
    Hu = -(pu * np.log(pu + eps)).sum()
    Hv = -(pv * np.log(pv + eps)).sum()
    log_term = np.log(p / (pu @ pv) + eps)
    log_term = np.where(p > 0, log_term, 0.0)
    MI = (p * log_term).sum()
    if Hu + Hv < eps:
        return 1.0
    return float(2.0 * MI / (Hu + Hv))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(
    seed: int = 0,
    H: int = 24,
    K: int = 3,
    T: int = 4,
    n_train: int = 1024,
    n_test: int = 128,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-3,
    n_shapes: int = 3,
    noise_p: float = 0.10,
    init_noise_std: float = 0.1,
    quick: bool = False,
    verbose: bool = True,
):
    """Train N-EM.

    noise_p: bit-flip noise applied to inputs during training.  Original
    Greff 2017 uses ~10% salt-and-pepper noise to break symmetry between
    slots and force per-object specialisation rather than a single
    permutation-invariant reconstruction.
    init_noise_std: std of Gaussian noise added on top of the learnable
    theta_init for each (image, slot).  Primary breaker of K-slot symmetry
    is theta_init itself; this just adds per-image jitter.
    """
    if quick:
        epochs = 3
        n_train = 256
        n_test = 64
    rng = np.random.default_rng(seed)
    D = CANVAS * CANVAS

    # Datasets (clean ground-truth)
    X_train, M_train = make_dataset(n_train, rng, n_shapes=n_shapes)
    X_test, M_test = make_dataset(n_test, rng, n_shapes=n_shapes)

    # Init
    params = init_params(rng, D, H, K)
    state = init_adam_state(params)

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_nmi": [],
    }
    # We keep a held-out batch for visualization (per-iteration gamma)
    viz_idx = np.arange(min(8, n_test))
    viz_x = X_test[viz_idx]
    viz_m = M_test[viz_idx]
    # fixed noise for viz so frames are comparable across epochs
    viz_noise = rng.normal(0, init_noise_std, size=(viz_x.shape[0], K, H))

    # And per-epoch snapshots of viz gamma over iterations
    viz_snapshots = []  # each: dict with (epoch, gamma_per_iter, mu_per_iter)

    n_batches = n_train // batch_size

    best_nmi = -1.0
    best_params = None
    best_epoch = -1
    t0 = time.time()
    for epoch in range(epochs):
        # Shuffle training set
        perm = rng.permutation(n_train)
        train_loss_sum = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x_b = X_train[idx].copy()
            # Bit-flip salt-and-pepper noise during training
            if noise_p > 0:
                flip_mask = rng.uniform(0, 1, size=x_b.shape) < noise_p
                x_b = np.where(flip_mask, 1.0 - x_b, x_b)
            noise = rng.normal(0, init_noise_std, size=(batch_size, K, H))
            loss, hist_, _ = forward(params, x_b, noise, T)
            grads = backward(params, x_b, hist_, T)
            # gradient clip (L2 norm 5.0)
            total_norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
            if total_norm > 5.0:
                scale = 5.0 / (total_norm + 1e-8)
                for k in grads:
                    grads[k] *= scale
            adam_step(params, grads, state, lr=lr)
            train_loss_sum += float(loss)
        train_loss = train_loss_sum / max(n_batches, 1)

        # Test loss + NMI on CLEAN inputs
        test_loss, test_nmi = evaluate(
            params, X_test, M_test, K=K, H=H, T=T, batch=64,
            seed=seed * 10 + epoch, init_noise_std=init_noise_std,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_nmi"].append(test_nmi)

        if test_nmi > best_nmi:
            best_nmi = test_nmi
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}

        # Snapshot viz: per-iteration gamma + mu using the fixed viz noise
        _, viz_hist, _ = forward(params, viz_x, viz_noise, T)
        viz_snapshots.append(
            dict(
                epoch=epoch,
                gamma_per_iter=[h["gamma"].copy() for h in viz_hist],
                mu_per_iter=[h["mu"].copy() for h in viz_hist],
                loss_per_iter=[float(h["loss"]) for h in viz_hist],
            )
        )

        if verbose:
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"test_loss {test_loss:.4f} | test_NMI {test_nmi:.3f}",
                flush=True,
            )

    wall = time.time() - t0

    # Best-NMI snapshot of the viz batch -- the headline picture should
    # come from the checkpoint with cleanest slot binding, not the noisy
    # late-training collapse-prone state.
    if best_params is None:
        best_params = params
        best_epoch = epochs - 1
        best_nmi = history["test_nmi"][-1]
    _, best_viz_hist, _ = forward(best_params, viz_x, viz_noise, T)
    best_viz_snapshot = dict(
        epoch=best_epoch,
        gamma_per_iter=[h["gamma"].copy() for h in best_viz_hist],
        mu_per_iter=[h["mu"].copy() for h in best_viz_hist],
        loss_per_iter=[float(h["loss"]) for h in best_viz_hist],
    )

    return dict(
        params=params,
        best_params=best_params,
        best_nmi=best_nmi,
        best_epoch=best_epoch,
        best_viz_snapshot=best_viz_snapshot,
        history=history,
        viz_x=viz_x,
        viz_m=viz_m,
        viz_noise=viz_noise,
        viz_snapshots=viz_snapshots,
        X_test=X_test,
        M_test=M_test,
        wall=wall,
        init_noise_std=init_noise_std,
    )


def evaluate(params, X, M, K, H, T, batch=64, seed=0, init_noise_std=0.1):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    losses = []
    per_img_nmis = []  # one entry per image (counted images only)
    for i in range(0, n, batch):
        x = X[i : i + batch]
        m = M[i : i + batch]
        noise = rng.normal(0, init_noise_std, size=(x.shape[0], K, H))
        loss, hist, gamma_T = forward(params, x, noise, T)
        losses.append(float(hist[-1]["loss"]) * x.shape[0])
        # per-image NMI -- recompute here so we can collect each image
        pred = gamma_T.argmax(axis=1)
        for b in range(x.shape[0]):
            fg = x[b].astype(bool)
            if fg.sum() < 2:
                continue
            true = m[b][fg]
            if np.unique(true).size < 2:
                continue
            per_img_nmis.append(_nmi(true, pred[b][fg]))
    test_loss = sum(losses) / n
    test_nmi = float(np.mean(per_img_nmis)) if per_img_nmis else 0.0
    return test_loss, test_nmi


# ----------------------------------------------------------------------
# Driver: save run.json with everything needed for viz + GIF
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="3 epochs, smaller dataset (smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--T", type=int, default=4, help="EM iterations")
    ap.add_argument("--K", type=int, default=3, help="number of slots")
    ap.add_argument("--H", type=int, default=24,
                    help="hidden dim per slot (small = bottleneck-forced specialisation)")
    ap.add_argument("--n-shapes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--noise-p", type=float, default=0.10,
                    help="bit-flip noise during training (Greff 2017 uses ~0.10)")
    ap.add_argument("--init-noise-std", type=float, default=0.1)
    ap.add_argument("--no-grad-check", action="store_true")
    ap.add_argument("--out", type=str, default="run.json")
    args = ap.parse_args()

    print(f"neural-em-shapes  seed={args.seed}  K={args.K}  T={args.T}  H={args.H}",
          flush=True)
    print(f"  env: {env_metadata()}", flush=True)

    if not args.no_grad_check:
        print("  gradient check ...", flush=True)
        rel = gradient_check(seed=0)
        print(f"    max relative error = {rel:.2e} (target < 1e-3)", flush=True)
        assert rel < 1e-3, f"gradient check failed: rel={rel}"

    epochs = args.epochs
    if epochs is None:
        epochs = 30 if not args.quick else 3

    out = train(
        seed=args.seed,
        K=args.K,
        T=args.T,
        H=args.H,
        epochs=epochs,
        n_shapes=args.n_shapes,
        lr=args.lr,
        noise_p=args.noise_p,
        init_noise_std=args.init_noise_std,
        quick=args.quick,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)

    final_test_loss = out["history"]["test_loss"][-1]
    final_test_nmi = out["history"]["test_nmi"][-1]
    print(
        f"  final  test_loss {final_test_loss:.4f}  "
        f"test_NMI {final_test_nmi:.3f}  ({out['wall']:.1f}s)",
        flush=True,
    )
    print(
        f"  best   epoch    {out['best_epoch']:3d}   "
        f"test_NMI {out['best_nmi']:.3f}  (used for viz)",
        flush=True,
    )

    # ----- Persist run.json (small: config + history + summary)
    snaps = _subsample_snapshots(out["viz_snapshots"], n_keep=8)
    payload = {
        "config": {
            "seed": args.seed, "K": args.K, "T": args.T, "H": args.H,
            "epochs": epochs, "lr": args.lr, "n_shapes": args.n_shapes,
            "noise_p": args.noise_p, "init_noise_std": args.init_noise_std,
            "canvas": CANVAS, "shape_size_min": SHAPE_SIZE_MIN,
            "shape_size_max": SHAPE_SIZE_MAX,
        },
        "env": env_metadata(),
        "history": out["history"],
        "wall_seconds": out["wall"],
        "final": {
            "test_loss": final_test_loss,
            "test_nmi": final_test_nmi,
        },
        "best": {
            "epoch": out["best_epoch"],
            "test_nmi": out["best_nmi"],
        },
        # Per-snapshot loss + epoch (NOT the heavy gamma/mu arrays).
        "viz_meta": {
            "snapshot_epochs": [s["epoch"] for s in snaps],
            "snapshot_loss_per_iter": [s["loss_per_iter"] for s in snaps],
            "best_viz_loss_per_iter": out["best_viz_snapshot"]["loss_per_iter"],
            "best_viz_epoch": out["best_viz_snapshot"]["epoch"],
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_path}", flush=True)

    # ----- Persist run_viz.npz (heavy arrays, binary, gzip-compressed)
    npz_path = os.path.join(here, "run_viz.npz")
    np.savez_compressed(
        npz_path,
        viz_x=out["viz_x"].astype(np.float32),
        viz_m=out["viz_m"].astype(np.int16),
        # Per-epoch snapshots: only gamma is needed (the GIF shows
        # responsibilities, not reconstructions).  float16 is plenty.
        snap_gamma=np.stack(
            [np.stack(s["gamma_per_iter"], axis=0) for s in snaps], axis=0
        ).astype(np.float16),
        snap_epochs=np.array([s["epoch"] for s in snaps], dtype=np.int64),
        # Best snapshot (single timestep): keep both gamma and mu for the
        # static slot-reconstruction PNG.
        best_gamma=np.stack(out["best_viz_snapshot"]["gamma_per_iter"], axis=0).astype(np.float32),
        best_mu=np.stack(out["best_viz_snapshot"]["mu_per_iter"], axis=0).astype(np.float32),
        best_epoch=np.array(out["best_viz_snapshot"]["epoch"], dtype=np.int64),
    )
    print(f"  wrote {npz_path}", flush=True)


def _subsample_snapshots(snaps, n_keep=12):
    if len(snaps) <= n_keep:
        return snaps
    idxs = np.linspace(0, len(snaps) - 1, n_keep).round().astype(int)
    return [snaps[i] for i in idxs]


if __name__ == "__main__":
    main()
