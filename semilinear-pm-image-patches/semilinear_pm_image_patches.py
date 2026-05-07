"""semilinear-pm-image-patches -- Schmidhuber, Eldracher, Foltin, *Semilinear
predictability minimization produces well-known feature detectors*, Neural
Computation 8(4):773--786, 1996.

The 1996 paper applies Predictability Minimization (PM, Schmidhuber 1992) with
a *semilinear* (one-nonlinearity) network to natural-image patches. The
encoder weights converge to **oriented edge / Gabor-like filters** strongly
resembling V1 simple-cell receptive fields and the qualitative ICA solution
later popularised by Bell-Sejnowski (1997) and Olshausen-Field (1996).

Algorithm (semilinear PM, "variance-decorrelation" variant)
-----------------------------------------------------------
Two adversarial sets of weights, sharing the same code:

    encoder W (M x D):      y   = W x             (LINEAR; rows orthonormal)
    predictor V (per unit): p_i = sum_{j != i} V_full[i, j] * z_j
                            where z_j = y_j^2 - E[y_j^2]    <-- the ONE nonlinearity

    L_pred = sum_i (p_i - z_i)^2

The *predictor* descends `L_pred` (it wants to predict each centred squared
code z_i from the others z_-i). The *encoder* ascends `L_pred` (it wants
its codes y_i to have **statistically independent variances**). For
zero-mean signals, variance-independence equals higher-order independence,
which is the ICA criterion. The "one nonlinearity" of the title is the
squaring of the codes inside the predictor -- equivalently, the predictor
input is the semilinear feature vector (y, y^2). With a purely linear
predictor and orthonormal W, codes are already decorrelated and the game
is degenerate; the squaring is what surfaces the higher-order signal.

On natural-image-statistics patches (1/f power spectrum + sparse oriented
structure), the equilibrium W rows are oriented edge / bar detectors --
the familiar V1 simple-cell template. This is the "well-known feature
detectors" of the title, and is qualitatively the same set of filters
that Bell-Sejnowski InfoMax ICA and Olshausen-Field sparse coding produce
on the same data.

Synthetic natural-image dataset
-------------------------------
We generate 64x64 base images by:

    1.  Pink-noise (1/f^beta) Gaussian field via FFT (gives natural-image
        power spectrum but is purely Gaussian -- has no higher-order
        structure for PM to find).
    2.  Add ~30 random Gaussian-windowed oriented bars per image. These
        sparse oriented edges inject the non-Gaussian higher-order
        statistics that ICA / PM can extract.
    3.  Whole-image standardisation (zero mean, unit std).

We then sample N random `patch_size`x`patch_size` patches, subtract per-patch
DC, and ZCA-whiten the patch pool. ZCA whitening is the standard
preprocessing for ICA / PM on images (Bell-Sejnowski 1997, Hyvarinen 2001):
it removes second-order correlations so the encoder's job is purely
higher-order independence.

Pure numpy + matplotlib, deterministic, runs in ~30 s on an M-series CPU.

CLI
---

    python3 semilinear_pm_image_patches.py --seed 0
    # ~30 s, prints headline metrics as JSON.

    python3 semilinear_pm_image_patches.py --grad-check
    # Numerical-vs-analytic gradient check on a tiny random batch.

The default recipe (seed 0, M=16, patch=8, n_steps=2500) reproduces the
headline reported in README.md / Section Results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

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


def excess_kurtosis(z: np.ndarray) -> float:
    """Fisher's kurtosis (excess over Gaussian = 0)."""
    z = z - z.mean()
    m2 = float((z ** 2).mean())
    m4 = float((z ** 4).mean())
    if m2 < 1e-12:
        return 0.0
    return m4 / (m2 ** 2) - 3.0


# ----------------------------------------------------------------------
# Synthetic natural-image patches
# ----------------------------------------------------------------------

def pink_noise_image(rng: np.random.Generator, size: int = 64,
                     beta: float = 2.0) -> np.ndarray:
    """1/f^beta power-spectrum Gaussian field.

    beta = 2 (the default) gives the empirical power-law of natural scenes
    (Field 1987). Pink noise alone is Gaussian and has no kurtotic
    structure: ICA / PM on pure pink noise is degenerate. We add sparse
    oriented bars in `add_random_bars` to inject higher-order statistics.
    """
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    fx = np.fft.fftfreq(size).reshape(1, -1)
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0  # avoid div-by-zero (DC handled separately below)
    amp = 1.0 / (f ** (beta / 2.0))
    F = (rng.standard_normal((size, size))
         + 1j * rng.standard_normal((size, size))) * amp
    F[0, 0] = 0.0  # zero mean
    return np.real(np.fft.ifft2(F))


def add_random_bars(rng: np.random.Generator, img: np.ndarray,
                    n_bars: int = 30) -> np.ndarray:
    """Add random oriented Gaussian-windowed bars (sparse edge structure).

    Each bar has a random centre, orientation, length, thickness, and
    contrast; the cross-section is Gaussian (a smooth bar, not a hard
    line). The per-image total of `n_bars` bars makes the patch
    distribution kurtotic in the direction of the dominant bar
    orientations -- exactly the higher-order structure that ICA / PM
    extracts as oriented filters.
    """
    H, W = img.shape
    out = img.copy()
    yy, xx = np.indices((H, W)).astype(np.float64)
    for _ in range(n_bars):
        cy = rng.uniform(0, H)
        cx = rng.uniform(0, W)
        theta = rng.uniform(0, np.pi)
        length = rng.uniform(3.0, 12.0)
        thickness = rng.uniform(0.7, 1.5)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        contrast = sign * rng.uniform(0.5, 2.5)
        d_perp = (yy - cy) * np.cos(theta) - (xx - cx) * np.sin(theta)
        d_para = (yy - cy) * np.sin(theta) + (xx - cx) * np.cos(theta)
        bar = (np.exp(-d_perp ** 2 / (2 * thickness ** 2))
               * np.exp(-d_para ** 2 / (2 * length ** 2)))
        out += contrast * bar
    return out


def make_dataset(rng: np.random.Generator, n_images: int = 30,
                 image_size: int = 64, patch_size: int = 8,
                 n_patches: int = 30000, beta: float = 2.0,
                 n_bars: int = 30):
    """Sample raw image patches from the synthetic natural-image pool."""
    images = np.empty((n_images, image_size, image_size))
    for i in range(n_images):
        img = pink_noise_image(rng, size=image_size, beta=beta)
        img = add_random_bars(rng, img, n_bars=n_bars)
        img = (img - img.mean()) / (img.std() + 1e-8)
        images[i] = img
    patches = np.empty((n_patches, patch_size * patch_size))
    for p in range(n_patches):
        i = int(rng.integers(0, n_images))
        cy = int(rng.integers(0, image_size - patch_size + 1))
        cx = int(rng.integers(0, image_size - patch_size + 1))
        patches[p] = images[i, cy:cy + patch_size,
                            cx:cx + patch_size].flatten()
    patches -= patches.mean(axis=1, keepdims=True)  # remove per-patch DC
    return patches, images


def zca_whiten(X: np.ndarray, eps: float = 1e-2):
    """Symmetric ZCA whitening: y = U diag(1/sqrt(lambda+eps)) U^T x.

    Standard preprocessing for ICA / PM on images (Bell-Sejnowski 1997).
    Returns the whitened data and the whitening matrix.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / Xc.shape[0]
    U, S, _ = np.linalg.svd(cov)
    Wzca = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return Xc @ Wzca.T, Wzca


# ----------------------------------------------------------------------
# Predictability-minimisation model
# ----------------------------------------------------------------------

def orthonormalize_rows(W: np.ndarray) -> np.ndarray:
    """Project W onto the Stiefel manifold: rows orthonormal.

    Polar projection via SVD: W -> U V^T where W = U S V^T. This is the
    nearest orthonormal matrix in Frobenius norm and keeps the encoder on
    the manifold throughout PM training. Without this, semilinear PM with
    a linear predictor degenerates to encoder saturation (it grows ||W||
    until tanh outputs saturate at +-1, which makes binary codes that the
    linear predictor cannot fit).
    """
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt


def init_params(rng: np.random.Generator, n_in: int, n_hidden: int):
    """Encoder W: M x D, orthonormal rows. Predictor V: M x (M-1)."""
    W = rng.standard_normal((n_hidden, n_in))
    W = orthonormalize_rows(W)
    V = rng.standard_normal((n_hidden, n_hidden - 1)) * 0.01
    return W, V


def make_Vfull(V: np.ndarray) -> np.ndarray:
    """Embed each predictor row into a M x M matrix with zero diagonal.

    `Vfull[j, i]` is the linear weight of `y_i` in the predictor for `y_j`,
    forced to zero for `i = j` so unit j cannot trivially predict itself.
    """
    M = V.shape[0]
    Vfull = np.zeros((M, M))
    for j in range(M):
        Vfull[j, :j] = V[j, :j]
        Vfull[j, j + 1:] = V[j, j:]
    return Vfull


def forward(W: np.ndarray, V: np.ndarray, X: np.ndarray):
    """X: (B, D) -> y, z, p, Vfull, mu_z, sigma_z.

    Linear encoder y = W x. Squared codes are *standardised* per coord:

        y2     = y * y
        mu_z   = E_b[y2]                 (detached batch mean)
        sigma_z = std_b[y2 - mu_z] + eps (detached batch std)
        z      = (y2 - mu_z) / sigma_z

    Standardisation keeps the predictor's input scale O(1) regardless of
    how kurtotic any particular code unit becomes; without it, a single
    rare-large y_k value blows up V's gradient. Mean and std are
    treated as constants in the gradient (a stop-grad), so the analytic
    chain rule below ignores their dependence on W.
    """
    Vfull = make_Vfull(V)
    y = X @ W.T
    y2 = y * y
    mu_z = y2.mean(axis=0, keepdims=True)
    centred = y2 - mu_z
    sigma_z = centred.std(axis=0, keepdims=True) + 1e-3
    z = centred / sigma_z
    p = z @ Vfull.T
    return y, z, p, Vfull, mu_z, sigma_z


def predictor_grad(z: np.ndarray, p: np.ndarray, V: np.ndarray) -> np.ndarray:
    """dL_pred / dV.

        L_pred = sum_b sum_j (p_{b,j} - z_{b,j})^2
              = sum_b sum_j (sum_l Vfull[j,l] z_{b,l} - z_{b,j})^2

    so
        dL_pred / dVfull[j, k] = 2 sum_b err_{b,j} z_{b,k}      (k != j)
                              = 2 (err.T @ z)[j, k]

    Diagonal is zero by construction (unit j does not predict its own
    squared code). The dV slot for predictor j skips the j-th column.
    """
    M = V.shape[0]
    err = p - z
    dVfull = (err.T @ z) * 2.0 / err.shape[0]
    np.fill_diagonal(dVfull, 0.0)
    dV = np.empty_like(V)
    for j in range(M):
        dV[j, :j] = dVfull[j, :j]
        dV[j, j:] = dVfull[j, j + 1:]
    return dV


def encoder_grad(W: np.ndarray, X: np.ndarray, y: np.ndarray,
                 z: np.ndarray, p: np.ndarray, Vfull: np.ndarray,
                 sigma_z: np.ndarray) -> np.ndarray:
    """dL_pred / dW with linear encoder y = W x and standardised squared
    codes z = (y^2 - mu_z) / sigma_z (mu_z, sigma_z stop-grad).

        err_{b,j} = p_{b,j} - z_{b,j}

        d z_{b,k} / d y_{b,k} = 2 y_{b,k} / sigma_z[k]    (chain through y^2)
        d err_{b,j} / d y_{b,k}:
            j == k: - 2 y_{b,k} / sigma_z[k]
            j != k:  Vfull[j, k] * 2 y_{b,k} / sigma_z[k]

        d L / d y_{b,k} = 2 sum_j err_{b,j} * d err_{b,j} / d y_{b,k}
                       = (4 y_{b,k} / sigma_z[k])
                         * ((err @ Vfull)_{b,k} - err_{b,k})

        d L / d W_{k,d} = (dL_dy.T @ X) / B

    Encoder *ascends* this gradient (maximises L_pred). The training loop
    applies W += lr_e * dW.
    """
    err = p - z
    dL_dy = (4.0 / sigma_z) * y * (err @ Vfull - err)
    return dL_dy.T @ X / X.shape[0]


def numerical_grad_check(seed: int = 0, eps: float = 1e-6) -> dict:
    """Sanity: max abs error of analytic vs central-difference gradient.

    Note: the encoder gradient treats the batch mean E[y^2] as detached
    (constant). The numerical check therefore freezes mu_z at the value
    computed from the unperturbed W; this matches the analytic gradient.
    """
    rng = np.random.default_rng(seed)
    D, M, B = 5, 4, 7
    W = rng.standard_normal((M, D)) * 0.3
    V = rng.standard_normal((M, M - 1)) * 0.3
    X = rng.standard_normal((B, D))
    y0, z0, p0, Vfull0, mu_z0, sigma_z0 = forward(W, V, X)
    dV_an = predictor_grad(z0, p0, V)
    dW_an = encoder_grad(W, X, y0, z0, p0, Vfull0, sigma_z0)

    def loss_with_stops(W_, V_, mu_z_, sigma_z_):
        Vfull_ = make_Vfull(V_)
        y_ = X @ W_.T
        z_ = (y_ * y_ - mu_z_) / sigma_z_
        p_ = z_ @ Vfull_.T
        return float(np.mean(np.sum((p_ - z_) ** 2, axis=1)))

    dV_num = np.zeros_like(V)
    for idx in np.ndindex(V.shape):
        V[idx] += eps
        Lp = loss_with_stops(W, V, mu_z0, sigma_z0)
        V[idx] -= 2 * eps
        Lm = loss_with_stops(W, V, mu_z0, sigma_z0)
        V[idx] += eps
        dV_num[idx] = (Lp - Lm) / (2 * eps)

    dW_num = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        W[idx] += eps
        Lp = loss_with_stops(W, V, mu_z0, sigma_z0)
        W[idx] -= 2 * eps
        Lm = loss_with_stops(W, V, mu_z0, sigma_z0)
        W[idx] += eps
        dW_num[idx] = (Lp - Lm) / (2 * eps)

    return {
        "max_err_V": float(np.max(np.abs(dV_num - dV_an))),
        "max_err_W": float(np.max(np.abs(dW_num - dW_an))),
    }


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(seed: int = 0, n_hidden: int = 16, patch_size: int = 8,
          n_patches: int = 30000, n_steps: int = 2500, batch: int = 256,
          lr_e: float = 0.01, lr_p: float = 0.02, n_p_inner: int = 2,
          beta: float = 2.0, n_bars: int = 30, n_images: int = 30,
          v_l2: float = 1e-3, grad_clip: float = 1.0,
          snap_every: int = 0):
    """Predictability-minimisation training on synthetic natural patches.

    Returns a dict with the trained W, the predictor V, the whitening
    matrix, the raw / whitened patches, the source images, the per-step
    history, and the encoder snapshots (when `snap_every > 0`).
    """
    rng = np.random.default_rng(seed)
    raw_patches, images = make_dataset(
        rng, n_images=n_images, image_size=64, patch_size=patch_size,
        n_patches=n_patches, beta=beta, n_bars=n_bars,
    )
    X, Wzca = zca_whiten(raw_patches, eps=1e-2)
    n_in = X.shape[1]
    W, V = init_params(rng, n_in, n_hidden)

    history: dict = {"step": [], "L_pred": [], "y_kurt_mean": []}
    snapshots = []
    N = X.shape[0]

    for step in range(n_steps):
        idx = rng.integers(0, N, size=batch)
        Xb = X[idx]

        # Predictor inner updates: descend L_pred wrt V.
        # Includes a small L2 penalty on V to keep the predictor bounded;
        # without it the predictor can drift unboundedly under the
        # squared-code regression and destabilise training.
        for _ in range(n_p_inner):
            y, z, p, _, _, _ = forward(W, V, Xb)
            dV = predictor_grad(z, p, V) + v_l2 * V
            V -= lr_p * dV

        # Encoder outer update: ascend L_pred wrt W (with grad-norm clip).
        y, z, p, Vfull, _, sigma_z = forward(W, V, Xb)
        L_pred = float(np.mean(np.sum((p - z) ** 2, axis=1)))
        # Track per-batch excess kurtosis of code as a proxy for "Gabor-ness".
        m2 = (y ** 2).mean(axis=0) + 1e-12
        m4 = (y ** 4).mean(axis=0)
        y_kurt = float((m4 / m2 ** 2 - 3.0).mean())
        dW = encoder_grad(W, Xb, y, z, p, Vfull, sigma_z)
        gnorm = float(np.linalg.norm(dW))
        if grad_clip > 0 and gnorm > grad_clip:
            dW = dW * (grad_clip / gnorm)
        W += lr_e * dW
        # Project rows back to the Stiefel manifold (orthonormal). With a
        # linear encoder this is a hard requirement: without orthonormality
        # the encoder maximises L_pred trivially by inflating ||W|| (since
        # variance scales quadratically with W). The Stiefel constraint
        # forces purely higher-order, ICA-style independence.
        W = orthonormalize_rows(W)

        history["step"].append(step)
        history["L_pred"].append(L_pred)
        history["y_kurt_mean"].append(y_kurt)

        if snap_every and (step % snap_every == 0 or step == n_steps - 1):
            snapshots.append((step, W.copy()))

    return {"W": W, "V": V, "Wzca": Wzca, "X": X, "raw_patches": raw_patches,
            "images": images, "history": history, "snapshots": snapshots,
            "patch_size": patch_size, "n_hidden": n_hidden}


# ----------------------------------------------------------------------
# Filter-quality evaluation (does the encoder look like Gabors?)
# ----------------------------------------------------------------------

def filter_orientation_metrics(W: np.ndarray, patch_size: int) -> list:
    """For each filter, report orientation, peak frequency, angular concentration.

    Method: 2-D FFT of the filter, zero-out DC. The pixel of largest
    magnitude defines the dominant frequency and orientation. The angular
    concentration is the fraction of total spectral energy within a
    +- 22.5 deg band of the dominant orientation. Gabor-like filters
    exhibit high concentration (> 0.5); blob / DC-like filters score low.
    """
    M, _ = W.shape
    out = []
    cy = patch_size / 2.0
    cx = patch_size / 2.0
    yy, xx = np.indices((patch_size, patch_size)).astype(np.float64)
    ang = np.arctan2(yy - cy, xx - cx) % np.pi
    for i in range(M):
        f = W[i].reshape(patch_size, patch_size)
        F = np.fft.fftshift(np.fft.fft2(f))
        mag = np.abs(F).copy()
        mag[int(cy), int(cx)] = 0.0
        py, px = np.unravel_index(np.argmax(mag), mag.shape)
        dy, dx = py - cy, px - cx
        theta = float(np.arctan2(dy, dx) % np.pi)
        diff = np.minimum(np.abs(ang - theta), np.pi - np.abs(ang - theta))
        in_band = (diff < np.pi / 8.0).astype(np.float64)
        in_band[int(cy), int(cx)] = 0.0
        denom = float(mag.sum()) + 1e-12
        conc = float((mag * in_band).sum() / denom)
        out.append({
            "orientation_rad": theta,
            "peak_freq_pix": float(np.hypot(dy, dx)),
            "concentration": conc,
        })
    return out


def code_kurtosis(W: np.ndarray, X: np.ndarray) -> dict:
    """Excess kurtosis of each linear-encoder code y_i = W_i^T x.

    Kurtosis > 0 = sparse / heavy-tailed code. Random orthonormal
    projection gives ~0 (close to Gaussian by central-limit on patches);
    Gabor-like projections of natural-image patches give kurtosis well
    above zero (typical: 2-5). The increase from random to trained is
    the standard quantitative signature of an ICA / sparse-coding fit.
    """
    y = X @ W.T
    ks = [excess_kurtosis(y[:, i]) for i in range(W.shape[0])]
    return {
        "y_kurtosis_mean": float(np.mean(ks)),
        "y_kurtosis_max": float(np.max(ks)),
        "y_kurtosis_min": float(np.min(ks)),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hidden", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--n-patches", type=int, default=30000)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr-e", type=float, default=0.01)
    parser.add_argument("--lr-p", type=float, default=0.02)
    parser.add_argument("--v-l2", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-p-inner", type=int, default=2)
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--n-bars", type=int, default=30)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--grad-check", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON results here.")
    args = parser.parse_args()

    if args.grad_check:
        errs = numerical_grad_check(seed=args.seed)
        print(json.dumps(errs, indent=2))
        return

    t0 = time.time()
    res = train(
        seed=args.seed, n_hidden=args.n_hidden, patch_size=args.patch_size,
        n_patches=args.n_patches, n_steps=args.n_steps, batch=args.batch,
        lr_e=args.lr_e, lr_p=args.lr_p, n_p_inner=args.n_p_inner,
        beta=args.beta, n_bars=args.n_bars, n_images=args.n_images,
        v_l2=args.v_l2, grad_clip=args.grad_clip, snap_every=0,
    )
    elapsed = time.time() - t0

    metrics = filter_orientation_metrics(res["W"], args.patch_size)
    concs = np.array([m["concentration"] for m in metrics])
    kurt = code_kurtosis(res["W"], res["X"])

    summary = {
        "config": vars(args),
        "wallclock_s": elapsed,
        "final_L_pred": res["history"]["L_pred"][-1],
        "initial_L_pred": res["history"]["L_pred"][0],
        "n_oriented_filters_conc_gt_0.5": int((concs > 0.5).sum()),
        "n_oriented_filters_conc_gt_0.4": int((concs > 0.4).sum()),
        "mean_concentration": float(concs.mean()),
        "median_concentration": float(np.median(concs)),
        "kurtosis": kurt,
        "env": {
            "git_commit": git_hash(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        payload = {
            "summary": summary,
            "history": res["history"],
            "filter_metrics": metrics,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f)


if __name__ == "__main__":
    main()
