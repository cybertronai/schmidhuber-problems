"""
lococode-ica - Feature extraction through LOCOCODE.

Hochreiter & Schmidhuber, "Feature extraction through LOCOCODE", Neural
Computation 11(3):679-714 (1999). Companion: Hochreiter & Schmidhuber,
"Flat minima", Neural Computation 9(1):1-42 (1997).

The LOCOCODE claim
------------------
A standard autoencoder trained to MSE reconstruction develops dense codes
that mix the underlying latent factors. If we instead bias training toward
"flat minima" - low-complexity weight configurations with few effective
parameters - the autoencoder is forced to use as few hidden units per
input as possible. On sparse-source / sparsely-coded data this produces
codes that are (i) sparse and (ii) statistically near-independent, i.e. an
ICA-like decomposition motivated from a Kolmogorov-complexity / minimum
description length perspective.

Faithful v1 reduction
---------------------
The paper's flat-minimum penalty involves second-order Hessian terms that
are awkward to evaluate in pure numpy on a laptop. The 2015 *Deep Learning
in Neural Networks* survey (Schmidhuber, NN 61, sec. 5.6.4) summarises
LOCOCODE as "low-complexity coding by a regulariser that prefers networks
with as few effective free parameters as possible, producing sparse codes".
The simplest faithful reduction (and the one that the LOCOCODE follow-up
literature converged on) is:

    L = || X - X_hat ||^2_F / n            # reconstruction MSE
      + lambda_act * | H |_1 / n           # activity sparsity (low complexity)
      + lambda_w   * ( ||W_enc||^2 + ||W_dec||^2 )  # weight decay

Sparsity on the hidden activities is the single most important component
of the flat-minimum penalty in linear / shallow LOCOCODE: it pushes the
network to use the smallest number of hidden units that suffices to
reconstruct each input, which is the algorithmic definition of "few
effective parameters". Weight decay damps the co-adaptive directions
that produce dense codes. Together they reproduce the paper's headline
finding on sparse-source data: an ICA-quality decomposition.

We compare LOCOCODE against two baselines:
  * PCA (linear, 2nd-order statistics only, recovers principal axes)
  * FastICA (whitening + tanh fixed-point, the canonical ICA algorithm)

on synthetic data where the true source structure is sparse Laplacian
with a known random orthogonal mixing. Headline metric is the **Amari
distance** between the recovered and true mixing matrices: 0 = perfect
permutation+scaling, larger = mixed.

CLI
---
    python3 lococode_ica.py --seed 0
    python3 lococode_ica.py --seed 0 --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def laplacian_sources(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """Sample n sparse independent sources, k of them, from a Laplace
    distribution (super-Gaussian, kurtosis = 3). Variance is normalised to 1."""
    u = rng.uniform(-0.5 + 1e-9, 0.5 - 1e-9, size=(n, k))
    s = -np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))     # Laplace, scale=1
    s = (s - s.mean(0, keepdims=True)) / s.std(0, keepdims=True)
    return s


def random_orthogonal(rng: np.random.Generator, k: int) -> np.ndarray:
    M = rng.standard_normal((k, k))
    Q, R = np.linalg.qr(M)
    # Fix sign so QR is deterministic up to seed
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1.0
    return Q * sign[np.newaxis, :]


def generate_dataset(seed: int = 0, k: int = 8, n_samples: int = 2000):
    """Generate (X, S, A) where S are independent Laplacian sources, A is a
    random orthogonal mixing matrix and X = S @ A.T is the observed signal."""
    rng = np.random.default_rng(seed)
    S = laplacian_sources(rng, k, n_samples)
    A = random_orthogonal(rng, k)
    X = S @ A.T
    # Center observations (sources are already zero-mean)
    X = X - X.mean(0, keepdims=True)
    return X, S, A


def whiten(X: np.ndarray):
    """Return (Z, K) where Z = X @ K.T has identity covariance.

    LOCOCODE / ICA-style whitening: this is the standard preprocessing that
    reduces the source-recovery problem to finding an orthogonal rotation.
    Without whitening, the L1 sparsity gradient has no scale anchor and the
    autoencoder collapses W toward zero (with W_dec compensating in scale).
    """
    n, k = X.shape
    Xc = X - X.mean(0, keepdims=True)
    cov = Xc.T @ Xc / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    K = (eigvecs / np.sqrt(eigvals)).T
    Z = Xc @ K.T
    return Z, K


# ---------------------------------------------------------------------------
# LOCOCODE: linear sparse autoencoder trained with L1 activity + weight decay
# ---------------------------------------------------------------------------

class LococodeAE:
    """Tied linear autoencoder Z -> H -> Z_hat operating on whitened input Z.

    Encoder:  H     = Z @ W^T          (W is k x k)
    Decoder:  Z_hat = H @ W            (tied: decoder is W^T's transpose = W)

    With whitened input, the optimal reconstruction is achieved by any
    orthogonal W (since W^T W = I gives Z_hat = Z). MSE alone has a flat
    minimum on the orthogonal manifold; L1 sparsity on H breaks the
    rotational symmetry by pulling W toward the orientation that makes the
    codes sparsest - which on Laplacian-source data is exactly the demixing
    direction. This is the LOCOCODE / flat-minimum-search story in its
    cleanest reduction.
    """

    def __init__(self, k: int, rng: np.random.Generator):
        # Init close to a random orthogonal matrix (good basin for FMS).
        M = rng.standard_normal((k, k))
        Q, R = np.linalg.qr(M)
        sign = np.sign(np.diag(R))
        sign[sign == 0] = 1.0
        self.W = Q * sign[np.newaxis, :]

    def encode(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.W.T

    def decode(self, H: np.ndarray) -> np.ndarray:
        return H @ self.W

    def forward(self, Z: np.ndarray):
        H = self.encode(Z)
        R = self.decode(H)
        return H, R

    # Compatibility with code that expects W_enc / W_dec (tied: both = W).
    @property
    def W_enc(self): return self.W
    @property
    def W_dec(self): return self.W.T


def train_lococode(
    X: np.ndarray,
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.05,
    lambda_act: float = 0.5,
    lambda_w: float = 1e-4,
    snapshot_every: int = 5,
    A_true: np.ndarray | None = None,
    S_true: np.ndarray | None = None,
):
    """Train the LOCOCODE autoencoder on X. Returns (model, history).

    Pipeline:
      1. Whiten X -> Z (cov(Z) = I).
      2. Train tied autoencoder Z -> H -> Z_hat with MSE + L1 + weight decay.
      3. Recovered demixer in original-X space is W @ K where Z = X @ K.T.
    """
    rng = np.random.default_rng(seed)
    n, k = X.shape
    Z, K_white = whiten(X)
    model = LococodeAE(k, rng)

    history = {
        "epoch": [],
        "recon_loss": [],
        "act_l1": [],
        "kurtosis_mean": [],
        "sparsity_frac": [],
        "amari": [],
        "snapshots": [],
        "K_white": K_white,
    }

    for epoch in range(epochs + 1):
        H_full, R_full = model.forward(Z)
        recon = float(np.mean((Z - R_full) ** 2))
        act = float(np.mean(np.abs(H_full)))
        kurt = float(np.mean(_kurtosis(H_full)))
        sp = float(np.mean(np.abs(H_full) < 0.2))
        # Demixer in the original X coordinates: H = X @ (W @ K_white).T
        W_xspace = model.W @ K_white
        amari = (
            float(amari_distance(W_xspace, A_true))
            if A_true is not None else float("nan")
        )

        history["epoch"].append(epoch)
        history["recon_loss"].append(recon)
        history["act_l1"].append(act)
        history["kurtosis_mean"].append(kurt)
        history["sparsity_frac"].append(sp)
        history["amari"].append(amari)
        if epoch % snapshot_every == 0 or epoch == epochs:
            history["snapshots"].append({
                "epoch": epoch,
                "W": model.W.copy(),
                "W_xspace": W_xspace.copy(),
            })

        if epoch == epochs:
            break

        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            zb = Z[b]
            H, R = model.forward(zb)             # H = Z W^T, R = H W = Z W^T W
            err = R - zb                          # (b, k)

            # Tied gradient of recon ||R - Z||^2 with R = Z W^T W:
            #   dL/dW = 2/b * ( H^T err + err^T (zb W^T) )      (k, k)
            # which equals 2/b * d/dW [Z W^T W - Z]^T [Z W^T W - Z]
            grad_W = 2.0 / zb.shape[0] * (H.T @ err + err.T @ (zb @ model.W.T))

            # L1 sparsity on H = Z W^T:
            #   d|H|/dW[a,b] = sign(H[:, a]).T @ zb[:, b] / b
            grad_W += lambda_act * (np.sign(H).T @ zb) / zb.shape[0]

            # Weight decay
            grad_W += 2.0 * lambda_w * model.W

            model.W -= lr * grad_W

    return model, history


def _kurtosis(H: np.ndarray) -> np.ndarray:
    """Excess kurtosis per column (Gaussian = 0, Laplace = 3)."""
    H_c = H - H.mean(0, keepdims=True)
    var = (H_c ** 2).mean(0)
    var = np.where(var < 1e-12, 1e-12, var)
    return (H_c ** 4).mean(0) / (var ** 2) - 3.0


# ---------------------------------------------------------------------------
# Baselines: PCA and a small FastICA
# ---------------------------------------------------------------------------

def pca_decompose(X: np.ndarray, k: int):
    """Return W_pca such that H = X @ W_pca.T are the principal components."""
    Xc = X - X.mean(0, keepdims=True)
    cov = Xc.T @ Xc / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Whitening style PC: components scaled to unit variance
    W_pca = (eigvecs / np.sqrt(eigvals + 1e-9)).T
    return W_pca[:k], eigvals


def fastica(X: np.ndarray, seed: int = 0, max_iter: int = 200, tol: float = 1e-6):
    """Symmetric FastICA with tanh contrast. Returns W_ica such that
    H = X @ W_ica.T are the recovered independent sources."""
    rng = np.random.default_rng(seed)
    n, k = X.shape
    Xc = X - X.mean(0, keepdims=True)

    # Whitening
    cov = Xc.T @ Xc / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    K = (eigvecs / np.sqrt(eigvals)).T              # (k, k) whitener
    Z = Xc @ K.T                                    # whitened observations

    # Symmetric decorrelation init
    W = rng.standard_normal((k, k))
    W, _ = np.linalg.qr(W)

    for _ in range(max_iter):
        WZ = Z @ W.T                                # (n, k)
        gW = np.tanh(WZ)
        gprime = 1.0 - gW ** 2
        W_new = (gW.T @ Z) / n - np.diag(gprime.mean(0)) @ W
        # symmetric decorrelation
        u, s, vt = np.linalg.svd(W_new, full_matrices=False)
        W_new = u @ vt
        if np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1.0)) < tol:
            W = W_new
            break
        W = W_new

    W_ica = W @ K                                   # demixer in original space
    return W_ica


# ---------------------------------------------------------------------------
# Amari distance
# ---------------------------------------------------------------------------

def amari_distance(W_recover: np.ndarray, A_true: np.ndarray) -> float:
    """Distance between recovered demixer W and true mixing A. P = W @ A.
    Returns 0 iff P is a generalised permutation (permutation x diagonal).
    Lower = better. The expression below is the standard Amari index."""
    P = W_recover @ A_true
    P = np.abs(P)
    k = P.shape[0]
    row_max = P.max(axis=1, keepdims=True)
    col_max = P.max(axis=0, keepdims=True)
    row_max = np.where(row_max < 1e-12, 1e-12, row_max)
    col_max = np.where(col_max < 1e-12, 1e-12, col_max)
    s_rows = (P / row_max).sum(axis=1) - 1.0
    s_cols = (P / col_max).sum(axis=0) - 1.0
    return float(s_rows.sum() + s_cols.sum()) / (2.0 * k * (k - 1))


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------

def run_one_seed(seed: int, k: int, n_samples: int, epochs: int):
    X, S, A = generate_dataset(seed=seed, k=k, n_samples=n_samples)

    t0 = time.time()
    model, hist = train_lococode(
        X, seed=seed, epochs=epochs, A_true=A, S_true=S,
    )
    t_loco = time.time() - t0

    K_white = hist["K_white"]
    W_loco_xspace = model.W @ K_white               # demixer in original X space

    # PCA baseline
    W_pca, _eig = pca_decompose(X, k)

    # FastICA baseline
    W_ica = fastica(X, seed=seed)

    H_loco = X @ W_loco_xspace.T
    H_pca = X @ W_pca.T
    H_ica = X @ W_ica.T

    # Reconstruction in original X coordinates: X_hat = (X @ W_loco^T) @ W_dec
    # In whitened space, R = Z W^T W; back to X: X_hat = R @ K^{-T}
    # Easier: just compute MSE on whitened reconstruction.
    Z_loco, _ = whiten(X)
    R_loco = (Z_loco @ model.W.T) @ model.W
    recon_z = float(np.mean((Z_loco - R_loco) ** 2))

    metrics = {
        "lococode": {
            "amari": amari_distance(W_loco_xspace, A),
            "kurtosis_mean": float(np.mean(_kurtosis(H_loco))),
            "sparsity_frac_lt_0.2": float(np.mean(np.abs(H_loco) < 0.2)),
            "recon_mse_whitened": recon_z,
            "wallclock_s": t_loco,
        },
        "pca": {
            "amari": amari_distance(W_pca, A),
            "kurtosis_mean": float(np.mean(_kurtosis(H_pca))),
            "sparsity_frac_lt_0.2": float(np.mean(np.abs(H_pca) < 0.2)),
        },
        "fastica": {
            "amari": amari_distance(W_ica, A),
            "kurtosis_mean": float(np.mean(_kurtosis(H_ica))),
            "sparsity_frac_lt_0.2": float(np.mean(np.abs(H_ica) < 0.2)),
        },
    }

    return {
        "X": X, "S": S, "A": A,
        "model": model, "history": hist,
        "W_loco_xspace": W_loco_xspace,
        "W_pca": W_pca, "W_ica": W_ica,
        "H_loco": H_loco, "H_pca": H_pca, "H_ica": H_ica,
        "metrics": metrics,
        "seed": seed,
        "k": k, "n_samples": n_samples, "epochs": epochs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8, help="number of sources/observations")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="if >1, run a sweep starting at --seed and report mean/std")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.n_seeds == 1:
        out = run_one_seed(args.seed, args.k, args.n_samples, args.epochs)
        m = out["metrics"]
        if not args.quiet:
            print(f"seed={args.seed} k={args.k} n={args.n_samples} epochs={args.epochs}")
            print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
                  f"{platform.platform()}")
            print()
            hdr = f"{'method':<10} {'Amari':>9} {'kurtosis':>10} {'sparsity':>10}"
            print(hdr)
            print("-" * len(hdr))
            for name in ("lococode", "pca", "fastica"):
                e = m[name]
                print(f"{name:<10} {e['amari']:>9.4f} {e['kurtosis_mean']:>10.3f} "
                      f"{e['sparsity_frac_lt_0.2']:>10.3f}")
            print()
            print(f"  LOCOCODE wallclock: {m['lococode']['wallclock_s']:.2f} s")
            print(f"  LOCOCODE whitened-recon MSE: "
                  f"{m['lococode']['recon_mse_whitened']:.4f}")
        return out

    rows = []
    for s in range(args.seed, args.seed + args.n_seeds):
        out = run_one_seed(s, args.k, args.n_samples, args.epochs)
        rows.append(out["metrics"])
    if not args.quiet:
        print(f"sweep over {args.n_seeds} seeds (Amari, lower = better)")
        for name in ("lococode", "pca", "fastica"):
            vals = np.array([r[name]["amari"] for r in rows])
            print(f"  {name:<10} mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")
    return rows


if __name__ == "__main__":
    main()
