"""em-segmentation-isbi: patch-based pixel classifier for membrane segmentation.

Reference paper:
  Cireşan, Giusti, Gambardella, Schmidhuber,
  *Deep neural networks segment neuronal membranes in electron microscopy
  images*, NIPS 2012.

The original paper trains a deep CNN with 65x65 patches on the ISBI 2012
EM stack (Drosophila ssTEM, 30 slices at 512x512). The SPEC for v1 forbids
external dataset downloads, so this stub substitutes a **synthetic
Voronoi-EM** dataset generated entirely in numpy:

  * cells   : random Voronoi tessellation of an HxW canvas
  * membrane: 1-pixel boundary where 4-neighbors disagree on cell id
              (this is the binary ground-truth mask)
  * texture : per-cell mean intensity + fine Gaussian noise + sparse
              dark organelles (small Gaussian blobs)
  * noise   : multiplicative gain + additive Gaussian
  * blur    : single 3x3 box blur to mimic optical PSF

Architecture: 32x32 grayscale patch -> 2-hidden-layer MLP
(1024 -> 256 -> 128 -> 1) trained with SGD + momentum. The MLP predicts
membrane probability for the **centre pixel** of the patch (sliding
window classifier), exactly like the paper's CNN but with a patch-MLP
substitute that fits the v1 numpy/CPU/<5min budget.

CLI: ``python3 em_segmentation_isbi.py --seed 0``

Determinism: every random source goes through ``np.random.default_rng``
seeded from --seed; running with the same seed twice on the same machine
gives bit-identical output (verified manually before commit).
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
from pathlib import Path
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Synthetic Voronoi-EM dataset
# --------------------------------------------------------------------------- #
def make_voronoi_em(
    h: int,
    w: int,
    n_cells: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image HxW float32 in [0,1], membrane HxW uint8, cell_id HxW int).

    Pure numpy. Builds a Voronoi tessellation by argmin Euclidean distance
    to ``n_cells`` random seed points, then derives the membrane mask from
    4-neighbor cell-id disagreement.
    """
    seeds = np.stack([
        rng.integers(0, h, size=n_cells),
        rng.integers(0, w, size=n_cells),
    ], axis=1).astype(np.float32)  # (n_cells, 2)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Squared distance to each seed: (h, w, n_cells).
    # Memory-friendly: do the argmin in a loop over batches if needed; for
    # the v1 sizes we use (<=128x128 with <=80 seeds), full broadcast is fine.
    dy = yy[:, :, None] - seeds[None, None, :, 0]
    dx = xx[:, :, None] - seeds[None, None, :, 1]
    d2 = dy * dy + dx * dx  # (h, w, n_cells)
    cell_id = d2.argmin(axis=2).astype(np.int32)  # (h, w)

    # Membrane = pixel whose right or down 4-neighbor lives in a different cell.
    diff_right = np.zeros_like(cell_id, dtype=bool)
    diff_down = np.zeros_like(cell_id, dtype=bool)
    diff_right[:, :-1] = cell_id[:, :-1] != cell_id[:, 1:]
    diff_down[:-1, :] = cell_id[:-1, :] != cell_id[1:, :]
    membrane = (diff_right | diff_down).astype(np.uint8)
    # Symmetrise: also mark the other side of the boundary so the membrane is
    # 1-px on both sides of the edge (matches the visual width of EM
    # membranes, which are darker than a single pixel boundary).
    membrane[:, 1:] |= diff_right[:, :-1]
    membrane[1:, :] |= diff_down[:-1, :]

    # Per-cell brightness mean (cells differ in cytoplasmic intensity).
    mean_per_cell = rng.uniform(0.55, 0.85, size=n_cells).astype(np.float32)
    image = mean_per_cell[cell_id]

    # Membranes are darker than cytoplasm.
    image = np.where(membrane.astype(bool), rng.uniform(0.05, 0.18, size=image.shape).astype(np.float32), image)

    # Fine intra-cell texture: low-amplitude Gaussian noise.
    image = image + rng.normal(0.0, 0.04, size=image.shape).astype(np.float32)

    # Sparse organelles: a handful of small dark Gaussian blobs scattered
    # inside cells (avoid placing them on top of the membrane).
    n_org = max(2, n_cells // 3)
    for _ in range(n_org):
        cy = rng.integers(2, h - 2)
        cx = rng.integers(2, w - 2)
        if membrane[cy, cx]:
            continue
        sigma = rng.uniform(0.8, 1.6)
        amp = rng.uniform(0.18, 0.35)
        # Tight 5x5 window
        ys = slice(max(0, cy - 3), min(h, cy + 4))
        xs = slice(max(0, cx - 3), min(w, cx + 4))
        gy, gx = np.mgrid[ys.start - cy:ys.stop - cy, xs.start - cx:xs.stop - cx]
        blob = np.exp(-(gy * gy + gx * gx) / (2.0 * sigma * sigma)).astype(np.float32)
        image[ys, xs] -= amp * blob

    # Multiplicative gain + additive noise.
    gain = 1.0 + rng.normal(0.0, 0.05, size=image.shape).astype(np.float32)
    image = image * gain + rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)

    # 3x3 box blur for a mild PSF.
    pad = np.pad(image, 1, mode="edge")
    blurred = np.zeros_like(image)
    for di in range(3):
        for dj in range(3):
            blurred += pad[di:di + h, dj:dj + w]
    image = blurred / 9.0

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, membrane, cell_id


def make_dataset(
    n_train: int,
    n_test: int,
    h: int,
    w: int,
    n_cells: int,
    seed: int,
):
    rng = np.random.default_rng(seed + 1)
    train_imgs, train_masks = [], []
    test_imgs, test_masks = [], []
    for _ in range(n_train):
        img, mask, _ = make_voronoi_em(h, w, n_cells, rng)
        train_imgs.append(img)
        train_masks.append(mask)
    for _ in range(n_test):
        img, mask, _ = make_voronoi_em(h, w, n_cells, rng)
        test_imgs.append(img)
        test_masks.append(mask)
    return (
        np.stack(train_imgs),
        np.stack(train_masks),
        np.stack(test_imgs),
        np.stack(test_masks),
    )


# --------------------------------------------------------------------------- #
# Patch sampling (sliding-window pixel classifier)
# --------------------------------------------------------------------------- #
def reflect_pad(image: np.ndarray, pad: int) -> np.ndarray:
    """Reflect-pad a single 2D image so that we can extract a (P,P) patch
    centred on every pixel without falling off the image."""
    return np.pad(image, pad, mode="reflect")


def gather_patch(
    padded: np.ndarray, cy: int, cx: int, patch: int
) -> np.ndarray:
    """Return a (patch, patch) window for the pixel at (cy, cx) in the
    original (un-padded) image coordinates. ``padded`` must already be
    padded by ``patch // 2`` on each side. The centre pixel sits at
    offset (patch//2, patch//2) inside the window (for even patch sizes
    the centre is biased one pixel toward the bottom-right, which is fine
    for the membrane-classification task — the network sees the full
    32x32 context around each pixel either way)."""
    return padded[cy:cy + patch, cx:cx + patch].copy()


def sample_balanced_patches(
    images: np.ndarray,
    masks: np.ndarray,
    n_patches: int,
    patch: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``n_patches`` patches roughly 50/50 membrane/non-membrane.

    Returns (X, y) with X shape (n_patches, patch*patch) and y shape (n_patches,).
    """
    n_pos_target = n_patches // 2
    n_neg_target = n_patches - n_pos_target
    half = patch // 2

    n_imgs, h, w = images.shape
    paddeds = [reflect_pad(images[i], half) for i in range(n_imgs)]

    # Pre-flatten coords by class for each image.
    pos_coords_per_img = [np.argwhere(masks[i] == 1) for i in range(n_imgs)]
    neg_coords_per_img = [np.argwhere(masks[i] == 0) for i in range(n_imgs)]

    X = np.empty((n_patches, patch * patch), dtype=np.float32)
    y = np.empty(n_patches, dtype=np.float32)

    for k in range(n_patches):
        i = int(rng.integers(0, n_imgs))
        if k < n_pos_target:
            cands = pos_coords_per_img[i]
            label = 1.0
        else:
            cands = neg_coords_per_img[i]
            label = 0.0
        if cands.shape[0] == 0:
            # Degenerate; fall through to the other class.
            cands = pos_coords_per_img[i] if label == 0.0 else neg_coords_per_img[i]
            label = 1.0 - label
        idx = int(rng.integers(0, cands.shape[0]))
        cy, cx = int(cands[idx, 0]), int(cands[idx, 1])
        X[k] = gather_patch(paddeds[i], cy, cx, patch).reshape(-1)
        y[k] = label

    # Shuffle so the two halves are mixed before SGD.
    perm = rng.permutation(n_patches)
    return X[perm], y[perm]


def all_patches_for_image(image: np.ndarray, patch: int) -> np.ndarray:
    """Return every pixel's patch for ``image``: shape (h*w, patch*patch)."""
    half = patch // 2
    padded = reflect_pad(image, half)
    h, w = image.shape
    out = np.empty((h * w, patch * patch), dtype=np.float32)
    k = 0
    for cy in range(h):
        for cx in range(w):
            out[k] = padded[cy:cy + patch, cx:cx + patch].reshape(-1)
            k += 1
    return out


# --------------------------------------------------------------------------- #
# 2-hidden-layer MLP pixel classifier
# --------------------------------------------------------------------------- #
@dataclass
class MLP:
    """input -> tanh -> tanh -> sigmoid; weights are stored as plain numpy."""
    Ws: list = field(default_factory=list)
    bs: list = field(default_factory=list)
    vWs: list = field(default_factory=list)  # momentum buffers
    vbs: list = field(default_factory=list)

    @classmethod
    def make(cls, layer_sizes, rng: np.random.Generator) -> "MLP":
        Ws, bs, vWs, vbs = [], [], [], []
        for fin, fout in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(1.0 / fin).astype(np.float32)
            Ws.append((rng.standard_normal((fin, fout)).astype(np.float32) * scale))
            bs.append(np.zeros(fout, dtype=np.float32))
            vWs.append(np.zeros_like(Ws[-1]))
            vbs.append(np.zeros_like(bs[-1]))
        return cls(Ws=Ws, bs=bs, vWs=vWs, vbs=vbs)

    def forward(self, x: np.ndarray):
        """Return (probs, cache) where cache is the list of pre/post acts."""
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = h @ W + b
            if i < len(self.Ws) - 1:
                h = np.tanh(z)
            else:
                # Sigmoid on the final scalar logit.
                h = 1.0 / (1.0 + np.exp(-z))
            acts.append(h)
        return h.reshape(-1), acts

    def backward(self, acts, y: np.ndarray):
        """Cross-entropy gradient through the network. Returns dWs, dbs."""
        # acts[-1]: (B, 1), y: (B,)
        p = acts[-1].reshape(-1)
        # dL/dz_last = p - y (sigmoid + BCE).
        dz = (p - y).reshape(-1, 1) / max(1, y.shape[0])
        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.Ws)
        for i in range(len(self.Ws) - 1, -1, -1):
            h_in = acts[i]  # (B, fin)
            dWs[i] = h_in.T @ dz
            dbs[i] = dz.sum(axis=0)
            if i > 0:
                # Through tanh.
                d_h = dz @ self.Ws[i].T
                tanh_out = acts[i]  # post-tanh activation of layer i-1
                dz = d_h * (1.0 - tanh_out * tanh_out)
        return dWs, dbs

    def sgd_step(self, dWs, dbs, lr: float, momentum: float, weight_decay: float) -> None:
        for i in range(len(self.Ws)):
            self.vWs[i] = momentum * self.vWs[i] - lr * (dWs[i] + weight_decay * self.Ws[i])
            self.Ws[i] = self.Ws[i] + self.vWs[i]
            self.vbs[i] = momentum * self.vbs[i] - lr * dbs[i]
            self.bs[i] = self.bs[i] + self.vbs[i]

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=np.float32)
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            p, _ = self.forward(xb)
            out[i:i + batch_size] = p
        return out


# --------------------------------------------------------------------------- #
# Baselines + metrics
# --------------------------------------------------------------------------- #
def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Hand-rolled 3x3 Sobel gradient magnitude in numpy."""
    pad = np.pad(image, 1, mode="edge")
    gx = (
        -1.0 * pad[:-2, :-2] + 1.0 * pad[:-2, 2:]
        - 2.0 * pad[1:-1, :-2] + 2.0 * pad[1:-1, 2:]
        - 1.0 * pad[2:, :-2] + 1.0 * pad[2:, 2:]
    )
    gy = (
        -1.0 * pad[:-2, :-2] - 2.0 * pad[:-2, 1:-1] - 1.0 * pad[:-2, 2:]
        + 1.0 * pad[2:, :-2] + 2.0 * pad[2:, 1:-1] + 1.0 * pad[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def edge_baseline_score(image: np.ndarray) -> np.ndarray:
    """Higher = more likely membrane. Membranes in our synthetic EM are
    darker than cytoplasm, so we additionally include the inverted
    intensity in the score for the simple-edge baseline. Returns the
    Sobel magnitude normalised to [0,1]."""
    s = sobel_magnitude(image)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    # Membranes are darker pixels; inverted intensity helps a thresholder.
    inten = 1.0 - image
    inten = (inten - inten.min()) / (inten.max() - inten.min() + 1e-9)
    score = 0.5 * s + 0.5 * inten
    return score


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (AUC, fpr, tpr) using the standard rank-based estimator."""
    order = np.argsort(-scores)
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.float32)
    # Add a sentinel at the start.
    n_pos = max(1.0, y_sorted.sum())
    n_neg = max(1.0, (1 - y_sorted).sum())
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1.0 - y_sorted) / n_neg
    # Trapezoid AUC.
    fpr_ext = np.concatenate([[0.0], fpr, [1.0]])
    tpr_ext = np.concatenate([[0.0], tpr, [1.0]])
    auc = float(np.trapezoid(tpr_ext, fpr_ext))
    return auc, fpr, tpr


def pixel_accuracy(probs: np.ndarray, labels: np.ndarray, thresh: float = 0.5) -> float:
    return float(((probs >= thresh) == (labels >= 0.5)).mean())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    seed: int = 0
    n_train_images: int = 8
    n_test_images: int = 4
    image_h: int = 96
    image_w: int = 96
    n_cells: int = 25
    patch: int = 32
    hidden_sizes: tuple = (256, 128)
    epochs: int = 12
    patches_per_epoch: int = 4096
    batch_size: int = 64
    lr: float = 0.05
    lr_decay: float = 0.92  # multiplicative decay per epoch
    momentum: float = 0.9
    weight_decay: float = 1e-5
    eval_pixels_per_image: int = 4096   # subsample for fast per-epoch eval
    log_every: int = 16


def env_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": commit,
    }


def prior_matching_threshold(probs: np.ndarray, target_pos_frac: float) -> float:
    """Return the threshold t such that mean(probs >= t) ≈ target_pos_frac.
    This is the canonical fair threshold when a class-balanced model is
    deployed on a class-imbalanced test distribution."""
    if probs.size == 0:
        return 0.5
    q = max(0.0, min(1.0, 1.0 - target_pos_frac))
    return float(np.quantile(probs, q))


def evaluate_full_image(
    model: MLP, image: np.ndarray, mask: np.ndarray, patch: int,
    target_pos_frac: float | None = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Return (prob_map HxW, acc@0.5, AUC, prior_matching_threshold, acc@prior).

    ``target_pos_frac`` should be the global training-set membrane fraction;
    when given we additionally report pixel accuracy at the threshold that
    matches that prior (a class-balanced trained MLP needs a higher
    threshold than 0.5 to predict the natural class frequencies)."""
    h, w = image.shape
    X = all_patches_for_image(image, patch)
    probs = model.predict_proba(X)
    prob_map = probs.reshape(h, w)
    labels = mask.reshape(-1).astype(np.float32)
    acc_05 = pixel_accuracy(probs, labels, thresh=0.5)
    auc, _, _ = roc_auc(probs, labels)
    if target_pos_frac is None:
        return prob_map, acc_05, auc, 0.5, acc_05
    thr = prior_matching_threshold(probs, target_pos_frac)
    acc_prior = pixel_accuracy(probs, labels, thresh=thr)
    return prob_map, acc_05, auc, thr, acc_prior


def evaluate_subsampled(
    model: MLP,
    images: np.ndarray,
    masks: np.ndarray,
    patch: int,
    n_pixels: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Subsampled accuracy + AUC across all images, balanced over class."""
    n_imgs, h, w = images.shape
    half_n = n_pixels // 2
    half_p = patch // 2
    paddeds = [reflect_pad(images[i], half_p) for i in range(n_imgs)]

    Xs, ys = [], []
    for img_i in range(n_imgs):
        pos = np.argwhere(masks[img_i] == 1)
        neg = np.argwhere(masks[img_i] == 0)
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            continue
        nP = max(1, half_n // n_imgs)
        nN = nP
        idx_p = rng.integers(0, pos.shape[0], size=nP)
        idx_n = rng.integers(0, neg.shape[0], size=nN)
        for r in pos[idx_p]:
            cy, cx = int(r[0]), int(r[1])
            Xs.append(paddeds[img_i][cy:cy + patch, cx:cx + patch].reshape(-1))
            ys.append(1.0)
        for r in neg[idx_n]:
            cy, cx = int(r[0]), int(r[1])
            Xs.append(paddeds[img_i][cy:cy + patch, cx:cx + patch].reshape(-1))
            ys.append(0.0)

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    probs = model.predict_proba(X)
    acc = pixel_accuracy(probs, y)
    auc, _, _ = roc_auc(probs, y)
    return acc, auc


def evaluate_edge_baseline(images: np.ndarray, masks: np.ndarray) -> Tuple[float, float]:
    """Pixel acc + AUC of the Sobel+intensity edge baseline across images."""
    all_scores = []
    all_labels = []
    for i in range(images.shape[0]):
        s = edge_baseline_score(images[i])
        all_scores.append(s.reshape(-1))
        all_labels.append(masks[i].reshape(-1).astype(np.float32))
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    # Use 0.5 threshold on the [0,1] score for accuracy.
    acc = pixel_accuracy(all_scores, all_labels, thresh=0.5)
    auc, _, _ = roc_auc(all_scores, all_labels)
    return acc, auc


def train(cfg: TrainConfig) -> dict:
    np.random.seed(cfg.seed)  # legacy global; we use Generator everywhere
    rng = np.random.default_rng(cfg.seed)

    print(f"[em-segmentation-isbi] seed={cfg.seed}")
    print(f"  config: {asdict(cfg)}")
    print(f"  env   : {env_info()}")

    # 1) Synthesize dataset.
    t0 = time.time()
    train_imgs, train_masks, test_imgs, test_masks = make_dataset(
        cfg.n_train_images, cfg.n_test_images,
        cfg.image_h, cfg.image_w, cfg.n_cells, cfg.seed,
    )
    print(f"  dataset: train {train_imgs.shape}, test {test_imgs.shape}, "
          f"membrane fraction (train) {train_masks.mean():.3f} ({time.time() - t0:.1f}s)")

    # 2) Build model.
    layer_sizes = (cfg.patch * cfg.patch, *cfg.hidden_sizes, 1)
    model = MLP.make(layer_sizes, rng)
    n_params = sum(W.size for W in model.Ws) + sum(b.size for b in model.bs)
    print(f"  model : layers {layer_sizes}, params {n_params}")

    # 3) Edge baseline (one-shot).
    edge_acc, edge_auc = evaluate_edge_baseline(test_imgs, test_masks)
    print(f"  edge baseline (Sobel+inv-intensity): test pixel acc {edge_acc*100:.2f}%, AUC {edge_auc:.4f}")

    # 4) Train.
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_acc_sub": [], "test_auc_sub": [], "wallclock_s": [],
    }
    t_train = time.time()
    eval_rng = np.random.default_rng(cfg.seed + 99)
    lr = cfg.lr
    for epoch in range(1, cfg.epochs + 1):
        # Resample patches at each epoch (the paper's recipe; reduces
        # memorization of any individual patch).
        Xtr, ytr = sample_balanced_patches(
            train_imgs, train_masks, cfg.patches_per_epoch, cfg.patch, rng,
        )
        order = rng.permutation(Xtr.shape[0])
        Xtr = Xtr[order]
        ytr = ytr[order]

        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for i in range(0, Xtr.shape[0], cfg.batch_size):
            xb = Xtr[i:i + cfg.batch_size]
            yb = ytr[i:i + cfg.batch_size]
            p, acts = model.forward(xb)
            # Binary cross-entropy.
            eps = 1e-7
            loss = -float(np.mean(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)))
            running_loss += loss * yb.shape[0]
            running_n += yb.shape[0]
            running_correct += int(((p >= 0.5) == (yb >= 0.5)).sum())
            dWs, dbs = model.backward(acts, yb)
            model.sgd_step(dWs, dbs, lr, cfg.momentum, cfg.weight_decay)

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)
        test_acc, test_auc = evaluate_subsampled(
            model, test_imgs, test_masks, cfg.patch,
            cfg.eval_pixels_per_image * test_imgs.shape[0], eval_rng,
        )
        wc = time.time() - t_train
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc_sub"].append(test_acc)
        history["test_auc_sub"].append(test_auc)
        history["wallclock_s"].append(wc)
        print(
            f"  epoch {epoch:2d}/{cfg.epochs}  lr {lr:.4f}  "
            f"loss {train_loss:.4f}  train_acc {train_acc*100:.2f}%  "
            f"test_acc {test_acc*100:.2f}%  test_AUC {test_auc:.4f}  "
            f"({wc:.1f}s elapsed)",
            flush=True,
        )
        lr *= cfg.lr_decay

    # 5) Final dense evaluation on every test image (full pixel grid).
    train_membrane_frac = float(train_masks.mean())
    final_accs05, final_accs_prior, final_aucs = [], [], []
    final_thrs, prob_maps = [], []
    for i in range(test_imgs.shape[0]):
        prob_map, acc_05, auc, thr, acc_prior = evaluate_full_image(
            model, test_imgs[i], test_masks[i], cfg.patch,
            target_pos_frac=train_membrane_frac,
        )
        final_accs05.append(acc_05)
        final_accs_prior.append(acc_prior)
        final_aucs.append(auc)
        final_thrs.append(thr)
        prob_maps.append(prob_map)
    final_acc_05 = float(np.mean(final_accs05))
    final_acc_prior = float(np.mean(final_accs_prior))
    final_auc = float(np.mean(final_aucs))

    print(f"  final dense test ROC AUC                       {final_auc:.4f}")
    print(f"  final dense test pixel acc @0.5                {final_acc_05*100:.2f}%")
    print(f"  final dense test pixel acc @prior-matched thr  {final_acc_prior*100:.2f}% "
          f"(mean thr {np.mean(final_thrs):.3f}, train pos frac {train_membrane_frac:.3f})")
    print(f"  edge baseline (re-stated)                       acc {edge_acc*100:.2f}%, AUC {edge_auc:.4f}")
    print(f"  total wallclock                                 {time.time() - t0:.1f}s")

    return {
        "config": asdict(cfg),
        "env": env_info(),
        "history": history,
        "edge_baseline": {"pixel_acc": edge_acc, "auc": edge_auc},
        "final_dense": {
            "pixel_acc_at_05": final_acc_05,
            "pixel_acc_at_prior": final_acc_prior,
            "auc": final_auc,
            "per_image_acc_05": final_accs05,
            "per_image_acc_prior": final_accs_prior,
            "per_image_auc": final_aucs,
            "per_image_threshold": final_thrs,
            "train_membrane_frac": train_membrane_frac,
        },
        "model": model,
        "train_imgs": train_imgs,
        "train_masks": train_masks,
        "test_imgs": test_imgs,
        "test_masks": test_masks,
        "prob_maps": np.stack(prob_maps),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patches-per-epoch", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--n-cells", type=int, default=25)
    p.add_argument("--image-h", type=int, default=96)
    p.add_argument("--image-w", type=int, default=96)
    p.add_argument("--n-train-images", type=int, default=8)
    p.add_argument("--n-test-images", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--save-results", type=str, default=None,
                   help="if given, dumps a small JSON of the metric scalars")
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        patches_per_epoch=args.patches_per_epoch,
        batch_size=args.batch_size,
        patch=args.patch,
        n_cells=args.n_cells,
        image_h=args.image_h,
        image_w=args.image_w,
        n_train_images=args.n_train_images,
        n_test_images=args.n_test_images,
        lr=args.lr,
    )
    result = train(cfg)

    if args.save_results:
        out = {
            "config": result["config"],
            "env": result["env"],
            "history": result["history"],
            "edge_baseline": result["edge_baseline"],
            "final_dense": result["final_dense"],
        }
        Path(args.save_results).write_text(json.dumps(out, indent=2))
        print(f"  wrote results JSON -> {args.save_results}")


if __name__ == "__main__":
    main()
