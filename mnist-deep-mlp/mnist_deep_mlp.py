"""mnist-deep-mlp — Deep, big, simple MLP on MNIST (numpy only).

Cireşan, Meier, Gambardella, Schmidhuber, *Deep, big, simple neural nets
excel on handwritten digit recognition*, Neural Computation 22(12), 2010.

Paper claim: a 5-hidden-layer MLP (e.g. 784-2500-2000-1500-1000-500-10),
trained with SGD on heavily-deformed MNIST (per-epoch elastic + affine),
reaches **0.35% test error** — best plain-MLP result on MNIST at the time
and competitive with then-current convolutional methods. The paper credits
the bulk of the gap over a vanilla MLP to the augmentation, not the depth.

This file is the v1 reproduction:
  * pure numpy (matplotlib only used by the visualization scripts)
  * MNIST loaded from the IDX gzips (cached under ~/.cache/schmidhuber-mnist/)
    — the v1 SPEC also allows torchvision.datasets.MNIST, but torchvision is
    not installed in this environment so we fall back to the equivalent
    raw-IDX path. Either way the model code is pure numpy.
  * smaller architecture so the run fits in the v1 <5 min CPU budget
    (default 784-512-256-10, ~537k weights, vs the paper's ~12M-weight nets)
  * on-the-fly augmentation: affine (small rotation + scale + translation)
    plus elastic deformation in the Simard et al. 2003 style (random
    displacement fields smoothed by a separable Gaussian and bilinearly
    interpolated)
  * SGD with Nesterov-style momentum and a step-decayed learning rate
  * tanh activation (paper's choice; documented in §Deviations)

Headline (default flags, --seed 0):
  ~1.6% test error after 15 epochs, ~3.4 min on a laptop CPU.
The 5x gap to the paper's 0.35% is expected at this scale: the paper's
~12M-weight network plus 800-1000 epochs is far outside the v1 budget.

The CLI entry point is `train()`; see the `--help` for flags. All flags
are optional; `python3 mnist_deep_mlp.py --seed 0` reproduces §Results.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader (raw IDX format, cached on disk)
# ----------------------------------------------------------------------

_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Multiple mirrors because yann.lecun.com has been intermittent.
_MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]


def _default_cache_dir() -> Path:
    # Prefer reusing a sibling project's cache if it exists, since these are
    # the same files. Otherwise stash under a project-named directory.
    sibling = Path.home() / ".cache" / "hinton-mnist"
    if sibling.is_dir():
        return sibling
    return Path.home() / ".cache" / "schmidhuber-mnist"


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-deep-mlp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def _ensure_files(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for fname in _MNIST_FILES.values():
        target = cache / fname
        if target.is_file() and target.stat().st_size > 1024:
            continue
        last_err = None
        for base in _MNIST_MIRRORS:
            try:
                _download(base + fname, target)
                last_err = None
                break
            except Exception as e:  # pragma: no cover (network)
                last_err = e
        if last_err is not None:
            raise RuntimeError(
                f"Failed to fetch {fname} from any mirror: {last_err}"
            )


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"bad image magic: {magic}"
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return arr.copy()


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2049, f"bad label magic: {magic}"
        n = int.from_bytes(f.read(4), "big")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).copy()


def load_mnist(cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y).

    Images are float32 in [0, 1], shape (N, 28, 28). Labels are int64.
    """
    cache = cache_dir if cache_dir is not None else _default_cache_dir()
    _ensure_files(cache)
    train_x = _read_idx_images(cache / _MNIST_FILES["train_images"]).astype(np.float32) / 255.0
    train_y = _read_idx_labels(cache / _MNIST_FILES["train_labels"]).astype(np.int64)
    test_x = _read_idx_images(cache / _MNIST_FILES["test_images"]).astype(np.float32) / 255.0
    test_y = _read_idx_labels(cache / _MNIST_FILES["test_labels"]).astype(np.int64)
    return train_x, train_y, test_x, test_y


# ----------------------------------------------------------------------
# Augmentation (affine + elastic, all numpy)
# ----------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def _separable_gaussian_2d(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth (B, H, W) along H and W with a separable kernel."""
    if sigma <= 0:
        return field
    k = _gaussian_kernel_1d(sigma)
    r = (k.size - 1) // 2
    # Pad with edge replication.
    padded = np.pad(field, ((0, 0), (r, r), (r, r)), mode="edge")
    # Convolve along W.
    out_w = np.zeros_like(field)
    for i, w in enumerate(k):
        out_w += w * padded[:, r:r + field.shape[1], i:i + field.shape[2]]
    # Convolve along H.
    padded2 = np.pad(out_w, ((0, 0), (r, r), (0, 0)), mode="edge")
    out = np.zeros_like(field)
    for i, w in enumerate(k):
        out += w * padded2[:, i:i + field.shape[1], :]
    return out


def _bilinear_sample(images: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Sample (B, H, W) at floating coordinates (gx, gy) of shape (B, H, W).

    Out-of-bounds pixels become 0 (MNIST background).
    """
    B, H, W = images.shape
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = gx - x0
    fy = gy - y0

    # Mask for any corner outside; clip indices and zero the contribution.
    def _gather(yi: np.ndarray, xi: np.ndarray) -> np.ndarray:
        mask = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        yi_c = np.clip(yi, 0, H - 1)
        xi_c = np.clip(xi, 0, W - 1)
        b = np.arange(B)[:, None, None]
        v = images[b, yi_c, xi_c]
        v = np.where(mask, v, 0.0)
        return v

    v00 = _gather(y0, x0)
    v01 = _gather(y0, x1)
    v10 = _gather(y1, x0)
    v11 = _gather(y1, x1)
    out = ((1 - fy) * ((1 - fx) * v00 + fx * v01)
           + fy * ((1 - fx) * v10 + fx * v11))
    return out.astype(np.float32)


def augment_batch(
    images: np.ndarray,
    rng: np.random.Generator,
    *,
    max_rotation_deg: float = 15.0,
    max_translation_px: float = 2.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    elastic_alpha: float = 8.0,
    elastic_sigma: float = 4.0,
) -> np.ndarray:
    """Apply per-image affine + elastic deformation.

    images: (B, H, W) float32 in [0, 1].
    Returns a new (B, H, W) float32 array, same shape.
    """
    B, H, W = images.shape
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    # Per-image affine parameters.
    angles = rng.uniform(-max_rotation_deg, max_rotation_deg, size=B).astype(np.float32)
    angles = np.deg2rad(angles)
    scales = rng.uniform(scale_range[0], scale_range[1], size=B).astype(np.float32)
    tx = rng.uniform(-max_translation_px, max_translation_px, size=B).astype(np.float32)
    ty = rng.uniform(-max_translation_px, max_translation_px, size=B).astype(np.float32)

    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    s = scales[:, None, None]

    # Map output coord -> source coord (inverse warp).
    yyc = yy[None] - cy
    xxc = xx[None] - cx
    src_y = (sin * xxc + cos * yyc) / s + cy + ty[:, None, None]
    src_x = (cos * xxc - sin * yyc) / s + cx + tx[:, None, None]

    # Elastic displacement field.
    if elastic_alpha > 0 and elastic_sigma > 0:
        dx = rng.uniform(-1.0, 1.0, size=(B, H, W)).astype(np.float32)
        dy = rng.uniform(-1.0, 1.0, size=(B, H, W)).astype(np.float32)
        dx = _separable_gaussian_2d(dx, elastic_sigma) * elastic_alpha
        dy = _separable_gaussian_2d(dy, elastic_sigma) * elastic_alpha
        src_y = src_y + dy
        src_x = src_x + dx

    return _bilinear_sample(images, src_x, src_y)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    seed: int = 0
    hidden_sizes: tuple[int, ...] = (512, 256)
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.05
    lr_decay: float = 0.95            # multiplicative per epoch
    momentum: float = 0.9
    weight_decay: float = 1e-5
    augment: bool = True
    eval_every: int = 1               # epochs
    log_every: int = 100              # batches
    cache_dir: str | None = None


class DeepMLP:
    """Plain MLP with tanh hidden units and softmax output, manual SGD.

    Forward / backward are implemented in a single class with per-layer
    weight matrices and bias vectors. Initialization is the
    `tanh`-tuned variant of "Glorot uniform":
       W ~ U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))).
    """

    def __init__(self, layer_sizes: list[int], rng: np.random.Generator):
        assert len(layer_sizes) >= 2
        self.layer_sizes = list(layer_sizes)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for fin, fout in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fin + fout))
            self.W.append(rng.uniform(-limit, limit, (fin, fout)).astype(np.float32))
            self.b.append(np.zeros((fout,), dtype=np.float32))
        # Velocities for momentum.
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    @property
    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """x: (B, 784). Returns (logits (B, 10), cache of activations)."""
        acts = [x]
        h = x
        # All hidden layers use tanh.
        for w, b in zip(self.W[:-1], self.b[:-1]):
            z = h @ w + b
            h = np.tanh(z)
            acts.append(h)
        # Output is linear (we apply softmax in the loss).
        logits = h @ self.W[-1] + self.b[-1]
        acts.append(logits)
        return logits, acts

    @staticmethod
    def softmax_xent_grad(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (dlogits, mean_loss) for softmax cross-entropy."""
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        p = ex / ex.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        loss = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean()
        dlogits = p
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        return dlogits.astype(np.float32), float(loss)

    def backward(
        self,
        acts: list[np.ndarray],
        dlogits: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backprop. Returns (dWs, dbs) layer by layer."""
        dWs: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        dbs: list[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]

        # Output layer.
        h_prev = acts[-2]
        dWs[-1] = h_prev.T @ dlogits
        dbs[-1] = dlogits.sum(axis=0)
        dh = dlogits @ self.W[-1].T

        # Hidden layers (tanh).
        for i in range(len(self.W) - 2, -1, -1):
            h = acts[i + 1]
            dz = dh * (1.0 - h * h)
            h_prev = acts[i]
            dWs[i] = h_prev.T @ dz
            dbs[i] = dz.sum(axis=0)
            if i > 0:
                dh = dz @ self.W[i].T
        return dWs, dbs

    def sgd_step(
        self,
        dWs: list[np.ndarray],
        dbs: list[np.ndarray],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        for i in range(len(self.W)):
            self.vW[i] = momentum * self.vW[i] + dWs[i] + weight_decay * self.W[i]
            self.vb[i] = momentum * self.vb[i] + dbs[i]
            self.W[i] -= lr * self.vW[i]
            self.b[i] -= lr * self.vb[i]

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        out = np.zeros(x.shape[0], dtype=np.int64)
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            logits, _ = self.forward(xb)
            out[i:i + xb.shape[0]] = logits.argmax(axis=1)
        return out


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def train(cfg: TrainConfig) -> dict:
    """Train a deep MLP on MNIST. Returns a dict of metrics."""
    np.random.seed(cfg.seed)  # belt-and-braces: some numpy paths still consult this
    rng = np.random.default_rng(cfg.seed)

    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    print(f"[mnist-deep-mlp] loading MNIST from {cache_dir or _default_cache_dir()}")
    train_x, train_y, test_x, test_y = load_mnist(cache_dir)
    print(f"  train: {train_x.shape}, test: {test_x.shape}")

    # Architecture: 784 -> hidden ... -> 10
    layer_sizes = [28 * 28, *cfg.hidden_sizes, 10]
    model = DeepMLP(layer_sizes, rng)
    print(f"  arch: {' -> '.join(str(s) for s in layer_sizes)}  ({model.n_params:,} weights)")

    n_train = train_x.shape[0]
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "train_err": [],
        "test_err": [],
        "lr": [],
        "wallclock_s": [],
    }

    t0 = time.time()
    lr = cfg.lr
    aug_rng = np.random.default_rng(cfg.seed + 1)
    perm_rng = np.random.default_rng(cfg.seed + 2)

    for epoch in range(cfg.epochs):
        order = perm_rng.permutation(n_train)
        running_loss = 0.0
        running_n = 0
        running_err = 0
        for i in range(0, n_train, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            if cfg.augment:
                xb = augment_batch(xb, aug_rng)
            xb_flat = _flatten(xb)
            logits, acts = model.forward(xb_flat)
            dlogits, loss = model.softmax_xent_grad(logits, yb)
            dWs, dbs = model.backward(acts, dlogits)
            model.sgd_step(dWs, dbs, lr, cfg.momentum, cfg.weight_decay)

            running_loss += loss * xb.shape[0]
            running_n += xb.shape[0]
            running_err += int((logits.argmax(axis=1) != yb).sum())

            if cfg.log_every and (i // cfg.batch_size) % cfg.log_every == 0:
                print(
                    f"  epoch {epoch+1:2d}  step {i//cfg.batch_size:4d}  "
                    f"loss {loss:.4f}  lr {lr:.4f}",
                    flush=True,
                )

        train_loss = running_loss / running_n
        train_err = running_err / running_n
        # Eval on test.
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            preds = model.predict(_flatten(test_x))
            test_err = float((preds != test_y).mean())
        else:
            test_err = float("nan")

        elapsed = time.time() - t0
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_err"].append(train_err)
        history["test_err"].append(test_err)
        history["lr"].append(lr)
        history["wallclock_s"].append(elapsed)
        print(
            f"epoch {epoch+1:2d}/{cfg.epochs}  "
            f"train_loss {train_loss:.4f}  "
            f"train_err {train_err*100:.2f}%  "
            f"test_err {test_err*100:.2f}%  "
            f"lr {lr:.4f}  "
            f"elapsed {elapsed:.1f}s",
            flush=True,
        )
        lr *= cfg.lr_decay

    final_test_err = history["test_err"][-1]
    print(
        f"\nFinal: test_err = {final_test_err*100:.2f}%  "
        f"(seed {cfg.seed}, {cfg.epochs} epochs, "
        f"{model.n_params:,} weights, "
        f"{history['wallclock_s'][-1]:.1f}s)"
    )
    return {
        "config": cfg.__dict__ | {"hidden_sizes": list(cfg.hidden_sizes)},
        "history": history,
        "final_test_err": final_test_err,
        "n_params": model.n_params,
        "model": model,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hidden", type=int, nargs="+", default=[512, 256],
        help="hidden layer sizes (e.g. --hidden 1024 512 256)",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-decay", type=float, default=0.95)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--save", type=str, default=None,
                   help="if set, write a JSON metrics file at this path")
    args = p.parse_args(argv)
    cfg = TrainConfig(
        seed=args.seed,
        hidden_sizes=tuple(args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        augment=not args.no_augment,
        cache_dir=args.cache_dir,
    )
    cfg._save = args.save  # type: ignore[attr-defined]
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    out = train(cfg)
    save_path = getattr(cfg, "_save", None)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                {k: v for k, v in out.items() if k != "model"},
                f, indent=2, default=str,
            )
        print(f"  wrote metrics to {save_path}")


if __name__ == "__main__":
    main()
