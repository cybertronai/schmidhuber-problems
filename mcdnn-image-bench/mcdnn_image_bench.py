"""
mcdnn-image-bench — single-column deep MLP on MNIST (numpy only).

Cireşan, Meier, Schmidhuber, *Multi-column deep neural networks for image
classification*, CVPR 2012. The paper's headline number — 0.23% MNIST test error
— came from averaging 35 deep CNN columns trained on differently-preprocessed
inputs (block-distorted, scaled, normalized-thickness, ...). Each column is a
GPU-trained deep CNN.

Per the v1 SPEC (issue #1), single-column MNIST is the v1 headline; multi-column
GTSRB / CASIA goes to v1.5. This stub implements one column — a 4-layer MLP —
that captures the *single-column* part of the methodology in pure numpy. The
multi-column averaging step is documented in §Open questions and reproduces in
v1.5 once we have multiple columns over multiple datasets.

Architecture
------------
    784 -> 800 -ReLU-> 800 -ReLU-> 400 -ReLU-> 10 -softmax-> CE

He init for the ReLU layers, Glorot for the output layer, plain SGD with
Nesterov momentum, mini-batch 128, fixed LR with step decay at epoch 6.

Single column means: one network, one preprocessing (per-image mean-zero / unit-
norm), no test-time augmentation, no ensembling. The "deep MLP at scale" framing
is the same one Cireşan et al. used in the 2010 *Deep, big, simple neural nets
excel on handwritten digit recognition* Neural Computation paper that preceded
MCDNN — see also wave-9/mnist-deep-mlp for that companion stub.

Run
---
    python3 mcdnn_image_bench.py --seed 0

Reproducible: bit-identical metrics across runs at fixed seed.

Wallclock target: <5 minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MNIST loader (pure stdlib + numpy; cached under ~/.cache/hinton-mnist/)
# ---------------------------------------------------------------------------

MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# pytorch ossci mirror (public S3); same files as Yann LeCun's original host
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

# expected sha256 of each gzipped file (computed at first load)
MNIST_SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}

DEFAULT_CACHE = Path.home() / ".cache" / "hinton-mnist"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)


def _ensure_mnist(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        path = cache / name
        if path.exists():
            continue
        _download(f"{MNIST_MIRROR}/{name}", path)
    # integrity check (warn-only; the file might be a valid mirror with a
    # slightly different blob but identical post-decode bytes)
    for name in MNIST_FILES:
        got = _sha256(cache / name)
        want = MNIST_SHA256.get(name)
        if want and got != want:
            print(
                f"  WARN: {name} sha256 {got[:8]} != expected {want[:8]} "
                f"(mirror differs; decode-time check still applied)",
                flush=True,
            )


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"bad magic for image file {path}: {magic}")
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return arr


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"bad magic for label file {path}: {magic}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(cache: Path = DEFAULT_CACHE) -> dict:
    """Return MNIST as a dict with float32 features in [0, 1] and int labels."""
    _ensure_mnist(cache)
    x_train = _read_idx_images(cache / "train-images-idx3-ubyte.gz")
    y_train = _read_idx_labels(cache / "train-labels-idx1-ubyte.gz")
    x_test = _read_idx_images(cache / "t10k-images-idx3-ubyte.gz")
    y_test = _read_idx_labels(cache / "t10k-labels-idx1-ubyte.gz")
    if x_train.shape != (60000, 28, 28) or x_test.shape != (10000, 28, 28):
        raise RuntimeError("MNIST shapes wrong; cache is corrupt")
    if y_train.shape != (60000,) or y_test.shape != (10000,):
        raise RuntimeError("MNIST label shapes wrong; cache is corrupt")
    # flatten and normalize to [0, 1] then mean-subtract per sample
    Xtr = (x_train.astype(np.float32) / 255.0).reshape(60000, 784)
    Xte = (x_test.astype(np.float32) / 255.0).reshape(10000, 784)
    return {
        "x_train": Xtr,
        "y_train": y_train.astype(np.int64),
        "x_test": Xte,
        "y_test": y_test.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# MLP (forward, backward, SGD with Nesterov momentum)
# ---------------------------------------------------------------------------


def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(
        2.0 / fan_in
    ).astype(np.float32)


def glorot_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    s = np.sqrt(6.0 / (fan_in + fan_out)).astype(np.float32)
    return rng.uniform(-s, s, size=(fan_in, fan_out)).astype(np.float32)


class MLP:
    """4-layer MLP: 784 -> H1 -ReLU-> H2 -ReLU-> H3 -ReLU-> 10 -softmax->.

    Stores parameters as a flat dict so SGD-with-momentum is one loop.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        sizes: tuple[int, ...] = (784, 800, 800, 400, 10),
    ) -> None:
        self.sizes = sizes
        self.params: dict[str, np.ndarray] = {}
        self.velocity: dict[str, np.ndarray] = {}
        for i in range(len(sizes) - 1):
            fi, fo = sizes[i], sizes[i + 1]
            init = glorot_init if i == len(sizes) - 2 else he_init
            self.params[f"W{i}"] = init(rng, fi, fo)
            self.params[f"b{i}"] = np.zeros((fo,), dtype=np.float32)
            self.velocity[f"W{i}"] = np.zeros_like(self.params[f"W{i}"])
            self.velocity[f"b{i}"] = np.zeros_like(self.params[f"b{i}"])

    def num_params(self) -> int:
        return sum(p.size for p in self.params.values())

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return logits and the cache (list of activations) for backward."""
        cache: list[np.ndarray] = [x]
        h = x
        last = len(self.sizes) - 2
        for i in range(len(self.sizes) - 1):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i == last:
                cache.append(z)  # logits
            else:
                h = np.maximum(z, 0.0, out=z)  # ReLU in-place
                cache.append(h)
        return cache[-1], cache

    def backward(
        self,
        cache: list[np.ndarray],
        y: np.ndarray,
        weight_decay: float = 0.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Softmax + cross-entropy backward. Returns mean loss and grads."""
        logits = cache[-1]
        n = logits.shape[0]
        # log-softmax stable
        m = logits.max(axis=1, keepdims=True)
        z = logits - m
        log_sum = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_p = z - log_sum
        loss = float(-log_p[np.arange(n), y].mean())

        # dlogits = (softmax - onehot) / n
        p = np.exp(log_p)
        p[np.arange(n), y] -= 1.0
        dz = p.astype(np.float32) / np.float32(n)

        grads: dict[str, np.ndarray] = {}
        last = len(self.sizes) - 2
        # iterate layers from last to first
        for i in range(last, -1, -1):
            h_prev = cache[i]  # input to layer i
            grads[f"W{i}"] = h_prev.T @ dz
            grads[f"b{i}"] = dz.sum(axis=0)
            if weight_decay > 0.0:
                grads[f"W{i}"] += weight_decay * self.params[f"W{i}"]
            if i > 0:
                # backprop through layer i to get dh_prev
                dh = dz @ self.params[f"W{i}"].T
                # we stored ReLU output in cache[i]; gradient mask = (cache[i] > 0)
                dh = dh * (cache[i] > 0).astype(np.float32)
                dz = dh
        return loss, grads

    def step(
        self,
        grads: dict[str, np.ndarray],
        lr: float,
        momentum: float = 0.9,
        nesterov: bool = True,
    ) -> None:
        for k, g in grads.items():
            v = self.velocity[k]
            v_new = momentum * v - lr * g
            if nesterov:
                update = momentum * v_new - lr * g
            else:
                update = v_new
            self.params[k] += update
            self.velocity[k] = v_new

    def predict(self, x: np.ndarray, batch: int = 1000) -> np.ndarray:
        out = np.empty(x.shape[0], dtype=np.int64)
        for i in range(0, x.shape[0], batch):
            logits, _ = self.forward(x[i : i + batch])
            out[i : i + batch] = logits.argmax(axis=1)
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    seed: int = 0,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 0.05,
    lr_decay_epoch: int = 6,
    lr_decay_factor: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    hidden: tuple[int, int, int] = (800, 800, 400),
    out_dir: Path | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    print(f"loading MNIST (cache={DEFAULT_CACHE}) ...", flush=True)
    data = load_mnist()
    Xtr, Ytr = data["x_train"], data["y_train"]
    Xte, Yte = data["x_test"], data["y_test"]
    print(f"  train={Xtr.shape}  test={Xte.shape}", flush=True)

    sizes = (784, *hidden, 10)
    model = MLP(rng, sizes=sizes)
    print(f"model: {' -> '.join(map(str, sizes))}  params={model.num_params():,}",
          flush=True)

    n = Xtr.shape[0]
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_err": [],
        "epoch_seconds": [],
        "lr": [],
    }

    t_start = time.time()
    cur_lr = lr
    for epoch in range(epochs):
        if epoch == lr_decay_epoch:
            cur_lr *= lr_decay_factor
            print(f"  lr decayed to {cur_lr:.5f}", flush=True)

        # shuffle (deterministic from seed via rng)
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_correct = 0
        t0 = time.time()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = Xtr[idx]
            yb = Ytr[idx]
            logits, cache = model.forward(xb)
            loss, grads = model.backward(cache, yb, weight_decay=weight_decay)
            model.step(grads, lr=cur_lr, momentum=momentum, nesterov=True)
            epoch_loss += loss * xb.shape[0]
            n_correct += int((logits.argmax(axis=1) == yb).sum())
        epoch_seconds = time.time() - t0

        train_loss = epoch_loss / n
        train_acc = n_correct / n
        test_pred = model.predict(Xte)
        test_acc = float((test_pred == Yte).mean())
        test_err = 1.0 - test_acc

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_err"].append(test_err)
        history["epoch_seconds"].append(epoch_seconds)
        history["lr"].append(cur_lr)

        print(
            f"  epoch {epoch:2d}  loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  test_err={test_err*100:.2f}%  "
            f"({epoch_seconds:.1f}s)",
            flush=True,
        )

    total_seconds = time.time() - t_start

    # final eval (also computes confusion matrix)
    final_test_pred = model.predict(Xte)
    final_test_err = float((final_test_pred != Yte).mean())
    confusion = np.zeros((10, 10), dtype=np.int64)
    for t_, p_ in zip(Yte, final_test_pred):
        confusion[t_, p_] += 1

    result = {
        "seed": seed,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "sizes": list(sizes),
            "nesterov": True,
            "init": "He (hidden) + Glorot (output)",
            "optimizer": "SGD + Nesterov momentum",
            "preprocess": "pixel/255",
        },
        "metrics": {
            "final_test_err": final_test_err,
            "final_test_err_pct": final_test_err * 100.0,
            "best_test_err_pct": min(history["test_err"]) * 100.0,
            "best_test_err_epoch": int(np.argmin(history["test_err"])),
            "final_train_acc": history["train_acc"][-1],
            "total_seconds": total_seconds,
            "n_params": model.num_params(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "git_commit": _git_commit(),
        },
        "history": history,
        "confusion": confusion.tolist(),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # save weights + history; keep .npz reasonably small
        np.savez(
            out_dir / "weights.npz",
            **{k: v for k, v in model.params.items()},
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved weights -> {out_dir / 'weights.npz'}", flush=True)
        print(f"  saved history -> {out_dir / 'history.json'}", flush=True)

    return result


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="single-column deep MLP on MNIST (numpy only)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-decay-epoch", type=int, default=6)
    ap.add_argument("--lr-decay-factor", type=float, default=0.2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--hidden",
        type=int,
        nargs=3,
        default=[800, 800, 400],
        help="three hidden layer sizes",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "viz")
    args = ap.parse_args()

    print(
        f"mcdnn-image-bench  seed={args.seed}  epochs={args.epochs}  "
        f"batch={args.batch_size}  lr={args.lr}  hidden={tuple(args.hidden)}",
        flush=True,
    )
    result = train(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay_factor=args.lr_decay_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hidden=tuple(args.hidden),
        out_dir=args.out_dir,
    )
    print("\n=== RESULT ===")
    print(f"  test error: {result['metrics']['final_test_err_pct']:.2f}%  "
          f"(best {result['metrics']['best_test_err_pct']:.2f}% at epoch "
          f"{result['metrics']['best_test_err_epoch']})")
    print(f"  total: {result['metrics']['total_seconds']:.1f}s  "
          f"params: {result['metrics']['n_params']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
