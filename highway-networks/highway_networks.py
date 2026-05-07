"""highway-networks -- Srivastava, Greff, Schmidhuber,
*Training Very Deep Networks*, NIPS 2015 (arXiv:1507.06228).

Highway transform per layer:
    y = H(x) * T(x) + x * (1 - T(x))
    H(x) = tanh(W_H x + b_H)            -- the "transform" branch
    T(x) = sigmoid(W_T x + b_T)         -- the "transform gate"
                                          (1 - T is the "carry gate")

The carry path lets information flow through unimpeded when T -> 0, so a
randomly initialised deep highway net behaves at init like a near-identity
chain. This sidesteps the vanishing-gradient pathology that prevents very
deep plain feedforward nets (with saturating activations) from training.

Headline contrast (this stub):
    A 30-layer highway net trains on MNIST.
    A 30-layer plain MLP at the same depth, same width, same activation,
    same optimiser fails to learn -- it stays at chance / fluctuates.

Architecture (matches the Srivastava 2015 setup, scaled down for laptop):
  * 28x28 -> 50 input projection (tanh)              [W_in, b_in]
  * N highway / plain layers of width 50             [stacked]
  * 50 -> 10 logits, softmax + cross-entropy         [W_out, b_out]

Per the paper (sec 3): T-gate bias is initialised negative (default -2.0)
so the network starts close to the identity. H uses standard small-init.
We use tanh in H to make the contrast crisp -- with ReLU, plain nets are
known to train at modest depth even without skip connections; tanh is
where the original deep-net failure mode lives, and where the highway
mechanism shines.

CLI:
    python3 highway_networks.py --seed 0           # train both, default 30 layers
    python3 highway_networks.py --seed 0 --quick   # depth=10, fewer epochs
    python3 highway_networks.py --seed 0 --depth 50
    python3 highway_networks.py --seed 0 --depths 10,20,30,50

MNIST is loaded from ~/.cache/hinton-mnist/ (idx files); if missing, we
download from the public Yann LeCun mirror to that cache.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
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


# ----------------------------------------------------------------------
# MNIST loader (idx format from ~/.cache/hinton-mnist/)
# ----------------------------------------------------------------------

MNIST_CACHE = os.path.expanduser("~/.cache/hinton-mnist")
MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _ensure_mnist_cache() -> None:
    os.makedirs(MNIST_CACHE, exist_ok=True)
    for name, url in MNIST_URLS.items():
        path = os.path.join(MNIST_CACHE, name)
        if not os.path.exists(path):
            print(f"  downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        buf = f.read(n * rows * cols)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return arr


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        buf = f.read(n)
        arr = np.frombuffer(buf, dtype=np.uint8)
    return arr


def load_mnist() -> Dict[str, np.ndarray]:
    _ensure_mnist_cache()
    Xtr = _read_idx_images(os.path.join(MNIST_CACHE, "train-images-idx3-ubyte.gz"))
    ytr = _read_idx_labels(os.path.join(MNIST_CACHE, "train-labels-idx1-ubyte.gz"))
    Xte = _read_idx_images(os.path.join(MNIST_CACHE, "t10k-images-idx3-ubyte.gz"))
    yte = _read_idx_labels(os.path.join(MNIST_CACHE, "t10k-labels-idx1-ubyte.gz"))
    # Normalise to [-1, 1] -- helps tanh nets at init
    Xtr = (Xtr.astype(np.float64) / 255.0) * 2.0 - 1.0
    Xte = (Xte.astype(np.float64) / 255.0) * 2.0 - 1.0
    return {"Xtr": Xtr, "ytr": ytr.astype(np.int64),
            "Xte": Xte, "yte": yte.astype(np.int64)}


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def _tanh(x):
    return np.tanh(x)


def _dtanh_from_y(y):
    return 1.0 - y * y


def _sigmoid(x):
    # numerically stable
    out = np.empty_like(x)
    pos = x >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _dsigmoid_from_y(y):
    return y * (1.0 - y)


# ----------------------------------------------------------------------
# Models: highway + plain. Same input/output projection, same depth.
# ----------------------------------------------------------------------

class DeepNet:
    """Deep MLP with shared input/output projection layers and a stack of
    `depth` hidden blocks at fixed width `hidden`.

    block = "highway"  ->  y = T*tanh(W_H x + b_H) + (1-T)*x,
                            T = sigmoid(W_T x + b_T)
    block = "plain"    ->  y = tanh(W x + b)        (no skip)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int,
        depth: int,
        block: str,
        rng: np.random.Generator,
        gate_bias_init: float = -2.0,
    ):
        assert block in ("highway", "plain")
        self.block = block
        self.depth = depth
        self.hidden = hidden

        # Input projection: d_in -> hidden, tanh
        s_in = 1.0 / np.sqrt(d_in)
        self.W_in = rng.uniform(-s_in, s_in, size=(hidden, d_in))
        self.b_in = np.zeros(hidden)

        # Hidden blocks
        s_h = 1.0 / np.sqrt(hidden)
        self.W_H = [rng.uniform(-s_h, s_h, size=(hidden, hidden)) for _ in range(depth)]
        self.b_H = [np.zeros(hidden) for _ in range(depth)]
        if block == "highway":
            # T-gate weights: small init, bias negative (paper sec 3)
            self.W_T = [rng.uniform(-s_h, s_h, size=(hidden, hidden)) for _ in range(depth)]
            self.b_T = [np.full(hidden, gate_bias_init) for _ in range(depth)]
        else:
            self.W_T = []
            self.b_T = []

        # Output projection: hidden -> d_out (logits)
        s_out = 1.0 / np.sqrt(hidden)
        self.W_out = rng.uniform(-s_out, s_out, size=(d_out, hidden))
        self.b_out = np.zeros(d_out)

        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self._params_dict().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params_dict().items()}
        self._t = 0

    # -- parameter access ----------------------------------------------

    def _params_dict(self) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {
            "W_in": self.W_in, "b_in": self.b_in,
            "W_out": self.W_out, "b_out": self.b_out,
        }
        for i in range(self.depth):
            d[f"W_H_{i}"] = self.W_H[i]
            d[f"b_H_{i}"] = self.b_H[i]
            if self.block == "highway":
                d[f"W_T_{i}"] = self.W_T[i]
                d[f"b_T_{i}"] = self.b_T[i]
        return d

    # -- forward / backward --------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass, cache intermediates, return logits (N, d_out)."""
        N = X.shape[0]
        cache: Dict[str, object] = {"X": X}
        # Input projection
        z_in = X @ self.W_in.T + self.b_in
        h0 = _tanh(z_in)
        cache["h0"] = h0
        # Stack hidden blocks
        h = h0
        layer_h = []
        layer_T = []
        layer_H = []
        for i in range(self.depth):
            zH = h @ self.W_H[i].T + self.b_H[i]
            H = _tanh(zH)
            if self.block == "highway":
                zT = h @ self.W_T[i].T + self.b_T[i]
                T = _sigmoid(zT)
                h_new = T * H + (1.0 - T) * h
                layer_T.append(T)
            else:
                h_new = H
                layer_T.append(None)
            layer_H.append(H)
            layer_h.append(h_new)
            h = h_new
        cache["layer_H"] = layer_H
        cache["layer_T"] = layer_T
        cache["layer_h"] = layer_h
        # Output projection (linear, softmax handled in loss)
        logits = h @ self.W_out.T + self.b_out
        cache["logits"] = logits
        self._cache = cache
        return logits

    def backward(self, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        """Given dL/dlogits, return parameter gradients."""
        cache = self._cache
        X = cache["X"]
        h0 = cache["h0"]
        layer_H = cache["layer_H"]
        layer_T = cache["layer_T"]
        layer_h = cache["layer_h"]
        N = X.shape[0]

        grads: Dict[str, np.ndarray] = {}

        # Output projection: logits = h_last @ W_out.T + b_out
        h_last = layer_h[-1] if self.depth > 0 else h0
        grads["W_out"] = dlogits.T @ h_last
        grads["b_out"] = dlogits.sum(axis=0)
        dh = dlogits @ self.W_out  # (N, hidden)

        # Walk hidden blocks in reverse
        for i in reversed(range(self.depth)):
            h_prev = layer_h[i - 1] if i > 0 else h0
            H = layer_H[i]
            if self.block == "highway":
                T = layer_T[i]
                # h_new = T * H + (1 - T) * h_prev
                # d/dT = (H - h_prev), d/dH = T, d/dh_prev (carry path) = (1 - T)
                dT = dh * (H - h_prev)
                dH = dh * T
                d_carry = dh * (1.0 - T)

                # T = sigmoid(zT); zT = h_prev @ W_T.T + b_T
                dzT = dT * _dsigmoid_from_y(T)
                grads[f"W_T_{i}"] = dzT.T @ h_prev
                grads[f"b_T_{i}"] = dzT.sum(axis=0)
                dh_from_T = dzT @ self.W_T[i]

                # H = tanh(zH); zH = h_prev @ W_H.T + b_H
                dzH = dH * _dtanh_from_y(H)
                grads[f"W_H_{i}"] = dzH.T @ h_prev
                grads[f"b_H_{i}"] = dzH.sum(axis=0)
                dh_from_H = dzH @ self.W_H[i]

                dh = d_carry + dh_from_T + dh_from_H
            else:
                # plain: h_new = tanh(zH); zH = h_prev @ W_H.T + b_H
                dzH = dh * _dtanh_from_y(H)
                grads[f"W_H_{i}"] = dzH.T @ h_prev
                grads[f"b_H_{i}"] = dzH.sum(axis=0)
                dh = dzH @ self.W_H[i]

        # Input projection: h0 = tanh(X @ W_in.T + b_in)
        dz_in = dh * _dtanh_from_y(h0)
        grads["W_in"] = dz_in.T @ X
        grads["b_in"] = dz_in.sum(axis=0)

        return grads

    # -- Adam update ---------------------------------------------------

    def adam_step(
        self, grads: Dict[str, np.ndarray],
        lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
        clip: float = 5.0,
    ) -> float:
        """Apply Adam to all params; return global grad norm before clipping."""
        # Global gradient L2 norm for clipping + diagnostics
        sqsum = 0.0
        for g in grads.values():
            sqsum += float((g * g).sum())
        gnorm = float(np.sqrt(sqsum))
        scale = 1.0 if gnorm < clip else (clip / (gnorm + 1e-12))

        self._t += 1
        params = self._params_dict()
        for k, p in params.items():
            g = grads[k] * scale
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)
            m_hat = self._m[k] / (1.0 - b1 ** self._t)
            v_hat = self._v[k] / (1.0 - b2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return gnorm


# ----------------------------------------------------------------------
# Loss + accuracy
# ----------------------------------------------------------------------

def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Stable softmax CE. Returns (mean loss, dL/dlogits)."""
    N, C = logits.shape
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    p = exp / exp.sum(axis=1, keepdims=True)
    ll = -np.log(p[np.arange(N), y] + 1e-12).mean()
    dL = p.copy()
    dL[np.arange(N), y] -= 1.0
    dL /= N
    return float(ll), dL


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train_one(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    *, hidden: int, depth: int, block: str,
    epochs: int, batch_size: int, lr: float,
    seed: int, eval_every: int = 1, log_layers: bool = False,
    label: str = "", verbose: bool = True,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    net = DeepNet(d_in=Xtr.shape[1], d_out=10,
                  hidden=hidden, depth=depth, block=block, rng=rng)
    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_acc": [], "grad_norm": [],
    }
    layer_T_history: List[Tuple[int, List[float]]] = []
    n_train = Xtr.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    t_start = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        ep_acc = 0.0
        ep_gn = 0.0
        n_steps = 0
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            logits = net.forward(xb)
            loss, dlog = softmax_cross_entropy(logits, yb)
            grads = net.backward(dlog)
            gn = net.adam_step(grads, lr=lr)
            ep_loss += loss
            ep_acc += accuracy(logits, yb)
            ep_gn += gn
            n_steps += 1
        ep_loss /= n_steps
        ep_acc /= n_steps
        ep_gn /= n_steps

        # Test accuracy (in chunks)
        te_acc = _eval_acc(net, Xte, yte, batch_size=512)
        history["epoch"].append(ep + 1)
        history["train_loss"].append(ep_loss)
        history["train_acc"].append(ep_acc)
        history["test_acc"].append(te_acc)
        history["grad_norm"].append(ep_gn)

        if log_layers and block == "highway":
            # Probe T-gate means per layer on first 256 test examples
            net.forward(Xte[:256])
            T_means = [float(t.mean()) for t in net._cache["layer_T"]]
            layer_T_history.append((ep + 1, T_means))

        if verbose:
            elapsed = time.time() - t_start
            print(f"  [{label}] ep {ep+1:3d}/{epochs}  loss {ep_loss:.4f}  "
                  f"trA {ep_acc:.3f}  teA {te_acc:.3f}  |g| {ep_gn:.3e}  "
                  f"t {elapsed:.1f}s", flush=True)
    elapsed = time.time() - t_start

    # Final per-layer T statistics for highway nets (one snapshot)
    final_T_per_layer: List[float] = []
    if block == "highway":
        net.forward(Xte[:1000])
        final_T_per_layer = [float(t.mean()) for t in net._cache["layer_T"]]

    return {
        "label": label,
        "block": block,
        "depth": depth,
        "hidden": hidden,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "final_train_loss": history["train_loss"][-1],
        "wallclock_sec": elapsed,
        "layer_T_history": layer_T_history,
        "final_T_per_layer": final_T_per_layer,
    }


def _eval_acc(net: DeepNet, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    n = X.shape[0]
    correct = 0
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        logits = net.forward(X[s:e])
        correct += int((logits.argmax(axis=1) == y[s:e]).sum())
    return correct / n


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

DEFAULT_HIDDEN = 50
DEFAULT_DEPTH = 30
DEFAULT_EPOCHS = 12
DEFAULT_BATCH = 128
DEFAULT_LR = 5e-3
DEFAULT_NTRAIN = 6000
DEFAULT_NTEST = 2000


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help="Depth (in hidden blocks) for the headline run.")
    p.add_argument("--depths", type=str, default="",
                   help="Comma-separated list of depths to sweep, e.g. 10,20,30,50.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--n-train", type=int, default=DEFAULT_NTRAIN)
    p.add_argument("--n-test", type=int, default=DEFAULT_NTEST)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: depth=10, epochs=5, n_train=2000.")
    p.add_argument("--out", type=str, default="run.json")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args(argv)


def run_pair(Xtr, ytr, Xte, yte, *, depth: int, args, log_layers: bool = False) -> Dict[str, object]:
    """Train one highway and one plain net at the same depth/seed/etc."""
    t0 = time.time()
    print(f"\n=== depth={depth}, hidden={args.hidden} ===", flush=True)
    print("--- highway ---", flush=True)
    res_hw = train_one(
        Xtr, ytr, Xte, yte,
        hidden=args.hidden, depth=depth, block="highway",
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        seed=args.seed, label=f"hw-{depth}",
        log_layers=log_layers,
    )
    print("--- plain ---", flush=True)
    res_pl = train_one(
        Xtr, ytr, Xte, yte,
        hidden=args.hidden, depth=depth, block="plain",
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        seed=args.seed, label=f"pl-{depth}",
    )
    return {"depth": depth, "highway": res_hw, "plain": res_pl,
            "wallclock_sec": time.time() - t0}


def main(argv=None):
    args = parse_args(argv)

    if args.quick:
        args.depth = 10
        args.epochs = 5
        args.n_train = 2000
        args.n_test = 1000

    print(f"highway-networks, seed={args.seed}, depth={args.depth}, "
          f"hidden={args.hidden}, epochs={args.epochs}, batch={args.batch}, "
          f"lr={args.lr}, n_train={args.n_train}, n_test={args.n_test}",
          flush=True)
    print(f"env: {env_metadata()}", flush=True)

    print("loading MNIST ...", flush=True)
    mnist = load_mnist()
    Xtr_full, ytr_full = mnist["Xtr"], mnist["ytr"]
    Xte_full, yte_full = mnist["Xte"], mnist["yte"]
    rng = np.random.default_rng(args.seed)
    tr_idx = rng.permutation(Xtr_full.shape[0])[: args.n_train]
    te_idx = rng.permutation(Xte_full.shape[0])[: args.n_test]
    Xtr, ytr = Xtr_full[tr_idx], ytr_full[tr_idx]
    Xte, yte = Xte_full[te_idx], yte_full[te_idx]
    print(f"  train {Xtr.shape}, test {Xte.shape}", flush=True)

    runs: List[Dict[str, object]] = []

    if args.depths:
        depths = [int(s) for s in args.depths.split(",") if s.strip()]
        for d in depths:
            runs.append(run_pair(Xtr, ytr, Xte, yte, depth=d, args=args,
                                 log_layers=(d == max(depths))))
    else:
        runs.append(run_pair(Xtr, ytr, Xte, yte, depth=args.depth, args=args,
                             log_layers=True))

    headline = runs[-1]
    print("\n=== HEADLINE ===")
    print(f"  highway depth={headline['depth']}: "
          f"final test acc {headline['highway']['final_test_acc']:.3f}  "
          f"({headline['highway']['wallclock_sec']:.1f}s)")
    print(f"  plain   depth={headline['depth']}: "
          f"final test acc {headline['plain']['final_test_acc']:.3f}  "
          f"({headline['plain']['wallclock_sec']:.1f}s)")

    out = {
        "seed": args.seed,
        "hidden": args.hidden,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "env": env_metadata(),
        "runs": runs,
    }
    if not args.no_save:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
