"""
compete-to-compute --- Srivastava, Masci, Kazerounian, Gomez, Schmidhuber,
*Compete to compute*, NIPS 2013.

Headline contrast
-----------------
Two MLPs with identical width / depth / optimiser are trained sequentially on
two disjoint MNIST class-splits (Task1 = digits 0-4; Task2 = digits 5-9; the
output head spans all 10 classes throughout). After Task2 finishes, accuracy
on Task1's test set (classes 0-4) is measured.

  * ``ReluMLP``  --- vanilla feed-forward MLP with ReLU activations. Every
    hidden unit responds to every batch, so Task2's gradient touches every
    weight. Task1's representation is overwritten -- catastrophic forgetting.
  * ``LwtaMLP``  --- Local Winner-Take-All blocks of size ``k``. Inside each
    block of ``k`` units the maximum activation is forwarded; the others
    output zero. Backprop only flows through the winner. With Task1 ~~ Task2
    statistics differing, different blocks specialise on different tasks,
    so Task2 only updates a strict subset of the parameters and Task1
    accuracy is preserved.

Both networks share identical seeds, initial weights, optimiser and learning
rate; the only difference is the activation rule.

CLI
---
    python3 compete_to_compute.py --seed 0

Outputs (default args, MacBook M-series CPU)
--------------------------------------------
* ``viz/snapshots.npz``   per-epoch test-accuracy + winner-frequency log
* training curves printed to stdout

The script targets <90s wallclock on a laptop CPU and records seed, config,
git hash, platform and library versions in ``results.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader (pure stdlib + numpy; cached under ~/.cache/hinton-mnist)
# ----------------------------------------------------------------------
CACHE = os.path.expanduser("~/.cache/hinton-mnist")
URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def load_mnist() -> dict:
    """Download (once) and decode MNIST. Floats in [0, 1]."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for k, url in URLS.items():
        path = os.path.join(CACHE, os.path.basename(url))
        if not os.path.exists(path):
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
        with gzip.open(path, "rb") as f:
            data = f.read()
        if "images" in k:
            out[k] = (np.frombuffer(data, np.uint8, offset=16)
                      .reshape(-1, 28, 28).astype(np.float32) / 255.0)
        else:
            out[k] = np.frombuffer(data, np.uint8, offset=8).astype(np.int64)
    return out


def balanced_subsample(images: np.ndarray, labels: np.ndarray,
                       n_per_class: int, rng: np.random.Generator,
                       classes=None):
    if classes is None:
        classes = sorted(np.unique(labels).tolist())
    cls = [np.where(labels == c)[0] for c in classes]
    n_per_class = min(n_per_class, min(len(c) for c in cls))
    idx = np.concatenate([rng.choice(cls[i], n_per_class, replace=False)
                          for i in range(len(classes))])
    rng.shuffle(idx)
    return images[idx], labels[idx]


def one_hot(y: np.ndarray, n_cls: int = 10) -> np.ndarray:
    out = np.zeros((y.size, n_cls), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


def split_by_classes(images: np.ndarray, labels: np.ndarray,
                     classes) -> tuple:
    """Sub-select rows whose label belongs to ``classes``."""
    mask = np.isin(labels, np.asarray(classes))
    return images[mask], labels[mask]


# ----------------------------------------------------------------------
# Activation primitives
# ----------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def lwta_mask(z: np.ndarray, k: int) -> np.ndarray:
    """Per-block winner mask. Ties broken by argmax (returns smallest index).

    z : (B, H), H must be divisible by k.
    """
    B, H = z.shape
    G = H // k
    z_g = z.reshape(B, G, k)
    winners = np.argmax(z_g, axis=2)             # (B, G)
    mask_g = np.zeros_like(z_g)
    np.put_along_axis(mask_g, winners[..., None], 1.0, axis=2)
    return mask_g.reshape(B, H)


# ----------------------------------------------------------------------
# Network: a configurable MLP with ReLU or LWTA activation
# ----------------------------------------------------------------------
class MLP:
    """L-1 hidden layers + 1 linear output. activation in {'relu', 'lwta'}.

    Identical SGD-with-momentum optimiser regardless of activation. The
    only difference between the two variants is the hidden-layer
    forward/backward.
    """

    def __init__(self, sizes, activation: str, k: int,
                 rng: np.random.Generator):
        self.sizes = list(sizes)
        self.activation = activation
        self.k = k
        if activation == "lwta":
            for h in self.sizes[1:-1]:
                assert h % k == 0, f"hidden {h} not divisible by k={k}"
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / n_in)
            self.W.append((rng.standard_normal((n_in, n_out)) * scale)
                          .astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))
        # Momentum buffers
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ---- forward ----------------------------------------------------
    def forward(self, x: np.ndarray):
        cache = {"a": [x], "mask": [None]}      # mask aligned with hidden layers
        h = x
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            if self.activation == "relu":
                mask = (z > 0).astype(np.float32)
                h = z * mask
            elif self.activation == "lwta":
                mask = lwta_mask(z, self.k)
                h = z * mask
            else:
                raise ValueError(self.activation)
            cache["a"].append(h)
            cache["mask"].append(mask)
        logits = h @ self.W[-1] + self.b[-1]
        cache["logits"] = logits
        return logits, cache

    # ---- loss + grad ------------------------------------------------
    def loss_and_grads(self, cache, y_onehot, head_mask=None):
        """``head_mask`` is an optional ``(C,)`` mask in {0,1}. When given,
        loss is computed over the masked logits only (softmax over the
        masked classes); gradient on the inactive output rows is zero.

        This implements the standard *multi-head* split-MNIST evaluation:
        Task1 only ever sees logits for its classes, Task2 only sees its
        own. The shared body of the network is what does or does not
        forget."""
        logits = cache["logits"]
        B = logits.shape[0]
        if head_mask is not None:
            # mask = 1 for active classes, else 0
            inactive = (1.0 - head_mask) * 1e9
            masked = logits - inactive               # softmax over active
            log_p = log_softmax(masked)
        else:
            log_p = log_softmax(logits)
        loss = -(y_onehot * log_p).sum() / B
        probs = np.exp(log_p)
        dlogits = (probs - y_onehot) / B          # (B, C)
        if head_mask is not None:
            dlogits = dlogits * head_mask[None, :]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        # Output layer
        gW[-1] = cache["a"][-1].T @ dlogits
        gb[-1] = dlogits.sum(0)
        da = dlogits @ self.W[-1].T
        # Hidden layers
        for i in range(len(self.W) - 2, -1, -1):
            mask = cache["mask"][i + 1]
            dz = da * mask                          # ReLU and LWTA both gate via mask
            gW[i] = cache["a"][i].T @ dz
            gb[i] = dz.sum(0)
            da = dz @ self.W[i].T
        return loss, gW, gb

    # ---- update -----------------------------------------------------
    def step(self, gW, gb, lr: float, momentum: float, weight_decay: float):
        for i in range(len(self.W)):
            g = gW[i] + weight_decay * self.W[i]
            self.vW[i] = momentum * self.vW[i] - lr * g
            self.W[i] += self.vW[i]
            self.vb[i] = momentum * self.vb[i] - lr * gb[i]
            self.b[i] += self.vb[i]

    # ---- inference + winner statistics ------------------------------
    def predict(self, x: np.ndarray, head_mask=None) -> np.ndarray:
        logits, _ = self.forward(x)
        if head_mask is not None:
            inactive = (1.0 - head_mask) * 1e9
            logits = logits - inactive
        return logits.argmax(1)

    def winner_freq(self, x: np.ndarray, layer: int = 0) -> np.ndarray:
        """Fraction of inputs that activate each unit in the given hidden
        layer. For LWTA: 1/k average over a balanced batch when units are
        distributed evenly. For ReLU: average over (z > 0)."""
        _, cache = self.forward(x)
        return cache["mask"][layer + 1].mean(0)


# ----------------------------------------------------------------------
# Training loop with eval-on-both-tasks bookkeeping
# ----------------------------------------------------------------------
def train_one_task(net: MLP, X: np.ndarray, Y: np.ndarray,
                   *, epochs: int, batch: int, lr: float, momentum: float,
                   weight_decay: float, rng: np.random.Generator,
                   eval_fns, head_mask=None):
    """Train ``net`` for ``epochs`` SGD passes on (X, Y).
    ``eval_fns`` is a dict of name -> callable(net) -> dict; called
    once per epoch and the per-epoch records are returned as a list.
    ``head_mask`` (optional) restricts loss / gradient to a subset of
    output classes (multi-head continual-learning protocol)."""
    log = []
    N = X.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(N)
        ep_loss = 0.0
        ep_correct = 0
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            xb = X[b]
            yb = Y[b]
            logits, cache = net.forward(xb)
            loss, gW, gb = net.loss_and_grads(cache, yb, head_mask=head_mask)
            net.step(gW, gb, lr, momentum, weight_decay)
            ep_loss += loss * xb.shape[0]
            if head_mask is not None:
                inact = (1.0 - head_mask) * 1e9
                pred = (logits - inact).argmax(1)
            else:
                pred = logits.argmax(1)
            ep_correct += int((pred == yb.argmax(1)).sum())
        rec = {"epoch": ep + 1,
               "train_loss": ep_loss / N,
               "train_acc": ep_correct / N}
        for name, fn in eval_fns.items():
            rec.update({f"{name}_{k}": v for k, v in fn(net).items()})
        log.append(rec)
    return log


def evaluate(net: MLP, X: np.ndarray, y: np.ndarray,
             head_mask=None, batch: int = 1024) -> dict:
    n_correct = 0
    for s in range(0, X.shape[0], batch):
        pred = net.predict(X[s:s + batch], head_mask=head_mask)
        n_correct += int((pred == y[s:s + batch]).sum())
    return {"acc": n_correct / X.shape[0]}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_one_seed(args, seed: int, mn: dict, *, save_snapshots: bool = True,
                 verbose: bool = True):
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"compete-to-compute  seed={seed}  "
              f"hidden={args.hidden}  k={args.k}  depth={args.depth}")

    # ------- data ---------------------------------------------------
    X_tr_full = mn["train_images"].reshape(-1, 28 * 28)
    y_tr_full = mn["train_labels"]
    X_te_full = mn["test_images"].reshape(-1, 28 * 28)
    y_te_full = mn["test_labels"]

    # Task1 / Task2 are disjoint class splits.
    X_tr_t1_full, y_tr_t1_full = split_by_classes(
        X_tr_full, y_tr_full, args.task1_classes)
    X_tr_t2_full, y_tr_t2_full = split_by_classes(
        X_tr_full, y_tr_full, args.task2_classes)
    X_tr_t1, y_tr_t1 = balanced_subsample(
        X_tr_t1_full, y_tr_t1_full, args.n_train_per_class, rng)
    X_tr_t2, y_tr_t2 = balanced_subsample(
        X_tr_t2_full, y_tr_t2_full, args.n_train_per_class, rng)

    X_te_t1, y_te_t1 = split_by_classes(X_te_full, y_te_full,
                                        args.task1_classes)
    X_te_t2, y_te_t2 = split_by_classes(X_te_full, y_te_full,
                                        args.task2_classes)
    # cap test set sizes for speed
    if args.n_test < X_te_t1.shape[0]:
        i = rng.permutation(X_te_t1.shape[0])[:args.n_test]
        X_te_t1, y_te_t1 = X_te_t1[i], y_te_t1[i]
    if args.n_test < X_te_t2.shape[0]:
        i = rng.permutation(X_te_t2.shape[0])[:args.n_test]
        X_te_t2, y_te_t2 = X_te_t2[i], y_te_t2[i]

    Y_tr_t1 = one_hot(y_tr_t1)
    Y_tr_t2 = one_hot(y_tr_t2)

    if verbose:
        print(f"  Task1 classes={args.task1_classes} -> "
              f"train={X_tr_t1.shape[0]}, test={X_te_t1.shape[0]}")
        print(f"  Task2 classes={args.task2_classes} -> "
              f"train={X_tr_t2.shape[0]}, test={X_te_t2.shape[0]}")

    # ------- networks (identical seeds, identical inits up to activation) -
    sizes = [28 * 28] + [args.hidden] * args.depth + [10]
    net_relu = MLP(sizes, "relu", args.k, np.random.default_rng(seed + 1))
    net_lwta = MLP(sizes, "lwta", args.k, np.random.default_rng(seed + 1))

    # multi-head: only the active task's logits participate in loss / pred.
    head1 = np.zeros(10, dtype=np.float32); head1[args.task1_classes] = 1.0
    head2 = np.zeros(10, dtype=np.float32); head2[args.task2_classes] = 1.0

    # eval fns evaluated once per epoch (after each gradient pass)
    eval_fns = {
        "t1_test": lambda n: evaluate(n, X_te_t1, y_te_t1, head_mask=head1),
        "t2_test": lambda n: evaluate(n, X_te_t2, y_te_t2, head_mask=head2),
    }

    # ------- training schedule --------------------------------------
    schedule = []
    t0 = time.time()
    for tag, net in [("relu", net_relu), ("lwta", net_lwta)]:
        rng_train = np.random.default_rng(seed + 100 if tag == "relu"
                                          else seed + 200)
        if verbose:
            print(f"\n--- {tag.upper()} : Task1 (digits {args.task1_classes}) ---")
        log_t1 = train_one_task(net, X_tr_t1, Y_tr_t1,
                                epochs=args.epochs, batch=args.batch,
                                lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                rng=rng_train, eval_fns=eval_fns,
                                head_mask=head1)
        for r in log_t1:
            r["task"] = 1; r["model"] = tag; schedule.append(r)
            if verbose:
                print(f"  ep{r['epoch']}  trL={r['train_loss']:.3f}  "
                      f"trA={r['train_acc']:.3f}  "
                      f"T1={r['t1_test_acc']:.3f}  T2={r['t2_test_acc']:.3f}")
        t1_final_acc = log_t1[-1]["t1_test_acc"]

        if verbose:
            print(f"--- {tag.upper()} : Task2 (digits {args.task2_classes}) ---")
        log_t2 = train_one_task(net, X_tr_t2, Y_tr_t2,
                                epochs=args.epochs, batch=args.batch,
                                lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                rng=rng_train, eval_fns=eval_fns,
                                head_mask=head2)
        for r in log_t2:
            r["task"] = 2; r["model"] = tag; schedule.append(r)
            if verbose:
                print(f"  ep{r['epoch']}  trL={r['train_loss']:.3f}  "
                      f"trA={r['train_acc']:.3f}  "
                      f"T1={r['t1_test_acc']:.3f}  T2={r['t2_test_acc']:.3f}")
        t1_after_t2_acc = log_t2[-1]["t1_test_acc"]
        t2_final_acc = log_t2[-1]["t2_test_acc"]
        forgetting = t1_final_acc - t1_after_t2_acc
        if verbose:
            print(f"  ==> {tag}: T1 before T2 = {t1_final_acc:.3f}, "
                  f"T1 after T2 = {t1_after_t2_acc:.3f}, "
                  f"forgetting = {forgetting:.3f}, "
                  f"T2 final = {t2_final_acc:.3f}")
    wall = time.time() - t0
    if verbose:
        print(f"wallclock: {wall:.1f}s")

    # ------- summary metrics ----------------------------------------
    def pick(model, task, ep):
        for r in schedule:
            if r["model"] == model and r["task"] == task and r["epoch"] == ep:
                return r
        return None

    summary = {}
    for m in ("relu", "lwta"):
        t1_pre = pick(m, 1, args.epochs)["t1_test_acc"]
        t1_post = pick(m, 2, args.epochs)["t1_test_acc"]
        t2_post = pick(m, 2, args.epochs)["t2_test_acc"]
        summary[m] = {
            "t1_after_t1_training": t1_pre,
            "t1_after_t2_training": t1_post,
            "t2_after_t2_training": t2_post,
            "forgetting": t1_pre - t1_post,
        }

    snapshot_data = None
    if save_snapshots:
        snapshot_data = dict(
            epochs=np.array([r["epoch"] for r in schedule]),
            tasks=np.array([r["task"] for r in schedule]),
            models=np.array([r["model"] for r in schedule]),
            t1_test_acc=np.array([r["t1_test_acc"] for r in schedule]),
            t2_test_acc=np.array([r["t2_test_acc"] for r in schedule]),
            train_loss=np.array([r["train_loss"] for r in schedule]),
            W1_relu=net_relu.W[0],
            W1_lwta=net_lwta.W[0],
            winner_freq_relu_t1=net_relu.winner_freq(X_te_t1, layer=0),
            winner_freq_relu_t2=net_relu.winner_freq(X_te_t2, layer=0),
            winner_freq_lwta_t1=net_lwta.winner_freq(X_te_t1, layer=0),
            winner_freq_lwta_t2=net_lwta.winner_freq(X_te_t2, layer=0),
        )
    return {"summary": summary, "schedule": schedule,
            "wallclock_s": wall, "snapshots": snapshot_data}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the headline single-seed run "
                        "(also the start of the multi-seed sweep)")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="run this many consecutive seeds and aggregate "
                        "(seed, seed+1, ..., seed+n-1)")
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--k", type=int, default=2,
                   help="LWTA block size (hidden must be divisible by k)")
    p.add_argument("--depth", type=int, default=2,
                   help="number of hidden layers")
    p.add_argument("--n-train-per-class", type=int, default=500)
    p.add_argument("--n-test", type=int, default=1000,
                   help="cap per-task test set size")
    p.add_argument("--task1-classes", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4])
    p.add_argument("--task2-classes", type=int, nargs="+",
                   default=[5, 6, 7, 8, 9])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out", default="results.json")
    p.add_argument("--snapshots", default="viz/snapshots.npz")
    args = p.parse_args()

    print(f"compete-to-compute  start_seed={args.seed}  n_seeds={args.n_seeds}  "
          f"hidden={args.hidden}  k={args.k}  depth={args.depth}")
    print("loading MNIST...")
    mn = load_mnist()

    seed_runs = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        save_snap = (i == 0)            # only snapshot the headline seed
        run = run_one_seed(args, seed, mn, save_snapshots=save_snap,
                           verbose=(args.n_seeds == 1 or save_snap))
        run["seed"] = seed
        seed_runs.append(run)
        if args.n_seeds > 1 and not save_snap:
            print(f"  seed={seed}: ReLU forget = "
                  f"{run['summary']['relu']['forgetting']:.3f}  "
                  f"LWTA forget = "
                  f"{run['summary']['lwta']['forgetting']:.3f}")

    # ------- save snapshot from the headline seed ------------------
    headline = seed_runs[0]
    if headline["snapshots"] is not None:
        os.makedirs(os.path.dirname(args.snapshots), exist_ok=True)
        np.savez(args.snapshots, **headline["snapshots"])

    # ------- aggregate stats --------------------------------------
    def _mean_std(vals):
        v = np.array(vals, dtype=np.float64)
        return float(v.mean()), float(v.std(ddof=0))

    relu_forgets = [r["summary"]["relu"]["forgetting"] for r in seed_runs]
    lwta_forgets = [r["summary"]["lwta"]["forgetting"] for r in seed_runs]
    relu_t2 = [r["summary"]["relu"]["t2_after_t2_training"] for r in seed_runs]
    lwta_t2 = [r["summary"]["lwta"]["t2_after_t2_training"] for r in seed_runs]

    aggregate = {
        "n_seeds": args.n_seeds,
        "relu_forgetting_mean": _mean_std(relu_forgets)[0],
        "relu_forgetting_std":  _mean_std(relu_forgets)[1],
        "lwta_forgetting_mean": _mean_std(lwta_forgets)[0],
        "lwta_forgetting_std":  _mean_std(lwta_forgets)[1],
        "relu_t2_acc_mean":     _mean_std(relu_t2)[0],
        "lwta_t2_acc_mean":     _mean_std(lwta_t2)[0],
        "lwta_wins_per_seed": int(sum(l < r for l, r in
                                      zip(lwta_forgets, relu_forgets))),
    }

    # ------- results.json -----------------------------------------
    out = {
        "config": {
            "seed": args.seed,
            "n_seeds": args.n_seeds,
            "hidden": args.hidden,
            "k": args.k,
            "depth": args.depth,
            "n_train_per_class": args.n_train_per_class,
            "n_test": args.n_test,
            "task1_classes": list(args.task1_classes),
            "task2_classes": list(args.task2_classes),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "git": git_hash(),
        },
        "headline_seed": args.seed,
        "headline_summary": headline["summary"],
        "headline_wallclock_s": round(headline["wallclock_s"], 2),
        "schedule": headline["schedule"],
        "aggregate": aggregate,
        "per_seed": [{"seed": r["seed"], "summary": r["summary"]}
                     for r in seed_runs],
    }

    def _coerce(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"unhandled {type(o).__name__}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=_coerce)
    print(f"wrote {args.out}")
    print(f"\nHEADLINE (seed={args.seed}): forgetting (T1 acc dropped after Task2)")
    print(f"  ReLU MLP : {headline['summary']['relu']['forgetting']:.3f}")
    print(f"  LWTA MLP : {headline['summary']['lwta']['forgetting']:.3f}")
    if args.n_seeds > 1:
        print(f"\nMULTI-SEED MEAN ({args.n_seeds} seeds):")
        print(f"  ReLU MLP : {aggregate['relu_forgetting_mean']:.3f} "
              f"+/- {aggregate['relu_forgetting_std']:.3f}")
        print(f"  LWTA MLP : {aggregate['lwta_forgetting_mean']:.3f} "
              f"+/- {aggregate['lwta_forgetting_std']:.3f}")
        print(f"  LWTA wins on {aggregate['lwta_wins_per_seed']}/"
              f"{args.n_seeds} seeds")
    if (aggregate["lwta_forgetting_mean"]
            < aggregate["relu_forgetting_mean"]):
        print("  LWTA forgets less than ReLU on average. (Reproduces.)")
    else:
        print("  LWTA does NOT forget less than ReLU on average.")


if __name__ == "__main__":
    main()
