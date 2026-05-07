"""predictable-stereo -- Schmidhuber & Prelinger,
*Discovering predictable classifications*, Neural Computation 5(4):625-635 (1993).
Companion to Becker & Hinton 1992, *Self-organizing neural network that
discovers surfaces in random-dot stereograms*, Nature 355:161-163.

Predictability MAXIMIZATION (the dual of predictability minimization).

Two networks, each with a different view of the same scene, train cooperatively
to produce scalar codes that maximally agree. Their only shared information
is a hidden binary "depth" variable; everything else in each view is
view-specific noise that the networks must learn to ignore.

Becker-Hinton IMAX objective (their 1992 paper, equation 4):
    I(y_L; y_R) = 0.5 * log( var(y_L + y_R) / var(y_L - y_R) )
which under the Gaussian assumption equals the mutual information between
the two scalar outputs. Maximizing it forces the codes to capture only the
shared signal: anything view-specific shows up as noise in (y_L - y_R) and
is penalized.

Schmidhuber & Prelinger 1993 take this from continuous codes to a binary
classification regime (their Figs 1-2 use binary stereo input). We train
with the continuous IMAX (BPTT-friendly), then threshold the scalar code at
0 to read off a binary classification, and measure recovery accuracy of the
hidden depth.

Synthetic binary stereo (constructed from scratch since the original
random-dot stereograms are heavyweight):
  * d_shared dims encode the hidden depth z in {-1, +1} via a sign flip
    on a fixed left-eye template (resp. right-eye template), plus low-amplitude
    bit-flip noise.
  * d_view dims per view are uncorrelated random per-sample bits ("view-specific
    distractors"). They look just like the shared dims marginally, so a
    network with no access to the partner cannot tell which dims to attend to.

The point of predictability maximization: the only way to make y_L and y_R
agree across the dataset is to read the shared dims. Networks discover
which input dims are shared without supervision.

Headlines (--seed 0, default config):
    Predictability MAX on real stereo  : ~1.00 binary recovery accuracy
    Predictability MAX on shuffled stereo (no shared variable): ~0.50
    Untrained random networks          : ~0.50

CLI:
    python3 predictable_stereo.py --seed 0
    python3 predictable_stereo.py --seed 0 --quick      # smoke test
    python3 predictable_stereo.py --seed 0 --shuffled   # baseline (no shared z)
    python3 predictable_stereo.py --seeds 0,1,2,3,4     # multi-seed sweep
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


# ----------------------------------------------------------------------
# Synthetic stereo dataset
# ----------------------------------------------------------------------

def make_stereo_dataset(
    n_samples: int,
    d_shared: int = 8,
    d_view: int = 8,
    flip_p: float = 0.10,
    seed: int = 0,
    shuffled: bool = False,
) -> Dict[str, np.ndarray]:
    """Build paired binary "stereo" inputs with a hidden shared depth variable.

    Each sample i has a depth bit z_i in {-1, +1}. The left and right views
    each have:
        d_shared dims = z_i * template(left or right) with each bit flipped
            independently with probability flip_p (low-amplitude observation noise)
        d_view dims = i.i.d. uniform {-1, +1} per sample, INDEPENDENT between
            views (view-specific distractors).

    The two templates are random {-1, +1} vectors, fixed across the dataset,
    different between the two views. A network must learn which input dims
    are shared (i.e. which dims encode z) without supervision.

    Args:
        n_samples : number of paired samples
        d_shared  : number of input dims encoding the shared depth
        d_view    : number of view-specific (distractor) input dims per view
        flip_p    : observation noise: per-bit flip probability on shared dims
        seed      : RNG seed
        shuffled  : if True, the right view's depth is reshuffled, breaking
                    the shared-variable structure. Used as a baseline: any
                    "agreement" the networks find here is spurious.

    Returns:
        dict with x_L (n, d_shared+d_view), x_R (n, d_shared+d_view), z (n,)
    """
    rng = np.random.default_rng(seed)
    z_L = rng.choice([-1.0, 1.0], size=n_samples)
    if shuffled:
        # Independent depth per view -> nothing shared.
        z_R = rng.permutation(z_L)  # same marginal but unrelated to z_L
    else:
        z_R = z_L

    template_L = rng.choice([-1.0, 1.0], size=d_shared)
    template_R = rng.choice([-1.0, 1.0], size=d_shared)

    # Shared-encoding dims: z_i * template, then flip each bit w.p. flip_p.
    base_L = np.outer(z_L, template_L)  # (n, d_shared)
    base_R = np.outer(z_R, template_R)
    flips_L = rng.random((n_samples, d_shared)) < flip_p
    flips_R = rng.random((n_samples, d_shared)) < flip_p
    shared_L = base_L * np.where(flips_L, -1.0, 1.0)
    shared_R = base_R * np.where(flips_R, -1.0, 1.0)

    # View-specific dims: independent random bits per view per sample.
    view_L = rng.choice([-1.0, 1.0], size=(n_samples, d_view))
    view_R = rng.choice([-1.0, 1.0], size=(n_samples, d_view))

    x_L = np.concatenate([shared_L, view_L], axis=1)
    x_R = np.concatenate([shared_R, view_R], axis=1)

    return {
        "x_L": x_L.astype(np.float64),
        "x_R": x_R.astype(np.float64),
        "z": z_L.astype(np.float64),  # ground-truth depth bits
        "template_L": template_L,
        "template_R": template_R,
        "d_shared": d_shared,
        "d_view": d_view,
    }


# ----------------------------------------------------------------------
# Two-layer scalar-output MLP (per-view network)
# ----------------------------------------------------------------------

class ViewNet:
    """Small MLP that maps a view to a scalar code y in [-1, 1]."""

    def __init__(self, d_in: int, d_hidden: int, rng: np.random.Generator):
        s1 = 1.0 / np.sqrt(d_in)
        s2 = 1.0 / np.sqrt(d_hidden)
        self.W1 = rng.uniform(-s1, s1, size=(d_hidden, d_in))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.uniform(-s2, s2, size=(1, d_hidden))
        self.b2 = np.zeros(1)
        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._t = 0

    def params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Returns y of shape (N,)."""
        self._cache_X = X
        self._cache_z1 = X @ self.W1.T + self.b1            # (N, d_h)
        self._cache_h = np.tanh(self._cache_z1)             # (N, d_h)
        self._cache_z2 = self._cache_h @ self.W2.T + self.b2  # (N, 1)
        self._cache_y = np.tanh(self._cache_z2).reshape(-1)
        return self._cache_y

    def backward(self, dy: np.ndarray) -> Dict[str, np.ndarray]:
        """Given dL/dy of shape (N,), return param gradients."""
        N = dy.shape[0]
        dz2 = (dy * (1.0 - self._cache_y ** 2)).reshape(N, 1)
        dW2 = dz2.T @ self._cache_h          # (1, d_h)
        db2 = dz2.sum(axis=0)                # (1,)
        dh = dz2 @ self.W2                   # (N, d_h)
        dz1 = dh * (1.0 - self._cache_h ** 2)
        dW1 = dz1.T @ self._cache_X          # (d_h, d_in)
        db1 = dz1.sum(axis=0)                # (d_h,)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step_adam(self, grads: Dict[str, np.ndarray], lr: float,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k, p in self.params().items():
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            p[...] -= lr * mhat / (np.sqrt(vhat) + eps)


# ----------------------------------------------------------------------
# IMAX loss (Becker-Hinton 1992 equation 4)
# ----------------------------------------------------------------------

def imax_loss_and_grads(
    yL: np.ndarray, yR: np.ndarray, eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """Returns -I(yL; yR) = -0.5 * log(var(yL+yR) / var(yL-yR)) and
    its gradients w.r.t. yL, yR.

    Minimizing this is maximizing the agreement-to-disagreement ratio.
    """
    N = yL.shape[0]
    s = yL + yR
    d = yL - yR
    mu_s = s.mean()
    mu_d = d.mean()
    var_s = ((s - mu_s) ** 2).mean() + eps
    var_d = ((d - mu_d) ** 2).mean() + eps
    loss = 0.5 * (np.log(var_d) - np.log(var_s))   # = -0.5 log(var_s/var_d)
    # Gradient: dL/dvar_s = -0.5/var_s, dL/dvar_d = +0.5/var_d
    # dvar_s/ds_i = (2/N)(s_i - mu_s); dvar_d/dd_i = (2/N)(d_i - mu_d)
    dyL = -(s - mu_s) / (N * var_s) + (d - mu_d) / (N * var_d)
    dyR = -(s - mu_s) / (N * var_s) - (d - mu_d) / (N * var_d)
    info = {
        "loss": float(loss),
        "var_signal": float(var_s),
        "var_noise": float(var_d),
        "I_nats": float(0.5 * np.log(var_s / var_d)),
    }
    return loss, dyL, dyR, info


# ----------------------------------------------------------------------
# Eval: binary recovery accuracy from scalar codes
# ----------------------------------------------------------------------

def recovery_accuracy(y: np.ndarray, z: np.ndarray) -> float:
    """Threshold y at 0 to get a binary code, compare with z (also binary).
    Account for sign ambiguity by taking max(acc, 1 - acc)."""
    pred = np.where(y >= 0.0, 1.0, -1.0)
    acc = float(np.mean(pred == z))
    return max(acc, 1.0 - acc)


def agreement(yL: np.ndarray, yR: np.ndarray) -> float:
    """Binary agreement after thresholding."""
    bL = np.where(yL >= 0.0, 1, -1)
    bR = np.where(yR >= 0.0, 1, -1)
    return float(np.mean(bL == bR))


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(
    seed: int = 0,
    n_samples: int = 1024,
    n_eval: int = 1024,
    d_shared: int = 8,
    d_view: int = 8,
    flip_p: float = 0.10,
    d_hidden: int = 16,
    lr: float = 0.03,
    n_epochs: int = 400,
    eval_every: int = 20,
    shuffled: bool = False,
    quick: bool = False,
    verbose: bool = True,
):
    if quick:
        n_epochs = 80
        n_samples = 256
        n_eval = 256
        eval_every = 10

    data = make_stereo_dataset(
        n_samples=n_samples, d_shared=d_shared, d_view=d_view,
        flip_p=flip_p, seed=seed, shuffled=shuffled,
    )
    # Held-out eval set: same generative process, fresh samples & noise.
    # The two templates are part of the world (kept fixed); only z, view-noise
    # and the view-specific distractor dims are resampled. This is the standard
    # "new draws from the same distribution" generalization check.
    eval_data = make_stereo_dataset(
        n_samples=n_eval, d_shared=d_shared, d_view=d_view,
        flip_p=flip_p, seed=seed + 9_999, shuffled=shuffled,
    )
    eval_data["template_L"] = data["template_L"]
    eval_data["template_R"] = data["template_R"]
    # Re-render the eval shared dims using the train templates so the world
    # parameters (which template encodes the depth) are shared.
    rng_eval = np.random.default_rng(seed + 1_234_567)
    z_evalL = rng_eval.choice([-1.0, 1.0], size=n_eval)
    if shuffled:
        z_evalR = rng_eval.permutation(z_evalL)
    else:
        z_evalR = z_evalL
    base_L = np.outer(z_evalL, data["template_L"])
    base_R = np.outer(z_evalR, data["template_R"])
    flips_L = rng_eval.random((n_eval, d_shared)) < flip_p
    flips_R = rng_eval.random((n_eval, d_shared)) < flip_p
    shared_L = base_L * np.where(flips_L, -1.0, 1.0)
    shared_R = base_R * np.where(flips_R, -1.0, 1.0)
    view_L = rng_eval.choice([-1.0, 1.0], size=(n_eval, d_view))
    view_R = rng_eval.choice([-1.0, 1.0], size=(n_eval, d_view))
    eval_data["x_L"] = np.concatenate([shared_L, view_L], axis=1)
    eval_data["x_R"] = np.concatenate([shared_R, view_R], axis=1)
    eval_data["z"] = z_evalL

    rng = np.random.default_rng(seed + 31_337)
    d_in = d_shared + d_view
    netL = ViewNet(d_in, d_hidden, rng)
    netR = ViewNet(d_in, d_hidden, rng)

    history = {
        "epoch": [], "loss": [], "I_nats": [],
        "recovery_acc_train": [], "agreement_train": [],
        "recovery_acc_eval": [], "agreement_eval": [],
    }

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        yL = netL.forward(data["x_L"])
        yR = netR.forward(data["x_R"])
        loss, dyL, dyR, info = imax_loss_and_grads(yL, yR)
        gL = netL.backward(dyL)
        gR = netR.backward(dyR)
        netL.step_adam(gL, lr)
        netR.step_adam(gR, lr)

        if ep == 1 or ep % eval_every == 0 or ep == n_epochs:
            yL_tr = netL.forward(data["x_L"])
            yR_tr = netR.forward(data["x_R"])
            yL_ev = netL.forward(eval_data["x_L"])
            yR_ev = netR.forward(eval_data["x_R"])
            acc_tr = recovery_accuracy(yL_tr, data["z"])
            agr_tr = agreement(yL_tr, yR_tr)
            acc_ev = recovery_accuracy(yL_ev, eval_data["z"])
            agr_ev = agreement(yL_ev, yR_ev)
            history["epoch"].append(ep)
            history["loss"].append(info["loss"])
            history["I_nats"].append(info["I_nats"])
            history["recovery_acc_train"].append(acc_tr)
            history["agreement_train"].append(agr_tr)
            history["recovery_acc_eval"].append(acc_ev)
            history["agreement_eval"].append(agr_ev)
            if verbose:
                print(f"  ep {ep:4d}  loss {info['loss']:+.4f}  "
                      f"I {info['I_nats']:.4f} nats  "
                      f"recov[tr/ev] {acc_tr:.3f}/{acc_ev:.3f}  "
                      f"agree[tr/ev] {agr_tr:.3f}/{agr_ev:.3f}")

    wall = time.time() - t0
    final = {
        "wallclock_sec": wall,
        "final_loss": history["loss"][-1],
        "final_I_nats": history["I_nats"][-1],
        "final_recovery_acc_train": history["recovery_acc_train"][-1],
        "final_agreement_train": history["agreement_train"][-1],
        "final_recovery_acc_eval": history["recovery_acc_eval"][-1],
        "final_agreement_eval": history["agreement_eval"][-1],
    }
    return {
        "netL": netL, "netR": netR, "data": data, "eval_data": eval_data,
        "history": history, "summary": final,
    }


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(seeds: List[int], **kwargs) -> List[Dict]:
    rows = []
    for s in seeds:
        kwargs2 = dict(kwargs)
        kwargs2["seed"] = s
        kwargs2["verbose"] = False
        out = train(**kwargs2)
        rows.append({
            "seed": s,
            "final_loss": out["summary"]["final_loss"],
            "final_I_nats": out["summary"]["final_I_nats"],
            "final_recovery_acc_train": out["summary"]["final_recovery_acc_train"],
            "final_recovery_acc_eval": out["summary"]["final_recovery_acc_eval"],
            "final_agreement_eval": out["summary"]["final_agreement_eval"],
        })
    return rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds for a sweep, e.g. 0,1,2,3,4")
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--d-shared", type=int, default=8)
    p.add_argument("--d-view", type=int, default=8)
    p.add_argument("--flip-p", type=float, default=0.10)
    p.add_argument("--d-hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--shuffled", action="store_true",
                   help="No shared depth between views (negative-control baseline)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="run.json")
    args = p.parse_args()

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        rows = sweep(
            seeds,
            n_samples=args.n_samples, d_shared=args.d_shared, d_view=args.d_view,
            flip_p=args.flip_p, d_hidden=args.d_hidden, lr=args.lr,
            n_epochs=args.n_epochs, eval_every=args.eval_every,
            shuffled=args.shuffled, quick=args.quick,
        )
        print("\nseed | loss     | I_nats | recov_train | recov_eval | agree_eval")
        for r in rows:
            print(f" {r['seed']:>3} | {r['final_loss']:+.4f}  | "
                  f"{r['final_I_nats']:6.3f} | "
                  f"{r['final_recovery_acc_train']:>10.3f}  | "
                  f"{r['final_recovery_acc_eval']:>9.3f}  | "
                  f"{r['final_agreement_eval']:>9.3f}")
        accs = [r["final_recovery_acc_eval"] for r in rows]
        print(f"\neval recovery acc: mean {np.mean(accs):.3f}  "
              f"min {np.min(accs):.3f}  max {np.max(accs):.3f}  n={len(accs)}")
        return

    print(f"== predictable-stereo (Becker-Hinton IMAX), seed={args.seed} "
          f"shuffled={args.shuffled} ==")
    out = train(
        seed=args.seed,
        n_samples=args.n_samples, d_shared=args.d_shared, d_view=args.d_view,
        flip_p=args.flip_p, d_hidden=args.d_hidden, lr=args.lr,
        n_epochs=args.n_epochs, eval_every=args.eval_every,
        shuffled=args.shuffled, quick=args.quick,
    )
    s = out["summary"]
    print(f"\nfinal: loss {s['final_loss']:+.4f}   I {s['final_I_nats']:.4f} nats")
    print(f"       recovery_acc  train {s['final_recovery_acc_train']:.3f}   "
          f"eval {s['final_recovery_acc_eval']:.3f}")
    print(f"       agreement     train {s['final_agreement_train']:.3f}   "
          f"eval {s['final_agreement_eval']:.3f}")
    print(f"       wallclock {s['wallclock_sec']:.2f} s")

    blob = {
        "args": vars(args),
        "env": env_metadata(),
        "history": out["history"],
        "summary": s,
    }
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
