"""curiosity-three-regions

Schmidhuber, "Adaptive confidence and adaptive curiosity" (TR FKI-149-91,
TUM, 1991) and "Curious model-building control systems" (IJCNN 1991, vol. 2,
pp. 1458-1463). Reconstructed from the IJCNN paper abstract, the 2010
"Formal theory of creativity, fun, and intrinsic motivation" review, and
the 2020 "Deep Learning: Our Miraculous Year 1990-1991" retrospective.

Setup
-----
A 1-D environment partitioned into three regions:

  A: deterministic, small.   K=4 contexts, fixed targets in {-1,0,+1}.
  B: random / unlearnable.   K=8 contexts, target = N(0, sigma_B) per visit.
  C: learnable but complex.  K=32 contexts, fixed but pseudo-random targets
                              from N(0, sigma_C).

At each step the agent picks one region and observes (context, y). The
context for region r is an internal counter c that cycles 0..K_r-1; this
makes coverage deterministic given the visit order and removes one source
of variance versus uniform sampling, while keeping the prediction problem
non-trivial. Documented in §Deviations.

A per-region tabular world model M[r][c] is updated online with EMA:
    M[r][c] <- M[r][c] + alpha * (y - M[r][c])
Per-step squared error err = (y - M[r][c])**2 is appended to that region's
error history.

Curiosity (per region, per step) is the windowed prediction-error
*reduction*, clipped at zero:
    curiosity_r(t) = max(0, mean(err_r[t-2W:t-W]) - mean(err_r[t-W:t]))
This is the 1991 paper's "improvement of M" signal in its simplest
discrete form. Clipping at zero matches the intent that an agent gets
no reward for getting *worse*.

Policy: epsilon-soft softmax over per-region curiosity. With probability
eps the agent picks a region uniformly (small permanent baseline of
exploration); otherwise it samples from softmax(beta * curiosity). During
the burn-in window the agent picks uniformly to seed all three regions
with a few samples before curiosity is meaningful.

Headline expectation
--------------------
After burn-in:
  - A converges to ~0 error in tens of visits, curiosity drops to 0.
  - B has constant high error (variance of the noise), so windowed
    curiosity is ~0 plus small finite-sample fluctuations.
  - C has many contexts to cover, so windowed curiosity stays positive
    for thousands of visits.

So the visit distribution at end-of-run should be roughly C >> B > A.
The C >> B gap is the curiosity-driven model-building effect; the B > A
gap is the finite-sample noise floor that distinguishes "unpredictable"
from "fully predicted".

Run
---
    python3 curiosity_three_regions.py --seed 0
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np


# ----------------------------- regions ---------------------------------

def make_region(name, K, kind, rng, target_scale):
    """Build a region descriptor.

    'deterministic': K contexts, simple repeating pattern of magnitude scale.
    'random':        K contexts, but target is resampled from N(0, scale)
                     each visit (the context is irrelevant -- pure noise).
    'learnable':     K contexts, fixed but pseudo-random targets ~ N(0, scale).
    """
    if kind == "deterministic":
        base = np.array([1.0, 0.0, -1.0, 0.0])
        targets = np.tile(base, (K + len(base) - 1) // len(base))[:K] * target_scale
    elif kind == "random":
        targets = None
    elif kind == "learnable":
        targets = rng.normal(0.0, target_scale, size=K)
    else:
        raise ValueError(f"unknown region kind: {kind}")
    return {
        "name": name,
        "K": int(K),
        "kind": kind,
        "targets": targets,
        "M": np.zeros(int(K)),
        "counter": 0,
        "target_scale": float(target_scale),
    }


def visit_region(region, rng, alpha):
    """Take one step inside a region. Returns (context, y, prediction, err)."""
    K = region["K"]
    c = region["counter"]
    region["counter"] = (c + 1) % K
    if region["kind"] == "random":
        y = float(rng.normal(0.0, region["target_scale"]))
    else:
        y = float(region["targets"][c])
    pred = float(region["M"][c])
    err = (y - pred) ** 2
    region["M"][c] = (1.0 - alpha) * region["M"][c] + alpha * y
    return c, y, pred, err


# ----------------------------- policy ----------------------------------

def curiosity_signal(err_history, W):
    """Windowed prediction-error reduction, clipped at zero."""
    if len(err_history) < 2 * W:
        return 0.0
    arr = np.asarray(err_history, dtype=np.float64)
    old = float(arr[-2 * W : -W].mean())
    new = float(arr[-W:].mean())
    return max(0.0, old - new)


def select_region(curiosities, rng, beta, eps):
    """Epsilon-soft softmax over curiosity values."""
    n = len(curiosities)
    if rng.random() < eps:
        return int(rng.integers(n))
    cs = np.asarray(curiosities, dtype=np.float64)
    z = beta * cs
    z -= z.max()
    p = np.exp(z)
    p = p / p.sum()
    return int(rng.choice(n, p=p))


# ----------------------------- experiment ------------------------------

DEFAULTS = dict(
    steps=5000,
    burn_in=200,
    window=50,
    alpha=0.05,
    beta=30.0,
    eps=0.02,
    K_det=4,
    K_rand=8,
    K_learn=128,
    sigma_det=1.0,
    sigma_rand=0.5,
    sigma_learn=2.0,
)


def run_experiment(seed=0, **kwargs):
    """Run one curiosity-three-regions episode. Returns a dict of logs."""
    cfg = {**DEFAULTS, **kwargs}
    rng = np.random.default_rng(seed)

    # The region's "targets" RNG is the shared one for the learnable region's
    # fixed pseudo-random targets; the random region pulls from rng each
    # visit. Determinism: rng is seeded once; consumption order is fixed.
    regions = [
        make_region("A_deterministic", cfg["K_det"], "deterministic",
                    rng, cfg["sigma_det"]),
        make_region("B_random",        cfg["K_rand"], "random",
                    rng, cfg["sigma_rand"]),
        make_region("C_learnable",     cfg["K_learn"], "learnable",
                    rng, cfg["sigma_learn"]),
    ]

    err_hist = [[] for _ in regions]      # per-region error history
    chosen = []                            # region index per step
    cur_log = []                           # curiosity per region per step

    for t in range(cfg["steps"]):
        cs = [curiosity_signal(err_hist[i], cfg["window"]) for i in range(len(regions))]
        cur_log.append(cs)

        if t < cfg["burn_in"]:
            a = int(rng.integers(len(regions)))
        else:
            a = select_region(cs, rng, cfg["beta"], cfg["eps"])
        chosen.append(a)

        _c, _y, _pred, err = visit_region(regions[a], rng, cfg["alpha"])
        err_hist[a].append(err)

    visit_counts = [chosen.count(i) for i in range(len(regions))]
    return {
        "regions": [{"name": r["name"], "K": r["K"], "kind": r["kind"]} for r in regions],
        "chosen": chosen,
        "err_hist": err_hist,
        "cur_log": cur_log,
        "visit_counts": visit_counts,
        "config": {"seed": int(seed), **cfg},
        "final_M": [r["M"].tolist() for r in regions],
        "targets": [r["targets"].tolist() if r["targets"] is not None else None
                    for r in regions],
    }


# ----------------------------- CLI -------------------------------------

def _env_info():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--burn-in", type=int, default=DEFAULTS["burn_in"], dest="burn_in")
    p.add_argument("--window", type=int, default=DEFAULTS["window"])
    p.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    p.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    p.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    p.add_argument("--out", type=str, default=None,
                   help="optional path to save a JSON results summary")
    args = p.parse_args()

    res = run_experiment(
        seed=args.seed,
        steps=args.steps,
        burn_in=args.burn_in,
        window=args.window,
        alpha=args.alpha,
        beta=args.beta,
        eps=args.eps,
    )

    names = [r["name"] for r in res["regions"]]
    vc = res["visit_counts"]
    total = sum(vc)
    print(f"seed={args.seed} steps={args.steps} burn_in={args.burn_in}")
    print("visit counts:")
    for n, v in zip(names, vc):
        print(f"  {n:18s}  {v:5d}  ({100.0 * v / total:5.1f}%)")
    print("final mean err (last 200 visits per region):")
    tail_err = []
    for n, h in zip(names, res["err_hist"]):
        if h:
            tail = h[-min(200, len(h)) :]
            te = float(np.mean(tail))
        else:
            te = float("nan")
        tail_err.append(te)
        print(f"  {n:18s}  {te:.4f}  (n_visits={len(h)})")

    a, b, c = vc
    headline_ok = c > b > a
    print(f"\nheadline (C_learnable > B_random > A_deterministic): "
          f"{'PASS' if headline_ok else 'FAIL'}")

    if args.out:
        out_payload = {
            "config": res["config"],
            "env": _env_info(),
            "regions": res["regions"],
            "visit_counts": res["visit_counts"],
            "tail_mean_err": tail_err,
            "headline_ok": bool(headline_ok),
        }
        Path(args.out).write_text(json.dumps(out_payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
