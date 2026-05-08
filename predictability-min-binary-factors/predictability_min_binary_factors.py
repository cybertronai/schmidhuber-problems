"""
predictability-min-binary-factors --- Schmidhuber, *Learning factorial codes
by predictability minimization*, Neural Computation 4(6):863-879 (1992).

Two adversarial networks on synthetic factorial binary patterns:

  * an **encoder** E maps an observation x in R^D to K sigmoid code units
    y in (0,1)^K (with a decoder D for information preservation);
  * for each code unit i, a separate **predictor** P_i maps the OTHER K-1
    units to a prediction y_hat_i in (0,1).

Predictors minimize       L_P  = mean_{b,i} (y_{b,i} - y_hat_{b,i})^2
Encoder/decoder minimize  L_E  = L_recon  -  lambda * L_P

The encoder therefore *maximises* L_P (it pushes each code unit AWAY from
its own predictor's guess), while the reconstruction term keeps y informative
about x.  At the fixed point, code components are mutually unpredictable
(=> approximately statistically independent given the dataset) yet jointly
informative -- a factorial code.

This is the proto-GAN: encoder vs predictor, 1992.

Synthetic data
--------------
K independent +/-1 binary factors b ~ Uniform({-1,+1})^K, mixed by a fixed
random matrix M in R^{D x K} (with unit-norm columns) plus optional Gaussian
observation noise:

    x = M @ b  +  sigma * eps,   eps ~ N(0, I_D)

With K=4, D=8 the observable lives near a 4-D linear subspace of R^8 and
recovering b modulo permutation+sign requires both information preservation
(reconstruction) and decorrelation (PM).

CLI
---
    python3 predictability_min_binary_factors.py --seed 0

Default config solves K=4, D=8 in roughly 2000 steps in well under a minute
on an M-series laptop.  Headline metric: pairwise MI between code components
collapses toward 0 while bit-recovery accuracy modulo permutation+sign
reaches >= 99.5% on a held-out batch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from itertools import permutations

import numpy as np


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_mixing(K: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Random D x K linear mixing matrix with unit-norm columns."""
    M = rng.standard_normal((D, K))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    return M


def sample_batch(batch_size: int,
                 K: int,
                 M: np.ndarray,
                 rng: np.random.Generator,
                 noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, b) where b is +/-1 factors and x = b @ M.T  +  noise."""
    b = (rng.integers(0, 2, size=(batch_size, K)) * 2 - 1).astype(np.float64)
    x = b @ M.T
    if noise > 0.0:
        x = x + noise * rng.standard_normal(x.shape)
    return x, b


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

PRED_PARAMS = ("Wp1", "bp1", "Wp2", "bp2")
ENC_DEC_PARAMS = ("We1", "be1", "We2", "be2", "Wd1", "bd1", "Wd2", "bd2")


class PMNet:
    """Encoder + decoder + K per-unit predictors, all 1-hidden-layer MLPs.

    Encoder      : D -> Henc (tanh) -> K (sigmoid)
    Decoder      : K -> Hdec (tanh) -> D (linear)
    Predictor i  : (K-1) -> Hpred (tanh) -> 1 (sigmoid),  for i = 0..K-1
    """

    def __init__(self,
                 D: int,
                 K: int,
                 Henc: int = 32,
                 Hdec: int = 32,
                 Hpred: int = 16,
                 rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.D, self.K = D, K
        self.Henc, self.Hdec, self.Hpred = Henc, Hdec, Hpred

        self.We1 = rng.standard_normal((D, Henc)) * np.sqrt(2.0 / D)
        self.be1 = np.zeros(Henc)
        # Smaller init for the sigmoid pre-activation -> code units start near 0.5
        self.We2 = rng.standard_normal((Henc, K)) * np.sqrt(2.0 / Henc) * 0.3
        self.be2 = np.zeros(K)

        self.Wd1 = rng.standard_normal((K, Hdec)) * np.sqrt(2.0 / K)
        self.bd1 = np.zeros(Hdec)
        self.Wd2 = rng.standard_normal((Hdec, D)) * np.sqrt(2.0 / Hdec)
        self.bd2 = np.zeros(D)

        self.Wp1 = rng.standard_normal((K, K - 1, Hpred)) * np.sqrt(2.0 / (K - 1))
        self.bp1 = np.zeros((K, Hpred))
        self.Wp2 = rng.standard_normal((K, Hpred, 1)) * np.sqrt(2.0 / Hpred) * 0.3
        self.bp2 = np.zeros((K, 1))

        # idx_others[i] = list of indices [0..K-1] without i, length K-1.
        self.idx_others = np.array([
            [j for j in range(K) if j != i] for i in range(K)
        ])

    # ---- Forward ----------------------------------------------------------

    def forward(self, x: np.ndarray) -> dict:
        # Encoder
        h_enc_pre = x @ self.We1 + self.be1
        h_enc = np.tanh(h_enc_pre)
        z_enc = h_enc @ self.We2 + self.be2
        y = sigmoid(z_enc)                              # (B, K)

        # Decoder
        h_dec_pre = y @ self.Wd1 + self.bd1
        h_dec = np.tanh(h_dec_pre)
        x_hat = h_dec @ self.Wd2 + self.bd2

        # Predictors: gather y[:, idx_others[i]] -> (K, B, K-1)
        y_in = np.transpose(y[:, self.idx_others], (1, 0, 2))
        hp_pre = (np.einsum("ibj,ijh->ibh", y_in, self.Wp1)
                  + self.bp1[:, None, :])
        hp = np.tanh(hp_pre)
        zp = (np.einsum("ibh,iho->ibo", hp, self.Wp2)
              + self.bp2[:, None, :])
        yhat = sigmoid(zp)                              # (K, B, 1)

        return dict(
            x=x, y=y,
            h_enc=h_enc, h_enc_pre=h_enc_pre, z_enc=z_enc,
            x_hat=x_hat, h_dec=h_dec, h_dec_pre=h_dec_pre,
            yhat=yhat, hp=hp, hp_pre=hp_pre, zp=zp, y_in=y_in,
        )

    @staticmethod
    def losses(cache: dict) -> tuple[float, float]:
        x, x_hat, y = cache["x"], cache["x_hat"], cache["y"]
        yhat = cache["yhat"][..., 0].T                  # (B, K)
        L_recon = float(((x - x_hat) ** 2).mean())
        L_pred = float(((y - yhat) ** 2).mean())
        return L_recon, L_pred

    # ---- Predictor backward (y treated as constant) -----------------------

    def grad_predictor(self, cache: dict) -> dict:
        y = cache["y"]                                  # (B, K)
        yhat_full = cache["yhat"]                       # (K, B, 1)
        yhat = yhat_full[..., 0].T                      # (B, K)
        B, K = y.shape

        # dL_P / dyhat_{i,b}  =  (yhat - y) * 2 / (B*K)   (treat y constant)
        d_yhat = (yhat - y) * (2.0 / (B * K))           # (B, K)
        d_zp = d_yhat.T[..., None] * yhat_full * (1.0 - yhat_full)   # (K, B, 1)

        dWp2 = np.einsum("ibh,ibo->iho", cache["hp"], d_zp)
        dbp2 = d_zp.sum(axis=1)
        d_hp = np.einsum("ibo,iho->ibh", d_zp, self.Wp2)
        d_hp_pre = d_hp * (1.0 - cache["hp"] ** 2)
        dWp1 = np.einsum("ibj,ibh->ijh", cache["y_in"], d_hp_pre)
        dbp1 = d_hp_pre.sum(axis=1)

        return dict(Wp1=dWp1, bp1=dbp1, Wp2=dWp2, bp2=dbp2)

    # ---- Encoder + decoder backward (predictor params treated as constant) -

    def grad_encoder_decoder(self, cache: dict, lam: float) -> dict:
        x, y, x_hat = cache["x"], cache["y"], cache["x_hat"]
        yhat_full = cache["yhat"]
        yhat = yhat_full[..., 0].T                      # (B, K)
        B, K = y.shape
        D = x.shape[1]

        # ---- Decoder backward, from L_recon = mean((x - x_hat)^2) ----
        d_xh = (x_hat - x) * (2.0 / (B * D))
        dWd2 = cache["h_dec"].T @ d_xh
        dbd2 = d_xh.sum(axis=0)
        d_h_dec = d_xh @ self.Wd2.T
        d_h_dec_pre = d_h_dec * (1.0 - cache["h_dec"] ** 2)
        dWd1 = y.T @ d_h_dec_pre
        dbd1 = d_h_dec_pre.sum(axis=0)
        dy_recon = d_h_dec_pre @ self.Wd1.T             # (B, K)

        # ---- L_pred contribution to dy ----
        # Direct  : dL_P / dy_{b,i}  =  2*(y - yhat)/(B*K)
        dy_direct = (y - yhat) * (2.0 / (B * K))
        # Indirect: chain through predictor i (yhat_i depends on y_{j != i}).
        # We reuse the same predictor backward chain but stop at the input y_in.
        d_yhat = (yhat - y) * (2.0 / (B * K))
        d_zp = d_yhat.T[..., None] * yhat_full * (1.0 - yhat_full)
        d_hp = np.einsum("ibo,iho->ibh", d_zp, self.Wp2)
        d_hp_pre = d_hp * (1.0 - cache["hp"] ** 2)
        d_y_in = np.einsum("ibh,ijh->ibj", d_hp_pre, self.Wp1)   # (K, B, K-1)
        dy_indirect = np.zeros_like(y)
        for i in range(K):
            for m, j in enumerate(self.idx_others[i]):
                dy_indirect[:, j] += d_y_in[i, :, m]

        dy_pred = dy_direct + dy_indirect               # full dL_P / dy
        dy = dy_recon - lam * dy_pred                   # L_E = L_recon - lam*L_P

        # ---- Encoder backward ----
        dz = dy * y * (1.0 - y)                         # sigmoid derivative
        dWe2 = cache["h_enc"].T @ dz
        dbe2 = dz.sum(axis=0)
        d_h_enc = dz @ self.We2.T
        d_h_enc_pre = d_h_enc * (1.0 - cache["h_enc"] ** 2)
        dWe1 = x.T @ d_h_enc_pre
        dbe1 = d_h_enc_pre.sum(axis=0)

        return dict(
            We1=dWe1, be1=dbe1, We2=dWe2, be2=dbe2,
            Wd1=dWd1, bd1=dbd1, Wd2=dWd2, bd2=dbd2,
        )


# ----------------------------------------------------------------------
# Adam (separate state per optimizer)
# ----------------------------------------------------------------------

def init_adam(net: PMNet, names: tuple[str, ...]) -> dict:
    state = {}
    for k in names:
        v = getattr(net, k)
        state[k] = dict(m=np.zeros_like(v), v=np.zeros_like(v), t=0)
    return state


def adam_step(net: PMNet,
              grads: dict,
              state: dict,
              lr: float,
              beta1: float = 0.9,
              beta2: float = 0.999,
              eps: float = 1e-8) -> None:
    for k, g in grads.items():
        s = state[k]
        s["t"] += 1
        s["m"] = beta1 * s["m"] + (1.0 - beta1) * g
        s["v"] = beta2 * s["v"] + (1.0 - beta2) * (g * g)
        m_hat = s["m"] / (1.0 - beta1 ** s["t"])
        v_hat = s["v"] / (1.0 - beta2 ** s["t"])
        getattr(net, k)[...] = (
            getattr(net, k) - lr * m_hat / (np.sqrt(v_hat) + eps)
        )


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def mi_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical mutual information (nats) between two binary 1-D arrays."""
    p = np.zeros((2, 2))
    for av in (0, 1):
        for bv in (0, 1):
            p[av, bv] = ((a == av) & (b == bv)).mean()
    p_a = p.sum(axis=1)
    p_b = p.sum(axis=0)
    mi = 0.0
    for av in (0, 1):
        for bv in (0, 1):
            if p[av, bv] > 0 and p_a[av] > 0 and p_b[bv] > 0:
                mi += p[av, bv] * np.log(p[av, bv] / (p_a[av] * p_b[bv]))
    return float(mi)


def evaluate(net: PMNet,
             M: np.ndarray,
             rng: np.random.Generator,
             n: int = 4096,
             threshold: float = 0.5,
             noise: float = 0.0) -> dict:
    """Compute all reportable metrics on a fresh batch."""
    x, b = sample_batch(n, net.K, M, rng, noise=noise)
    cache = net.forward(x)
    L_recon, L_pred = net.losses(cache)

    y = cache["y"]
    y_bin = (y > threshold).astype(np.int64)
    b_bin = (b > 0).astype(np.int64)

    K = net.K
    pmi = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            v = mi_binary(y_bin[:, i], y_bin[:, j])
            pmi[i, j] = v
            pmi[j, i] = v
    upper = np.triu_indices(K, k=1)
    mean_pair_mi = float(pmi[upper].mean()) if upper[0].size > 0 else 0.0

    mi_yb = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            mi_yb[i, j] = mi_binary(y_bin[:, i], b_bin[:, j])

    best_score, best_perm = -np.inf, tuple(range(K))
    for perm in permutations(range(K)):
        score = sum(mi_yb[i, perm[i]] for i in range(K))
        if score > best_score:
            best_score = score
            best_perm = perm

    bit_correct = 0
    bit_total = 0
    signs = []
    for i in range(K):
        j = best_perm[i]
        match_pos = float((y_bin[:, i] == b_bin[:, j]).mean())
        if match_pos >= 0.5:
            correct = int((y_bin[:, i] == b_bin[:, j]).sum())
            signs.append(+1)
        else:
            correct = int((y_bin[:, i] != b_bin[:, j]).sum())
            signs.append(-1)
        bit_correct += correct
        bit_total += n
    bit_acc = bit_correct / bit_total

    code_var = float(y.var(axis=0).mean())
    code_mean = float(y.mean(axis=0).mean())

    return dict(
        L_recon=L_recon, L_pred=L_pred,
        pairwise_mi=mean_pair_mi, pairwise_mi_matrix=pmi,
        bit_acc=bit_acc, mi_yb=mi_yb,
        best_perm=best_perm, signs=signs,
        code_var=code_var, code_mean=code_mean,
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(D: int = 8,
          K: int = 4,
          Henc: int = 32,
          Hdec: int = 32,
          Hpred: int = 16,
          n_steps: int = 2500,
          batch_size: int = 128,
          lr_pred: float = 0.01,
          lr_ed: float = 0.005,
          n_pred_steps: int = 3,
          lam: float = 1.0,
          lam_warmup: int = 400,
          noise: float = 0.05,
          seed: int = 0,
          log_every: int = 50,
          eval_n: int = 2048,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True) -> tuple[PMNet, dict, np.ndarray]:
    """Run the full alternating PM training loop and return (net, history, M)."""
    seed_seq = np.random.SeedSequence(seed)
    data_seed, init_seed, batch_seed, eval_seed = seed_seq.spawn(4)
    data_rng = np.random.default_rng(data_seed)
    net_rng = np.random.default_rng(init_seed)
    batch_rng = np.random.default_rng(batch_seed)
    eval_rng = np.random.default_rng(eval_seed)

    M = make_mixing(K, D, data_rng)
    net = PMNet(D, K, Henc=Henc, Hdec=Hdec, Hpred=Hpred, rng=net_rng)

    pred_state = init_adam(net, PRED_PARAMS)
    ed_state = init_adam(net, ENC_DEC_PARAMS)

    history = dict(step=[], L_recon=[], L_pred=[],
                   pairwise_mi=[], bit_acc=[], lam=[],
                   code_var=[], code_mean=[])

    if verbose:
        print(f"# predictability minimisation  D={D} K={K}  "
              f"steps={n_steps}  batch={batch_size}  lam_max={lam}")

    for step in range(n_steps):
        cur_lam = lam * min(1.0, step / max(lam_warmup, 1))

        # Predictor steps -- chase the encoder's current code distribution.
        for _ in range(n_pred_steps):
            x, _ = sample_batch(batch_size, K, M, batch_rng, noise=noise)
            cache = net.forward(x)
            grads_p = net.grad_predictor(cache)
            adam_step(net, grads_p, pred_state, lr_pred)

        # Encoder + decoder step -- minimise L_recon - cur_lam * L_pred.
        x, _ = sample_batch(batch_size, K, M, batch_rng, noise=noise)
        cache = net.forward(x)
        grads_ed = net.grad_encoder_decoder(cache, cur_lam)
        adam_step(net, grads_ed, ed_state, lr_ed)

        if step % log_every == 0 or step == n_steps - 1:
            ev_rng = np.random.default_rng(int(eval_rng.integers(0, 2**31 - 1)))
            metrics = evaluate(net, M, ev_rng, n=eval_n, noise=noise)
            history["step"].append(step)
            history["L_recon"].append(metrics["L_recon"])
            history["L_pred"].append(metrics["L_pred"])
            history["pairwise_mi"].append(metrics["pairwise_mi"])
            history["bit_acc"].append(metrics["bit_acc"])
            history["lam"].append(cur_lam)
            history["code_var"].append(metrics["code_var"])
            history["code_mean"].append(metrics["code_mean"])
            if verbose:
                print(f"step {step:5d}  "
                      f"L_rec={metrics['L_recon']:.4f}  "
                      f"L_pred={metrics['L_pred']:.4f}  "
                      f"pMI={metrics['pairwise_mi']:.4f}  "
                      f"bit_acc={metrics['bit_acc']*100:5.1f}%  "
                      f"lam={cur_lam:.2f}")

        if snapshot_callback is not None and (
                step % snapshot_every == 0 or step == n_steps - 1):
            snapshot_callback(step, net, history, M)

    return net, history, M


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--henc", type=int, default=32)
    p.add_argument("--hdec", type=int, default=32)
    p.add_argument("--hpred", type=int, default=16)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr-pred", type=float, default=0.01)
    p.add_argument("--lr-ed", type=float, default=0.005)
    p.add_argument("--pred-steps", type=int, default=3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lam-warmup", type=int, default=400)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=2048)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    net, history, M = train(
        D=args.D, K=args.K,
        Henc=args.henc, Hdec=args.hdec, Hpred=args.hpred,
        n_steps=args.steps, batch_size=args.batch,
        lr_pred=args.lr_pred, lr_ed=args.lr_ed,
        n_pred_steps=args.pred_steps,
        lam=args.lam, lam_warmup=args.lam_warmup,
        noise=args.noise, seed=args.seed,
        log_every=args.log_every, eval_n=args.eval_n,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    # Final evaluation on a deterministic held-out RNG.
    final = evaluate(
        net, M,
        np.random.default_rng(args.seed + 12345),
        n=4096, noise=args.noise,
    )
    print(f"\nFinal:  L_recon={final['L_recon']:.4f}  "
          f"L_pred={final['L_pred']:.4f}  "
          f"pairwise_MI={final['pairwise_mi']:.4f}  "
          f"bit_acc={final['bit_acc']*100:.2f}%")
    print(f"Best perm (y_i -> b_j): {final['best_perm']}   "
          f"Signs: {final['signs']}")
    print(f"Wall time: {elapsed:.1f}s")

    results = dict(
        config=vars(args),
        final=dict(
            L_recon=final["L_recon"], L_pred=final["L_pred"],
            pairwise_mi=final["pairwise_mi"],
            bit_acc=final["bit_acc"],
            best_perm=list(final["best_perm"]),
            signs=list(final["signs"]),
        ),
        wallclock_s=elapsed,
        env=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            platform=platform.platform(),
            processor=platform.processor() or "unknown",
            git=git_hash(),
        ),
    )
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
