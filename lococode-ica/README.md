# lococode-ica

Hochreiter & Schmidhuber, *Feature extraction through LOCOCODE*,
Neural Computation 11(3):679–714 (1999). Companion: Hochreiter &
Schmidhuber, *Flat minima*, Neural Computation 9(1):1–42 (1997).

![LOCOCODE-ICA training animation](lococode_ica.gif)

## Problem

LOCOCODE is the unsupervised-feature-extraction outcome of training an
autoencoder while regularising it toward "flat minima" — weight
configurations with low Kolmogorov complexity / few effective free
parameters. The headline claim is that on sparse inputs the resulting
hidden codes are sparse and statistically near-independent: an ICA-like
decomposition motivated from minimum-description-length rather than from
higher-order-statistic maximisation.

We test this on a synthetic ICA benchmark:

- `k = 8` independent **Laplacian** sources (`S ∈ R^{n × k}`,
  super-Gaussian, kurtosis = 3).
- A random orthogonal mixing matrix `A ∈ R^{k × k}`.
- Observations `X = S A^T`, `n = 2000` samples.
- Whitened input `Z = X K^T` so that `cov(Z) = I` (standard ICA / LOCOCODE
  preprocessing).

The autoencoder has tied weights `W ∈ R^{k × k}` with encoder `H = Z W^T`
and decoder `Z_hat = H W`, trained on:

```
L = ||Z - Z_hat||^2 + λ_act |H|_1 + λ_w ||W||^2
```

The L1 sparsity term is the LOCOCODE / flat-minimum-search reduction:
forcing the hidden code to be sparse pushes the network to use as few
hidden units per input as possible, which is the algorithmic definition
of "few effective parameters". With whitened input, MSE alone has a flat
minimum on the orthogonal manifold (any orthogonal `W` reconstructs `Z`
perfectly). After each gradient update we use the polar/SVD retraction
`W = U V^T` to remain on that manifold. The L1 penalty can therefore select
the rotation whose codes are sparsest without shrinking the code scale or
trading away reconstruction — which on Laplacian sources is exactly the
demixing direction.

We compare against two baselines:

- **PCA** — top-`k` eigenvectors of the covariance matrix. Uses only
  second-order statistics; cannot resolve rotations of the source
  distribution and so cannot recover ICA components.
- **FastICA** — symmetric tanh fixed-point with whitening. The canonical
  ICA algorithm we benchmark against.

## Files

| File | Purpose |
|---|---|
| `lococode_ica.py` | data generation, manifold-constrained LOCOCODE autoencoder, PCA + FastICA baselines, Amari distance, CLI. `python3 lococode_ica.py --seed N [--n-seeds K] [--k 8] [--epochs 200]`. |
| `visualize_lococode_ica.py` | trains once, saves five static PNGs in `viz/`. |
| `make_lococode_ica_gif.py` | trains once, saves `lococode_ica.gif` showing training dynamics. |
| `lococode_ica.gif` | animated training (≤ 600 KB). |
| `viz/` | training curves, Amari comparison, hidden-unit histograms, recovered demixers, source-recovery cross-correlations. |

## Running

```bash
python3 lococode_ica.py --seed 0
```

Reproduces the headline numbers in **§Results** in ~0.8 s wallclock on the
validation host (the network itself trains in ~0.52 s; the rest is NumPy
import + FastICA baseline).

To regenerate visualisations:

```bash
python3 visualize_lococode_ica.py --seed 0 --outdir viz
python3 make_lococode_ica_gif.py --seed 0 --snapshot-every 5 --fps 8
```

To run a 10-seed sweep:

```bash
python3 lococode_ica.py --seed 0 --n-seeds 10
```

To reproduce the previous unconstrained-autoencoder ablation:

```bash
python3 lococode_ica.py --seed 0 --n-seeds 10 --no-orthogonal-retraction
```

## Results

Headline (seed 0, default hyperparameters, k = 8, n = 2000, 200 epochs):

| Method | Amari ↓ | mean kurtosis | sparsity (\|h\|<0.2) |
|---|---:|---:|---:|
| **LOCOCODE** (L1 + tied AE + orthogonal retraction) | **0.014** | 3.19 | 0.249 |
| PCA (2nd-order) | 0.388 | 1.08 | 0.182 |
| FastICA (tanh fp) | 0.022 | 3.22 | 0.247 |

LOCOCODE wallclock: 0.52 s on the validation host (training only). Whitened
reconstruction MSE at convergence: `1.1e-30`; the Frobenius error of
`W W^T` from identity is `3.0e-15`.

10-seed sweep (seeds 0–9, same hyperparameters):

| Method | Amari mean | std | min | max |
|---|---:|---:|---:|---:|
| **LOCOCODE + retraction** | **0.0172** | 0.0018 | 0.0141 | 0.0198 |
| LOCOCODE without retraction (same-host ablation) | 0.115 | 0.032 | 0.079 | 0.193 |
| PCA | 0.423 | 0.034 | 0.371 | 0.478 |
| FastICA | 0.021 | 0.002 | 0.019 | 0.025 |

**Headline finding** — constraining the whitened tied autoencoder to its
natural orthogonal manifold reduces mean Amari error from 0.117 in the
published catalog result (0.115 in the same-host ablation) to 0.0172: an
85.3% reduction. LOCOCODE now slightly outperforms FastICA's 0.0208 mean,
with Laplace-like kurtosis (3.19) and no reconstruction tradeoff. A held-out
audit on seeds 10--29 gives LOCOCODE 0.0179 +/- 0.0028 versus FastICA
0.0208 +/- 0.0029, confirming the result was not specific to seeds 0--9.
Across all 30 seeds, LOCOCODE averages 0.01767 versus FastICA's 0.02081
and wins the paired comparison on 28/30 seeds.

Hyperparameters used:

```
k = 8, n_samples = 2000, epochs = 200, batch_size = 64,
lr = 0.05, lambda_act = 0.5, lambda_w = 1e-4
sources: Laplace(0, 1), standardised; mixing: random orthogonal
preprocessing: zero-mean, ZCA whitening on observations
constraint: polar/SVD retraction W = U V^T after every minibatch update
```

## Visualizations

### Training curves
![training curves](viz/training_curves.png)

Four panels over 200 epochs. **Top-left**: whitened reconstruction MSE stays
at floating-point zero because every update is retracted to an orthogonal
matrix. **Top-right**: mean `|H|` decays from 0.766 to 0.703 as the rotation
becomes sparser. **Bottom-left**: mean excess kurtosis climbs from 0.92 to
3.19, essentially the Laplace-source value. **Bottom-right**: Amari distance
falls from 0.379 to 0.014; separation improves without sacrificing
reconstruction.

### Amari + kurtosis comparison
![amari comparison](viz/amari_comparison.png)

LOCOCODE slightly leads FastICA on Amari (0.014 vs 0.022) and matches its
source kurtosis (3.19 vs 3.22); PCA remains far behind at 0.388 Amari and
1.08 kurtosis. The sparse autoencoder has moved from an approximate
ICA-like result to clean source separation.

### Hidden-unit activation histograms
![hidden distributions](viz/hidden_distributions.png)

The most-kurtotic unit per method, z-scored, with Laplace (purple
dashed) and Gaussian (grey dotted) reference curves. **LOCOCODE** unit 0
(excess `k = 4.60`) and **FastICA** unit 3 (`k = 4.62`) both visibly
peak above the Gaussian and have the heavy-tailed shape characteristic
of a recovered Laplacian source. The most-kurtotic **PCA** unit (`k =
2.19`) is closer to Gaussian — PCA finds an axis of maximum variance, not
of maximum non-Gaussianity, so even its "best" unit is closer to a
mixture than to a pure source.

### Recovered demixers
![recovered demixers](viz/recovered_demixers.png)

`|W_recovered @ A_true|` after row-normalisation and a greedy row
permutation. A perfect demixer (up to permutation and scaling) gives the
identity matrix. **LOCOCODE** and **FastICA** are both essentially
permutation matrices. **PCA** is a dense mixture in every column because
second-order statistics cannot break rotational symmetry.

### Source recovery
![source recovery](viz/source_recovery.png)

Cross-correlation `|corr(S_true, H_recovered)|` after greedy row
permutation. Same story as the demixer view but expressed through the
recovered codes themselves: LOCOCODE and FastICA both show near-unit
diagonal correlations with negligible cross-talk; PCA mixes sources across
the grid.

### GIF: training dynamics
The animation walks through the same training run frame-by-frame: top-
left shows `|W @ A|` resolving from a dense pattern at epoch 0 to a near
permutation by roughly epoch 25; top-right shows the chosen hidden unit's
distribution sharpening from Gaussian-like to heavy-tailed; the bottom
panel shows the Amari distance dropping while kurtosis rises in lock-
step.

## Deviations from the original

1. **Flat-minimum penalty is L1-on-activations, not the paper's
   activation-Hessian regulariser.** The 1997 *Flat minima* paper defines
   FMS as a penalty on the determinant of the output Jacobian's Hessian
   — second-order in the activations. We approximate this with the
   first-order surrogate `λ_act |H|_1 + λ_w ||W||^2`, which the LOCOCODE
   follow-up literature (Olshausen-Field-style sparse coding,
   sparse-autoencoder regularisers) converged on as the practically
   equivalent reduction on linear / shallow architectures. The 2015
   *Deep Learning in Neural Networks* survey (Schmidhuber, NN 61, sec.
   5.6.4) describes LOCOCODE in terms of "as few effective free
   parameters as possible" — which a hidden-code L1 penalty enforces
   directly. We document it explicitly because it's the largest
   methodological deviation.
2. **Pre-whitening of the input.** The paper's experiments on natural
   image patches did not whiten explicitly (the FMS regulariser on a
   non-trivial nonlinear architecture eats the conditioning problem
   itself). On a linear `k → k` architecture without whitening, the L1
   sparsity gradient has no scale anchor and the network collapses
   `W → 0` with a compensating `W_dec` rescaling. ZCA whitening of the
   observations restores a clean orthogonal manifold and is the same
   preprocessing FastICA uses; we apply it to both for fairness.
3. **Tied weights** (encoder = decoder transpose). The 1999 paper allows
   untied weights; with whitened input the tied case is provably
   equivalent at the optimum (any orthogonal `W` is its own inverse) and
   training is much more stable.
4. **Orthogonal-manifold retraction.** The paper does not project weights
   after each update. In this reduced square, whitened, tied-weight problem,
   however, every reconstruction optimum is orthogonal and the remaining
   problem is purely rotational. The polar step `W = U V^T` enforces that
   known constraint and prevents L1 from improving sparsity by shrinking
   scale. `--no-orthogonal-retraction` reproduces the previous 0.115 Amari
   same-host ablation.
5. **Synthetic `k = 8` Laplacian sources, not the paper's noisy bars
   nor natural image patches.** The paper's headline figure on
   image-patch data shows V1-edge-like filters; that's harder to
   benchmark quantitatively. Using synthetic sources with a known
   ground-truth mixing matrix lets us report Amari distance — the
   standard ICA evaluation metric — and a 10-seed sweep. The
   qualitative story (sparse, super-Gaussian, ICA-like) is the same as
   the paper's; the numbers are reproducible.
6. **No `numpy`-prohibited dependencies.** Pure numpy + matplotlib +
   PIL (only inside `make_lococode_ica_gif.py` to assemble the GIF,
   which the v1 SPEC explicitly allows).

## Open questions / next experiments

- **Exact flat-minimum objective.** Orthogonal retraction closes the measured
  FastICA gap for this L1 surrogate, but the paper's activation-Hessian
  regularizer remains unimplemented. Comparing that exact objective against
  L1 on the same constrained manifold would isolate the approximation.
- **Natural-image-patch experiment.** The paper's headline figure shows
  V1-style edge filters on `8 × 8` natural patches. We did not include
  this because it requires either a small natural-image dataset
  (`olshausen-field` patches) or an external image. A v1.5 follow-up:
  add a `--data patches --image-path X` mode that reads a single
  greyscale photo, extracts patches, and demonstrates the
  edge-like-filter result.
- **Noisy bars problem.** The paper also tests LOCOCODE on the noisy
  bars problem (Földiák 1990). Easy to add as a second `--data bars`
  mode in `lococode_ica.py`; visualising the recovered bars would be a
  nice complement to the histograms.
- **Higher-dim sources.** We test `k = 8`. The original paper reports
  on roughly that scale. How does LOCOCODE scale to `k = 32` or `k =
  64`? Test whether the small advantage over FastICA survives as the
  orthogonal manifold grows. PCA should remain uniformly worst.
- **v2 hook.** Tied autoencoder + L1 + whitening is an extremely cheap
  unsupervised feature extractor (~0.5 s for `k = 8, n = 2000`). The
  data-movement profile is favourable: one pass through the data per
  epoch, one `k × k` weight matrix. A clean candidate for ByteDMD
  comparison against PCA (1 cov + 1 eigh) and FastICA (whiten + 200-
  iter fixed-point) on the same problem.
- **Citation gap on the FMS regulariser.** The 1997 *Flat minima* paper
  PDF is retrievable but the exact form of the penalty involves
  notational variants that differ between paper and 2015 survey. We
  use the L1 surrogate without claiming faithful reproduction of the
  Hessian-based form. The right way to close this is to implement the
  Hessian penalty exactly on a 1-hidden-layer net and compare on the
  same synthetic benchmark.

## Sources

- Hochreiter, S., & Schmidhuber, J. (1999). *Feature extraction through
  LOCOCODE*. Neural Computation, 11(3), 679–714.
- Hochreiter, S., & Schmidhuber, J. (1997). *Flat minima*. Neural
  Computation, 9(1), 1–42.
- Schmidhuber, J. (2015). *Deep Learning in Neural Networks: An
  Overview*. Neural Networks, 61, 85–117 (sec. 5.6.4 summarises LOCOCODE
  as flat-minimum-search-based unsupervised feature extraction).
- Hyvärinen, A. (1999). *Fast and robust fixed-point algorithms for
  independent component analysis*. IEEE TNN 10(3) — for the FastICA
  baseline.
- Amari, S., Cichocki, A., & Yang, H. H. (1996). *A new learning
  algorithm for blind signal separation*. NIPS 8 — for the Amari
  distance evaluation metric.
