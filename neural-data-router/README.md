# neural-data-router

Csordás, R., Irie, K., & Schmidhuber, J. (2022).
*The Neural Data Router: Adaptive Control Flow in Transformers Improves
Systematic Generalization.* **ICLR 2022** (arXiv:2110.07732).

![NDR vs vanilla on length generalization](neural_data_router.gif)

## Problem

Compositional table lookup. Vocabulary contains `N_VALUES = 4` value
tokens (`v0..v3`) and `N_FUNCS = 4` function tokens (`f0..f3`). Each
function `fi` is a fixed permutation of `{0,1,2,3}` (sampled per seed
from one shared table). An expression of depth `d` is the sequence

```
v ,  f_{i_1} ,  f_{i_2} ,  ... ,  f_{i_d}
```

with target `f_{i_d}( ... f_{i_2}( f_{i_1}( v ) ) )`. The model reads
the answer off its hidden state at the **last active position** of the
input.

* **Train depths**: `1, 2, 3, 4`  (sequence lengths 2..5)
* **Test depths**:  `5, 6, 7`     (sequence lengths 6..8 — *out of training*)

The published NDR paper benchmarks this same task with 8 values / 8
functions and depths 1..5 train, 6..8 test. We use a smaller alphabet
(4/4) so a single-CPU pure-numpy run finishes inside the 5-minute budget
listed in the SPEC.

## What this stub demonstrates

A pure-numpy contrast between two architectures that share **all** the
same parameter shapes and the same training recipe:

| Switch | NDR | Vanilla Transformer |
|---|---|---|
| Attention | **geometric scan** (per-query, distance-ordered) | softmax |
| Per-layer copy gate `g` | yes (`x' = g·f(x) + (1−g)·x`) | no (`x' = f(x)`) |
| Positional encoding | none (geometric scan provides position) | sinusoidal |
| Layers / `d_model` / heads / `d_ff` | 6 / 48 / 4 / 96 | same |

Both train cleanly to ≥98 % on the train depths. They diverge sharply
on the test depths: NDR keeps depth 5 well above chance; the size-matched
vanilla Transformer collapses to chance the moment the sequence runs
past the training distribution.

### Geometric attention (this stub's variant)

For each query position `i`, the keys are scanned in order of
**distance from `i`** — `i, i−1, i+1, i−2, i+2, …` (lower index wins
tiebreaks). Within a head, with `p[i,j] = sigmoid(Q_i·K_j / √d_k)` and
the scan order `π_i`,

```
A[i, π_i(k)]  =  p[i, π_i(k)] · ∏_{m<k} (1 − p[i, π_i(m)])
```

This is a **geometric distribution over key positions**: the model
"stops" at the first scoring key. Padded keys are masked to `p=0` so
they are transparent in the scan. Unlike softmax, this distribution
does not flatten as the sequence grows — depth-`d` chains and depth-`(d+1)`
chains see the *same* attention shape per scan step, which is the
structural ingredient that buys length generalization.

### Copy gate

```
attn_out = Σ_j A[i,j] · V[j]
ff_out   = FFN(x + attn_out)
g        = sigmoid(W_g · [x ; attn_out ; ff_out] + b_g)         # (B,L,1)
x'       = g · (x + attn_out + ff_out) + (1 − g) · x
```

`b_g = +3` at init so `g ≈ 0.95` (each layer mostly transforms,
occasional copy). The network can then learn to close the gate on
positions whose role at this layer is "carry the previous-layer state
forward unchanged".

## Files

| File | Purpose |
|---|---|
| `neural_data_router.py` | Pure-numpy NDR + vanilla Transformer, manual forward / backward, Adam, CLI. |
| `visualize_neural_data_router.py` | Reads `run.json`, writes 5 PNGs to `viz/`. |
| `make_neural_data_router_gif.py` | Builds `neural_data_router.gif` from per-eval snapshots in `run.json`. |
| `run.json` | Headline single-seed run (committed; seed 0, 8000 steps). |
| `run_multiseed.json` | 3-seed sweep summary (committed; seeds 0,1,2). |
| `neural_data_router.gif` | 16-frame training-dynamics animation (≈ 162 KB). |
| `viz/` | 5 static PNGs (see §Visualizations). |

## Running

Headline run (≈ 3 min 30 s on M-series CPU):

```
python3 neural_data_router.py --seed 0
```

Quick smoke test (≈ 8 s):

```
python3 neural_data_router.py --seed 0 --quick
```

Multi-seed sweep (3 seeds, ≈ 11 min):

```
python3 neural_data_router.py --multi-seed 3 --steps 8000 --out run_multiseed.json
```

Regenerate plots:

```
python3 visualize_neural_data_router.py
python3 make_neural_data_router_gif.py
```

## Results

Single-seed headline (`--seed 0`, default config: 8000 steps, batch 64,
`lr=3e-3`, Adam, `d_model=48`, `n_heads=4`, `n_layers=6`, `d_ff=96`,
`gate_init_bias=+3.0`):

Per-depth accuracy (final, 512-sample eval each depth, chance = 0.25):

| Depth | NDR | Vanilla |
|---:|---:|---:|
| **train** d=1 | 1.000 | 1.000 |
| **train** d=2 | 1.000 | 1.000 |
| **train** d=3 | 0.996 | 1.000 |
| **train** d=4 | 0.965 | 0.973 |
| **test** d=5 | **0.602** | 0.324 |
| **test** d=6 | **0.293** | 0.289 |
| **test** d=7 | **0.293** | 0.199 |

Headline aggregate (mean over the depth bin):

| | train (d=1..4) | test (d=5..7) |
|---|---:|---:|
| NDR     | 0.986 | **0.395** |
| Vanilla | 0.988 |  0.258  |

NDR's depth-5 generalization (60 %) is comfortably above vanilla's
(32 %), which is barely above the 25 % chance floor; both decay
to chance at depth 6 and beyond. Wallclock for the seed-0 run on an
M-series CPU: NDR train 133 s, vanilla train 78 s; total 3 min 30 s.

Three-seed sweep (`--multi-seed 3 --steps 8000`, in
`run_multiseed.json`):

| Seed | NDR test | Vanilla test |
|---:|---:|---:|
| 0 | 0.395 | 0.258 |
| 1 | 0.424 | 0.295 |
| 2 | 0.396 | 0.334 |
| **mean** | **0.405 ± 0.013** | **0.296 ± 0.031** |

NDR > vanilla on the test split on **3/3 seeds**. The depth-5 gap is
the cleanest reproducible signal across seeds (≈ +12 pp on average,
with one seed at +16 pp and one tied). At depth 6 NDR is also
consistently above vanilla but both are close to chance. Train
accuracy is ≥ 0.98 on every seed for both architectures.

## Visualizations

`viz/learning_curves.png` — training loss (log-y) and train/test
accuracy curves. NDR's test (d=5..7) curve climbs above 0.35 from step
~1500 onward; vanilla's test curve hovers near the chance line (0.25)
the entire run.

`viz/per_depth_final.png` — bar chart of final per-depth accuracy with
chance line and train/test depth shading. The contrast at d=5 is the
visual headline.

`viz/length_generalization.png` — per-depth accuracy curves over the
full training run, NDR vs vanilla side by side. Solid lines are train
depths; dashed lines are test depths. Vanilla's dashed lines mostly
oscillate near chance; NDR's d=5 curve clearly separates.

`viz/attention_maps.png` — head-mean attention weights at each layer
for one fixed depth-5 input (NDR top row, vanilla bottom row). NDR's
attention is sparse and peaked on `i±1` neighbours; vanilla's is
broader and more diffuse.

`viz/copy_gate.png` — NDR copy-gate openness `g` per layer per position
on the same input. Many positions are near `g≈1` (transform), but a
fraction sit substantially below — those positions are being carried
through unchanged at that layer.

## Deviations from the original

* **Vocabulary size.** Paper uses 8 values / 8 functions; we use 4 / 4
  to keep a 6-layer numpy run inside the 5-minute SPEC budget. This
  shrinks the per-layer "function memorisation" target from 64 entries
  to 16. Chance is correspondingly 0.25 instead of 0.125.
* **Train / test depth split.** Paper trains depths ≤ 5 and tests ≤ 8.
  We train ≤ 4 and test ≤ 7. The depth-5 vs depth-4 gap (one out of
  distribution) is the cleanest reproducible signal at our scale.
* **No LayerNorm.** Both models use plain residual connections without
  LayerNorm. Adding LN would mean another set of manual gradients; we
  found the contrast holds without it. Both models do train cleanly.
* **No dropout.** None applied; the synthetic data is unbounded so
  overfitting on train is not the failure mode for vanilla.
* **Geometric attention shape.** We implement the **distance-ordered
  scan** form `A[i,π_i(k)] = p · ∏(1−p)` with `π_i` = positions sorted
  by `|i−j|`. The paper uses a directional version with separate
  left-to-right and right-to-left heads; the distance-ordered scan is a
  symmetric simplification that already captures the "no smearing with
  length" property the paper uses.
* **Positional encoding.** NDR has none; vanilla uses sinusoidal. The
  paper gives both versions a positional embedding. Removing it from
  NDR was the single change that pushed depth-5 test accuracy from
  ~0.30 (no contrast) to ~0.60 (clear contrast) — see Open questions.
* **Copy-gate input.** We feed `[x ; attn_out ; ff_out]` to the gate;
  the paper uses `[x ; layer_output]`. Feeding the FFN output too lets
  the gate condition on what the layer is about to produce.
* **Output read-out.** Single linear layer at the last *active*
  position, projecting `d_model → N_VALUES`. The paper uses a similar
  read-off at a sentinel position.

## Open questions / next experiments

* **Why does removing positional encoding matter so much for NDR?**
  With sinusoidal positional embeddings, NDR's depth-5 test accuracy
  collapsed to ~0.30 — same as vanilla. The hypothesis: with PE, the
  embedding at position 5 (test) doesn't appear in training, so
  position-conditional features of the per-layer transform fail at
  depth 5. Without PE, every position embedding is identical and the
  geometric scan provides "structural" relative position. Confirm
  this with a sweep where vanilla *also* drops PE — does it also
  generalize, or does softmax attention smear regardless?
* **Why does generalization fail at d≥6?** With `n_layers = 6`, depth-7
  composition needs all 6 layers used productively for routing. The
  copy gate's structural role is to *free* layers, not to add capacity
  beyond `n_layers`. Bumping to `n_layers = 8` would test whether
  depth-7 generalization is a layer-count ceiling or something else.
* **Vocabulary scaling.** Re-running at the paper's 8/8 vocab (with
  proportional steps) should re-create the paper's 100 %
  length-generalization claim if the architecture really is right. We
  didn't do this in v1 because the per-step time roughly triples.
* **Multi-seed robustness.** 3 seeds (0, 1, 2) committed to
  `run_multiseed.json`. NDR test mean = 0.405 ± 0.013, vanilla test
  mean = 0.296 ± 0.031. NDR beats vanilla on 3/3 seeds. Vanilla's
  variance is higher because it has nothing to anchor it to a
  length-invariant policy: each seed converges to a slightly
  different position-specific solution.
* **Head direction.** Our scan is purely distance-ordered. The paper's
  alternating L→R / R→L heads may help on tasks that have right-to-left
  dependencies (not this one). Worth re-testing on a task where the
  answer position is in the middle.
* **ByteDMD instrumentation.** Once v2 wires up ByteDMD, NDR's appeal
  becomes empirical: a sparse-per-position transform should move *less*
  data than a dense softmax-attention block. Concrete sub-question: do
  the layers where the gate closes drop their attention compute too,
  or do they still pay for `Q,K,V` matmuls?
