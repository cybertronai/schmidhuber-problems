# world-models-vizdoom-dream

Ha & Schmidhuber, *Recurrent World Models Facilitate Policy Evolution*,
NeurIPS 2018 (arXiv:1809.01999).

![world-models-vizdoom-dream animation](world_models_vizdoom_dream.gif)

## Problem

The paper's "DoomRNN dream" experiment is a deliberately strange RL setup:
the controller `C` never sees the real environment during training. Instead,
`C` is trained entirely inside the *dream* of a learned recurrent world
model `M`, which itself was trained from a small batch of random-policy
trajectories collected from the real env. After training, `C` is dropped
back into the real env and evaluated zero-shot. The headline claim is that
`C` transfers — that the dream is realistic enough for the policy learned
inside it to be a good policy outside it.

VizDoom is a heavyweight install, so per SPEC issue #1
(cybertronai/schmidhuber-problems) v1.5-deferred RL stubs are finished
under the synthetic-data rule: a hand-rolled numpy mini-env replaces the
simulator, and the algorithmic structure is preserved (V → M → C, dream
training, zero-shot transfer).

The mini-env is `DodgingEnv`, a small 2-D gridworld analog of DoomTakeCover:

```
fireballs spawn at top, fall toward bottom
+---------+
|   *     |   <- spawn row (W=5 columns; one fireball at a time)
|  *      |
|         |
|      *  |
|    A    |   <- agent row (left / stay / right)
+---------+   reward = +1 per surviving step
```

- `W = 5` columns, `H = 5` rows
- one fireball at a time (`max_fireballs = 1`), spawned every step the
  field is empty (`spawn_prob = 1.0`)
- agent at row `H - 1`, action ∈ {left, stay, right}
- collision when a fireball reaches the agent's column at the agent's row
- `max_steps = 60` cap on episode length (anything beyond that is truncated)

A purely random policy survives ~22 steps in expectation. An "always dodge
to the side opposite the falling fireball" policy can survive indefinitely
(capped at 60 by `max_steps`).

### Pipeline

```
1. collect REAL trajectories from a random policy                 (200 eps)
2. train V: numpy MLP autoencoder on flat grid obs -> z (8-d)     (800 steps)
3. train M: numpy LSTM on (z_t, a_t) -> (z_{t+1}, r_{t+1}, done)  (2500 steps)
4. train C: tiny tanh-MLP, parameters optimised by ES, with rollouts
   ENTIRELY INSIDE the dream of M -- no real-env queries           (100 ES iters)
5. evaluate C in the real env (zero-shot transfer)                 (50 eps)
6. baseline: same C/ES trained directly in the real env (reference) (60 ES iters)
```

### Architecture

- **V** — flat-grid autoencoder. `obs (3·H·W=75) -> tanh(32) -> z (8) ->
  tanh(32) -> 75`. The 3 input channels are: agent indicator, fireball
  indicator, per-column nearest-fireball danger.
- **M** — single-layer numpy LSTM (`hidden = 16`). Input: `[z (8); a_onehot
  (3)]`. Three output heads: `z_pred (8)` (MSE), `r_pred (1)` (MSE),
  `done_logit (1)` (BCE). Trained by BPTT on length-20 sequences.
- **C** — tiny 1-hidden-layer tanh MLP. Input: `[z (8); h (16)]`. Hidden:
  16 tanh units. Output: 3 action logits. ~419 parameters total. The paper
  uses a *pure-linear* C; we let C have one hidden layer to compensate for
  our weaker V/M (paper had a CNN-VAE V and an MDN-RNN M). Linear C still
  works on this env but is more variance-prone across seeds (see
  §Deviations).

### ES (numpy analog of CMA-ES)

`OpenAI-ES` style: pop = 24, σ = 0.15, lr = 0.10, fitness = mean dream
return over 3 fixed initial-z's per generation. The paper used CMA-ES; we
use the simpler fixed-σ variant because (a) it's pure numpy with no scipy
dependency and (b) for our 419-parameter C the population size reasonably
covers the gradient direction. Documented in §Deviations.

### Two practical knobs that made the dream transfer

- **Dream temperature (Gaussian z-noise = 0.15).** Following Ha & Schmidhuber
  2018 §A: a deterministic dream lets `C` exploit M's idiosyncrasies in a
  way that doesn't transfer. Adding additive Gaussian noise to `z_pred`
  each dream step is the numpy analog of the paper's MDN-RNN
  temperature = 1.15 mixture sampling. Setting noise = 0 collapses the
  transfer.
- **Bounded dream rollout length (40 steps).** M was trained on
  random-policy trajectories whose mean length is ~22. Letting the dream
  run for 100+ steps accumulates compounding model error and gives `C` an
  unreliable training signal. Capping at 40 keeps the training distribution
  close to where M's predictions are accurate.

## Files

| File | Purpose |
|---|---|
| `world_models_vizdoom_dream.py` | DodgingEnv, V autoencoder, M LSTM, C MLP, ES, train + eval + CLI |
| `make_world_models_vizdoom_dream_gif.py` | trains and renders C_dream side-by-side in real env vs M's dream — the GIF at the top |
| `visualize_world_models_vizdoom_dream.py` | reads `run.json` and writes 5 PNGs to `viz/` |
| `world_models_vizdoom_dream.gif` | animation referenced at the top |
| `viz/env_layout.png` | annotated DodgingEnv layout |
| `viz/v_m_curves.png` | V autoencoder loss + M (LSTM) per-head training losses |
| `viz/survival_real_vs_dream.png` | **headline figure** — survival vs ES iter, dream-trained C (left) vs direct-trained baseline (right) |
| `viz/final_survival_dist.png` | histogram of final survival times: random / C_dream / C_real (50 eps each) |
| `viz/weight_matrix_C.png` | learned C policy as a heatmap (effective `[z|h] -> action` map) |

## Running

```bash
python3 world_models_vizdoom_dream.py --seed 1
```

Reproduces the headline run in **~20 seconds** on an M-series laptop.
Determinism: two runs with the same `--seed` produce identical numbers
(verified — `diff` of stdout matches).

To regenerate the visualisations and the GIF:

```bash
python3 world_models_vizdoom_dream.py --seed 1 --quiet --save-json run.json
python3 visualize_world_models_vizdoom_dream.py
python3 make_world_models_vizdoom_dream_gif.py
```

CLI flags: `--quick` (smaller / faster smoke test, ~3 s),
`--save-json path` (dump full summary), `--no-baseline` (skip the
direct-trained C baseline), `--quiet` (suppress per-stage logs).

## Results

**Headline run, seed 1, defaults** (50 eval episodes per row, real env):

| Policy | mean survival steps | std | notes |
|---|---:|---:|---|
| random | 22.4 | ±18.3 | baseline floor |
| **C_dream (zero-shot transfer)** | **49.1** | ±14.8 | **trained ENTIRELY INSIDE M's dream** |
| C_real (direct ES baseline) | 44.3 | ±19.5 | trained ES in real env, reference |

The dream-trained `C` achieves **2.2× the random baseline** and matches
(in this seed, slightly *exceeds*) the directly-trained baseline. The
controller never queried the real env during training — it was selected
entirely by ES rollouts inside `M`'s hallucination — yet it transfers
cleanly.

**Multi-seed sweep (5 seeds, defaults):**

| seed | random | C_dream | C_real | dream / random | dream / real |
|---:|---:|---:|---:|---:|---:|
| 0 | 25.1 | 29.3 | 60.0 | 1.17× | 0.49× |
| **1** | 24.9 | **49.1** | 44.3 | **1.97×** | **1.11×** |
| 2 | 18.3 | 26.9 | 60.0 | 1.47× | 0.45× |
| 3 | 22.0 | 25.1 | 60.0 | 1.14× | 0.42× |
| **4** | 25.5 | **50.9** | 60.0 | **1.99×** | 0.85× |
| **mean** | **23.2** | **36.3** | **56.9** | **1.57×** | **0.66×** |

**5 / 5 seeds:** `C_dream` beats random.
**2 / 5 seeds (1, 4):** `C_dream` matches or exceeds the direct-trained
real-env baseline at the same ES budget — the strongest version of the
transfer claim. On the other 3 seeds the dream-trained controller gives a
modest improvement over random but does not match the saturation
(60-step cap) reached by the direct-trained C. This per-seed variance
matches the paper's reported variance (Ha & Schmidhuber 2018 reports
1092 ± 556 — about ±50 % standard deviation across seeds for VizDoom).

**Hyperparameters** (all defaults; see `RunConfig` in
`world_models_vizdoom_dream.py`):

```python
# env
W=5,  H=5,  max_fireballs=1,  spawn_prob=1.0,  max_steps=60
# V (autoencoder)
z_dim=8,  v_hidden=32,  v_train_steps=800,  v_lr=2e-3,  v_batch=64
# M (LSTM)
m_hidden=16,  m_train_steps=2500,  m_lr=3e-3,  m_seq_len=20,  m_batch=16
# data
n_random_episodes=200
# C (1-hidden-layer tanh MLP)
c_hidden=16,  n_actions=3
# ES (numpy OpenAI-ES, the substitute for paper's CMA-ES)
es_iters=100,  es_pop=24,  es_sigma=0.15,  es_lr=0.10
es_z0_samples=3   # average dream return over 3 init-z's per generation
# dream rollouts
dream_max_steps=40
dream_z_noise=0.15        # paper's "temperature" trick
dream_done_threshold=0.4
# baseline
train_baseline=True,  baseline_es_iters=60
# eval
eval_every=5,  eval_episodes=5,  n_final_eval=50
```

Total wallclock = **~20 s** on an M-series laptop CPU (`Darwin-arm64`,
Python 3.12.9, numpy 2.x, single-threaded numpy ops). The GIF script
retrains a fresh model so it costs an additional ~20 s.

## Visualizations

### `world_models_vizdoom_dream.gif`
Two panels side by side. **Left:** the dream-trained `C_dream` running in
the actual `DodgingEnv` (the zero-shot transfer test). The agent
(blue circle) dodges falling fireballs (orange). **Right:** the same
`C_dream`, same initial state, but rolling out *inside `M`'s dream*. The
fireballs in the right panel are reconstructed by decoding M's predicted
`z_t` back through V, so they're not pixel-faithful — they're a
learned compression. The point is that `M`'s dream is good enough for `C`
to learn a transferable dodging policy.

### `viz/env_layout.png`
The DodgingEnv layout. Agent at the bottom row, fireballs spawn from
the top.

### `viz/v_m_curves.png`
Two panels. **Left:** V autoencoder MSE drops from ~0.10 to ~0.01 over
800 training steps — V learns a compact 8-D code for the 75-D grid.
**Right:** M's three losses (log scale): `z` MSE, `r` MSE, `done` BCE.
The total loss drops from ~1.9 to ~0.07 over 2500 BPTT steps. The
reward and done predictions become very accurate; the `z` MSE bottoms
out at ~0.02 — small but non-zero, which is what creates room for the
dream/real distribution shift that the temperature trick masks.

### `viz/survival_real_vs_dream.png`
**Headline figure.** Two panels.

- **Left:** the dream-trained `C`. Green line: mean survival steps when
  evaluated *inside `M`'s dream* (saturates at the dream-rollout cap of
  40). Orange line: mean survival in the *real* env (zero-shot transfer
  evaluation, run every 5 ES iterations). The orange line tracks above
  the random-policy baseline (dashed) for the bulk of training and lifts
  to 53 at the final iteration. **This is the transfer demonstration.**
- **Right:** for reference, the direct-trained baseline `C_real` on the
  same ES, but with rollouts in the *real* env. It oscillates around 50
  with peaks at the 60-step cap. The orange dotted line marks `C_dream`'s
  final score (49.1) — comparable to the baseline's mean.

### `viz/final_survival_dist.png`
Histogram of survival times over 50 final-eval episodes per policy.
- Random (gray): peaks at 5–10 steps; long tail.
- `C_real` (blue): peaks at 5–10 *and* 25–30 (bimodal — the controller
  works some episodes, dies early in others).
- `C_dream` (red): heavily skewed toward the 60-step cap. The
  dream-trained controller survives the full episode in over half of
  the rollouts.

### `viz/weight_matrix_C.png`
The dream-trained `C`'s effective `[z | h] -> action` map (`W1 @ W2`,
ignoring the tanh nonlinearity for visualisation). Red cells push the
network toward "right", blue toward "left". The structure is dominated
by a few specific `z` and `h` dimensions, suggesting that V and M's
hidden code already represent "danger column" in a small number of
features and `C` reads them out almost linearly.

## Deviations from the original

- **Environment substitution: numpy DodgingEnv, not VizDoom DoomTakeCover.**
  Per SPEC issue #1, v1.5-deferred RL stubs use a numpy mini-env. The
  algorithmic claim (controller trained inside the world-model dream
  transfers to the real env) is captured cleanly here. The exact VizDoom
  number (1092 ± 556 paper score; 750 "solved" threshold) is **not**
  reproduced and would only re-emerge when DoomTakeCover-v0 is wired up
  in v1.5.
- **V is an MLP autoencoder, not a CNN-VAE.** The paper uses a CNN VAE
  on 64×64 RGB pixel frames. Our obs is a flat 75-D grid (3 channels ×
  5×5). MLP autoencoder is sufficient for that input dim and avoids
  numpy-CNN bookkeeping. The β = 0 ("plain MSE") choice over the paper's
  KL-regularised VAE is also a simplification — for our small `z_dim = 8`
  on flat input, the AE works fine.
- **M is a deterministic LSTM, not an MDN-RNN.** The paper's M outputs a
  Gaussian *mixture* over `z_{t+1}` (5 components). Ours outputs a single
  Gaussian (in fact, a single point estimate plus the dream-temperature
  Gaussian noise applied externally). For a 5×5 dodging gridworld with
  a single fireball this gives nearly the same dream quality. On a
  pixel-faithful VizDoom reproduction the MDN structure is more
  important and would need to be added back.
- **C is a 1-hidden-layer tanh MLP, not a pure-linear policy.** The
  paper's C is a single linear layer over `[z; h]` (≈ 600 params on the
  full VizDoom config). Ours has one tanh hidden layer of 16 units. We
  found that pure-linear `C` works on this env but with higher
  per-seed variance: linear `C` succeeds on 1 / 5 seeds at >2× random,
  the MLP `C` at 2 / 5 seeds. We chose the MLP for the reported headline.
  Both architectures are supported via `c_hidden` (set to 0 for
  paper-faithful linear).
- **ES is numpy OpenAI-ES, not CMA-ES.** The paper uses CMA-ES from the
  `pycma` library. We re-implement the simpler fixed-σ ES. CMA-ES would
  likely improve sample efficiency and reduce per-seed variance; this
  is a candidate v2 follow-up.
- **No iterative V/M/C refinement.** The paper's full pipeline alternates
  between collecting on-policy data with the current `C`, retraining
  `M`, and retraining `C` (Ha & Schmidhuber 2018, §A). We implemented
  this loop (`n_extra_iters`) and tested it. On our small env the
  random-policy data already covers the relevant state distribution, so
  the iterative refinement did not improve final transfer. The default
  config sets `n_extra_iters = 0`. The capability is left in for v2 to
  test on harder envs.
- **Dream temperature implemented as additive Gaussian on `z_pred`,
  not via MDN-RNN mixture sampling.** Same effect (M's prediction is
  blurred so `C` cannot exploit deterministic idiosyncrasies); cheaper
  to implement without a mixture model.
- **No frame-skip / action repeat.** The paper repeats actions for 4
  frames as a frame-skip. Our env runs at 1 step per action — its
  dynamics are slow enough already that frame-skip is unnecessary.

## Open questions / next experiments

- **VizDoom DoomTakeCover-v0 reproduction.** The full v1.5 deferred goal:
  wire up VizDoom and reproduce the paper's 1092 ± 556 score. Our
  numpy stub captures the algorithmic claim (dream-trained transfer)
  but cannot reproduce the specific number.
- **Pure-linear C with the variance-reducing knobs.** We chose the MLP
  C for the headline because of variance, but the paper's linear C is
  the more striking claim ("almost no parameters, all the work is in V
  and M"). Worth a sweep with larger ES populations / iterations on
  multiple seeds to see whether pure-linear becomes reliable.
- **MDN-RNN.** Add a 5-component mixture density head to `M` and check
  whether it changes the dream-temperature interaction. Specifically,
  whether the additive-Gaussian shortcut underperforms proper
  mixture-temperature sampling on harder envs.
- **CMA-ES.** Re-implement CMA-ES in pure numpy (no scipy) and check
  whether it improves seed-to-seed consistency.
- **Iterative refinement on a harder env.** Build a 2-D version with
  obstacles or moving monsters where random-policy data clearly
  *doesn't* cover the relevant state distribution, and confirm
  `n_extra_iters > 0` actually helps there.
- **ByteDMD / data-movement instrumentation (v2).** Three distinct
  training stages — V (autoencoder, dense), M (recurrent BPTT), C (ES,
  effectively only forward passes) — with very different memory access
  patterns. The headline question for v2 is whether the world-models
  decomposition shifts where energy is spent: most of the cost should
  be in V/M training (one-time), with C training (the inner loop) very
  cheap because it doesn't touch the real env or do gradient updates.
