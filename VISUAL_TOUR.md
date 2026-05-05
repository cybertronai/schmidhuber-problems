# Visual tour

A picture-first walk through the implemented problems in this catalog.
This is the companion to [`hinton-problems`](../hinton-problems)' visual
tour, structured the same way: per-stub embedded GIF + short note on what
the visualization is meant to show.

> **Status.** This repository is at the **scaffold stage** — 58 problem
> stubs raise `NotImplementedError`, none have been implemented yet, and
> there are no GIFs or `viz/` folders to embed. This file is the
> *template* the tour will follow as stubs get filled in. The skeleton
> below mirrors the `hinton-problems/VISUAL_TOUR.md` layout.

---

## Table of contents

- [How to read this page](#how-to-read-this-page)
- [1980s — Local rules and the Neural Bucket Brigade](#1980s--local-rules-and-the-neural-bucket-brigade)
- [1990 — Controller + world-model + flip-flop](#1990--controller--world-model--flip-flop)
- [1991 — Curiosity, subgoals, the chunker](#1991--curiosity-subgoals-the-chunker)
- [1992 — Neural Computation triple](#1992--neural-computation-triple)
- [1993 — Predictable classifications, self-reference, very deep chunking](#1993--predictable-classifications-self-reference-very-deep-chunking)
- [1995–1997 — Levin search and the LSTM benchmark suite](#19951997--levin-search-and-the-lstm-benchmark-suite)
- [Mid-90s — Evolutionary, RL, and feature detection](#mid-90s--evolutionary-rl-and-feature-detection)
- [2000–2002 — LSTM follow-ups](#20002002--lstm-follow-ups)
- [2002–2010 — Evolutionary RL, OOPS, BLSTM+CTC](#20022010--evolutionary-rl-oops-blstmctc)
- [2010–2017 — Deep learning at scale](#20102017--deep-learning-at-scale)
- [2018–2025 — World models, fast-weight Transformers, systematic generalization](#20182025--world-models-fast-weight-transformers-systematic-generalization)

---

## How to read this page

A few conventions will show up across the catalog and are worth fixing
once.

**The Schmidhuber signature is the controlled difficulty sweep.** Most
visualizations in this repository, when implemented, will show a
*sequence-length × distractor-count* grid rather than a single
qualitative figure. The 1997 LSTM paper's Tables 10–11 (lag q × distractor
p) are the prototype; the 2017 *Search Space Odyssey*'s 5,400-experiment
fANOVA-analyzed grid is the apotheosis. Expect tour entries to embed
training-curves PNGs that vary one hyperparameter at a time and report
"sequences-to-success" rather than absolute accuracy.

**Long-time-lag tasks need waterfall plots, not snapshot frames.** For
the flip-flop, the 22-symbol chunker, the adding problem, the temporal-
order tasks, and the a^n b^n c^n family, a useful GIF is one that shows
*activation traces over a single test sequence* — input pulse, gate
openings, cell state. Frame-by-frame snapshots of training are less
informative than they are for representational tasks (the network state
doesn't visibly change shape; what changes is whether one cell latches
correctly).

**Reproduces? badges.** When stubs get implemented they should follow
the hinton-problems convention: `yes` = matches paper qualitatively or
quantitatively; `partial` = method works, paper-config gap documented;
`no` = paper claim does not replicate.

**Status legend used below.** Until implementations exist, each entry is
marked with one of:

- `stub` — folder exists, README + problem.py raising `NotImplementedError`.
- `partial` — code runs, results not yet matching the paper.
- `done` — implemented, GIF + viz/ committed, headline metric reported.

All entries below are currently `stub`.

---

## 1980s — Local rules and the Neural Bucket Brigade

### Schmidhuber (1989) — A local learning algorithm for dynamic feedforward and recurrent networks

#### nbb-xor — `stub`

[`nbb-xor/`](nbb-xor/) — XOR via the Neural Bucket Brigade. Strictly
local-in-space-and-time, winner-take-all dissipative learning rule.
Quotable target: **619 pattern presentations on average across 20 runs.**

When implemented, the GIF should show the WTA competition resolving on
each tick across the 6-tick presentation window — and the long-run
weight-substance flow visualized as arrows in the reverse direction of
activation.

#### nbb-moving-light — `stub`

[`nbb-moving-light/`](nbb-moving-light/) — 1-D moving-light direction
discrimination. Two competing recurrent output units encoding left→right
vs right→left. Target: **~223 cycles per sequence** in 9/10 runs.

The natural visualization is the activation race between the two output
units across the 5-tick sequence, with the weight matrix's recurrent
loops growing visibly across training.

---

## 1990 — Controller + world-model + flip-flop

### Schmidhuber (1990) — Making the world differentiable

#### flip-flop ★ — `stub`

[`flip-flop/`](flip-flop/) — *The* canonical Schmidhuber latch. Output 1
whenever 'B' first follows 'A' with arbitrary delay; only a scalar pain
signal as feedback; no episode boundaries.

When implemented this is the centerpiece visualization for the entire
1990–1997 era. The GIF should show: (a) input stream A/B/X across many
hundred ticks; (b) the controller's continuous output drifting up after
A, latching, then coming down on B; (c) the model's pain prediction
tracking actual pain. The contrast between successful and failed runs is
itself a useful figure (6/10 sequential, 20/30 parallel).

#### pole-balance-non-markov — `stub`

[`pole-balance-non-markov/`](pole-balance-non-markov/) — Cart-pole with
hidden velocities. Target: **>1000-step survival in 17/20 runs.**

A two-panel GIF — the cart-pole physics on the left, the controller's
internal recurrent activations on the right (the inferred
representation of the missing velocities) — is the right
visualization.

#### pole-balance-markov-vac — `stub`

[`pole-balance-markov-vac/`](pole-balance-markov-vac/) — Markov cart-
pole with vector-valued adaptive critic.

Useful as a baseline panel next to the non-Markov variant, showing the
critic's predictions across the state space.

#### saccadic-target-detection — `stub`

[`saccadic-target-detection/`](saccadic-target-detection/) — Controller
+ model learn to shift a fovea over a 2-D scene.

The natural GIF traces the fovea trajectory across the scene with the
target highlighted — the original "differentiable attention" visual.

---

## 1991 — Curiosity, subgoals, the chunker

### Schmidhuber (1991) — Adaptive confidence and adaptive curiosity

#### curiosity-three-regions — `stub`

[`curiosity-three-regions/`](curiosity-three-regions/) — The "no joy in
pure noise, no joy in pure knowledge" demonstration.

The expected visualization: a 2-D map of the discrete environment
colour-coded by visit frequency, alongside a per-region model-error
trace. The agent's visits should pile up on the *learnable-but-not-yet-
learned* region.

### Schmidhuber (1991) — Learning to generate sub-goals

#### subgoal-obstacle-avoidance — `stub`

[`subgoal-obstacle-avoidance/`](subgoal-obstacle-avoidance/) — 2-D
continuous obstacle avoidance with a subgoal-generator RNN.

GIF: the agent's trajectory through the obstacle field, with the
subgoals as marked way-points. The interesting thing to watch is the
subgoals moving over training to land on the *right* side of obstacles.

### Schmidhuber (1991) — RL in Markovian and non-Markovian environments

#### pomdp-flag-maze — `stub`

[`pomdp-flag-maze/`](pomdp-flag-maze/) — Small partially-observable
maze; recurrent model+controller disambiguates hidden state.

Useful viz: the agent's path through the maze with its hidden-state
trace shown alongside, illustrating where the recurrence is doing work.

### Schmidhuber (1991/92) — Neural sequence chunkers

#### chunker-22-symbol ★ — `stub`

[`chunker-22-symbol/`](chunker-22-symbol/) — The 20-step-lag prediction-
and-classification task. Conventional RTRL/BPTT fail after 1M sequences;
chunker solves 13/17 runs in <5000 sequences.

When implemented this is the second-most-important visualization in the
catalog (after flip-flop). The natural GIF is a side-by-side: BPTT's
loss flatlining vs the chunker hierarchy converging, with a small inset
showing the high-level chunker firing once per ~21-symbol input window.

---

## 1992 — Neural Computation triple

### Schmidhuber (1992) — Learning to control fast-weight memories

#### fast-weights-unknown-delay — `stub`

[`fast-weights-unknown-delay/`](fast-weights-unknown-delay/) — Pattern
association across an unknown gap.

The expected visualization is the fast-weight matrix as a Hinton diagram
animated across a single test sequence — you should see it write at the
storage step and read out at the recall step.

#### fast-weights-key-value ★ — `stub`

[`fast-weights-key-value/`](fast-weights-key-value/) — Key/value
temporary variable binding. **The 1991 unnormalized linear-Transformer
ancestor.**

The most important visualization here is the Schlag/Irie/Schmidhuber
2021 mathematical equivalence diagram — same outer-product update,
same multiplicative read-out — alongside the trained fast-weight
trajectory. Useful side-by-side with `linear-transformers-fwp` once that
is implemented.

### Schmidhuber (1992) — Learning factorial codes by predictability minimization

#### predictability-min-binary-factors — `stub`

[`predictability-min-binary-factors/`](predictability-min-binary-factors/)
— Proto-GAN on synthetic factorial binary patterns.

The natural visualization is per-unit predictability over training,
showing the encoder driving each unit's predictability down toward its
floor while marginal information is preserved. Bars-and-stripes / V1
filters belong in `semilinear-pm-image-patches` (1996), not here.

---

## 1993 — Predictable classifications, self-reference, very deep chunking

### Schmidhuber & Prelinger (1993) — Discovering predictable classifications

#### predictable-stereo — `stub`

[`predictable-stereo/`](predictable-stereo/) — Becker–Hinton binary
stereo via predictability **maximization**. The single direct point of
contact between the Schmidhuber and Hinton experimental lineages.

A useful comparison panel side-by-side with
[`hinton-problems/random-dot-stereograms/`](../hinton-problems/random-dot-stereograms/)
once both are implemented — same task, different objective.

### Schmidhuber (1993) — A self-referential weight matrix

#### self-referential-weight-matrix — `stub`

[`self-referential-weight-matrix/`](self-referential-weight-matrix/) —
RNN reads/writes its own weight matrix as activations.

The natural visualization is the weight matrix itself animated across
training — *changes to the weights driven by the network reading the
weights themselves*. Best paired with a baseline showing what the same
architecture does with the self-reference channels lesioned.

### Schmidhuber (1993) Habilitationsschrift

#### chunker-very-deep-1200 — `stub`

[`chunker-very-deep-1200/`](chunker-very-deep-1200/) — Credit assignment
over ~1200 virtual layers (Schmidhuber's "Very Deep Learning of 1993").

The natural visualization is the chunker hierarchy depth growing across
training and a per-level activity trace showing each level firing at its
characteristic input rate.

---

## 1995–1997 — Levin search and the LSTM benchmark suite

### Schmidhuber (1995/97) — Discovering low-Kolmogorov-complexity solutions

#### levin-count-inputs — `stub`

[`levin-count-inputs/`](levin-count-inputs/) — Linear unit, target =
popcount, 3 training examples, 100 inputs. Optimal: w_i = 1.

The natural visualization is the Levin search frontier — programs of
each length sorted by probability — converging on the length-4 solution
`[1, 0, 2, 0]`.

#### levin-add-positions — `stub`

[`levin-add-positions/`](levin-add-positions/) — Same setup, target =
sum of indices. Optimal: w_i = i.

Same visualization style as count-inputs; the discovered length-8
program is the headline.

### Hochreiter & Schmidhuber (1996) — RS attacks

#### rs-two-sequence — `stub`

[`rs-two-sequence/`](rs-two-sequence/) — Random weight guessing solves
the Bengio-94 latch in ~718 trials (vs Bengio's 6,400-trial multigrid).

The natural visualization is a histogram of trials-to-solution comparing
RS vs the published gradient methods.

#### rs-parity — `stub`

[`rs-parity/`](rs-parity/) — RS solves long-sequence parity in 250 trials
(vs Bengio's SA at 810,000).

Same style histogram as `rs-two-sequence`.

#### rs-tomita — `stub`

[`rs-tomita/`](rs-tomita/) — RS attacks Tomita grammars #1, #2, #4. The
generic punch line: these "long-time-lag" benchmarks are trivial — solutions are sparse but discoverable in weight space.

#### adding-problem ★ — `stub`

[`adding-problem/`](adding-problem/) — First non-trivial LSTM benchmark.
T=100/lag=50: 74k sequences; T=1000/lag=500: 853k sequences.

When implemented this is the canonical LSTM-era visualization. Three
panels: input pair stream (real values + markers), cell-state trace
across the sequence, and the (T, lag) sweep table from the 1997 paper.

### Hochreiter & Schmidhuber (1997) — Long Short-Term Memory

#### embedded-reber — `stub`

[`embedded-reber/`](embedded-reber/) — Experiment 1, short-lag baseline.
LSTM solves 148/150 trials at mean 8,440 sequences.

A Reber-FSA diagram with the embedded outer frame, plus per-symbol
prediction probabilities, is the canonical figure.

#### noise-free-long-lag — `stub`

[`noise-free-long-lag/`](noise-free-long-lag/) — Experiment 2, three
sub-variants. Sweep: (q=50,p=50) → (q=1000,p=1000) costs 30k → 49k
sequences for LSTM; BPTT/RTRL fail at p=100.

The signature visualization is the q × p grid as a heatmap, with each
cell labelled by sequences-to-success or "fails." Plot the same grid
for BPTT, RTRL, chunker, and LSTM side-by-side.

#### two-sequence-noise — `stub`

[`two-sequence-noise/`](two-sequence-noise/) — Experiment 3, three sub-
variants with target noise σ=0.32 in 3c.

Useful figure: the network's output distribution converging to the
correct *conditional expectation* in 3c, not just the discrete labels.

#### multiplication-problem — `stub`

[`multiplication-problem/`](multiplication-problem/) — Experiment 5, ×
instead of + on the adding-problem setup.

The cell-state trace is the headline: the cell must implement a
multiplicative running register, which a single LSTM block can do via
input-gate modulation.

#### temporal-order-3bit — `stub`

[`temporal-order-3bit/`](temporal-order-3bit/) — Experiment 6a, 4-class.

A nice visualization is the cell-state distribution across 2,560 test
sequences — the sign of the internal state encodes the first X/Y, and
the input gate's response on the second X/Y is conditional on whether
the cell is empty.

#### temporal-order-4bit — `stub`

[`temporal-order-4bit/`](temporal-order-4bit/) — Experiment 6b, 8-class.
The hardest 1997 benchmark.

Three cell blocks of size 2 carry the three latched bits; a per-block
activity heatmap is the right figure.

---

## Mid-90s — Evolutionary, RL, and feature detection

### Salustowicz & Schmidhuber (1997) — PIPE

#### pipe-symbolic-regression — `stub`

[`pipe-symbolic-regression/`](pipe-symbolic-regression/) — Koza's
f(x) = x⁴ + x³ + x² + x.

A useful figure is the probabilistic prototype tree itself, animated
across generations as it concentrates on the correct subtree shape.

#### pipe-6-bit-parity — `stub`

[`pipe-6-bit-parity/`](pipe-6-bit-parity/) — 6-bit even parity via PIPE.
Same visualization style.

### Schmidhuber, Zhao, Wiering (1997) — SSA

#### ssa-bias-transfer-mazes — `stub`

[`ssa-bias-transfer-mazes/`](ssa-bias-transfer-mazes/) — Sequence of POM
mazes; SSA backtracks on policy modifications not followed by reward
acceleration.

The natural visualization is wallclock-to-success per maze with and
without SSA enabled, showing the bias-transfer acceleration.

### Wiering & Schmidhuber (1997) — HQ-learning

#### hq-learning-pomdp — `stub`

[`hq-learning-pomdp/`](hq-learning-pomdp/) — Hierarchical Q(λ) with
memoryless reactive subagents.

The natural figure is the subgoal sequence the HQ-table assigns,
overlaid on the maze.

### Schmidhuber, Eldracher, Foltin (1996) — Semilinear PM

#### semilinear-pm-image-patches — `stub`

[`semilinear-pm-image-patches/`](semilinear-pm-image-patches/) — V1-style
filters from natural patches.

A 4×4 grid of learned filters showing oriented Gabor edges and centre-
surround detectors is the canonical figure (1996 paper Figure 1).

### Hochreiter & Schmidhuber (1999) — LOCOCODE

#### lococode-ica — `stub`

[`lococode-ica/`](lococode-ica/) — Flat-minimum search produces
ICA-like sparse codes.

Side-by-side filter galleries for ICA, PCA, and LOCOCODE on the same
mixture distribution is the natural visualization.

---

## 2000–2002 — LSTM follow-ups

### Gers, Schmidhuber, Cummins (2000) — Forget gate

#### continual-embedded-reber — `stub`

[`continual-embedded-reber/`](continual-embedded-reber/) — Forget gate
solves continual streams that vanilla LSTM blows up on.

The natural figure: cell-state magnitude over time for vanilla vs
forget-gate LSTM on the same continual stream.

### Gers & Schmidhuber (2001) — CFL/CSL

#### anbn-anbncn — `stub`

[`anbn-anbncn/`](anbn-anbncn/) — First RNN result on a context-sensitive
language. Trained on n ≤ 10; generalizes to n in the hundreds.

The headline visualization is the *generalization curve* — accuracy at
test n=20, 50, 100, 500, 1000 — alongside the cell-state value, which
is provably tracking n.

### Gers, Schraudolph, Schmidhuber (2002) — Precise timing

#### timing-counting-spikes — `stub`

[`timing-counting-spikes/`](timing-counting-spikes/) — MSD / GTS / PFG.
Networks cannot solve GTS without peephole connections.

A peephole-vs-no-peephole side-by-side on GTS is the natural figure.

### Eck & Schmidhuber (2002) — Blues improvisation

#### blues-improvisation — `stub`

[`blues-improvisation/`](blues-improvisation/) — 12-bar bebop blues.

Piano-roll plots of free-running compositions over 96-step choruses,
plus chord-tracking accuracy, are the natural figures. A short audio
file of one composition is even better.

---

## 2002–2010 — Evolutionary RL, OOPS, BLSTM+CTC

### Schmidhuber, Wierstra, Gomez (2005/07) — Evolino

#### evolino-sines-mackey-glass — `stub`

[`evolino-sines-mackey-glass/`](evolino-sines-mackey-glass/) — Sum of
2/3/4/5 sines + Mackey-Glass τ=17.

The natural figure is the prediction overlay (target vs predicted) for
each sine count, plus the Mackey-Glass attractor reconstruction.

### Gomez & Schmidhuber (2005) — Co-evolving recurrent neurons

#### double-pole-no-velocity — `stub`

[`double-pole-no-velocity/`](double-pole-no-velocity/) — Canonical hard
non-Markov RL benchmark.

The double-pole physics animation alongside the controller's recurrent
state is the natural GIF.

### Graves et al. (2005/06) — BLSTM and CTC

#### timit-blstm-ctc — `stub`

[`timit-blstm-ctc/`](timit-blstm-ctc/) — TIMIT phoneme recognition.

A spectrogram with frame-level phoneme posterior trace and CTC alignment
output is the canonical figure.

### Graves et al. (2009) — Unconstrained handwriting

#### iam-handwriting — `stub`

[`iam-handwriting/`](iam-handwriting/) — IAM-OnDB online + IAM-DB
offline. ICDAR 2009 winner.

A handwriting strip with the network's character-level output overlaid,
plus a confusion matrix on the 80-character output set.

### Schmidhuber (2002–2004) — OOPS

#### oops-towers-of-hanoi — `stub`

[`oops-towers-of-hanoi/`](oops-towers-of-hanoi/) — Universal solver up
to n=30.

The natural figure is the prefix-reuse log: how each new task's solution
is a wrapper around frozen prefixes from earlier tasks. Plot search
budget per disk count vs non-incremental Levin baseline.

---

## 2010–2017 — Deep learning at scale

### Cireşan et al. (2010) — Deep, big, simple nets

#### mnist-deep-mlp — `stub`

[`mnist-deep-mlp/`](mnist-deep-mlp/) — MNIST 0.35% with a plain MLP.

Standard MNIST training-curve + sample-misclassifications grid.

### Cireşan, Meier, Schmidhuber (2012) — MCDNN

#### mcdnn-image-bench — `stub`

[`mcdnn-image-bench/`](mcdnn-image-bench/) — Multi-column ensemble
across MNIST/GTSRB/CIFAR/CASIA.

The natural figure is the per-column-vs-ensemble accuracy bar chart on
each benchmark, plus the GTSRB 0.54% vs 1.16%-human comparison.

### Cireşan et al. (2012) — EM segmentation

#### em-segmentation-isbi — `stub`

[`em-segmentation-isbi/`](em-segmentation-isbi/) — Won ISBI 2012.

Natural figure: a 512×512 EM patch with the ground-truth membrane mask
and the network's pixel-wise probability map overlaid.

### Srivastava et al. (2013) — Compete to compute

#### compete-to-compute — `stub`

[`compete-to-compute/`](compete-to-compute/) — LWTA + catastrophic-
forgetting benchmark.

The headline figure is the sequential-task curve: task-1 accuracy as
task-2 trains. ReLU collapses; LWTA holds.

### Srivastava, Greff, Schmidhuber (2015) — Highway Networks

#### highway-networks — `stub`

[`highway-networks/`](highway-networks/) — 100-layer FC nets train.

Two natural figures: (a) MNIST test error vs depth for plain vs highway
nets, (b) gating-T-distribution heatmap across depth showing where the
nets actually transform.

### Greff et al. (2017) — LSTM Search Space Odyssey

#### lstm-search-space-odyssey — `stub`

[`lstm-search-space-odyssey/`](lstm-search-space-odyssey/) — 8 variants
× 3 tasks × 5,400 experiments.

The fANOVA decomposition plot from the paper — variance explained by
each hyperparameter — is the headline figure.

### Koutník et al. (2014) — Clockwork RNN

#### clockwork-rnn — `stub`

[`clockwork-rnn/`](clockwork-rnn/) — Multi-rate hidden modules.

A natural figure is per-module activity heatmap aligned with the input
waveform — slow modules carry slow envelope, fast modules carry pitch.

### Koutník et al. (2013) — TORCS evolution

#### torcs-vision-evolution — `stub`

[`torcs-vision-evolution/`](torcs-vision-evolution/) — TORCS from raw
pixels via DCT-compressed weight evolution.

A driving GIF + the DCT-coefficient distribution converging across
generations.

### Greff, van Steenkiste, Schmidhuber (2017) — Neural EM

#### neural-em-shapes — `stub`

[`neural-em-shapes/`](neural-em-shapes/) — Static / flying shapes /
flying MNIST.

A natural GIF is the per-pixel object-assignment soft-mask animated
across EM iterations on a single image.

### van Steenkiste et al. (2018) — Relational Neural EM

#### relational-nem-bouncing-balls — `stub`

[`relational-nem-bouncing-balls/`](relational-nem-bouncing-balls/) —
Bouncing balls with occlusion and extrapolation.

The natural GIF is multi-ball trajectory with occlusion curtain crossed,
showing the network maintaining identity through the occlusion.

---

## 2018–2025 — World models, fast-weight Transformers, systematic generalization

### Ha & Schmidhuber (2018) — World Models

#### world-models-carracing ★ — `stub`

[`world-models-carracing/`](world-models-carracing/) — V+M+C on
CarRacing-v0. **906 ± 21 — first reported solution.**

Multi-panel GIF: real game frame, V's reconstruction, M's predicted
next-frame, C's action. The decoded latent plus predicted-next-frame is
the most visually compelling part of the original paper.

#### world-models-vizdoom-dream ★ — `stub`

[`world-models-vizdoom-dream/`](world-models-vizdoom-dream/) — Train in
the dream, transfer zero-shot to actual VizDoom.

The headline figure is *paired* gameplay: same controller, same seed,
in DoomRNN dream and in real VizDoom side-by-side. Plus the temperature
sweep τ ∈ {0.10, 0.5, 1.0, 1.15, 1.30}.

### Schmidhuber et al. (2019) — Upside-Down RL

#### upside-down-rl — `stub`

[`upside-down-rl/`](upside-down-rl/) — RL as supervised learning
conditioned on desired return.

The natural figure is the *desired-return-vs-achieved-return* scatter at
test time on LunarLanderSparse-v2, showing the agent honouring command
inputs even where A2C/DQN fail.

### Schlag, Irie, Schmidhuber (2021) — Linear Transformers as FWPs

#### linear-transformers-fwp ★ — `stub`

[`linear-transformers-fwp/`](linear-transformers-fwp/) — Mathematical
equivalence to the 1991 fast-weight programmer.

This is the *closing-the-loop* visualization. The natural figure is a
side-by-side: 1991 FWP outer-product update, 2021 linearized self-
attention update — same equation, different decade. Pair with
`fast-weights-key-value` for the historical arc.

### Csordás, Irie, Schmidhuber (2022) — Neural Data Router

#### neural-data-router — `stub`

[`neural-data-router/`](neural-data-router/) — Copy gate + geometric
attention for systematic generalization.

The natural figure is the depth-generalization curve on compositional
table lookup — train depths ≤5, test depths up to 8 — and the
attention-pattern heatmap showing the geometric attention's locality
structure.

---

## How to add a new section as stubs get implemented

The skeleton matches `hinton-problems/VISUAL_TOUR.md`. When a stub
graduates from `stub` to `partial` or `done`:

1. Replace the placeholder bullet with an embedded GIF:
   ```markdown
   ![<slug>](<slug>/<slug>.gif)
   ```
2. Replace the "When implemented this is …" forward-looking sentence
   with what the visualization actually shows.
3. Update the status badge from `stub` to the appropriate value.
4. Link to the most informative static PNG in `<slug>/viz/`.

Per-stub folder conventions when implementing:

```
problem-folder/
├── README.md                  source paper, problem, results, deviations
├── <slug>.py                  dataset + model + train + eval
├── visualize_<slug>.py        training curves + weight viz (writes to viz/)
├── make_<slug>_gif.py         animated GIF (writes <slug>.gif)
├── <slug>.gif                 committed animation
└── viz/                       committed PNGs
```

This is the same convention as `hinton-problems`, so visualization
scripts can be ported with minimal changes.
