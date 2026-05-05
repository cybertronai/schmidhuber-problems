# schmidhuber-problems

Stubs for the synthetic learning problems and toy datasets that appear in Jürgen Schmidhuber's papers (and those of his students/collaborators) from his 1987 diploma thesis through 2025.

Companion to [`hinton-problems`](../hinton-problems): same scaffold style, different lineage. Each problem lives in its own folder containing:
- `README.md` — source paper, brief description, what it demonstrates
- `problem.py` — skeleton for dataset generation + model + training (raises `NotImplementedError`)

The catalog focuses on **synthetic toy problems** Schmidhuber (or close collaborators) designed to isolate an algorithmic capability. The signature Schmidhuber-style toy is the **long-time-lag temporal indexing task** — from the 1990 flip-flop through the 1992 chunker's 22-symbol task to the 1997 LSTM benchmark suite. Folders are flat; the catalog below is grouped by year for readability.

> **Visual tour:** [`VISUAL_TOUR.md`](VISUAL_TOUR.md) is the picture-first walk through the same catalog (skeletal until problems get implemented; structured to grow as GIFs and viz folders are added per stub).

## Catalog

### 1980s — Local rules and the Neural Bucket Brigade

**Schmidhuber (1989)** — A local learning algorithm for dynamic feedforward and recurrent networks
- [nbb-xor](nbb-xor/) — XOR via NBB, the static sanity-check
- [nbb-moving-light](nbb-moving-light/) — 1-D moving-light direction discrimination

### 1990 — Controller + world-model + flip-flop

**Schmidhuber (1990)** — Making the world differentiable (FKI-126-90)
- [flip-flop](flip-flop/) — output 1 iff B follows A with arbitrary delay (the canonical LSTM-precursor latch)
- [pole-balance-non-markov](pole-balance-non-markov/) — cart-pole with hidden velocities, perfect differentiable model

**Schmidhuber (1990)** — Recurrent networks adjusted by adaptive critics
- [pole-balance-markov-vac](pole-balance-markov-vac/) — Markov cart-pole with vector-valued adaptive critic

**Schmidhuber & Huber (1990)** — Learning to generate focus trajectories (FKI-128-90)
- [saccadic-target-detection](saccadic-target-detection/) — controller + model learn to shift a fovea over a 2-D scene

### 1991 — Curiosity, subgoals, the chunker

**Schmidhuber (1991)** — Adaptive confidence and adaptive curiosity (FKI-149-91)
- [curiosity-three-regions](curiosity-three-regions/) — deterministic / random / learnable-but-unlearned partition

**Schmidhuber (1991)** — Learning to generate sub-goals for action sequences (ICANN-91)
- [subgoal-obstacle-avoidance](subgoal-obstacle-avoidance/) — 2-D continuous obstacle avoidance

**Schmidhuber (1991)** — Reinforcement learning in Markovian and non-Markovian environments (NIPS-3)
- [pomdp-flag-maze](pomdp-flag-maze/) — recurrent model+controller disambiguates hidden state

**Schmidhuber (1991/1992)** — Neural sequence chunkers / *Learning complex extended sequences using the principle of history compression*
- [chunker-22-symbol](chunker-22-symbol/) — 22-symbol alphabet, 20-step lag, no episode boundaries

### 1992 — Neural Computation triple

**Schmidhuber (1992)** — Learning to control fast-weight memories (NC 4(1))
- [fast-weights-unknown-delay](fast-weights-unknown-delay/) — pattern association across an unknown gap
- [fast-weights-key-value](fast-weights-key-value/) — key/value temporary variable binding (the linear-Transformer ancestor)

**Schmidhuber (1992)** — Learning factorial codes by predictability minimization (NC 4(6))
- [predictability-min-binary-factors](predictability-min-binary-factors/) — proto-GAN on synthetic factorial binary patterns

### 1993 — Predictable classifications, self-reference, very deep chunking

**Schmidhuber & Prelinger (1993)** — Discovering predictable classifications (NC 5(4))
- [predictable-stereo](predictable-stereo/) — Becker–Hinton binary stereo via predictability **maximization**

**Schmidhuber (1993)** — A self-referential weight matrix (ICANN-93)
- [self-referential-weight-matrix](self-referential-weight-matrix/) — RNN reads/writes its own weight matrix

**Schmidhuber (1993)** — Habilitationsschrift, *Netzwerkarchitekturen, Zielfunktionen und Kettenregel*
- [chunker-very-deep-1200](chunker-very-deep-1200/) — credit assignment over ~1200 virtual layers

### 1995–1997 — Levin search and the LSTM benchmark suite

**Schmidhuber (1995/1997)** — Discovering solutions with low Kolmogorov complexity (ICML / NN 10)
- [levin-count-inputs](levin-count-inputs/) — 100-bit input, target = popcount, 3 training examples
- [levin-add-positions](levin-add-positions/) — 100-bit input, target = sum of indices

**Hochreiter & Schmidhuber (1996)** — LSTM can solve hard long time lag problems (NIPS 9)
- [rs-two-sequence](rs-two-sequence/) — random-weight-guessing breaks the Bengio-94 latch
- [rs-parity](rs-parity/) — random-weight-guessing on long-sequence parity
- [rs-tomita](rs-tomita/) — RS attacks Tomita grammars #1, #2, #4
- [adding-problem](adding-problem/) — first non-trivial LSTM benchmark (Experiment 4)

**Hochreiter & Schmidhuber (1997)** — Long Short-Term Memory (NC 9(8)) — the canonical 6-experiment battery
- [embedded-reber](embedded-reber/) — Experiment 1 — short-lag baseline
- [noise-free-long-lag](noise-free-long-lag/) — Experiment 2 — three sub-variants, lags up to 1000 steps
- [two-sequence-noise](two-sequence-noise/) — Experiment 3 — three sub-variants with target noise
- [multiplication-problem](multiplication-problem/) — Experiment 5 — adding-problem with × instead of +
- [temporal-order-3bit](temporal-order-3bit/) — Experiment 6a — 4-class, two embedded {X, Y}
- [temporal-order-4bit](temporal-order-4bit/) — Experiment 6b — 8-class, three embedded {X, Y}

### Mid-90s — Evolutionary, RL, and feature detection

**Salustowicz & Schmidhuber (1997)** — Probabilistic Incremental Program Evolution
- [pipe-symbolic-regression](pipe-symbolic-regression/) — Koza's f(x) = x⁴ + x³ + x² + x
- [pipe-6-bit-parity](pipe-6-bit-parity/) — 6-bit even parity via PIPE

**Schmidhuber, Zhao, Wiering (1997)** — Shifting inductive bias with SSA (ML 28)
- [ssa-bias-transfer-mazes](ssa-bias-transfer-mazes/) — POM mazes with SSA-driven task transfer

**Wiering & Schmidhuber (1997)** — HQ-learning (Adaptive Behavior 6(2))
- [hq-learning-pomdp](hq-learning-pomdp/) — hierarchical Q(λ), 28-step optimal POM

**Schmidhuber, Eldracher, Foltin (1996)** — Semilinear PM produces well-known feature detectors (NC 8(4))
- [semilinear-pm-image-patches](semilinear-pm-image-patches/) — V1-style filters from natural patches

**Hochreiter & Schmidhuber (1999)** — Feature extraction through LOCOCODE (NC 11)
- [lococode-ica](lococode-ica/) — flat-minimum search produces ICA-like sparse codes

### 2000–2002 — LSTM follow-ups

**Gers, Schmidhuber, Cummins (2000)** — Learning to forget (NC 12(10))
- [continual-embedded-reber](continual-embedded-reber/) — forget gate solves continual streams

**Gers & Schmidhuber (2001)** — Context-free and context-sensitive languages (IEEE TNN 12(6))
- [anbn-anbncn](anbn-anbncn/) — first RNN result on a CSL

**Gers, Schraudolph, Schmidhuber (2002)** — Learning precise timing (JMLR 3)
- [timing-counting-spikes](timing-counting-spikes/) — peephole connections; MSD / GTS / PFG

**Eck & Schmidhuber (2002)** — Blues improvisation with LSTM (NNSP)
- [blues-improvisation](blues-improvisation/) — 12-bar bebop blues, free-running composition

### 2002–2010 — Evolutionary RL, OOPS, BLSTM+CTC

**Schmidhuber, Wierstra, Gomez (2005/2007)** — Evolino
- [evolino-sines-mackey-glass](evolino-sines-mackey-glass/) — superimposed sines + Mackey-Glass

**Gomez & Schmidhuber (2005)** — Co-evolving recurrent neurons (GECCO)
- [double-pole-no-velocity](double-pole-no-velocity/) — canonical hard non-Markov RL benchmark

**Graves et al. (2005/2006)** — BLSTM and Connectionist Temporal Classification
- [timit-blstm-ctc](timit-blstm-ctc/) — TIMIT phoneme recognition

**Graves, Liwicki, Fernández, Bertolami, Bunke, Schmidhuber (2009)** — Unconstrained handwriting (TPAMI)
- [iam-handwriting](iam-handwriting/) — IAM-OnDB online + IAM-DB offline; ICDAR 2009 winner

**Schmidhuber (2002–2004)** — Optimal Ordered Problem Solver (ML 54)
- [oops-towers-of-hanoi](oops-towers-of-hanoi/) — universal solver up to n=30

### 2010–2017 — Deep learning at scale

**Cireşan, Meier, Gambardella, Schmidhuber (2010)** — Deep, big, simple nets (NC 22(12))
- [mnist-deep-mlp](mnist-deep-mlp/) — MNIST 0.35% with plain MLP + heavy augmentation

**Cireşan, Meier, Schmidhuber (2012)** — Multi-column deep neural networks (CVPR)
- [mcdnn-image-bench](mcdnn-image-bench/) — MNIST/GTSRB/CIFAR/CASIA Chinese — sweep-all-benchmarks era

**Cireşan, Giusti, Gambardella, Schmidhuber (2012)** — EM segmentation (NIPS)
- [em-segmentation-isbi](em-segmentation-isbi/) — won ISBI 2012; only method beating second human observer

**Srivastava, Masci, Kazerounian, Gomez, Schmidhuber (2013)** — Compete to compute (NIPS)
- [compete-to-compute](compete-to-compute/) — LWTA + catastrophic-forgetting benchmark

**Srivastava, Greff, Schmidhuber (2015)** — Training very deep networks (NIPS)
- [highway-networks](highway-networks/) — y = H(x)·T(x) + x·(1−T(x)); 100-layer FC nets train

**Greff, Srivastava, Koutník, Steunebrink, Schmidhuber (2017)** — LSTM: a search space odyssey (TNNLS)
- [lstm-search-space-odyssey](lstm-search-space-odyssey/) — TIMIT/IAM/JSB; 8 variants × 5,400 experiments

**Koutník, Greff, Gomez, Schmidhuber (2014)** — A clockwork RNN (ICML)
- [clockwork-rnn](clockwork-rnn/) — multi-rate hidden modules; audio gen, raw-audio TIMIT word

**Koutník, Cuccu, Schmidhuber, Gomez (2013)** — Vision-based RL via evolution (GECCO)
- [torcs-vision-evolution](torcs-vision-evolution/) — TORCS from raw pixels, >1M weights in DCT space

**Greff, van Steenkiste, Schmidhuber (2017)** — Neural Expectation Maximization (NIPS)
- [neural-em-shapes](neural-em-shapes/) — static shapes / flying shapes / flying MNIST

**van Steenkiste, Chang, Greff, Schmidhuber (2018)** — Relational Neural EM (ICLR)
- [relational-nem-bouncing-balls](relational-nem-bouncing-balls/) — bouncing-balls, occlusion, extrapolation

### 2018–2025 — World models, fast-weight Transformers, systematic generalization

**Ha & Schmidhuber (2018)** — Recurrent World Models Facilitate Policy Evolution (NeurIPS)
- [world-models-carracing](world-models-carracing/) — V+M+C on CarRacing-v0
- [world-models-vizdoom-dream](world-models-vizdoom-dream/) — controller trained inside DoomRNN, transferred zero-shot

**Schmidhuber et al. (2019)** — Reinforcement Learning Upside Down (arXiv)
- [upside-down-rl](upside-down-rl/) — RL as supervised learning conditioned on desired return

**Schlag, Irie, Schmidhuber (2021)** — Linear Transformers are secretly fast weight programmers (ICML)
- [linear-transformers-fwp](linear-transformers-fwp/) — equates linear self-attention with the 1991 FWP

**Csordás, Irie, Schmidhuber (2022)** — The Neural Data Router (ICLR)
- [neural-data-router](neural-data-router/) — copy gate + geometric attention for systematic generalization

## Structure

```
problem-folder/
├── README.md      one paragraph: source + property
└── problem.py     stubs: generate_dataset, build_model, train
```

The stubs raise `NotImplementedError`. Fill in the parts you need.

## Methodological caveat

Many of the early TUM technical-report PDFs (FKI-125-90, FKI-129-90, FKI-148-91, FKI-149-91, the 1993 Habilitationsschrift, Hochreiter's 1991 diploma thesis) are difficult to retrieve in original form. Stub READMEs reconstruct the experiments from corroborated secondary sources — Schmidhuber's *Deep Learning: Our Miraculous Year 1990–1991* (2020), the 1997 LSTM paper's literature review, the 2001 Hochreiter/Bengio/Frasconi/Schmidhuber chapter *Gradient Flow in Recurrent Nets*, and relevant NeurIPS/Springer abstracts — and flag claims that rest on secondary citation rather than verbatim quotation.

## Schmidhuber vs Hinton: what's different

The companion catalog [`hinton-problems`](../hinton-problems) emphasizes **representational** toy tasks: small benchmarks (4-2-4 encoder, family trees, shifter) designed to expose what kind of internal representation a network develops. Hidden-unit inspection is the experimental payoff.

Schmidhuber's lineage emphasizes **algorithmic** capability: long-time-lag indexing (flip-flop, chunker, adding, temporal-order, a^n b^n c^n), key-value binding (1992 fast-weights → 2021 linear Transformers), Kolmogorov-complexity search (Levin → OOPS), and controller+model+curiosity loops in tiny stochastic environments (1990 pole-balance → 2018 World Models). The signature methodological move is the controlled difficulty sweep — (q=50, p=50) → (q=1000, p=1000) in the 1997 LSTM paper, the 5,400-experiment grid in the 2017 *Search Space Odyssey*. See the closing thematic synthesis in `docs/lineage.md` if/when populated, or the per-stub `What it demonstrates` sections.
