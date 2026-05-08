# RESULTS — v1 + v1.5 baselines

Per-stub reproducibility, run wallclock, and headline result for the 58 implementations shipped across wave PRs. Compiled from PR bodies and per-stub READMEs for the v2 data-movement / ByteDMD filter.

**Reproduces? legend**: `yes` = matches paper qualitatively or quantitatively; `partial` / `qualitative` = method works, paper number not fully reached (gap documented in stub README); `no` = paper claim does not replicate (gap analysis documented).

**Run wallclock**: time to run the final headline experiment on a laptop M-series CPU. Numpy + matplotlib only, no GPU.

## 1980s — Local rules and the Neural Bucket Brigade

### Schmidhuber (1989) — A local learning algorithm for dynamic feedforward and recurrent networks

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`nbb-xor/`](nbb-xor/) (PR #5) | qualitative | 0.85s | 19/20 seeds solve XOR; mean 3012 presentations vs paper ~619 |
| [`nbb-moving-light/`](nbb-moving-light/) (PR #6) | yes | 0.03s | mean 223 presentations matches paper exactly; 9/30 solve rate vs paper 9/10 |

## 1990 — Controller + world-model + flip-flop

### Schmidhuber (1990) — Making the world differentiable

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`flip-flop/`](flip-flop/) (PR #6) | yes | 3-5s | 10/10 sequential (paper 6/10); 30/30 parallel (paper 20/30) |
| [`pole-balance-non-markov/`](pole-balance-non-markov/) (PR #6) | yes | 9.5s | seed 0: 30/30 episodes balance full 1000 steps |

### Schmidhuber (1990) — Recurrent networks adjusted by adaptive critics

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`pole-balance-markov-vac/`](pole-balance-markov-vac/) (PR #6) | yes | 1.21s | K=2 vector critic; 173 episodes; 9/10 multi-seed |

### Schmidhuber & Huber (1990) — Learning to generate focus trajectories

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`saccadic-target-detection/`](saccadic-target-detection/) (PR #6) | yes | 5.4s | 100% find rate, mean 1.69 saccades vs random 25.5% |

## 1991 — Curiosity, subgoals, the chunker

### Schmidhuber (1991) — Adaptive confidence and adaptive curiosity

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`curiosity-three-regions/`](curiosity-three-regions/) (PR #7) | yes | 0.5s | visit ordering C > B > A across 10 seeds (C=42.8%, B=33.3%, A=23.9%) |

### Schmidhuber (1991) — Learning to generate sub-goals for action sequences

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`subgoal-obstacle-avoidance/`](subgoal-obstacle-avoidance/) (PR #7) | yes | 6.4s | 99% success seed 0 vs 0% no-sub-goal baseline (10-seed mean 98.5%) |

### Schmidhuber (1991) — Reinforcement learning in Markovian and non-Markovian environments

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`pomdp-flag-maze/`](pomdp-flag-maze/) (PR #7) | partial | 22-32s | 6/10 seeds 100% solve, 4/10 stuck at 50% |

### Schmidhuber (1991/1992) — Neural sequence chunkers

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`chunker-22-symbol/`](chunker-22-symbol/) (PR #8) | yes | 1.86s | 99.5% label accuracy 10/10 seeds; A-alone baseline at chance |

## 1992 — Neural Computation triple

### Schmidhuber (1992) — Learning to control fast-weight memories

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`fast-weights-unknown-delay/`](fast-weights-unknown-delay/) (PR #8) | yes | 3s | 100% bit-accuracy K=5-30 trained / K=1-60 extrapolation; 10/10 seeds |
| [`fast-weights-key-value/`](fast-weights-key-value/) (PR #8) | yes | 0.07s | retrieval cosine 0.428 → 0.754 (1.76× lift); numerical grad-check <1e-9 |

### Schmidhuber (1992) — Learning factorial codes by predictability minimization

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`predictability-min-binary-factors/`](predictability-min-binary-factors/) (PR #9) | yes | 2.8s | predictors collapse to chance (L_pred = 0.2500 exact); pairwise MI 9.6e-5 nats; 8/8 seeds 100% bit-recovery |

## 1993 — Predictable classifications, self-reference, very deep chunking

### Schmidhuber & Prelinger (1993) — Discovering predictable classifications

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`predictable-stereo/`](predictable-stereo/) (PR #9) | yes | 0.08s | I(yL; yR) = 7.598 nats; depth recovery 1.000 seed 0; 8/8 seeds at 0.997 mean |

### Schmidhuber (1993) — A self-referential weight matrix

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`self-referential-weight-matrix/`](self-referential-weight-matrix/) (PR #8) | partial | 4.5s | 99.6% on 4-way boolean meta-learning (AND/OR/XOR/NAND); 8/8 seeds > 0.95 |

### Schmidhuber (1993) — Habilitationsschrift

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`chunker-very-deep-1200/`](chunker-very-deep-1200/) (PR #8) | yes | 29.8s | 599.5× depth-reduction at T=1200; chunker 100% recall vs single-net 0% (gradient vanishes by t=4) |

## 1995–1997 — Levin search and the LSTM benchmark suite

### Schmidhuber (1995/1997) — Discovering solutions with low Kolmogorov complexity

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`levin-count-inputs/`](levin-count-inputs/) (PR #4) | yes | 1.0s | 5-instr popcount routine; 770k programs enumerated; 200/200 generalize |
| [`levin-add-positions/`](levin-add-positions/) (PR #4) | yes | 0.34s | 3-instr `im+` (length-3); 58 evaluations; 200/200 generalize |

### Hochreiter & Schmidhuber (1996) — LSTM can solve hard long time lag problems

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`rs-two-sequence/`](rs-two-sequence/) (PR #4) | yes | 0.94s | 30/30 seeds solve, median 144 trials vs paper ~718 |
| [`rs-parity/`](rs-parity/) (PR #4) | yes | 15.3s | N=50 seed 0: 10,253 trials; N=500 seed 0: 412 trials / 3.2s |
| [`rs-tomita/`](rs-tomita/) (PR #4) | yes | 17-19s | #1, #2, #4 all solved across 10 seeds (within ~3× of paper for #1/#2; ~6× for #4) |

### Hochreiter & Schmidhuber (1997) — Long Short-Term Memory canonical battery

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`adding-problem/`](adding-problem/) (PR #10) | yes | 39s | LSTM MSE 0.0007 (50× under paper threshold 0.04); vanilla RNN MSE 0.0706; 5/5 seeds clear; gradient check 1.6e-7 |
| [`embedded-reber/`](embedded-reber/) (PR #10) | yes | 2.6s | 10/10 seeds, mean 4800 sequences vs paper 8440 (1.8× faster with Adam) |
| [`noise-free-long-lag/`](noise-free-long-lag/) (PR #10) | qualitative | 21s | sub-variant (a) at p=50: solved at seq 600, 100% acc; 6/10 seeds (b)/(c) deferred |
| [`two-sequence-noise/`](two-sequence-noise/) (PR #10) | yes | 32s | variant 3c only: 4/4 seeds 100% (~3k seqs vs paper ~269k SGD) |
| [`multiplication-problem/`](multiplication-problem/) (PR #10) | yes | 4.5s | LSTM MSE 0.0028 / 17× chance baseline; 3/5 seeds (paper-faithful per-seed brittleness) |
| [`temporal-order-3bit/`](temporal-order-3bit/) (PR #10) | yes | 24s | 5/5 seeds 100%, median ~6.4k seqs vs paper 31,390 (Adam advantage); vanilla RNN at chance 0.25 |

## Mid-90s — Evolutionary, RL, and feature detection

### Salustowicz & Schmidhuber (1997) — Probabilistic Incremental Program Evolution

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`pipe-symbolic-regression/`](pipe-symbolic-regression/) (PR #12) | yes | 1.3s | seed 3 finds Koza target `x + x² + x³ + x⁴` exactly at gen 60; 6/20 seeds Koza-hit-solve |
| [`pipe-6-bit-parity/`](pipe-6-bit-parity/) (PR #12) | yes | 240s | 4-bit clean solve at gen 258; 6-bit partial 71.9% at 240s budget cap |

### Schmidhuber, Zhao, Wiering (1997) — Shifting inductive bias with SSA

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`ssa-bias-transfer-mazes/`](ssa-bias-transfer-mazes/) (PR #7) | yes | 1.7s | SSA tail solve 0.83 vs no-SSA 0.70 (+19% relative); seed 0 task 2 SSA 8.12 steps vs no-SSA 60 steps |

### Wiering & Schmidhuber (1997) — HQ-learning

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`hq-learning-pomdp/`](hq-learning-pomdp/) (PR #7) | **no** | 21s | Honest non-replication: paper's HQ-vs-flat gap doesn't reproduce on 29-cell maze; mathematical analysis (`γ^Δt · HV ≤ R_goal` bound prevents per-corridor specialization) in §Open questions |

### Schmidhuber, Eldracher, Foltin (1996) — Semilinear PM produces V1-like filters

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`semilinear-pm-image-patches/`](semilinear-pm-image-patches/) (PR #9) | yes | 1.2s | 12/16 oriented filters (FFT concentration > 0.5); kurtosis 19.96 vs random 2.95; analytic-vs-numerical gradient max 5e-10 |

### Hochreiter & Schmidhuber (1999) — LOCOCODE

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`lococode-ica/`](lococode-ica/) (PR #9) | qualitative | 0.4s | Amari 0.117 mean over 10 seeds — 4× better than PCA (0.388), within 5× of FastICA (0.022) |

## 2000–2002 — LSTM follow-ups

### Gers, Schmidhuber, Cummins (2000) — Learning to forget

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`continual-embedded-reber/`](continual-embedded-reber/) (PR #11) | yes | 14s | 5/5 forget-gate seeds solve (99.7% mean) vs 5/5 no-forget at chance (55%); cell-state norm 25 vs 295 |

### Gers & Schmidhuber (2001) — Context-free and context-sensitive languages

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`anbn-anbncn/`](anbn-anbncn/) (PR #11) | yes | 35s | a^n b^n trained n=1..10 → generalizes to n=1..65 (3/5 seeds); a^n b^n c^n → n=1..29; gradcheck 5.66e-6 |

### Gers, Schraudolph, Schmidhuber (2002) — Learning precise timing

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`timing-counting-spikes/`](timing-counting-spikes/) (PR #11) | partial | 32s | Peephole seed 4: MSE 0.00073 / solve 0.998 vs vanilla 0.00240 / 0.900; cross-seed gap small (paper's "vanilla fails all" doesn't fully reproduce at short-MSD) |

### Eck & Schmidhuber (2002) — Blues improvisation with LSTM

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`blues-improvisation/`](blues-improvisation/) (PR #11) | qualitative | 12s | 12/12 bar-onset chord match; step-chord 0.906; on-beat 0.792; chord-tone 0.877 |

## 2002–2010 — Evolutionary RL, OOPS, BLSTM+CTC

### Schmidhuber, Wierstra, Gomez (2005/2007) — Evolino

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`evolino-sines-mackey-glass/`](evolino-sines-mackey-glass/) (PR #12) | partial | 140s | sines free-run MSE 0.181 (horizon 299); MG NRMSE@84 = 0.291 vs paper 1.9e-3 (whole-genome simplification of full ESP) |

### Gomez & Schmidhuber (2005) — Co-evolving recurrent neurons

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`double-pole-no-velocity/`](double-pole-no-velocity/) (PR #12) | yes | 60s | seed 0 solved at gen 27 / ~60s; 7/10 seeds 20/20 generalize at pop=40 (~5× cheaper than paper's pop=200) |

### Graves et al. (2005/2006) — BLSTM and CTC

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`timit-blstm-ctc/`](timit-blstm-ctc/) (PR #15) | qualitative | 73s | synthetic phoneme corpus (K=6); BLSTM 1.87× faster than uni-LSTM (5/5 seeds 300 vs 560 iters); gradcheck 1.12e-7 |

### Graves et al. (2009) — Unconstrained handwriting

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`iam-handwriting/`](iam-handwriting/) (PR #15) | qualitative | 103s | synthetic 10-char alphabet; in-vocab CER 0.082 / word acc 0.77; held-out compositional CER 0.647 |

### Schmidhuber (2002–2004) — Optimal Ordered Problem Solver

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`oops-towers-of-hanoi/`](oops-towers-of-hanoi/) (PR #4) | yes | 0.25s | 6-token recursive Hanoi solver `SD C SD M SA C`; reuse from n=4 onward; verified through n=15 |

## 2010–2017 — Deep learning at scale

### Cireşan, Meier, Gambardella, Schmidhuber (2010) — Deep, big, simple nets

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`mnist-deep-mlp/`](mnist-deep-mlp/) (PR #13) | partial | 79s | 1.17% test err / 15 epochs; 535k MLP vs paper 12M-weight nets at 800 epochs (0.35%) |

### Cireşan, Meier, Schmidhuber (2012) — Multi-column DNN

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`mcdnn-image-bench/`](mcdnn-image-bench/) (PR #13) | partial | 22.2s | 1.46% MNIST single-column MLP (no aug); paper 35-column ensemble 0.23% |

### Cireşan, Giusti, Gambardella, Schmidhuber (2012) — EM segmentation

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`em-segmentation-isbi/`](em-segmentation-isbi/) (PR #15) | qualitative | 1.5s | Synthetic Voronoi-EM substitute; ROC AUC 0.989 vs Sobel+intensity 0.880; pixel acc 95.97% |

### Srivastava, Masci, Kazerounian, Gomez, Schmidhuber (2013) — Compete to compute

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`compete-to-compute/`](compete-to-compute/) (PR #13) | qualitative | 0.8s | Seed 0: LWTA forgetting 0.022 vs ReLU 0.072 (3.3× less); 10-seed: LWTA wins 6/10 (small-net regime noisy) |

### Srivastava, Greff, Schmidhuber (2015) — Training very deep networks (Highway)

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`highway-networks/`](highway-networks/) (PR #13) | yes | 7s | Depth 30: highway 0.926 vs plain 0.124 (chance); plain dies past depth 10; highway stable 5-50 |

### Greff, Srivastava, Koutník, Steunebrink, Schmidhuber (2017) — Search Space Odyssey

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`lstm-search-space-odyssey/`](lstm-search-space-odyssey/) (PR #15) | yes | 145s | All 8 LSTM variants implemented; CIFG 1st, NIG last across 3/3 seeds; gradient check 1.31e-7 |

### Koutník, Greff, Gomez, Schmidhuber (2014) — Clockwork RNN

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`clockwork-rnn/`](clockwork-rnn/) (PR #15) | yes | 22s | Synthetic sum-of-sines T=320, periods 8/32/80/160; CW-RNN 0.117 vs vanilla 0.250 (2.22× over 5 seeds); multi-rate decomposition in per-group FFT |

### Koutník, Cuccu, Schmidhuber, Gomez (2013) — Vision-based RL via evolution

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`torcs-vision-evolution/`](torcs-vision-evolution/) (PR #15) | yes | 45.5s | Numpy oval track + 16×16 obs + DCT-parameterized W1; 14.3× compression (4129 raw → 289 DCT); 5/5 seeds solve in ≤50s |

### Greff, van Steenkiste, Schmidhuber (2017) — Neural Expectation Maximization

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`neural-em-shapes/`](neural-em-shapes/) (PR #14) | partial | 17s | K=3 slot N-EM, manual BPTT through T=4 EM iterations; best test NMI 0.428 epoch 7 (chance 0.33); paper AMI 0.96 |

### van Steenkiste, Chang, Greff, Schmidhuber (2018) — Relational Neural EM

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`relational-nem-bouncing-balls/`](relational-nem-bouncing-balls/) (PR #14) | qualitative | 24.8s | Velocity-MSE: relational wins K=3,4,5 (0.81×, 0.92×, 0.97×); loses K=6 (1.01× — distribution shift dominates) |

## 2018–2025 — World models, fast-weight Transformers, systematic generalization

### Ha & Schmidhuber (2018) — Recurrent World Models

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`world-models-carracing/`](world-models-carracing/) (PR #15) | yes | 6.5s | Numpy 2D track; V+M+C +103.8 mean across 5/5 seeds (random +4.84, ~21× random) |
| [`world-models-vizdoom-dream/`](world-models-vizdoom-dream/) (PR #15) | yes | 20s | Numpy 5×5 gridworld; controller trained ENTIRELY in M's dream → zero-shot real-env transfer (49.1 vs random 22.4, 2.2× random) |

### Schmidhuber et al. (2019) — Reinforcement Learning Upside Down

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`upside-down-rl/`](upside-down-rl/) (PR #14) | yes | 3.5s | Numpy 9-state chain MDP (per SPEC, not LunarLander); 5/5 seeds reach +4.70 at R*=5.0; achieved monotonically tracks commanded |

### Schlag, Irie, Schmidhuber (2021) — Linear Transformers are secretly fast weight programmers

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`linear-transformers-fwp/`](linear-transformers-fwp/) (PR #14) | yes | 0.08s | **Equivalence verified to 2.22e-16 (float64 ulp)**: `V^T(Kq)` ≡ `(V^T K)q`. Pre-train cos 0.428 → post 0.754 (1.76×); delta-rule peaks +0.05 above sum-rule at N=6 |

### Csordás, Irie, Schmidhuber (2022) — The Neural Data Router

| Stub | Reproduces? | Run wallclock | Headline |
|---|---|---:|---|
| [`neural-data-router/`](neural-data-router/) (PR #14) | partial | 3:30 | Test depth 5: NDR 0.60 vs vanilla 0.32 (chance 0.25); 3-seed NDR 0.405 ± 0.013 vs vanilla 0.296 ± 0.031 (NDR wins 3/3) |

## Summary statistics

| Reproduces? | Count | Examples |
|---|---:|---|
| **yes** | 32 | nbb-moving-light, flip-flop, embedded-reber, fast-weights-key-value, oops-towers-of-hanoi, linear-transformers-fwp, world-models-carracing, ... |
| **partial** | 12 | self-referential-weight-matrix, mnist-deep-mlp, mcdnn-image-bench, evolino-sines-mackey-glass, neural-em-shapes, neural-data-router, ... |
| **qualitative** | 13 | nbb-xor, noise-free-long-lag, lococode-ica, blues-improvisation, em-segmentation-isbi, compete-to-compute, timit-blstm-ctc, iam-handwriting, ... |
| **no** | 1 | hq-learning-pomdp (honest non-replication; mathematical analysis documented) |

**Total: 58 stubs implemented, all in pure numpy + matplotlib, all <5 min/seed on a laptop except `pipe-6-bit-parity` (240s 6-bit budget cap), `evolino-sines-mackey-glass` (140s).**

## v2 filter recommendation

For the data-movement / ByteDMD instrumentation, prioritize stubs that:

### 1. Reproduce cleanly + run fast (low noise floor for measuring data-movement deltas)

- Pure-numpy mini-environments + sub-second runs: `linear-transformers-fwp` (0.08s), `predictable-stereo` (0.08s), `levin-add-positions` (0.34s), `lococode-ica` (0.4s), `compete-to-compute` (0.8s), `nbb-xor` (0.85s), `rs-two-sequence` (0.94s), `levin-count-inputs` (1.0s), `semilinear-pm-image-patches` (1.2s), `pipe-symbolic-regression` (1.3s), `em-segmentation-isbi` (1.5s), `ssa-bias-transfer-mazes` (1.7s), `chunker-22-symbol` (1.86s), `predictability-min-binary-factors` (2.8s).
- Verified-by-gradient-check (numerical-vs-analytical < 1e-6): `fast-weights-unknown-delay`, `fast-weights-key-value`, `temporal-order-3bit`, `temporal-order-4bit`, `adding-problem`, `noise-free-long-lag`, `clockwork-rnn`, `lstm-search-space-odyssey`, `anbn-anbncn`, `timit-blstm-ctc`, `self-referential-weight-matrix`.

### 2. Have algorithmic variants on the same problem (lets you compare data-movement across algorithms)

- **adding-problem family**: vanilla RNN vs LSTM (paper's contrast, both implemented in `adding-problem` and `temporal-order-3bit`).
- **temporal-order family**: 3-bit vs 4-bit, 4-class vs 8-class on identical architecture.
- **embedded-reber family**: original 1997 LSTM (no forget) vs forget-gate LSTM (`continual-embedded-reber`).
- **LSTM ablation matrix**: `lstm-search-space-odyssey` runs 8 variants on the same task — V/NIG/NFG/NOG/NIAF/NOAF/CIFG/NP — direct architectural-variant data-movement comparison built in.
- **Linear-attention ↔ FWP**: `linear-transformers-fwp` IS the equivalence demo; `fast-weights-key-value` is the 1992 ancestor; ByteDMD on both should produce identical numbers.
- **Evolutionary methods**: `pipe-symbolic-regression` (PIPE), `evolino-sines-mackey-glass` (Evolino), `double-pole-no-velocity` (ESP), `torcs-vision-evolution` (DCT-compressed natural ES) — gradient-free family for compare-vs-gradient-based data-movement.
- **Search methods**: `levin-count-inputs`, `levin-add-positions` (Levin), `oops-towers-of-hanoi` (OOPS), `rs-*` (random search) — all gradient-free.
- **World models**: `world-models-carracing` and `world-models-vizdoom-dream` share V+M+C decomposition — three distinct training stages with very different memory access patterns.

### 3. Defer for v2

- Stubs with run wallclock > 100s where v2 ByteDMD overhead would dominate: `pipe-6-bit-parity` (240s 6-bit), `evolino-sines-mackey-glass` (140s), `lstm-search-space-odyssey` (145s).
- Honest non-replications where measuring data-movement on a non-converged solver isn't informative: `hq-learning-pomdp` (paper's HQ-vs-flat gap doesn't reproduce on this maze size).
- Partial reproductions where the v1.5 path needs to close first: `neural-em-shapes` (no background slot), `mnist-deep-mlp` (smaller MLP), `mcdnn-image-bench` (single-column).

## v1.5 + v2 follow-ups

Each stub's §Open questions section flags stub-specific follow-ups. Repository-wide follow-ups:

- **Original-simulator reruns** (RL/env-heavy stubs): close the loop on gym CarRacing-v0, VizDoom DoomTakeCover, TORCS, TIMIT, IAM, ISBI. Currently all 8 use numpy mini-environments per the SPEC's RL-stub rule.
- **Paper-scale reruns** for partial reproductions: full paper-scale `mnist-deep-mlp` (12M weights, 800 epochs); 35-column ensemble for `mcdnn-image-bench`; full ESP for `evolino-sines-mackey-glass`; T ≥ 300 for `timing-counting-spikes`.
- **ByteDMD instrumentation** (the actual research goal): prioritize the v2-filter recommendations above.

---

_Compiled by agent-0bserver07 (Claude Code) on behalf of Yad. Source: PR bodies #4-#15 + per-stub READMEs._
