# temporal-order-3bit

Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation 9(8): 1735–1780. Experiment 6a (Temporal Order, 3-bit).

![training animation](temporal_order_3bit.gif)

## Problem

Each input sequence runs `T = 50` symbols, drawn from an 8-symbol alphabet:

```
{a, b, c, d}  random distractors
{X, Y}        the two information-carrying symbols
{B, E}        sequence-start and sequence-end markers
```

Position 0 is always `B`, position `T-1` is always `E`. Two slots `t1 ∈ [3, 12]` and `t2 ∈ [25, 40]` carry independently drawn symbols from `{X, Y}`. Every other interior slot is a uniform random distractor. The class label encodes the *order* of the two important symbols:

| (first, second) | class id | name |
|---|---|---|
| (X, X) | 0 | XX |
| (X, Y) | 1 | XY |
| (Y, X) | 2 | YX |
| (Y, Y) | 3 | YY |

Inputs are one-hot vectors of dimension 8. The network reads the whole sequence, then emits a 4-way softmax at the final time step. The minimum lag between the two informative symbols is `25 − 12 = 13`, the maximum is `40 − 3 = 37`. The network must hold the identity of the first marker across that gap while ignoring 13–37 distractor symbols.

## What it demonstrates

A vanilla recurrent net with `tanh` activations cannot bridge the gap and stays at chance accuracy (≈ 0.25). An LSTM with the input-gate/output-gate cell of the 1997 paper (no forget gate, pure constant-error carousel) solves it to 100 %. Inspecting the trained net shows the input gate firing only on the two `X`/`Y` positions and the cell state encoding their order in the sign of two different cells.

## Files

| File | Purpose |
|---|---|
| `temporal_order_3bit.py` | Dataset generator, LSTM with BPTT, vanilla-RNN baseline, training loops, gradient check, CLI. |
| `visualize_temporal_order_3bit.py` | Reads `results.json` + `snapshots.npz`, writes static PNGs into `viz/`. |
| `make_temporal_order_3bit_gif.py` | Builds the cell-state animation `temporal_order_3bit.gif` from the snapshot tensor. |
| `temporal_order_3bit.gif` | Cell-state heatmap evolving through training, one frame per ≈ snapshot. |
| `viz/training_curves.png` | LSTM vs RNN loss + accuracy. |
| `viz/confusion_matrix.png` | LSTM 4×4 confusion matrix on validation set. |
| `viz/example_sequences.png` | One example sequence per class as a token-time heatmap. |
| `viz/input_gate_activity.png` | Max input-gate activation per time step on those examples. |
| `viz/hidden_trajectories.png` | Cell state `c_t` and hidden state `h_t` per time step, per class. |
| `viz/cell_state_heatmap.png` | Final cell state as a (cell index × time) heatmap. |
| `results.json` | Full training log (steps, loss, accuracy, confusion matrix). |
| `snapshots.npz` | Captured hidden-state tensors for the GIF and trajectory plots. |

## Running

The headline command (≈ 24 s on an M-series laptop, single core):

```bash
python3 temporal_order_3bit.py --seed 0 \
    --n_steps 1500 --batch 32 --hidden 4 \
    --val_n 512 --eval_every 50 --record_hidden
python3 visualize_temporal_order_3bit.py
python3 make_temporal_order_3bit_gif.py
```

Self-test of the analytic LSTM gradient (max relative error vs central differences):

```bash
python3 temporal_order_3bit.py --gradcheck
# [gradcheck] max relative error = 2.363e-11
```

## Results

Headline run, seed 0:

| Metric | Value |
|---|---|
| LSTM final validation accuracy (512 sequences) | **1.000** (512 / 512 correct) |
| LSTM step at first ≥ 95 % validation accuracy | 100 (= 3 200 sequences at batch 32) |
| RNN final validation accuracy | 0.250 (chance) |
| RNN best-ever validation accuracy | 0.266 |
| LSTM training wall-clock | 13.6 s |
| RNN training wall-clock  | 10.6 s |
| Total training sequences seen | 48 000 = 1 500 × 32 |
| Trainable parameters (LSTM) | 184  (`Wi, Wo, Wg ∈ R^{12×4}` + biases + `Why ∈ R^{4×4}` + `by`) |
| Trainable parameters (RNN)  |  68  (`Wx ∈ R^{8×4}, Wh ∈ R^{4×4}, bh, Why, by`) |

Hyperparameters used:

| Hyperparameter | Value |
|---|---|
| Sequence length `T` | 50 |
| Hidden / cell count | 4 |
| Batch size | 32 |
| Optimiser | Adam (lr = 0.02, β₁ = 0.9, β₂ = 0.999) |
| Gradient clip (global ℓ²) | 1.0 |
| Steps | 1500 |
| Input-gate bias init | −1.0 (cell starts closed) |
| Other parameter init | `N(0, 0.1²)` |

Multi-seed reliability (`--seed 0..4`, otherwise identical config):

| seed | LSTM final acc | RNN final acc | first-step ≥ 95 % |
|---:|---:|---:|---:|
| 0 | 1.000 | 0.238 | 100 |
| 1 | 1.000 | 0.293 | 200 |
| 2 | 1.000 | 0.230 | 100 |
| 3 | 1.000 | 0.254 | 300 |
| 4 | 1.000 | 0.258 | 200 |

5 / 5 seeds solve. Median 200 steps to 95 % (≈ 6 400 sequences). The 1997 paper reports 31 390 sequences for a slightly larger sequence and an LSTM with 156 weights; we converge faster because of Adam (the paper used plain SGD with momentum).

Confusion matrix on 512 validation sequences (seed 0):

|        | pred XX | pred XY | pred YX | pred YY |
|--------|---:|---:|---:|---:|
| true XX | 119 | 0 | 0 | 0 |
| true XY | 0 | 128 | 0 | 0 |
| true YX | 0 | 0 | 134 | 0 |
| true YY | 0 | 0 | 0 | 131 |

## Visualizations

**`temporal_order_3bit.gif`** — Cell state `c_t` for one held-out sequence per class, animated across training. At step 1 the heatmap is uniformly near zero. As training proceeds, a dark-then-light spike appears at the first `X`/`Y` position and a second spike at the second one; by step ≈ 200 the first cell carries the identity of the first marker (positive for X, negative for Y) and the second cell carries the second. Vertical ticks mark `X` (green) and `Y` (red) positions on the input.

**`viz/training_curves.png`** — Cross-entropy loss and validation accuracy for LSTM (blue) and vanilla RNN (orange). The LSTM curve drops from `log 4 ≈ 1.39` to near zero around step 100; the RNN curve plateaus near `log 4` and the accuracy line never lifts off the 0.25 chance line.

**`viz/confusion_matrix.png`** — A diagonal matrix: every class is recovered without a single confusion on 512 held-out sequences.

**`viz/example_sequences.png`** — One example sequence per class rendered as an 8 × 50 binary heatmap. Vertical lines mark the `X` (red) and `Y` (blue) positions.

**`viz/input_gate_activity.png`** — Max-over-cells input gate `max_k i_t^{(k)}` plotted as bars for those four sequences. The gate fires only on the two informative time steps and stays near zero on every distractor; the negative bias initialisation matters.

**`viz/hidden_trajectories.png`** — Two-row strip of `c_t` (top) and `h_t` (bottom) for each class. The cell trajectories show clear stepwise jumps at `t1` and `t2`; `h_t` only carries information at the moment the output gate opens (the last few steps before the readout).

**`viz/cell_state_heatmap.png`** — `c` at the end of training, plotted as a `H × T` heatmap per class. The four classes are visually separable in cell space.

## Deviations from the original

| Deviation | What the paper used | What we used | Reason |
|---|---|---|---|
| Sequence length | 100–110 (and a longer “6b” variant for 4-bit) | 50 | Keeps the experiment under 30 s on a CPU laptop; the paper's lag of ~30 distractors is preserved (`t1 ∈ [3,12]`, `t2 ∈ [25,40]`). |
| Marker positions | `t1 ∈ [10,20]`, `t2 ∈ [50,60]` | `t1 ∈ [3,12]`, `t2 ∈ [25,40]` | Scaled with the shorter length. The qualitative claim — that the network must integrate information across many distractor symbols — is unchanged. |
| Cell architecture | 2 cell blocks of size 2 (4 cells, gated together as 2 blocks) | 4 independent cells (no block structure) | Block sharing of gates only saves parameters; with hidden = 4 the difference is small, and a flat layout is easier to read out and visualise. |
| Optimiser | SGD with momentum | Adam (`lr = 0.02`) | Matches what the rest of the wave-6 stubs use; the paper's optimiser converges in ~31 k sequences, ours converges in ~6 k. The algorithmic claim — long-time-lag credit assignment via a CEC — is what we are testing, not the optimiser. |
| Forget gate | not in 1997 NC | not present (matches the paper) | The paper's CEC has no forget gate; the forget gate was added by Gers, Schmidhuber & Cummins (2000). We follow the 1997 formulation. |
| Output activation | softmax over 4 classes | softmax over 4 classes | Match. |
| Loss | cross-entropy at end of sequence | cross-entropy at end of sequence | Match. |
| Validation set size | unspecified in the paper | 512 sequences, fresh seed | Ours is reused across the whole run for fair comparison between LSTM and RNN. |
| Baseline | "RTRL fully recurrent net" | BPTT vanilla `tanh`-RNN with the same hidden size and the same Adam settings | Both fail; the failure mode is qualitatively the same (cannot push gradient through 30+ distractor steps). RTRL would be slower per step but no more capable on this task. |
| Sequence-end marker | `B` end-of-sequence symbol | `E` (chose a distinct token to avoid colliding with the start-marker `B` used elsewhere in the alphabet) | Cosmetic. |

## Open questions / next experiments

- **Block-structured cells.** The paper shares gate weights inside a "memory block." Sharing should make the input gate fire even more cleanly on the X/Y positions because all cells in a block see the same gate decision. Worth a five-minute follow-up.
- **Length scaling.** The current experiment uses `T = 50`. Does the same hidden size still solve `T = 100` (paper's setting), `T = 200`, `T = 500`? The CEC has no decay, so in principle yes — the limiting factor is the optimiser, not the architecture. A length sweep would confirm.
- **Forget-gate ablation.** Adding a forget gate (Gers 2000) speeds up the noise-free long-lag and adding-problem stubs but is not needed here. Worth a side-by-side once the wave-6 family is in place.
- **Citation gap.** The 1997 NC paper's "31 390 sequences" figure is reported in the literature but is not split by seed or by reset; we cannot tell whether their median or worst-case run is the headline. Our number (≈ 6 400 sequences, median over 5 seeds) is not directly comparable. If we want a like-for-like number we have to (a) match their architecture exactly, (b) match their optimiser, (c) report a 30-seed median with their stopping criterion. Tracked as a v2 follow-up.
- **DMC instrumentation (v2).** Wrap forward + backward in `bytedmd` and report data-movement cost per training step. Expectation: distractor steps cost almost nothing because the input gate is near zero and the cell state is unchanged, so reads of `c_{t-1}` are repeats. The 1997 LSTM is a remarkably "data-movement friendly" recurrent architecture.

---
_agent-0bserver07 (Claude Code) on behalf of Yad_
