# Optional v1.5 paper-scale path: measured partial reproduction

The existing `mnist_deep_mlp.py`, laptop settings and reported117/10000 result
remain available. The optional GPU-scale arm requested in issue18 completed
800 epochs at seed0 and returned51/10000 test mistakes (0.51%) versus the
paper's35/10000 (0.35%). It remains partial under the choices below. See
[README Results](README.md#results) and [RERUN.md](RERUN.md).

Primary source: Cireşan et al., [arXiv1003.0358v1](https://arxiv.org/abs/1003.0358),
sections2–5, Table1 row4 and GPU appendix. The paper specifies online backprop
without momentum, all-neuron scaled tanh, uniform weights, decaying learning
rate, and regenerated deformations. Its validation set is the original60000
training images, not a held-out training subset. Select the epoch with smallest
original-training validation error (earliest epoch on ties), then evaluate the
10000 test examples once. Do not select by test error or run another full arm
because of the test outcome. Compare the resulting mistake count with35/10000;
report an interval, runtime/hardware and limitations, whether successful or not.

## Selected choices and explicit approximations

| Item | Choice / evidence boundary |
|---|---|
| Widths | 2500–2000–1500–1000–500–10, one network, no ensemble |
| Input |Selected29x29 resized bilinearly from28x28: appendix describes29, giving12,107,500matrix weights (rounds to Table1's12.11M) plus7,510biases =12,115,010total parameters. Table1 does not separately account for biases. Section2 instead says28x28=784pixels feed the input; that gives11,965,000matrix weights plus7,510biases =11,972,510total. The selected input is a disclosed resolution of a source inconsistency, not a recovered fact. |
| Numeric precision |FP32, no TF32/autocast; scaled tanh1.7159*tanh(.6666*a) at every layer including output |
| Initialization |Every weight, including bias, independently uniform[-.05,.05] from private fixed generator |
| Update |One example, all backward deltas from old weights, then rank-one SGD update; no momentum, weight decay or accumulation |
| Loss / targets |Half sum squared error, bipolar targets+1/-1. **Reconstruction:** source does not specify output target values or explicit objective formula. Not a claim of recovered original code. |
| Schedule |800epochs of60000updates, seed0; lr=1e-3*(1e-3)^(epoch/799), ending1e-6. **Reconstruction:** issue supplies800; paper supplies endpoints/multiplicative decay, not this exact factor. |
| Order |Fresh deterministic shuffle each epoch. **Reconstruction:** original shuffle order is not supplied. |
| Deformation |Once per epoch, fixed sigma5.5/alpha37/scaling±17.5%, within stated ranges. Rotation OR horizontal shear with50/50probability, beta±7.5° for1/7 and±15° otherwise; independent X/Y scale. Choice probabilities/exact constants are reconstructed. |
| Interpolation |Inverse centered affine mapping plus smoothed displacement, zero image/field padding,21x21 normalized Gaussian, bilinear align-corners grid sampling. The original tiled300x300field boundary convention is not recovered; document this difference. No unsupported curriculum. |

## Implementation and honest gates

`python3 paper_online.py --smoke` and `python3 test_paper_online.py` use stdlib
only, synthetic tiny data and no network/GPU. The scalar float64 oracle checks
all weight/bias gradients. It is a correctness oracle, not the FP32 trainer.

`paper_cuda.py` imports torch only inside explicit native calls. It computes
FP32 matrix-vector propagation/backprop and sequential rank-one updates. A
CUDA graph contains32**ordered single-example updates**, not a minibatch.
Warmup/capture restore every parameter before counted training. The learning
rate remains a device scalar, so graph replay can respect the frozen schedule.

Before any timing, `check_paper_native.py --device cpu --scratch <owned-dir>`
must pass real Torch/scalar parity, warping and actual tensor checkpoint tests.
Then its CUDA variant must pass graph/eager parity, untouched warmup state,
actual bitwise interrupted/resumed continuation and exact checkpoint negatives.
The recorded run passed11 native CPU controls and13 CUDA controls, including
actual tensor restart checks. Sustained startup also checked full-size32/16
read-only captured/eager forwards bitwise and unchanged parameters. Run GPU
commands under an independent finite resource watchdog.

`paper_inputs.py` has no network fallback. Independently acquired approved
provider parquet files must match the pinned revision/size/SHA and every decoded
image/label must match canonical independently sourced IDX ordering. Training
uses only the two training byte files under the identity manifest. The full
driver opens test images only after800 validations and an exclusive final-test
reservation. Identity checking of both raw splits does not perform model
inference. Provider card attribution is retained; images/checkpoints are not
part of the code/results submission.

The pilot is at most600seconds after a separate<=60sgraph warmup, at most60000
updates, default4096. It never scores test data, never claims epoch selection,
and marks validation pending. Immutable checkpoints retain source/data/config/
backend, exact epoch/cursor/update count, LR and private epoch-addressed RNG
scheme; no mutable random draws occur between these checkpoints. A resume must
match bindings, regenerate the same epoch, verify model parity and continue at
the saved next-example cursor. Partial/uncommitted checkpoints are not resumes.
Unique temporary names let a later save survive orphaned partials. If only
pointer promotion failed, reuse a committed checkpoint only after checking its
full metadata and tensor equality; never overwrite an earlier commit. Windows
access/sharing errors get at most0.5sretry on this writer's own temporary path;
persistent or other errors remain failures with the partial retained.

Run native controls/pilot in a private offline environment with explicit resource
limits and a finite independent watchdog. Measure update wall time, CUDA graph
elapsed time, deformation, serialization and true peak memory separately before
starting a sustained run. Do not infer kernel duty time from total wall time.

`paper_train.py` implements the800-epoch arm, undeformed-training validation
selection, bounded rolling checkpoint retention and exclusive final-test ledger.
The recorded earliest validation minimum was49/60000 at the669th epoch; one
subsequent test pass made51/10000 mistakes. `RESULTS.md` therefore remains
**partial**. The Wilson95% test-set interval0.388–0.670% is conditional on this
one selected checkpoint and does not measure seed or reconstruction variability.
