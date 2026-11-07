# Sustained single-arm execution

This supplements the source choices in `PAPER_PROTOCOL.md`. The
`paper_train.py` is an optional offline executable, requiring a separately
reviewed operator resource watchdog. Importing it does not import Torch or run
a device. The recorded800-epoch seed0 run is partial at51/10000 test mistakes;
see [README Results](README.md#results) and [RERUN.md](RERUN.md).

Continue the verified seed0 4096-update pilot prefix into the same29x29 arm:
800epochs,60000 ordered online updates per epoch, the frozen geometric LR
schedule, architecture and deformations. Continuing the prefix avoids repeating
or discarding counted training updates. The command line identifies its genuine
commit digest. The pilot and sustained driver share the update/augmentation kernels.

At each completed epoch evaluate all60000 original, undeformed training images
using the same bilinear29x29 resize and[-1,1] normalization. The evaluator captures
sequential FP32 GEMV forwards without parameter updates, with a startup equality
check against eager full-size forwards and unchanged parameter tensors. It does
not use batched GEMM reductions. Choose the minimum integer validation mistake
count, with the earliest epoch winning ties. No test statistic can change this
rule, architecture, hyperparameters, seed, run length or chosen input dimension.

Every checkpoint stores exact epoch/cursor/update counts, next LR, source/config/
data/runtime binding, seed/epoch-addressed RNG scheme and shuffled-order digest.
At a60000-update boundary the checkpoint has next epoch/cursor0 plus an explicit
pending-validation flag; this retains the original tensor checkpoint contract.
The full800-entry validation journal must agree with the selected snapshot before
test admission. Interrupted validation can reuse its exact committed observation
or repeat that read-only training validation; it cannot create a second test pass.

Checkpoint thresholds are every6000updates, rounded up to the next completed
32-example chunk (e.g.6016), plus epoch validation boundaries and controlled stop.
The store retains latest, its recoverable predecessor and best, at most three
committed generations, with a fourth only during publication. Each is independently
content-verified. Before deleting an obsolete generation, a durable exact-member
intent identifies its files, sizes and digests; an interruption can remove only
the remaining matching files. Unknown/replaced files stop retirement. Old partial
writer artifacts are preserved by reversible relocation, never silently removed.
Actual footprint is checked before/after publication against512MiB, with an
additional64MiB allowance and the operator's preserved15GiB disk reserve. Journals
are bounded; the driver cannot accumulate8000 full model copies.

The final test requires an explicit flag, exactly800 validations and48million
training updates, and an exclusive durable reservation before either test stream
is opened. Load only the validation-selected best; evaluate exactly10000 examples
once (32-example chunks plus one16-example remainder graph). A failed/interrupted
test reservation stays consumed and is not automatically reset or retried.
Without the explicit final flag, training ends with an awaiting-test receipt.

Recorded sustained wall time was14,260.73s (~3.96h), continuing the4,096-update
pilot rather than restarting it. Per-invocation elapsed timers recorded6,680.32s
updates,3,628.40s validation,3,519.81s checkpoints and234.02s deformation; other
startup, bookkeeping and final-test work accounts for the remaining wall time.
The prior preparatory pilot phase took116.37s. These are elapsed times, not
metered active GPU-hours.

The store published8,801 fixed-shape snapshots. Their derived logical tensor
payload volume is426,530,528,671bytes, while only latest/predecessor/best remain.
This derives from committed-generation coverage and the invariant uncompressed
12-tensor serialization size; it does not measure physical-device, cache,
metadata or retry write traffic. Checkpoint overhead is included in the wall time.

The result retains the appendix29x29 versus main-text28x28 discrepancy,
12,107,500matrix weights +7,510biases, the disclosed objective/target/deformation
reconstructions, one chosen seed and modern FP32 reduction differences. Its
51/10000 test mistakes compare with the publication's35/10000 and the smaller
reference's117/10000. They do not recover unspecified original details or
establish variability across seeds. No second arm was selected from the test outcome.
