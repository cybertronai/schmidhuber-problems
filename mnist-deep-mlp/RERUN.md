# Offline paper-scale rerun

Run commands from this stub directory, using fresh `artifacts/`, `data/` and
`run/` directories. The recorded arm
used Python3.13.14, an SM120-enabled PyTorch2.14.0+cu130 build (CUDA13.0),
NumPy2.5.3, PyArrow26.0.1 and Pillow13.0.0. Acquire a compatible official CUDA
build separately. The original NumPy entry point remains independent of Torch.

Use one GPU and an independent finite resource watchdog for explicit CUDA
commands. The model caps its allocator at4GiB; that is not a cap on all driver
memory. Snapshot publication requires15GiB free disk space plus the next64MiB
snapshot allowance. `--seconds` is a cooperative work budget. The commands below assume
appropriate external resource/time supervision and a new `run/` directory.

## Acquire and verify data

The [pinned provider revision](https://huggingface.co/datasets/ylecun/mnist/tree/77f3279092a1c1579b2250db8eafed0ad422088c)
declares MIT. The bridge checks each provider image and label against original
IDX ordering. It checks the pinned Parquet sizes/hashes, exact60000/10000 counts,
28×28 image bytes, labels, headers and complete payload lengths. This raw data
identity check performs no model inference. Keep images and trained checkpoints
out of the code/results contribution.

For example, acquire the six public inputs separately:

```bash
mkdir -p artifacts run
idx=https://ossci-datasets.s3.amazonaws.com/mnist
for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl --fail --location "$idx/$file" --output "artifacts/$file"
done
provider=https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist
curl --fail --location "$provider/train-00000-of-00001.parquet" --output artifacts/train-00000-of-00001.parquet
curl --fail --location "$provider/test-00000-of-00001.parquet" --output artifacts/test-00000-of-00001.parquet
```

Then materialize a new offline `data/` directory; this refuses to overwrite an
existing output:

```bash
python - <<'PY'
from pathlib import Path
from paper_inputs import materialize
p = Path('artifacts')
materialize(
    {'train': p/'train-00000-of-00001.parquet',
     'test': p/'test-00000-of-00001.parquet'},
    {'train': [p/'train-images-idx3-ubyte.gz', p/'train-labels-idx1-ubyte.gz'],
     'test': [p/'t10k-images-idx3-ubyte.gz', p/'t10k-labels-idx1-ubyte.gz']},
    Path('data'))
PY
DATA_SHA=$(python -c "from paper_inputs import digest; print(digest('data/MANIFEST.json'))")
```

## Controls, counted prefix and full arm

The scalar/fixture commands need no Torch or dataset. Native commands explicitly
import Torch and require the prepared environment; CUDA is never a CPU fallback.

```bash
python -B test_paper_online.py
python -B test_checkpoint_publication.py
python -B test_paper_runstate.py
python -B check_paper_native.py --device cpu --scratch run/native-cpu

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
python -B check_paper_native.py --device cuda --scratch run/native-cuda

python -B paper_pilot.py --data data --data-manifest-sha256 "$DATA_SHA" \
  --output run/pilot --stop-file run/STOP --updates 4096 --seconds 540
```

Proceed only if `run/pilot/PILOT.json` reports4096 completed updates,
`stopReason="update-budget"`, finite parameters and zero test evaluations. The
pilot is a counted training prefix; continuing it avoids repeated/discarded
updates. Bind its actual immutable commit:

```bash
PREFIX_SHA=$(python -c "from paper_inputs import digest; print(digest('run/pilot/checkpoints/checkpoint-000004096/COMMIT.json'))")
python -B paper_train.py --data data --data-manifest-sha256 "$DATA_SHA" \
  --root run/arm --stop-file run/STOP --seconds 27000 --resume pilot \
  --pilot-checkpoint run/pilot/checkpoints/checkpoint-000004096 \
  --initial-prefix-commit-sha256 "$PREFIX_SHA" --final-test-approved
```

This is the fixed seed0,29×29,800-epoch arm; no parameter sweep occurs. The driver
verifies full-size read-only forward capture against eager results, regenerates
deformations per epoch and applies each of the48million updates sequentially.
All800 validations use the original undeformed training set. Minimum integer
mistakes, with earliest epoch on ties, selects the final model before a single
test pass. `STATE.json`, `VALIDATION.json`, `RETENTION.json`, `FINAL_TEST.json`
and `RESULT.json` preserve those boundaries. `bestEpoch` is zero-based.

Omit `--final-test-approved` to stop after completed training/selection without
opening test images. A later explicitly approved invocation can resume that
state for the one test pass.

## Resume and retention

Before final-test reservation, resume an interrupted arm using its existing
code, data, backend and original pilot commit identity:

```bash
python -B paper_train.py --data data --data-manifest-sha256 "$DATA_SHA" \
  --root run/arm --stop-file run/STOP --seconds 27000 --resume latest \
  --initial-prefix-commit-sha256 "$PREFIX_SHA" --final-test-approved
```

`run/STOP` requests a cooperative stop. Resolve an operator stop before an
approved resume. Do not delete or reset `FINAL_TEST.json`: an interrupted test
reservation remains consumed and a completed test cannot be repeated.

Latest, its recoverable predecessor and the validation-selected best are kept,
with at most one pending publication. The bounded512MiB store verifies exact
owned contents before retiring old payloads and preserves unknown/interrupted
artifacts. See [SUSTAINED_PROTOCOL.md](SUSTAINED_PROTOCOL.md) for the transaction
and timing scope, and [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) for reconstructed
choices and the29×29/28×28 source discrepancy.
