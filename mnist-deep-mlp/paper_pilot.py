"""Bounded optional v1.5 pilot. No download and no test-set inference.

Run only under an independently reviewed operator resource/authority supervisor.
This pilot alone is not a paper reproduction or permission for a sustained run.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_online import PaperConfig, epoch_order
from paper_inputs import read_train, digest
from paper_cuda import OnlineMLP, augment, configure_cuda, load_checkpoint, save_checkpoint


def source_digest():
    root = Path(__file__).resolve().parent
    files = {p: digest(root / p) for p in ('paper_online.py', 'paper_inputs.py', 'paper_cuda.py', 'paper_pilot.py')}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--data-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=4096)
    parser.add_argument('--seconds', type=int, default=600)
    args = parser.parse_args()
    cfg = PaperConfig()
    cfg.validate()
    if not 32 <= args.updates <= 60000 or args.updates % 32 or not 1 <= args.seconds <= 600:
        raise ValueError('Pilot is finite32..60000updates and<=600seconds')
    if args.stop_file.exists():
        raise RuntimeError('Operator stop is already active')
    images, labels = read_train(args.data, args.data_manifest_sha256)
    args.output.mkdir(exist_ok=False)
    # Configure before any CUDA operation. Actual supervisor additionally bounds
    # aggregate driver VRAM, host RAM/CPU, temperature/power and the single lease.
    import torch
    runtime = configure_cuda(torch)
    runtime['python'] = platform.python_version()
    model = OnlineMLP(cfg.sizes, cfg.seed, 'cuda')
    binding = {'configSHA256': cfg.digest(), 'sourceSHA256': source_digest(),
               'dataSHA256': args.data_manifest_sha256, 'runtime': runtime,
               'rngScheme': 'init=CPUtorch(seed);shuffle=PythonRandom((seed<<32)+epoch);augmentation=CUDAgenerator(seed+1000003*epoch+block)'}
    epoch, cursor, updates = 0, 0, 0
    if args.resume:
        meta = load_checkpoint(model, args.resume, binding)
        epoch, cursor, updates = (meta[k] for k in ('epoch', 'cursor', 'updates'))
        if epoch >= cfg.epochs or cursor % 32 or updates != epoch * 60000 + cursor:
            raise ValueError('Invalid resumed cursor')
    # Static capture warms only scratch updates and restores every weight/bias.
    capture_started = time.monotonic()
    model.capture()
    capture_seconds = time.monotonic() - capture_started
    if capture_seconds > 60:
        raise TimeoutError('Separate finite warmup/capture budget exhausted')
    pilot_started = time.monotonic()
    train_images = torch.frombuffer(images, dtype=torch.uint8).reshape(60000, 1, 28, 28).to('cuda', torch.float32).div_(255)
    train_labels = torch.frombuffer(labels, dtype=torch.uint8).to('cuda', torch.int64)
    import torch.nn.functional as f
    original = f.interpolate(train_images, size=(29, 29), mode='bilinear', align_corners=True)
    del train_images
    augmented = torch.empty_like(original)
    generation_started = time.monotonic()
    for block, first in enumerate(range(0, 60000, 256)):
        if args.stop_file.exists():
            raise RuntimeError('Operator stopped before training')
        if time.monotonic() - pilot_started >= args.seconds:
            raise TimeoutError('Finite pilot deadline exhausted during deformation')
        augmented[first:first+256] = augment(torch, original[first:first+256], train_labels[first:first+256], cfg.seed + 1000003*epoch + block)
    augmented.mul_(2).sub_(1)
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - generation_started
    order = torch.tensor(epoch_order(cfg.seed, epoch), dtype=torch.int64, device='cuda')
    target = torch.full((60000, 10), -1., device='cuda')
    target.scatter_(1, train_labels[:, None], 1.)
    initial_updates = updates
    max_updates = min(args.updates, 60000 - cursor)
    kernel_ms, checkpoints = 0., []
    started = time.monotonic()
    reason = 'update-budget'
    while updates - initial_updates < max_updates:
        if args.stop_file.exists():
            reason = 'operator-stop'
            break
        if time.monotonic() - pilot_started >= args.seconds:
            reason = 'time-budget'
            break
        selected = order[cursor:cursor+32]
        xs = augmented[selected].reshape(32, -1)
        ys = target[selected]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.replay(xs, ys, cfg.learning_rate(epoch))
        end.record()
        end.synchronize()
        kernel_ms += start.elapsed_time(end)
        cursor += 32
        updates += 32
    elapsed = time.monotonic() - started
    if not model.finite():
        raise RuntimeError('Nonfinite actual model state')
    # Avoid labeling the post-update boundary as an epoch validated checkpoint:
    # the pilot has no model-selection or final-test authority.
    if cursor == 60000:
        epoch, cursor = epoch + 1, 0
    meta = {**binding, 'epoch': epoch, 'cursor': cursor, 'updates': updates,
            'learningRate': cfg.learning_rate(min(epoch, cfg.epochs-1)),
            'validationPending': True, 'testEvaluations': 0,
            'epochOrderSHA256': hashlib.sha256(json.dumps(epoch_order(cfg.seed, epoch), separators=(',', ':')).encode()).hexdigest(),
            'mutableRNGState': None, 'rngExplanation': 'All random generation is private and addressed by fixed seed/epoch/block; no random draws occur between update checkpoints.'}
    saved = time.monotonic()
    checkpoint = save_checkpoint(model, args.output / 'checkpoints', meta)
    checkpoint_seconds = time.monotonic() - saved
    result = {'pilotOnly': True, 'completePaperRun': False, 'config': asdict(cfg),
              'parameters': cfg.parameters, **binding, 'updates': updates-initial_updates,
              'matrixWeights': cfg.matrix_weights, 'biasParameters': cfg.bias_parameters,
              'totalUpdates': updates, 'stopReason': reason, 'wallSeconds': elapsed,
              'totalPostCaptureSeconds': time.monotonic()-pilot_started,
              'cudaGraphElapsedMilliseconds': kernel_ms,
              'kernelActiveMilliseconds': None,
              'kernelTimingLimitation': 'CUDA stream event elapsed time is not an active-kernel duty-time measurement; a separately validated profiler trace is required for that metric.',
              'captureSeconds': capture_seconds, 'deformationSeconds': generation_seconds,
              'checkpointSeconds': checkpoint_seconds, 'checkpoint': checkpoint.name,
              'peakAllocatedBytes': torch.cuda.max_memory_allocated(),
              'peakReservedBytes': torch.cuda.max_memory_reserved(), 'testEvaluations': 0}
    with (args.output / 'PILOT.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
