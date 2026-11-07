"""Explicit native parity gates; run only in a separately granted task runtime.

No real MNIST, training result, test inference, download or automatic GPU fallback.
"""
import argparse
import copy
import json
import math
import platform
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_online import ReferenceMLP, bilinear, targets
from paper_cuda import OnlineMLP, augment, configure_cuda, load_checkpoint, sample_grid, save_checkpoint
import paper_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    import torch
    runtime = configure_cuda(torch) if args.device == 'cuda' else {'torch': torch.__version__, 'device': 'cpu'}
    if args.device == 'cpu':
        torch.set_num_threads(1)
    runtime.update(python=platform.python_version(),isolated=sys.flags.isolated,noSite=sys.flags.no_site,dontWriteBytecode=sys.dont_write_bytecode)
    checks = []

    def require(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def equal(a, b, atol=2e-6):
        return all(torch.allclose(x, y, rtol=1e-4, atol=atol) for x, y in zip([*a.weights, *a.biases], [*b.weights, *b.biases]))

    reference = ReferenceMLP([4, 5, 3], 7)
    native = OnlineMLP([4, 5, 3], device=args.device, reference=reference)
    xs = [[((i*7+j*3)%19-9)/10 for j in range(4)] for i in range(64)]
    ys = [targets(i % 3, 3) for i in range(64)]
    x = torch.tensor(xs, dtype=torch.float32, device=args.device)
    y = torch.tensor(ys, dtype=torch.float32, device=args.device)
    for a, b in zip(xs, ys):
        reference.step(a, b, .001)
    native.online(x, y, .001)
    require('64 sequential FP32 updates match independent scalar oracle',
            all(torch.allclose(v.cpu(), torch.tensor(expected, dtype=torch.float32), rtol=1e-4, atol=2e-6)
                for v, expected in zip([*native.weights, *native.biases], [*reference.weights, *reference.biases])))
    require('native output and all weights finite', native.finite())

    image = torch.arange(29*29, device=args.device, dtype=torch.float32).reshape(1, 1, 29, 29) / (29*29)
    coords = [((j + .5) % 32 - 1, (i + .25) % 32 - 1) for i in range(29) for j in range(29)]
    grid = torch.tensor([[(u/14-1, v/14-1) for u, v in coords]], device=args.device, dtype=torch.float32).reshape(1, 29, 29, 2)
    pixels = image[0, 0].cpu().tolist()
    expected = torch.tensor([bilinear(pixels, u, v) for u, v in coords], dtype=torch.float32).reshape(29, 29)
    actual = sample_grid(image, grid)[0, 0].cpu()
    require('841 bilinear warp coordinates incl zero background match scalar oracle', torch.allclose(actual, expected, atol=4e-6, rtol=2e-5))
    pair = image.repeat(2, 1, 1, 1)
    labels = torch.tensor([1, 4], device=args.device)
    state = torch.get_rng_state().clone()
    a, b = augment(torch, pair, labels, 11), augment(torch, pair, labels, 11)
    require('affine elastic generation finite and local-seed repeatable', torch.equal(a, b) and torch.isfinite(a).all())
    require('augmentation leaves global CPU RNG untouched', torch.equal(state, torch.get_rng_state()))

    base = ReferenceMLP([4, 5, 3], 7)
    uninterrupted = OnlineMLP([4, 5, 3], device=args.device, reference=base)
    split = OnlineMLP([4, 5, 3], device=args.device, reference=base)
    continuation = OnlineMLP([4, 5, 3], device=args.device, reference=base)
    if args.device == 'cuda':
        originals = [v.clone() for v in [*split.weights, *split.biases]]
        split.capture()
        require('graph warmup and capture preserve every original tensor', all(torch.equal(v, old) for v, old in zip([*split.weights, *split.biases], originals)))
        split.replay(x[:32], y[:32], .001)
        uninterrupted.capture()
        uninterrupted.replay(x[:32], y[:32], .001)
        uninterrupted.replay(x[32:], y[32:], .001)
        require('captured64 and eager64 updates agree', equal(uninterrupted, native))
    else:
        split.online(x[:32], y[:32], .001)
        uninterrupted.online(x, y, .001)

    args.scratch.mkdir(parents=True, exist_ok=True)
    binding = {'configSHA256': 'a'*64, 'sourceSHA256': 'b'*64, 'dataSHA256': 'c'*64, 'runtime': runtime}
    meta = {**binding, 'epoch': 0, 'cursor': 32, 'updates': 32, 'learningRate': .001,
            'rngScheme': 'closed deterministic fixture order', 'mutableRNGState': None}
    with tempfile.TemporaryDirectory(prefix='native-controls-', dir=args.scratch) as temp:
        root = Path(temp)
        (root / 'LATEST.partial').write_bytes(b'legacy interrupted pointer')
        (root / 'checkpoint-000000032.partial').mkdir()
        with mock.patch.object(paper_cuda.os, 'replace', side_effect=OSError('injected pointer-promotion interruption')):
            try:
                save_checkpoint(split, root, meta)
            except OSError:
                pass
            else:
                raise AssertionError('Pointer interruption was not exercised')
        committed = root / 'checkpoint-000000032'
        before = {p.name: p.read_bytes() for p in committed.iterdir()}
        saved = save_checkpoint(split, root, meta)
        require('interrupted pointer promotion reuses identical real tensor commit', before == {p.name: p.read_bytes() for p in saved.iterdir()})
        require('legacy partial artifacts remain intact without blocking publication', (root/'LATEST.partial').read_bytes() == b'legacy interrupted pointer' and (root/'checkpoint-000000032.partial').is_dir())
        require('genuine checkpoint metadata readback', load_checkpoint(continuation, saved, binding) == meta)
        if args.device == 'cuda':
            continuation.capture()
            continuation.replay(x[32:], y[32:], .001)
            torch.cuda.synchronize()
        else:
            continuation.online(x[32:], y[32:], .001)
        require('actual tensor checkpoint resume equals uninterrupted state bitwise', all(torch.equal(v, w) for v, w in zip([*continuation.weights, *continuation.biases], [*uninterrupted.weights, *uninterrupted.biases])))
        try:
            load_checkpoint(continuation, saved, {**binding, 'dataSHA256': 'd'*64})
        except ValueError:
            checks.append('substituted data binding refused')
        else:
            raise AssertionError('Data substitution accepted')
        payload = saved / 'tensors.pt'
        content = bytearray(payload.read_bytes())
        content[-1] ^= 1
        payload.write_bytes(content)
        try:
            load_checkpoint(continuation, saved, binding)
        except ValueError:
            checks.append('corrupted checkpoint refused before tensor load')
        else:
            raise AssertionError('Corrupt checkpoint accepted')
    print(json.dumps({'passed': True, 'runtime': runtime, 'checks': checks,
                      'actualMNISTLoaded': False, 'modelTestEvaluations': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
