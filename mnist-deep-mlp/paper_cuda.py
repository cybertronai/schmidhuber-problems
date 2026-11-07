"""Optional v1.5 FP32 online kernels; importing this module never imports torch.

This is not a runtime supervisor or permission to use a GPU. A separately
reviewed, source-bound pilot must establish numerical parity and resource bounds.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time


class OnlineMLP:
    def __init__(self, sizes, seed=0, device='cpu', reference=None):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.sizes = tuple(sizes)
        self.weights, self.biases = [], []
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            if reference is None:
                w = torch.empty((b, a), dtype=torch.float32).uniform_(-.05, .05, generator=generator)
                bias = torch.empty(b, dtype=torch.float32).uniform_(-.05, .05, generator=generator)
            else:
                w = torch.tensor(reference.weights[i], dtype=torch.float32)
                bias = torch.tensor(reference.biases[i], dtype=torch.float32)
            self.weights.append(w.to(self.device))
            self.biases.append(bias.to(self.device))
        self.acts = [torch.empty(n, device=self.device) for n in sizes]
        self.pre = [torch.empty(n, device=self.device) for n in sizes[1:]]
        self.deltas = [torch.empty(n, device=self.device) for n in sizes[1:]]
        self.derivatives = [torch.empty(n, device=self.device) for n in sizes[1:]]
        self.steps = [torch.empty(n, device=self.device) for n in sizes[1:]]
        self.lr = torch.zeros((), dtype=torch.float32, device=self.device)
        self.graph = None
        self.buffer_x = self.buffer_y = None
        self.chunk = None

    def _forward(self, x):
        t = self.torch
        self.acts[0].copy_(x)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            t.mv(w, self.acts[i], out=self.pre[i])
            self.pre[i].add_(b).mul_(.6666)
            t.tanh(self.pre[i], out=self.acts[i+1])
            self.acts[i+1].mul_(1.7159)

    def _step(self, x, target):
        t = self.torch
        self._forward(x)
        for derivative, activation in zip(self.derivatives, self.acts[1:]):
            t.div(activation, 1.7159, out=derivative)
            derivative.square_().neg_().add_(1).mul_(1.7159 * .6666)
        t.sub(self.acts[-1], target, out=self.deltas[-1])
        self.deltas[-1].mul_(self.derivatives[-1])
        for i in range(len(self.weights)-2, -1, -1):
            t.mv(self.weights[i+1].t(), self.deltas[i+1], out=self.deltas[i])
            self.deltas[i].mul_(self.derivatives[i])
        # Rank-one update after *all* backward deltas: no minibatch reduction,
        # momentum, weight decay, optimizer state, TF32 or mixed precision.
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            t.mul(self.deltas[i], self.lr, out=self.steps[i])
            w.addcmul_(self.steps[i][:, None], self.acts[i][None, :], value=-1.)
            b.sub_(self.steps[i])

    def online(self, xs, targets, lr):
        t = self.torch
        if xs.ndim != 2 or targets.shape != (len(xs), self.sizes[-1]) or xs.shape[1] != self.sizes[0]:
            raise ValueError('Exact input and target shapes required')
        if not 0 < lr <= 1 or not t.isfinite(xs).all().item() or not t.isfinite(targets).all().item():
            raise ValueError('Finite inputs/targets and bounded LR required')
        with t.no_grad():
            self.lr.fill_(lr)
            for x, target in zip(xs, targets):
                self._step(x, target)

    def capture(self, chunk=32):
        t = self.torch
        if self.device.type != 'cuda' or self.graph is not None or chunk != 32:
            raise ValueError('One fixed 32-example CUDA graph on CUDA required')
        self.chunk = chunk
        self.buffer_x = t.zeros((chunk, self.sizes[0]), device=self.device)
        self.buffer_y = t.zeros((chunk, self.sizes[-1]), device=self.device)
        originals = [x.clone() for x in [*self.weights, *self.biases]]
        # Warmup/capture must not consume training updates or alter initial state.
        self.lr.fill_(1e-3)
        stream = t.cuda.Stream(device=self.device)
        stream.wait_stream(t.cuda.current_stream(self.device))
        with t.no_grad(), t.cuda.stream(stream):
            for _ in range(3):
                self._step(self.buffer_x[0], self.buffer_y[0])
        t.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = t.cuda.CUDAGraph()
        with t.no_grad(), t.cuda.graph(self.graph):
            for i in range(chunk):
                self._step(self.buffer_x[i], self.buffer_y[i])
        with t.no_grad():
            for dst, original in zip([*self.weights, *self.biases], originals):
                dst.copy_(original)
        t.cuda.synchronize(self.device)

    def replay(self, xs, targets, lr):
        if self.graph is None or xs.shape != self.buffer_x.shape or targets.shape != self.buffer_y.shape:
            raise ValueError('Exact captured block required')
        if not 0 < lr <= 1:
            raise ValueError('Bounded positive learning rate required')
        self.buffer_x.copy_(xs)
        self.buffer_y.copy_(targets)
        self.lr.fill_(lr)
        self.graph.replay()

    def finite(self):
        return all(self.torch.isfinite(v).all().item() for v in [*self.weights, *self.biases])


def configure_cuda(torch):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Reviewed deterministic cuBLAS environment required before import')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One isolated actual CUDA device required')
    # Allocation ceiling is an additional bound, not a claim about total driver VRAM.
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0) != (12, 0) or 'sm_120' not in torch.cuda.get_arch_list():
        raise RuntimeError('Reviewed SM120 device and native backend support required')
    torch.cuda.set_per_process_memory_fraction(min(1., (4 * 1024**3) / props.total_memory), 0)
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': props.name,
            'capability': list(torch.cuda.get_device_capability(0)), 'dtype': 'float32',
            'tf32': False, 'deterministic': True}


def augment(torch, images, labels, seed, sigma=5.5, alpha=37., scale_percent=17.5):
    """Fixed paper-range recipe; affine choice/boundary conventions are disclosed.

    Images are raw [0,1], NCHW at29px. Caller regenerates the whole epoch once,
    in fixed256-image blocks; inputs are normalized to[-1,1] only after warping.
    """
    import math
    import torch.nn.functional as f
    if images.ndim != 4 or images.shape[1:] != (1, 29, 29) or len(labels) != len(images):
        raise ValueError('Exact resized MNIST image/label shapes required')
    device, n, side = images.device, len(images), images.shape[-1]
    generator = torch.Generator(device=device).manual_seed(seed)
    rand = lambda shape: torch.rand(shape, device=device, generator=generator)
    beta = torch.where((labels == 1) | (labels == 7), 7.5, 15.)
    angle = (rand((n,)) * 2 - 1) * beta * (math.pi / 180)
    sx = 1 + (rand((n,)) * 2 - 1) * scale_percent / 100
    sy = 1 + (rand((n,)) * 2 - 1) * scale_percent / 100
    rotation = rand((n,)) < .5
    theta = torch.zeros((n, 2, 3), dtype=torch.float32, device=device)
    # Inverse (source-sampling) affine transform around centered coordinates.
    theta[:, 0, 0] = torch.where(rotation, torch.cos(angle) / sx, 1 / sx)
    theta[:, 0, 1] = torch.where(rotation, torch.sin(angle) / sx, -torch.tan(angle) / sx)
    theta[:, 1, 0] = torch.where(rotation, -torch.sin(angle) / sy, torch.zeros_like(angle))
    theta[:, 1, 1] = torch.where(rotation, torch.cos(angle) / sy, 1 / sy)
    grid = f.affine_grid(theta, images.shape, align_corners=True)
    # Literal21x21 Gaussian, separable evaluation; zero field padding disclosed.
    axis = torch.arange(-10, 11, dtype=torch.float32, device=device)
    kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
    kernel /= kernel.sum()
    noise = rand((n * 2, 1, side, side)) * 2 - 1
    field = f.conv2d(noise, kernel.view(1, 1, 1, 21), padding=(0, 10))
    field = f.conv2d(field, kernel.view(1, 1, 21, 1), padding=(10, 0))
    field = field.reshape(n, 2, side, side).permute(0, 2, 3, 1)
    grid += field * (alpha * 2 / (side - 1))
    return sample_grid(images, grid)


def sample_grid(images, grid):
    import torch.nn.functional as f
    return f.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def save_checkpoint(model, root, metadata):
    """Immutable tensor payload + hashed JSON commit, then atomic latest pointer.

    No executable pickle objects are accepted on load; local tensor payload is
    bounded/hash-verified before weights_only deserialization.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('Owned ordinary checkpoint directory required')
    epoch, cursor, updates = (metadata[k] for k in ('epoch', 'cursor', 'updates'))
    if any(type(x) is not int or x < 0 for x in (epoch, cursor, updates)) or cursor >= 60000 or cursor % 32 or updates != epoch * 60000 + cursor:
        raise ValueError('Exact sequential checkpoint cursor required')
    name = f'checkpoint-{updates:09d}'
    final = root / name
    # Canonical JSON comparison also rejects nonfinite/unserializable metadata
    # before creating any payload. A committed checkpoint is never overwritten.
    canonical_metadata = json.dumps(metadata, sort_keys=True, allow_nan=False)
    if final.exists():
        value, state = _read_checkpoint(model, final)
        if json.dumps(value['metadata'], sort_keys=True, allow_nan=False) != canonical_metadata or not all(model.torch.equal(state[k], v.detach().cpu()) for k, v in _pairs(model)):
            raise ValueError('Existing committed checkpoint differs at the same update')
        _promote_pointer(root, final)
        return final
    # Old partial artifacts do not block a new publication and remain evidence.
    temp = Path(tempfile.mkdtemp(prefix=name + '.partial-', dir=root))
    payload = temp / 'tensors.pt'
    state = {f'w{i}': v.detach().cpu() for i, v in enumerate(model.weights)}
    state.update({f'b{i}': v.detach().cpu() for i, v in enumerate(model.biases)})
    with payload.open('xb') as file:
        model.torch.save(state, file)
        file.flush()
        os.fsync(file.fileno())
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    value = {'schemaVersion': 1, 'complete': True, 'metadata': metadata,
             'sizes': list(model.sizes), 'payloadBytes': payload.stat().st_size, 'payloadSHA256': digest}
    with (temp / 'COMMIT.json').open('x', encoding='utf-8') as file:
        json.dump(value, file, sort_keys=True, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    if final.exists():
        raise FileExistsError('Never replace an existing checkpoint')
    _publish_owned(temp, final, replace=False)
    _promote_pointer(root, final)
    return final


def _promote_pointer(root, final):
    descriptor, temporary = tempfile.mkstemp(prefix='LATEST.partial-', dir=root)
    pointer = Path(temporary)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
        json.dump({'checkpoint': final.name, 'commitSHA256': hashlib.sha256((final/'COMMIT.json').read_bytes()).hexdigest()}, file)
        file.flush()
        os.fsync(file.fileno())
    _publish_owned(pointer, root / 'LATEST.json', replace=True)


def _publish_owned(source, target, *, replace, clock=time.monotonic, pause=time.sleep):
    """Only this writer's unique temporary: finite Windows sharing retry.

    Does not reclaim a stale directory, overwrite a committed checkpoint, change
    permissions or infer another process owns a handle. Other errors propagate.
    """
    deadline = clock() + .5
    while True:
        try:
            (os.replace if replace else os.rename)(source, target)
            return
        except PermissionError as error:
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32) or (not replace and target.exists()) or clock() >= deadline:
                raise
            pause(.02)


def _pairs(model):
    return [(f'w{i}', v) for i, v in enumerate(model.weights)] + [(f'b{i}', v) for i, v in enumerate(model.biases)]


def _read_checkpoint(model, checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or {p.name for p in checkpoint.iterdir()} != {'COMMIT.json', 'tensors.pt'}:
        raise ValueError('Exact ordinary checkpoint content required')
    value = json.loads((checkpoint / 'COMMIT.json').read_text())
    if value.get('schemaVersion') != 1 or value.get('complete') is not True or value.get('sizes') != list(model.sizes):
        raise ValueError('Checkpoint schema/architecture differs')
    payload = checkpoint / 'tensors.pt'
    if payload.is_symlink() or payload.stat().st_size != value['payloadBytes'] or not 0 < value['payloadBytes'] <= 128 * 1024**2:
        raise ValueError('Bounded regular checkpoint payload required')
    data = payload.read_bytes()
    if len(data) != value['payloadBytes'] or hashlib.sha256(data).hexdigest() != value['payloadSHA256']:
        raise ValueError('Checkpoint tensor digest differs')
    state = model.torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    pairs = _pairs(model)
    if set(state) != {k for k, _ in pairs}:
        raise ValueError('Exact checkpoint tensor set required')
    # Validate *all* tensors before changing any live state.
    for key, target in pairs:
        tensor = state[key]
        if tensor.shape != target.shape or tensor.dtype != model.torch.float32 or not model.torch.isfinite(tensor).all().item():
            raise ValueError('Checkpoint tensor shape/dtype/finiteness differs')
    return value, state


def load_checkpoint(model, checkpoint, expected):
    value, state = _read_checkpoint(model, checkpoint)
    if any(value['metadata'].get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint scientific source/data/runtime binding differs')
    for key, target in _pairs(model):
        target.copy_(state[key])
    return value['metadata']
