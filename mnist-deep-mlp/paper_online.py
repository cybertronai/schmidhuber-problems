"""Small offline oracle and frozen v1.5 online-BP protocol (no imports/downloads).

Original NumPy laptop entry point is unchanged. See PAPER_PROTOCOL.md for the
paper's stated settings and explicitly reconstructed missing details.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class PaperConfig:
    seed: int = 0
    side: int = 29
    hidden: tuple[int, ...] = (2500, 2000, 1500, 1000, 500)
    outputs: int = 10
    epochs: int = 800
    examples: int = 60000
    initial_lr: float = 1e-3
    final_lr: float = 1e-6
    activation_a: float = 1.7159
    activation_b: float = 0.6666
    sigma: float = 5.5
    alpha: float = 37.0
    scale_percent: float = 17.5
    graph_chunk: int = 32
    checkpoint_updates: int = 6000

    @property
    def sizes(self):
        return (self.side * self.side, *self.hidden, self.outputs)

    @property
    def parameters(self):
        return sum((a + 1) * b for a, b in zip(self.sizes, self.sizes[1:]))

    @property
    def matrix_weights(self):
        return sum(a * b for a, b in zip(self.sizes, self.sizes[1:]))

    @property
    def bias_parameters(self):
        return sum(self.sizes[1:])

    def learning_rate(self, epoch):
        if not 0 <= epoch < self.epochs or self.epochs < 2:
            raise ValueError('Epoch outside frozen schedule')
        return self.initial_lr * (self.final_lr / self.initial_lr) ** (epoch / (self.epochs - 1))

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Nonnegative integer seed required')
        if (self.side, self.hidden, self.outputs, self.epochs, self.examples) != (29, (2500, 2000, 1500, 1000, 500), 10, 800, 60000):
            raise ValueError('Only the frozen paper-scale arm is supported')
        if self.examples % self.graph_chunk or self.graph_chunk != 32 or self.checkpoint_updates < self.graph_chunk:
            raise ValueError('Sequential graph/checkpoint bounds')
        if not all(math.isfinite(x) and x > 0 for x in (self.initial_lr, self.final_lr, self.activation_a, self.activation_b, self.sigma, self.alpha, self.scale_percent)):
            raise ValueError('Finite positive protocol constants required')
        # Do not let a changed constant retain the description of the frozen arm.
        if asdict(self) != asdict(PaperConfig(seed=self.seed)):
            raise ValueError('Scientific settings differ from the frozen arm')


def targets(label, count=10):
    if not isinstance(label, int) or not 0 <= label < count:
        raise ValueError('Label outside output domain')
    return [1.0 if j == label else -1.0 for j in range(count)]


class ReferenceMLP:
    """Tiny float64 scalar oracle; deliberately not a paper-scale CPU trainer."""
    def __init__(self, sizes, seed=0):
        if len(sizes) < 2 or any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError('Positive integer layer sizes required')
        if sum(a * b for a, b in zip(sizes, sizes[1:])) > 10000:
            raise ValueError('Offline reference is bounded to tiny fixtures')
        rng = random.Random(seed)
        self.weights = [[[rng.uniform(-.05, .05) for _ in range(a)] for _ in range(b)] for a, b in zip(sizes, sizes[1:])]
        self.biases = [[rng.uniform(-.05, .05) for _ in range(b)] for b in sizes[1:]]

    def forward(self, x):
        if len(x) != len(self.weights[0][0]) or not all(math.isfinite(v) for v in x):
            raise ValueError('Finite input vector with exact width required')
        acts = [list(x)]
        for w, b in zip(self.weights, self.biases):
            acts.append([1.7159 * math.tanh(.6666 * (sum(v * h for v, h in zip(row, acts[-1])) + bias)) for row, bias in zip(w, b)])
        return acts

    def loss(self, x, y):
        a = self.forward(x)[-1]
        if len(y) != len(a) or not all(math.isfinite(v) for v in y):
            raise ValueError('Finite target vector with exact width required')
        return .5 * sum((v - t) ** 2 for v, t in zip(a, y))

    def gradients(self, x, y):
        acts = self.forward(x)
        if len(y) != len(acts[-1]) or not all(math.isfinite(v) for v in y):
            raise ValueError('Finite target vector with exact width required')
        derivative = lambda v: 1.7159 * .6666 * (1 - (v / 1.7159) ** 2)
        deltas = [None] * len(self.weights)
        deltas[-1] = [(v - t) * derivative(v) for v, t in zip(acts[-1], y)]
        # Compute *all* deltas using old weights before any weight is changed.
        for layer in range(len(self.weights) - 2, -1, -1):
            deltas[layer] = [sum(self.weights[layer + 1][k][j] * d for k, d in enumerate(deltas[layer + 1])) * derivative(v) for j, v in enumerate(acts[layer + 1])]
        return [[[d * a for a in acts[layer]] for d in delta] for layer, delta in enumerate(deltas)], deltas

    def step(self, x, y, lr):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('Finite positive learning rate required')
        gw, gb = self.gradients(x, y)
        for w, b, dw, db in zip(self.weights, self.biases, gw, gb):
            for row, grad in zip(w, dw):
                for j, value in enumerate(grad):
                    row[j] -= lr * value
            for j, value in enumerate(db):
                b[j] -= lr * value


def bilinear(image, x, y):
    """Scalar zero-background bilinear oracle; coordinates are pixel centers."""
    if not image or not image[0] or any(len(row) != len(image[0]) for row in image):
        raise ValueError('Nonempty rectangular image required')
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('Finite source coordinates required')
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    pixel = lambda i, j: image[j][i] if 0 <= j < len(image) and 0 <= i < len(image[0]) else 0.0
    return ((1-fx)*(1-fy)*pixel(x0,y0) + fx*(1-fy)*pixel(x0+1,y0)
            + (1-fx)*fy*pixel(x0,y0+1) + fx*fy*pixel(x0+1,y0+1))


def epoch_order(seed, epoch, count=60000):
    if min(seed, epoch) < 0 or count < 1:
        raise ValueError('Nonnegative seed/epoch and positive sample count required')
    order = list(range(count))
    random.Random((seed << 32) + epoch).shuffle(order)
    return order


def smoke():
    model = ReferenceMLP([4, 3, 2], 0)
    x, y = [.1, -.2, .3, -.4], targets(1, 2)
    before = model.loss(x, y)
    for _ in range(8):
        model.step(x, y, .01)
    after = model.loss(x, y)
    if not math.isfinite(after) or not after < before:
        raise RuntimeError('Tiny online update failed')
    return {'offlineSmokePassed': True, 'updates': 8, 'before': before, 'after': after, 'gpu': False, 'datasetLoaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(smoke(), sort_keys=True))
