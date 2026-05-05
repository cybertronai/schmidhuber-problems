# mnist-deep-mlp

Cireşan, Meier, Gambardella, Schmidhuber, *Deep, big, simple neural nets excel on handwritten digit recognition*, Neural Computation 22(12).

## Problem

MNIST with plain GPU-accelerated MLPs (no convolution), 5 hidden layers, up to ~12M weights. Per-epoch on-the-fly affine + elastic deformations.

## What it demonstrates

MNIST 0.35% test error — best at the time. Plain MLPs + GPU + extensive deformation augmentation.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
