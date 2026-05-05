# curiosity-three-regions

Schmidhuber, *Adaptive confidence and adaptive curiosity*, TR FKI-149-91; *Curious model-building control systems*, IJCNN 1991, vol. 2, pp. 1458–1463.

## Problem

Discrete environment with three classes of state: deterministically predictable, intrinsically random/unlearnable, and learnable-but-not-yet-learned. Intrinsic reward = improvement of model accuracy.

## What it demonstrates

The canonical 'no joy in pure noise, no joy in pure knowledge' demonstration. Curious agent converges on the third class. Seed of artificial curiosity / GAN-like minimax.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
