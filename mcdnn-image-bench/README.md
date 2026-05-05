# mcdnn-image-bench

Cireşan, Meier, Schmidhuber, *Multi-column deep neural networks for image classification*, CVPR 2012.

## Problem

Deep CNN columns trained on differently-preprocessed inputs and averaged. MNIST, NIST SD 19, CASIA Chinese characters, GTSRB, CIFAR-10, NORB.

## What it demonstrates

MNIST 0.23% (35-net MCDNN). GTSRB 0.54% test error (vs 1.16% human). CASIA Chinese 6.5%/5.61%. The 'sweep all benchmarks' methodological pattern.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
