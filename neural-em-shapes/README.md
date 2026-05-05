# neural-em-shapes

Greff, van Steenkiste, Schmidhuber, *Neural Expectation Maximization*, NIPS 2017.

## Problem

(a) Static shapes — 28×28 binary, 3 random shapes ▲◆●. (b) Flying shapes — T=5 video at 28×28. (c) Flying MNIST — 24×24 grey, 2 down-sampled MNIST digits.

## What it demonstrates

N-EM AMI ≈ 0.96 on static shapes, beating Tagger. Unsupervised perceptual grouping via a differentiable EM.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
