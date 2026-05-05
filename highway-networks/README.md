# highway-networks

Srivastava, Greff, Schmidhuber, *Training very deep networks*, NIPS 2015.

## Problem

y = H(x)·T(x) + x·(1−T(x)); transform gate T sigmoid, biased −1 to −4. 10/20/50/100-layer FC nets on MNIST; conv highway 'C' on CIFAR-10/100.

## What it demonstrates

10/20/50/100-layer plain nets fail above ~20 layers; highway nets train cleanly. CIFAR-10 7.6%, CIFAR-100 32.24%.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
