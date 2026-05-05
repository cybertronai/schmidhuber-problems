# em-segmentation-isbi

Cireşan, Giusti, Gambardella, Schmidhuber, *Deep neural networks segment neuronal membranes in electron microscopy images*, NIPS 2012.

## Problem

ISBI 2012 EM segmentation challenge: pixel-wise membrane vs. non-membrane on Drosophila ssTEM (512×512×30 stack at ~4 nm/pixel, 50 nm slices).

## What it demonstrates

Won the challenge; the only method outperforming a second human observer on pixel error.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
