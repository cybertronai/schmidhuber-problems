# nbb-xor

Schmidhuber, *A local learning algorithm for dynamic feedforward and recurrent networks*, Connection Science 1(4):403–412.

## Problem

XOR via the Neural Bucket Brigade — strictly local-in-space-and-time, winner-take-all dissipative learning rule. 3-input retina (incl. bias) → 3 hidden → 2 output (one for XOR=1, one for XOR=0). Each pattern presented for 6 ticks; activations reset every cycle.

## What it demonstrates

A strictly local rule (no BP, no RTRL) can solve a non-linearly-separable static task. Average ~619 pattern presentations across 20 runs.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
