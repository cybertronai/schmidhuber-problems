# embedded-reber

Hochreiter & Schmidhuber 1997 (NC), Experiment 1.

## Problem

Reber grammar embedded in an outer 'B[T|P]…[T|P]E' frame; alphabet {B, T, P, S, X, V, E}. Predict the set of possible next symbols at each step. Predicting the second-to-last symbol requires remembering the second symbol of the string.

## What it demonstrates

Short-lag (~9 step) benchmark. LSTM (4×1 or 3×2 cell blocks) solves 148/150 trials at mean 8,440 sequences. Output gates protect short-lag predictions from long-lag interference.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
