# continual-embedded-reber

Gers, Schmidhuber, Cummins, *Learning to forget: continual prediction with LSTM*, Neural Computation 12(10).

## Problem

Original Embedded Reber strings concatenated without resets; continual versions of the noisy distractor sequences from 1997.

## What it demonstrates

Introduces the **forget gate** (giving 'Vanilla LSTM'). Standard LSTM fails as state grows unboundedly; LSTM-with-forget-gate solves them.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
