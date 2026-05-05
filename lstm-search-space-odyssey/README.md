# lstm-search-space-odyssey

Greff, Srivastava, Koutník, Steunebrink, Schmidhuber, *LSTM: A search space odyssey*, IEEE TNNLS.

## Problem

8 LSTM variants × 3 tasks × random search ≈ 5,400 experiments, ~15 CPU-years. Tasks: TIMIT (frame-level); IAM Online Handwriting; JSB Chorales (382 4-part Bach chorales, 88-binary polyphonic next-step prediction).

## What it demonstrates

fANOVA-analyzed grid. Forget gate and output activation are critical; momentum and peepholes unimportant; CIFG and NP simplify without hurting.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
