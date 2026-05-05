# predictability-min-binary-factors

Schmidhuber, *Learning factorial codes by predictability minimization*, Neural Computation 4(6):863–879 (TR CU-CS-565-91).

## Problem

Small synthetic binary input distributions with known factorial structure. K representational units; each is paired with a predictor that tries to predict it from the others; each unit's encoder is trained to *minimize* its own predictability while preserving information.

## What it demonstrates

The proto-GAN: explicit adversarial framing (predictor vs code unit). Network rediscovers underlying factors. Bars-and-stripes / V1 filters arrive only in 1996.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
