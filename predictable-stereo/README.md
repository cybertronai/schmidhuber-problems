# predictable-stereo

Schmidhuber & Prelinger, *Discovering predictable classifications*, Neural Computation 5(4):625–635 (TR CU-CS-626-92).

## Problem

Becker–Hinton-style binary stereo task: a pair of binary 'stereo' image patches with shifted dot patterns; the class is the disparity (shift). Modules learn to classify so that labels are predictable from neighboring modules' outputs.

## What it demonstrates

Predictability *maximization* (counterpoint to PM minimization). Direct point of contact with the Hinton lineage — Schmidhuber takes a Becker-Hinton benchmark and runs it on his own architecture.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
