# anbn-anbncn

Gers & Schmidhuber, *LSTM recurrent networks learn simple context-free and context-sensitive languages*, IEEE TNN 12(6).

## Problem

(a) a^n b^n (CFL) and (b) a^n b^n c^n (CSL). Trained on n ∈ {1,…,10}; tested for generalization to much larger n. Sequences delivered as one-hot character streams; targets are next-symbol probabilities.

## What it demonstrates

First RNN result on a context-sensitive language. LSTM with peephole connections generalizes to n in the hundreds.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
