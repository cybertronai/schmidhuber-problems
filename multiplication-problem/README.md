# multiplication-problem

Hochreiter & Schmidhuber 1997, Experiment 5.

## Problem

Same setup as adding-problem, but real components ∈ [0,1] and target = X1 × X2.

## What it demonstrates

T=100/lag=50: 482k sequences, MSE 0.0223. Demonstrates LSTM is not biased only to integration tasks; handles multiplicative relationships.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
