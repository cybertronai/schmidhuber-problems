# temporal-order-4bit

Hochreiter & Schmidhuber 1997, Experiment 6b.

## Problem

Three X/Y at t1, t2, t3 ∈ [10,20], [33,43], [66,76]; eight classes.

## What it demonstrates

LSTM 3 cell blocks of size 2, 308 weights. 571,100 sequences. The hardest LSTM-1997 benchmark.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
