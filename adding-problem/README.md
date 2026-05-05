# adding-problem

Hochreiter & Schmidhuber 1996 (NIPS 9); refined in Hochreiter & Schmidhuber 1997, *Long Short-Term Memory*, Neural Computation 9(8):1735–1780, Experiment 4.

## Problem

Each timestep input pair: (real ∈ [-1,1], marker ∈ {-1, 0, 1}). Two pairs marked with marker=1 (one in first 10, one in first T/2−1 unmarked); first & last markers = -1. Target at end: 0.5 + (X1+X2)/4.

## What it demonstrates

First non-trivial LSTM benchmark. T=100/lag=50: 74k sequences; T=1000/lag=500: 853k sequences. The de-facto evaluation for any RNN paper 1997–2010.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
