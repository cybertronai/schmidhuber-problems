# two-sequence-noise

Hochreiter & Schmidhuber 1997, Experiment 3.

## Problem

Three subtasks. 3a: Bengio-94 setup (signal + noise on same channel). 3b: Gaussian noise on information-carrying elements too. 3c: targets 0.2/0.8 with target noise σ=0.32.

## What it demonstrates

LSTM with 102 weights, output-gate biases −2/−4/−6. T=100: 27k sequences (3a) → 269k (3c). The paper concedes RS solves 3a faster than any gradient method.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
