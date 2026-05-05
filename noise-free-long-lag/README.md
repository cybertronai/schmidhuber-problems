# noise-free-long-lag

Hochreiter & Schmidhuber 1997, Experiment 2.

## Problem

Three sub-variants. (a) Two locally-encoded sequences (y, a₁,…,a_{p−1}, y) and (x, a₁,…,a_{p−1}, x); predict every next symbol. (b) Distractor block randomized — no local regularities. (c) Long lags + many distractors, the hardest.

## What it demonstrates

p=100: BPTT/RTRL → 0% success; chunker → 33%; LSTM → 100% in 5,040 sequences. (q=1000, p=1000): 49,000 sequences. No other algorithm at the time solved q ≥ 10.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
