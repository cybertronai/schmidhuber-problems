# chunker-22-symbol

Schmidhuber, *Neural sequence chunkers* (TR FKI-148-91); *Learning complex extended sequences using the principle of history compression*, Neural Computation 4(2):234–242 (1992).

## Problem

22-symbol alphabet {a, x, b1, …, b20}. Two possible input sequences: a·b1…b20 and x·b1…b20, presented in random order with no episode boundaries. Network must (a) predict next input and (b) at the *21st* symbol, output 1 if the prefix was 'a', else 0 — a 20-step lag.

## What it demonstrates

Multi-level hierarchy of recurrent predictors with history compression. Conventional RTRL/BPTT fails after 1M sequences; chunker solves 13/17 runs in <5000 sequences.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
