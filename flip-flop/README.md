# flip-flop

Schmidhuber, *Making the world differentiable*, TR FKI-126-90 (revised Nov 1990); IJCNN 1990 San Diego, vol. 2, pp. 253–258.

## Problem

Controller C must behave like a flip-flop: output 1 whenever event 'B' occurs for the first time after the last 'A', else 0. Arbitrary time lags between A and B; no episode boundaries; only a scalar pain signal as feedback. 5 inputs (A, B, X, bias, pain), 1 probabilistic real-valued output. World-model M predicts the pain unit.

## What it demonstrates

First explicit synthetic latching benchmark — an LSTM-precursor task a full year before Hochreiter's vanishing-gradient analysis. Sequential regime: 6/10 solved; parallel regime: 20/30 within 10⁶ steps.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
