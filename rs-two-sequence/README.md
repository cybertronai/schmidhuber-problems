# rs-two-sequence

Hochreiter & Schmidhuber, *LSTM can solve hard long time lag problems*, NIPS 9, pp. 473–479.

## Problem

Bengio-94 latch problem: 1 real-valued input over 500–600 timesteps; only first element conveys class (+1 vs -1); rest is N(0, 0.04) noise; targets 0/1.

## What it demonstrates

Random weight guessing (RS) over weights init U[-100, 100] solves it in ~718 trials — much faster than Bengio's 6,400-trial multigrid search. Punch line: this 'long-time-lag' benchmark is actually trivial.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
