# rs-parity

Hochreiter & Schmidhuber 1996 (same paper).

## Problem

Sequence of ±1 over 500–600 timesteps; classify by parity of count.

## What it demonstrates

RS A2 without self-connections solves in 250 trials. Bengio's simulated annealing needed ~810,000 trials. Same punch line as `rs-two-sequence`.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
