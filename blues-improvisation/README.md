# blues-improvisation

Eck & Schmidhuber, *Finding temporal structure in music: blues improvisation with LSTM*, NNSP 2002.

## Problem

12-bar bebop blues. Standard chord progression (C7, F7, C7, …) over 12 bars; time-step = one eighth note (96 steps per chorus). Multi-hot pitch vectors. Two experiments: chord-only and chord + improvised pentatonic melody.

## What it demonstrates

After training, the network is run free-running to compose new blues choruses.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
