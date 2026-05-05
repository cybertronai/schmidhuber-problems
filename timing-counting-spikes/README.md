# timing-counting-spikes

Gers, Schraudolph, Schmidhuber, *Learning precise timing with LSTM recurrent networks*, JMLR 3:115–143.

## Problem

Three task families: (a) Measure Spike Delays — input spike trains at intervals F+I(n), output integer offset I(n); (b) Generate Timed Spikes — reverse roles, network produces stable spike train; (c) Periodic Function Generation — sinusoid/triangle/rectangular targets, no input.

## What it demonstrates

Introduces **peephole connections**. Networks cannot solve GTS without peepholes. Stable generation for 1000 cycles, RMSE ~0.12–0.18.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
