# pole-balance-non-markov

Schmidhuber 1990 (same paper).

## Problem

Cart-pole balancing with a hand-coded *perfect differentiable model* M. The non-Markovian twist: only x and θ are visible — temporal derivatives (dx/dt, dθ/dt) are hidden. Recurrent controller C must infer velocities from history. Failure when |θ| > 0.21 rad or |x| > 2.4 m.

## What it demonstrates

Non-Markovian variant of Barto-Sutton-Anderson — partial observability forces representation learning. 17/20 runs achieve >1000-step survival within a few hundred trials.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
