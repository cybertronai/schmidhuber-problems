# pomdp-flag-maze

Schmidhuber, *Reinforcement learning in Markovian and non-Markovian environments*, NIPS-3 (1991), pp. 500–506.

## Problem

Small partially-observable maze (the 'flag' task). Hidden state means a non-recurrent agent fails; the recurrent model+controller pair disambiguates.

## What it demonstrates

Two interacting fully-recurrent continually-running networks (model + controller). Adaptive randomness; vector-valued adaptive critics.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
