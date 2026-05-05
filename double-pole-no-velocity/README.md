# double-pole-no-velocity

Gomez & Schmidhuber, *Co-evolving recurrent neurons learn deep memory POMDPs*, GECCO 2005.

## Problem

Cart with two stacked poles; observe positions only (no velocities). Canonical hard non-Markov RL benchmark.

## What it demonstrates

Cooperative co-evolution of individual neurons. The 2005 ICANN companion applies the same family to a 3-wheeled robot balancing two stacked poles in 3D physics.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
