# relational-nem-bouncing-balls

van Steenkiste, Chang, Greff, Schmidhuber, *Relational Neural EM*, ICLR 2018.

## Problem

Bouncing-balls physics: 4 balls 64×64; variants with 2 ball types (one 6× heavier, 1.25× larger), with curtain-occlusion across the middle. Extrapolation to 6–8 balls when trained on 4.

## What it demonstrates

Adds object-pair interaction module to N-EM. Tracks objects through occlusion and generalizes to more objects than seen in training.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
