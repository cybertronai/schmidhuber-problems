# world-models-vizdoom-dream

Ha & Schmidhuber 2018 (same paper).

## Problem

VizDoom Take Cover-v0. Controller trained *entirely inside the dream DoomRNN* at temperature τ=1.15; transferred zero-shot to actual VizDoom.

## What it demonstrates

Best agent: 1,092 ± 556 in real VizDoom (vs 750 'solved' threshold; leaderboard 820±58). The original 'training in a dream' demonstration.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
