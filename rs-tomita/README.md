# rs-tomita

Hochreiter & Schmidhuber 1996 (same paper).

## Problem

Tomita grammars #1, #2, #4 (Miller & Giles 1993) recast as long-time-lag benchmarks.

## What it demonstrates

RS solves Tomita #1 in 182/288 trials, #2 in 1,511/17,953, #4 in 13,833/35,610. The 'long-time-lag' benchmarks are again attackable by random search.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
