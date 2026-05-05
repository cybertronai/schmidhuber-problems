# fast-weights-key-value

Schmidhuber 1992 (same paper).

## Problem

Adaptive variable binding / key-value retrieval: a 'key' and 'value' pattern arrive at unknown times; later the same key reappears and the system must output the bound value.

## What it demonstrates

Direct ancestor of modern key/value attention. Trained with truncated BPTT.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
