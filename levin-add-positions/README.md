# levin-add-positions

Schmidhuber 1995/1997 (same paper).

## Problem

Linear unit with 100 binary inputs; target = sum of indices of 'on' bits. Optimal weight vector: w_i = i (a ramp).

## What it demonstrates

Levin search discovers a length-8 program (ALLOCATE; INCREMENT; OUTPUT; JUMP) that produces the ramp. Standard backprop cannot generalize from sparse data.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
