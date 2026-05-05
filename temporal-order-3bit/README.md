# temporal-order-3bit

Hochreiter & Schmidhuber 1997, Experiment 6a.

## Problem

Sequences begin E, end B; in between, random {a,b,c,d} with two embedded {X, Y} at t1∈[10,20], t2∈[50,60]; length [100,110]. Four classes encoding (X,X)(X,Y)(Y,X)(Y,Y).

## What it demonstrates

LSTM 2 cell blocks of size 2, 156 weights. 31,390 sequences. Typical solutions: sign of internal state encodes first X/Y; input gate opens conditional on whether cell is empty.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
