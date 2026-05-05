# fast-weights-unknown-delay

Schmidhuber, *Learning to control fast-weight memories: An alternative to dynamic recurrent networks*, Neural Computation 4(1):131–139.

## Problem

Two arbitrary input patterns must be associated across a time gap of *unknown* length. A slow programmer net S writes into the fast weights of a fast net F; reads back when triggered.

## What it demonstrates

The 1992 unnormalized linear Transformer / fast-weight programmer precursor. 'FROM/TO' would later be renamed KEY/VALUE.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
