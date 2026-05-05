# neural-data-router

Csordás, Irie, Schmidhuber, *The Neural Data Router*, ICLR 2022.

## Problem

(a) Compositional table lookup — train depths ≤5, test depths ≤8. (b) Simple parenthesised modular arithmetic at deeper test trees. (c) ListOps depth-generalization variant.

## What it demonstrates

Adds copy gate + geometric attention to Transformer. 100% length generalization on compositional table lookup; near-perfect on deeper depths of simple arithmetic and ListOps.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
