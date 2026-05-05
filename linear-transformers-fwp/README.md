# linear-transformers-fwp

Schlag, Irie, Schmidhuber, *Linear Transformers are secretly fast weight programmers*, ICML 2021.

## Problem

Synthetic associative-retrieval / memorization (overflow-capacity); WikiText-103 LM (16-layer, ~44M params); WMT'14 EN→DE.

## What it demonstrates

Mathematically equates linearised self-attention with the 1991 fast-weight programmer. Adds delta-rule update + DPFP feature map.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
