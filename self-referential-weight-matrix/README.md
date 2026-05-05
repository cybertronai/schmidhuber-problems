# self-referential-weight-matrix

Schmidhuber, *A self-referential weight matrix*, ICANN-93, pp. 446–451; *An introspective network that can learn to run its own weight change algorithm*, ICANN-93 Brighton.

## Problem

A recurrent net whose I/O channels include the ability to *read and write its own weight matrix encoded as activations*. The weight-change algorithm is itself learnable. A single small toy sequence-learning experiment as proof of concept.

## What it demonstrates

First proof of concept that a self-referential weight-update rule is learnable end-to-end. Re-explored in Irie et al. 2022's modern SRWM.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
