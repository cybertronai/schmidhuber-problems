# pole-balance-markov-vac

Schmidhuber, *Recurrent networks adjusted by adaptive critics*, IJCNN 1990 Washington DC.

## Problem

Markovian (full-state) cart-pole balancing using a vector-valued adaptive critic and a recurrent controller; critic predictions fed back into the controller.

## What it demonstrates

Introduces vector-valued adaptive critics — precursor to general value functions. Companion baseline to the non-Markovian variant.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
