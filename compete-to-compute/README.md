# compete-to-compute

Srivastava, Masci, Kazerounian, Gomez, Schmidhuber, *Compete to compute*, NIPS 2013.

## Problem

Local Winner-Take-All blocks (size 2; only winner forwards). Permutation-invariant MNIST, CIFAR-10, plus a sequential-task **catastrophic-forgetting benchmark** (MNIST → second-task switch).

## What it demonstrates

LWTA preserves performance on the first task vs ReLU/sigmoid baselines that catastrophically forget.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
