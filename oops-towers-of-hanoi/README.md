# oops-towers-of-hanoi

Schmidhuber, *Optimal Ordered Problem Solver*, TR IDSIA-12-02; Machine Learning 54:211–254 (2004).

## Problem

Universal solver for Towers of Hanoi with arbitrary n disks; demonstrated up to n=30 (minimal solution length 2³⁰−1 ≈ 10⁹). After learning a context-free symmetry/palindrome task, OOPS reuses prefixes that invoke recursion.

## What it demonstrates

Bias-optimal incremental Levin search; ~1000× speedup over non-incremental baselines via prefix freezing.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
