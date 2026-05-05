# pipe-symbolic-regression

Salustowicz & Schmidhuber, *Probabilistic Incremental Program Evolution*, Evolutionary Computation 5(2):123–141.

## Problem

Symbolic regression of f(x) = x⁴ + x³ + x² + x — Koza's classic GP benchmark. 20 fitness cases; terminals/functions {+, −, ×, ÷, x}.

## What it demonstrates

Population-based incremental learning over a probabilistic prototype tree. No crossover. Matches or outperforms standard Koza GP.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
