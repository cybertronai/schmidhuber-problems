# levin-count-inputs

Schmidhuber, *Discovering solutions with low Kolmogorov complexity and high generalization capability*, ICML 1995; Neural Networks 10(5):857–873 (1997).

## Problem

Linear unit with 100 binary inputs; target = number of 'on' bits. Optimal weight vector: w_i = 1 ∀i. Only 3 training examples — gradient descent fails.

## What it demonstrates

Probabilistic Levin search over self-sizing programs in a 13-instruction assembler. Discovers a length-4 program that outputs all-ones weights and generalizes perfectly.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
