# ssa-bias-transfer-mazes

Schmidhuber, Zhao, Wiering, *Shifting inductive bias with success-story algorithm, adaptive Levin search, and incremental self-improvement*, Machine Learning 28(1):105–130 (1997).

## Problem

Sequence of partially-observable grid worlds with increasing complexity. Limited local sensors (4-direction wall perception); fixed goal cell. Success-Story Algorithm (SSA) periodically undoes policy modifications not followed by reward acceleration.

## What it demonstrates

Knowledge accumulated from earlier mazes accelerates later ones via SSA. State spaces 'far bigger than most reported in the POE literature'.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
