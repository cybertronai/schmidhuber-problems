# hq-learning-pomdp

Wiering & Schmidhuber, *HQ-learning*, Adaptive Behavior 6(2):219–246.

## Problem

Hierarchical Q(λ) decomposing a POMDP into an ordered sequence of subtasks. Each subtask solved by a memoryless reactive subagent; an HQ-table assigns subgoals to subagents. POM with 28-step optimal solution requiring ≥3 reactive subagents.

## What it demonstrates

Memoryless agents + subgoal hierarchy outperform recurrent Q-learning on long-horizon partially-observable mazes.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
