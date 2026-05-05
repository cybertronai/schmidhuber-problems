# subgoal-obstacle-avoidance

Schmidhuber, *Learning to generate sub-goals for action sequences*, ICANN-91, pp. 967–972.

## Problem

2-D continuous obstacle avoidance: a point agent must navigate from a start to a goal around obstacles. A subgoal-generator RNN emits intermediate way-points; an evaluator predicts cost-to-go. Cost = sum of segment costs.

## What it demonstrates

Canonical end-to-end gradient-based hierarchical-RL task. Extended in Schmidhuber & Wahnsiedler 1993 with point-mass navigation around simple obstacle layouts.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
