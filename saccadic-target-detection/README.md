# saccadic-target-detection

Schmidhuber & Huber, *Learning to generate focus trajectories for attentive vision*, TR FKI-128-90, April 1990.

## Problem

Controller and world-model networks learn to shift a 'fovea' over a 2-D scene to detect a target. Inputs are local foveated patches; outputs are shift commands.

## What it demonstrates

Differentiable attention via controller+model. Compared explicitly to Nguyen-Widrow's truck-backer-upper.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
