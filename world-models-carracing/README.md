# world-models-carracing

Ha & Schmidhuber, *Recurrent World Models Facilitate Policy Evolution*, NeurIPS 2018.

## Problem

OpenAI Gym CarRacing-v0: 64×64×3 RGB; 3 continuous actions (steer, gas, brake); reward = tiles visited per second. V (Convolutional VAE, z∈ℝ³²) + M (MDN-LSTM, 5-mixture, 256 hidden) + C (linear, 867 params, evolved with CMA-ES).

## What it demonstrates

First reported solution: 906 ± 21 (vs DQN 343, A3C 591, leaderboard best 838). 'Solved' threshold: ≥900 over 100 trials.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
