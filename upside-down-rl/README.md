# upside-down-rl

Schmidhuber & Srivastava et al., *Reinforcement Learning Upside Down*, arXiv 1912.02875 / 1912.02877.

## Problem

RL framed as supervised learning: feed desired return (and horizon) as input commands. Tasks: LunarLander-v2; LunarLanderSparse-v2 (rewards delayed to last step); VizDoom TakeCover-v0.

## What it demonstrates

On LunarLanderSparse-v2, A2C/DQN/LSTM-DQN fail; UDRL still trains and scores well.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
