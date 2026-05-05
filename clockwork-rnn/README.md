# clockwork-rnn

Koutník, Greff, Gomez, Schmidhuber, *A clockwork RNN*, ICML 2014.

## Problem

Hidden layer partitioned into modules at clock rates 1, 2, 4, 8, …; slower modules connect down to faster modules. Audio waveform generation (320-sample), TIMIT spoken-word classification (raw audio), online handwriting.

## What it demonstrates

CW-RNN beats SRN and LSTM at matched parameter count on these tasks.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
