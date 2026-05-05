# timit-blstm-ctc

Graves & Schmidhuber, *Framewise phoneme classification with bidirectional LSTM*, Neural Networks 18 (2005); Graves et al., *Connectionist Temporal Classification*, ICML 2006.

## Problem

TIMIT corpus: 462-speaker training, 50-speaker validation, 24-speaker core test; 61 phonemes folded to 39. Features: 39 MFCC-style features per 10-ms frame.

## What it demonstrates

Bidirectional LSTM beats uni-LSTM, BRNN, and time-windowed MLPs. CTC introduces the 'blank' output unit and a forward-backward decoder.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
