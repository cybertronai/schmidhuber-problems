# iam-handwriting

Graves, Liwicki, Fernández, Bertolami, Bunke, Schmidhuber, *A novel connectionist system for unconstrained handwriting recognition*, IEEE TPAMI 31(5).

## Problem

(a) IAM-OnDB online whiteboard handwriting: train 5,364 lines; test 3,859 lines; 25 features per pen-coordinate sample. (b) IAM-DB offline scanned forms: train 6,161 lines; test 2,781 lines; 9 sliding-window features per pixel-column.

## What it demonstrates

BLSTM + CTC + token-passing decoder against 20K-word dictionary + bigram LM. Online word accuracy 79.7% (vs HMM 65.0%); offline 74.1% (vs HMM 64.5%). Won ICDAR 2009 in Arabic, French, Farsi.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
