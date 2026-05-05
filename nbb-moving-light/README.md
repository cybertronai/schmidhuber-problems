# nbb-moving-light

Schmidhuber 1989 (same paper) — *The neural bucket brigade*, in Pfeifer et al., *Connectionism in Perspective*, Elsevier, pp. 439–446.

## Problem

1-D moving-light direction discrimination via NBB. 5 input units + bias on a 1-D 'retina'; 2 competing recurrent output units encoding left→right vs right→left. Sequence length 5 ticks.

## What it demonstrates

Strictly local rule solves a simple temporal task with hidden recurrent units. ~223 cycles per sequence in 9/10 runs.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
