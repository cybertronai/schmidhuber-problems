# evolino-sines-mackey-glass

Schmidhuber, Wierstra, Gomez, *Evolino*, IJCAI 2005 / 2007.

## Problem

(a) Multiple superimposed sine waves — sum of 2/3/4/5 sines with incommensurate frequencies. (b) Mackey-Glass time-series prediction with delay τ=17.

## What it demonstrates

Evolves nonlinear (LSTM) hidden-layer weights via Enforced SubPopulations; computes linear output mapping in closed form. ESN baseline cannot solve >2 sines; Evolino-LSTM learns up to 5.

## Files

| File | Purpose |
|---|---|
| `problem.py` | dataset + model + training stubs (raise `NotImplementedError`) |
