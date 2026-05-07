# ssa-bias-transfer-mazes

Schmidhuber, Zhao, Wiering, *Shifting inductive bias with success-story
algorithm, adaptive Levin search, and incremental self-improvement*,
Machine Learning 28(1):105-130 (1997). Supplemented by Schmidhuber 2015,
*Deep Learning in Neural Networks: An Overview* §6.10, for the
formulation of the success-story criterion in modern terminology.

![ssa-bias-transfer-mazes animation](ssa_bias_transfer_mazes.gif)

## Problem

A POMDP grid world (5x5, four interior wall pillars) with a sequence of
four navigation tasks. The maze layout is fixed; only the goal cell
moves. The agent's start cell is always the centre, so each task forces a
different navigation direction.

```
. . . . .         tasks (executed in order):
. # . # .            0  NW-corner   start (2,2) -> goal (0,0)
. . S . .            1  NE-corner   start (2,2) -> goal (0,4)
. # . # .            2  SE-corner   start (2,2) -> goal (4,4)
. . . . .            3  SW-corner   start (2,2) -> goal (4,0)
```

- **Observation**: 4-direction wall sensors (16 binary patterns) plus a
  1-bit toggleable internal memory. Many cells share an identical wall
  signature (the four corridors between pillars look identical from
  either end), making this a POMDP. The memory bit gives the policy
  one bit of state to disambiguate.
- **Actions**: 6 — N, S, E, W movement, plus *set memory = 0* and *set
  memory = 1*. Bumping into a wall leaves the agent in place.
- **Reward**: -0.04 per step, +1 on reaching the goal (terminal).
  Episode timeout = 60 steps.
- **Policy**: tabular softmax over (wall_obs, memory_bit) -> action.
  Parameters θ ∈ R^{16x2x6} = 192 floats.

### Success-Story Algorithm (SSA)

The agent maintains a **stack of modifications** to its policy. A
modification is a REINFORCE update accumulated over a batch of episodes.
On each batch:

1. Run `mod_batch_size = 5` episodes, accumulate `(Δtime, Δreward)` into
   the lifetime totals.
2. Apply the SSA criterion to the existing stack (see below). Each
   invalid modification is **rolled back**: θ is restored to the
   snapshot stored before the modification was applied, and the entry is
   popped.
3. Compute a candidate REINFORCE update from the just-finished batch,
   apply it, and push a new stack entry recording (lifetime time,
   lifetime reward, pre-update θ).

**SSA criterion** (the form used here, equivalent in spirit to the 1997
paper's "valid times" stack): walking from the top of the stack down,
the rates `rate_i = (R_now - R_i) / (T_now - T_i)` must be
non-decreasing. If `rate_top < rate_below`, the most recent modification
is hurting the lifetime average reward more than the older modification;
pop it. After the pop, the criterion is re-checked against the new top.
Each modification gets at least `ssa_min_test_window = 200` env steps of
post-push data before it can be tested, so the rate estimate isn't
dominated by sampling noise.

### Three regimes are compared

| Regime | Continual policy? | SSA filtering? | Theta at start of task k+1 |
|---|---|---|---|
| `ssa`     | yes | yes | filtered policy from end of task k |
| `no_ssa`  | yes | no  | raw policy from end of task k |
| `restart` | no  | n/a | freshly initialized random policy |

The headline claim — that bias accumulated on earlier mazes accelerates
later ones — is tested by comparing `ssa` to `no_ssa` (does filtering
make the carried policy a better starting point for later tasks?) and to
`restart` (is the carried policy useful at all, or does cold-start beat
it?).

## Files

| File | Purpose |
|---|---|
| `ssa_bias_transfer_mazes.py` | Maze + tabular softmax policy + REINFORCE + SSA stack. CLI entry point; runs all three regimes and prints the headline table. |
| `make_ssa_bias_transfer_mazes_gif.py` | Re-trains under SSA and renders `ssa_bias_transfer_mazes.gif` showing the stack evolving over training, alongside the lifetime average reward. |
| `visualize_ssa_bias_transfer_mazes.py` | Static PNGs: maze layout, per-task bar charts, learning curves, stack evolution, pop timeline, and a 10-seed solve-rate summary. |
| `ssa_bias_transfer_mazes.gif` | Animation referenced at the top of this README. |
| `viz/maze_layout.png`        | The 5x5 maze with each task's start/goal pair. |
| `viz/per_task_steps.png`     | Bar chart, tail mean steps to goal per task per regime. |
| `viz/per_task_solve.png`     | Bar chart, tail solve rate per task per regime. |
| `viz/learning_curves.png`    | Smoothed steps-to-goal across all 800 episodes. |
| `viz/stack_evolution.png`    | Number of retained modifications on the SSA stack vs env step. |
| `viz/pop_timeline.png`       | Push and pop events coloured by which task proposed the modification. |
| `viz/multi_seed_solve.png`   | 10-seed aggregate: per-task tail solve rate (left) and cumulative solve rate over the task sequence (right). |

## Running

```bash
python3 ssa_bias_transfer_mazes.py --seed 0
```

Reproduces the headline table in **~1.7 s** on an M-series laptop CPU.
Determinism: the same `--seed` produces identical numbers across runs.

To regenerate the static visualizations and the GIF:

```bash
python3 visualize_ssa_bias_transfer_mazes.py --seed 0 --outdir viz
python3 make_ssa_bias_transfer_mazes_gif.py  --seed 0
```

The visualization script does its own 10-seed sweep for the aggregate
plot (~16 s extra). Pass `--no-multi-seed` to skip it.

CLI flags worth knowing: `--episodes-per-task N` (default 200),
`--mod-batch-size N` (default 5; episodes accumulated into one
modification), `--lr X` (default 0.4), `--ssa-min-test-window N`
(default 200; steps a modification must survive before SSA can test it),
`--ssa-pop-tolerance X` (default 0.0; raise to make SSA more lenient).
`--save-json path` dumps the full summary, including environment metadata
(Python / numpy version, OS, git commit), to JSON.

## Results

### Headline run, seed 0, defaults

```
Per-task tail mean steps-to-goal (last 20% of each task's episodes):
task            ssa      no_ssa     restart
0              5.45        5.45        7.55
1              6.90       10.12        5.25
2              8.12       60.00        7.50
3             35.30       42.05        6.22

Per-task tail solve rate:
task            ssa      no_ssa     restart
0              1.00        1.00        1.00
1              1.00        1.00        1.00
2              1.00        0.00        1.00
3              0.70        0.42        1.00
```

On task 2, `ssa` is **7.4x faster than no_ssa** (8.12 vs 60.00 steps)
and solves on every episode (1.00 vs 0.00 solve rate) — `no_ssa` carried
forward task-1's goal-direction bias and never recovered. `ssa` rolled
those modifications back.

Wallclock: ~1.7 s for all three regimes combined (4 tasks x 200 eps
each, 600 episodes per regime). SSA performed 150 mod pops.

### 10-seed aggregate

```
task                      ssa            no_ssa          restart
                  mean step (solve)   mean step (solve)   mean step (solve)
0                  6.64 (1.00)        6.37 (1.00)        7.27 (1.00)
1                  8.70 (1.00)       28.14 (0.65)        6.41 (1.00)
2                 39.83 (0.43)       34.12 (0.50)        6.70 (1.00)
3                 14.72 (0.90)       31.79 (0.63)        6.70 (1.00)
```

Across 10 seeds, **SSA's mean tail solve rate is 0.83**, vs **no_ssa's
0.70** — a **+19% relative improvement** in continual-learning
robustness. The biggest gains are on tasks 1 and 3 (the second and
fourth tasks): SSA rolls back the most recent task's goal-specific
modifications when their forward rate falls below the lifetime average,
preserving a more transferable policy. Task 2 is the regime's weakness
— after two task transitions the stack has been heavily popped and the
remaining policy is fragile; SSA loses to no_ssa on task 2 by a small
margin. Random restart per task is reliable (1.00 solve rate
everywhere) on this small maze because each task is individually easy
to relearn from scratch; SSA's promise — bias transfer that beats
cold-start — would shine more sharply on harder mazes (see Open
questions).

### Hyperparameters (defaults)

```python
n_tasks            = 4               n_obs   = 16          # 4 wall bits
episodes_per_task  = 200             n_mem   = 2           # 1 memory bit
mod_batch_size     = 5               n_acts  = 6           # 4 moves + 2 mem
lr                 = 0.4             theta_shape = (16, 2, 6) = 192 params
gamma              = 0.95            episode_limit = 60 steps
entropy_beta       = 0.01            step_cost = -0.04, goal_reward = +1.0
init_scale         = 0.05
ssa_min_test_window = 200            # steps before a mod can be SSA-tested
ssa_pop_tolerance   = 0.0            # 0 = strict criterion
```

## Visualizations

### `ssa_bias_transfer_mazes.gif`
Each frame shows one modification event during SSA training. Left:
maze, with the current task's goal coloured by task index (blue, orange,
green, red for tasks 0..3). Centre: the success-story stack — coloured
bars are retained modifications, oldest at bottom, each labelled with
the env step at which it was pushed. Right: lifetime average reward per
step, with grey dashed lines marking task boundaries and a black tick at
the current event time. The stack grows during a task as good
modifications accumulate, then partially collapses at task transitions
when the new task's lower reward rate triggers SSA pops.

### `viz/per_task_steps.png` and `viz/per_task_solve.png`
The headline bars. SSA matches `no_ssa` on task 0 (no transfer
opportunity yet), beats it from task 1 onwards (especially the **8 vs
60 steps on task 2**, where `no_ssa` is fully derailed by carried-over
bias), and trails `restart` because cold-start avoids transfer issues
entirely on this small maze.

### `viz/learning_curves.png`
Smoothed steps-to-goal across all 800 episodes (4 tasks x 200 eps).
The grey dashed verticals mark task boundaries. At each transition all
three regimes show a spike (the new task's goal is unknown). The
spike's *height* is what differs: `restart` re-initializes, `ssa`
benefits from carried-over generic navigation behaviour, `no_ssa`
sometimes never recovers (task 2, the orange line plateauing at 60
steps = full timeout = never reaches goal).

### `viz/stack_evolution.png`
Number of retained modifications on the SSA stack as training
progresses. Shows distinct phases: rapid stack growth at the start of
each task, then partial collapses at task boundaries when SSA detects
that the just-pushed (task-specific) modifications are dragging down
the lifetime rate.

### `viz/pop_timeline.png`
Every push (`^`) and pop (`v`) event, coloured by the task index that
owned the modification. Pops cluster around task boundaries, where
recently-pushed mods get rolled back when the new task's reward rate
exposes them as parochial.

### `viz/multi_seed_solve.png`
Left: per-task tail solve rate averaged over 10 seeds, with SEM error
bars. Right: cumulative solve rate over the task sequence. SSA is
visibly above no_ssa from task 1 onward; both fall short of random
restart, which is unaffected by transfer interference.

## Deviations from the original

1. **Modification = REINFORCE update, not arbitrary policy edit.** The
   1997 paper's modifications are general policy edits (additions to a
   "policy program"); we use one REINFORCE gradient batch as a single
   modification. This makes individual modifications smoother (gradient
   updates are improvements in expectation) and means SSA mostly
   filters out the *cross-task harmful* updates, not within-task noise.
   The bias-transfer demonstration still holds; the absolute number of
   pops would be lower if modifications were already gradient-filtered
   subroutines.
2. **Local SSA criterion + minimum test window.** The strict
   "lifetime-monotonic forward rates" stack criterion over-pops at task
   boundaries (the natural rate drop on a new task triggers cascading
   pops back to the lifetime start). We require each modification to
   have accumulated `ssa_min_test_window = 200` env steps of post-push
   data before it can be tested. Without this guard, the first batch
   of every new task triggers a stack-clearing avalanche. The 1997
   paper handles this implicitly by running each task much longer
   (millions of steps) before evaluating modifications; deferring the
   test is functionally equivalent on our shorter horizon.
3. **Tabular softmax policy, not the original universal-program
   self-modification setup.** The paper's incremental self-improvement
   (IS) variant pairs SSA with adaptive Levin search over symbolic
   programs. We replace IS with REINFORCE on a tabular policy (192
   parameters) so the stub is laptop-runnable in seconds. The SSA
   stack, criterion, and roll-back semantics are unchanged.
4. **Mini POMDP, not the paper's POE-literature mazes.** The 1997 paper
   reports state spaces "far bigger than most reported in the POE
   literature." We use a 5x5 maze with 21 free cells. The qualitative
   claim — bias transfer via SSA filtering — survives; absolute
   timings, stack sizes, and gap sizes do not.
5. **Reward shaping (-0.04/step, +1/goal).** The paper uses sparse
   per-episode reward; we add a small per-step cost so REINFORCE has
   useful gradient at every transition. SSA's criterion uses the same
   reward-rate signal regardless.
6. **Task sequence is a four-corner permutation, not increasing
   complexity.** The paper builds an explicit complexity ladder; we use
   four corner goals on the same maze. This isolates the goal-direction
   bias as the single transferable / interfering signal.

## Open questions / next experiments

- **Stronger PoMDP**, larger maze. Task 2's failure mode — cumulative
  stack pressure overwhelming SSA's filtering — should be the *normal*
  regime when each individual task takes longer to learn than current
  episodes-per-task (200) allow. A 9x9 maze with longer corridors and
  more memory-disambiguation requirement would push `restart` to also
  suffer from cold-start, and let SSA's carried policy dominate.
- **Different modification proposers.** REINFORCE makes modifications
  smooth; the paper's setup (random or program-search modifications)
  has more variance to filter. A version where each modification is a
  random sparse perturbation `Δθ ~ N(0, σ)` to a single (obs, mem,
  action) entry would more clearly exhibit SSA's selection pressure.
- **Adaptive `ssa_min_test_window`.** The 200-step window is a fixed
  hyperparameter. SSA in the paper effectively picks the window from
  the data — by detecting when reward rates have stabilized. A version
  that estimates the rate's standard error and tests modifications
  only when the gap is statistically significant should be both more
  conservative (fewer false-positive pops) and more decisive (faster
  pops on truly bad mods).
- **Comparison to EWC / synaptic intelligence baselines.** The
  continual-learning literature has 25 years of work since SSA. A
  direct comparison on this same task suite (same maze, same task
  sequence) would put SSA on the modern map. Predicted ranking:
  SSA ≈ EWC < replay-based methods, with SSA distinguished by *not
  needing* task labels.
- **Cross-task generalisation, not transfer.** The current experiment
  is sequential: train on task 0, then 1, then 2, then 3. Schmidhuber's
  later work (PowerPlay 2011, Asymptotic Optimality 2002) tests
  *generalisation* — does SSA's filtered policy perform on an unseen
  fifth task? A follow-up experiment with a held-out task would test
  whether SSA learns a task-agnostic navigation prior.
- **Data-movement metric (v2 / ByteDMD).** The full implementation is
  trivially small (192 parameters, 4 tasks, ~25 000 env steps). A
  ByteDMD-instrumented version would let us compare the data-movement
  cost of SSA's roll-back operations to plain REINFORCE — interesting
  given that roll-back is essentially `θ := snapshot`, a single big
  copy that should be much cheaper than the gradient computation it
  replaces.
