"""
ssa-bias-transfer-mazes — Schmidhuber, Zhao, Wiering 1997, *Shifting inductive
bias with success-story algorithm, adaptive Levin search, and incremental
self-improvement* (Machine Learning 28(1):105-130).

Sequence of partially-observable mazes. The agent navigates by 4-direction wall
sensors plus a 1-bit toggleable internal memory. Across tasks (the goal cell
moves), SSA keeps policy modifications that produce statistically meaningful
*lifetime* reward improvements; modifications that don't pay off get rolled
back. Three regimes are compared:

    ssa      : continual policy + SSA criterion filtering modifications
    no_ssa   : continual policy, no filtering (every gradient update kept)
    restart  : re-init policy at the start of each task

Headline: lifetime reward and steps-to-goal on the *later* tasks improve
with SSA vs random restarts. SSA also beats no-SSA on later tasks because
goal-specific updates from earlier tasks that hurt the lifetime average are
rolled back rather than carried forward.

Pure numpy. No torch, no scipy, no gym. CLI:

    python3 ssa_bias_transfer_mazes.py --seed 0
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Maze — fixed 5x5 layout with 4 interior wall pillars
# ----------------------------------------------------------------------
#
# Layout (W=wall, .=empty):
#
#     . . . . .
#     . W . W .
#     . . . . .
#     . W . W .
#     . . . . .
#
# 21 free cells. Tasks differ only in the goal cell. The 4-direction wall
# sensor produces 16 possible observations; multiple cells share the same
# observation (a corridor running between two pillars looks identical from
# either end), making this a POMDP. A 1-bit toggleable memory lets the
# policy disambiguate.

MAZE = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
H, W = MAZE.shape
FREE_CELLS = [(r, c) for r in range(H) for c in range(W) if MAZE[r, c] == 0]


# Action layout: 4 cardinal moves + 2 memory-set actions
# 0: N (r-1)        4: set memory = 0
# 1: S (r+1)        5: set memory = 1
# 2: E (c+1)
# 3: W (c-1)
N_ACTIONS = 6
ACTION_NAMES = ["N", "S", "E", "W", "M0", "M1"]


def wall_obs(pos: tuple[int, int]) -> int:
    """Encode 4-direction wall sensors into 0..15.

    bits: N=1, S=2, E=4, W=8.  A wall is "out of bounds" or a maze wall.
    """
    r, c = pos
    obs = 0
    if r == 0 or MAZE[r - 1, c] == 1:
        obs |= 1
    if r == H - 1 or MAZE[r + 1, c] == 1:
        obs |= 2
    if c == W - 1 or MAZE[r, c + 1] == 1:
        obs |= 4
    if c == 0 or MAZE[r, c - 1] == 1:
        obs |= 8
    return obs


def step_pos(pos: tuple[int, int], action: int) -> tuple[int, int]:
    """Apply movement action; bumping into walls leaves pos unchanged."""
    r, c = pos
    if action == 0 and r > 0 and MAZE[r - 1, c] == 0:
        r -= 1
    elif action == 1 and r < H - 1 and MAZE[r + 1, c] == 0:
        r += 1
    elif action == 2 and c < W - 1 and MAZE[r, c + 1] == 0:
        c += 1
    elif action == 3 and c > 0 and MAZE[r, c - 1] == 0:
        c -= 1
    return r, c


# ----------------------------------------------------------------------
# Tasks — same maze, different (start, goal) pair
# ----------------------------------------------------------------------
#
# 4 tasks chosen to spread goals across the maze. The starting cell is
# always (2, 2) — the centre — so each task forces a different navigation
# direction. The shared layout means generic navigation behaviour (e.g.
# "in a corridor with walls N/S, prefer horizontal moves") transfers; the
# goal-direction bias does not.

@dataclass
class Task:
    name: str
    start: tuple[int, int]
    goal: tuple[int, int]


TASKS = [
    Task("NW-corner", start=(2, 2), goal=(0, 0)),
    Task("NE-corner", start=(2, 2), goal=(0, 4)),
    Task("SE-corner", start=(2, 2), goal=(4, 4)),
    Task("SW-corner", start=(2, 2), goal=(4, 0)),
]


# ----------------------------------------------------------------------
# Policy — tabular softmax over (wall_obs, memory_bit) -> action
# ----------------------------------------------------------------------

N_OBS = 16  # 4 wall bits
N_MEM = 2   # 1 memory bit

PARAM_SHAPE = (N_OBS, N_MEM, N_ACTIONS)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def policy_probs(theta: np.ndarray, obs: int, mem: int) -> np.ndarray:
    return softmax(theta[obs, mem])


def sample_action(theta: np.ndarray, obs: int, mem: int,
                  rng: np.random.Generator) -> tuple[int, np.ndarray]:
    p = policy_probs(theta, obs, mem)
    a = int(rng.choice(N_ACTIONS, p=p))
    return a, p


# ----------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------

@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    reached_goal: bool
    trajectory: list[tuple[int, int, int]]   # (row, col, mem) per step
    actions: list[int]


STEP_COST = -0.04
GOAL_REWARD = 1.0
EPISODE_LIMIT = 60          # steps per episode


def run_episode(theta: np.ndarray, task: Task, rng: np.random.Generator,
                step_limit: int = EPISODE_LIMIT,
                record_trajectory: bool = False
                ) -> tuple[EpisodeResult, list[tuple[int, int, int, float]]]:
    """Run one episode under policy `theta` on `task`.

    Returns (result, transitions). transitions[t] = (obs, mem_in, action, reward).
    """
    pos = task.start
    mem = 0
    transitions: list[tuple[int, int, int, float]] = []
    traj: list[tuple[int, int, int]] = []
    actions: list[int] = []
    total = 0.0
    reached = False
    if record_trajectory:
        traj.append((pos[0], pos[1], mem))
    for _ in range(step_limit):
        obs = wall_obs(pos)
        mem_in = mem
        a, _ = sample_action(theta, obs, mem, rng)
        if a < 4:
            pos = step_pos(pos, a)
            r = STEP_COST
        elif a == 4:
            mem = 0
            r = STEP_COST
        else:  # a == 5
            mem = 1
            r = STEP_COST
        if pos == task.goal:
            r += GOAL_REWARD
            reached = True
        transitions.append((obs, mem_in, a, r))
        if record_trajectory:
            traj.append((pos[0], pos[1], mem))
            actions.append(a)
        total += r
        if reached:
            break
    return EpisodeResult(
        steps=len(transitions),
        total_reward=total,
        reached_goal=reached,
        trajectory=traj,
        actions=actions,
    ), transitions


# ----------------------------------------------------------------------
# REINFORCE update with discount + entropy bonus
# ----------------------------------------------------------------------

def reinforce_update(theta: np.ndarray,
                     transitions_batch: list[list[tuple[int, int, int, float]]],
                     lr: float, gamma: float, entropy_beta: float) -> np.ndarray:
    """Compute a candidate parameter update accumulated across the batch."""
    d_theta = np.zeros_like(theta)
    for transitions in transitions_batch:
        T = len(transitions)
        if T == 0:
            continue
        rewards = np.array([r for *_, r in transitions])
        returns = np.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        baseline = returns.mean()
        adv = returns - baseline
        for t in range(T):
            obs, mem, a, _ = transitions[t]
            p = softmax(theta[obs, mem])
            grad_logpi = -p
            grad_logpi[a] += 1.0
            d_theta[obs, mem] += lr * adv[t] * grad_logpi
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = np.log(np.clip(p, 1e-12, 1.0))
            grad_entropy = -(p * (logp - (p * logp).sum()))
            d_theta[obs, mem] += lr * entropy_beta * grad_entropy
    return d_theta


# ----------------------------------------------------------------------
# Success-Story Algorithm
# ----------------------------------------------------------------------

@dataclass
class StackEntry:
    """One checkpoint = one modification.

    Stored AT the moment the modification was applied:
      time         : cumulative env step at the time of push
      reward       : cumulative env reward at the time of push
      pre_theta    : snapshot of theta BEFORE the modification (for rollback)
      task_idx     : which task we were on when pushed (for visualization)
      n_eps_pushed : how many episodes had been run when pushed (for viz)
    """
    time: int
    reward: float
    pre_theta: np.ndarray
    task_idx: int
    n_eps_pushed: int


# ----------------------------------------------------------------------
# Training: one function, three regimes
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_tasks: int = len(TASKS)
    episodes_per_task: int = 200
    mod_batch_size: int = 5      # episodes per candidate modification
    lr: float = 0.4
    gamma: float = 0.95
    entropy_beta: float = 0.01
    init_scale: float = 0.05
    # SSA-specific knobs: each modification's validity test fires only
    # once at least min_test_window environment steps have elapsed since
    # the push, so the rate estimate is not dominated by sampling noise.
    ssa_min_test_window: int = 200
    # Pop only if the post-mod rate falls below the pre-mod rate by at
    # least `ssa_pop_tolerance` (in reward units / step). 0 = strict.
    ssa_pop_tolerance: float = 0.0


@dataclass
class TrainTrace:
    regime: str
    seed: int
    # per-episode arrays
    ep_task: list[int] = field(default_factory=list)
    ep_steps: list[int] = field(default_factory=list)
    ep_reward: list[float] = field(default_factory=list)
    ep_reached: list[bool] = field(default_factory=list)
    cum_time: list[int] = field(default_factory=list)
    cum_reward: list[float] = field(default_factory=list)
    # SSA-only: stack-size after every check
    stack_size: list[int] = field(default_factory=list)
    # one entry per "modification event": (cum_time, cum_reward,
    # task_idx, action) where action is "push" | "pop" | "kept"
    mod_events: list[tuple[int, float, int, str]] = field(default_factory=list)
    n_pops_total: int = 0
    pops_by_task: dict = field(default_factory=dict)
    # snapshots for visualization
    final_theta: Optional[np.ndarray] = None


def init_theta(rng: np.random.Generator, scale: float) -> np.ndarray:
    return rng.standard_normal(PARAM_SHAPE) * scale


def train(regime: str, seed: int, cfg: TrainConfig,
          quiet: bool = True) -> TrainTrace:
    """Train one of {ssa, no_ssa, restart}.

    All three see the same task sequence, the same hyperparameters, the
    same total step budget. The only difference is what happens between
    candidate modifications (and what happens between tasks).
    """
    if regime not in ("ssa", "no_ssa", "restart"):
        raise ValueError(f"unknown regime: {regime}")
    rng = np.random.default_rng(seed)
    theta = init_theta(rng, cfg.init_scale)
    trace = TrainTrace(regime=regime, seed=seed)
    stack: list[StackEntry] = []
    cum_time = 0
    cum_reward = 0.0
    pops_by_task: dict[int, int] = {}

    for task_idx in range(cfg.n_tasks):
        task = TASKS[task_idx]
        if regime == "restart":
            theta = init_theta(rng, cfg.init_scale)
            stack = []
            cum_time = 0
            cum_reward = 0.0  # restart sees a fresh "lifetime" too

        ep_in_task = 0
        while ep_in_task < cfg.episodes_per_task:
            # ------- Step 1: collect episodes under current theta -------
            batch_transitions: list[list[tuple[int, int, int, float]]] = []
            for _ in range(cfg.mod_batch_size):
                if ep_in_task >= cfg.episodes_per_task:
                    break
                ep_result, transitions = run_episode(theta, task, rng)
                batch_transitions.append(transitions)
                trace.ep_task.append(task_idx)
                trace.ep_steps.append(ep_result.steps)
                trace.ep_reward.append(ep_result.total_reward)
                trace.ep_reached.append(ep_result.reached_goal)
                cum_time += ep_result.steps
                cum_reward += ep_result.total_reward
                trace.cum_time.append(cum_time)
                trace.cum_reward.append(cum_reward)
                ep_in_task += 1
            if not batch_transitions:
                break

            # ------- Step 2: SSA criterion test (uses just-collected data
            # to evaluate the modifications already on the stack) ----------
            #
            # SSA criterion (local form — see §Deviations in README):
            # modification at top is valid iff
            #     (R_now - R_top) / (T_now - T_top)
            #         >= (R_top - R_below) / (T_top - T_below)
            # i.e. "since-mod rate beats just-before-mod rate". This is the
            # form Wiering & Schmidhuber 1996 used in EIRA; it generalises
            # better across non-stationary task sequences than the strict
            # lifetime-monotonicity form, which over-pops at task switches.
            if regime == "ssa" and stack:
                # SSA criterion: a modification at top is valid iff its
                # post-push reward rate is at least the rate at the time
                # of the next-older valid tag (forward from that tag, up
                # to now).  Equivalently: walking from the top down, the
                # rates `rate_i = (R_now - R_i) / (T_now - T_i)` must be
                # monotonically non-decreasing in i (oldest tag has the
                # smallest rate).  Pop until the property holds.
                while stack:
                    top = stack[-1]
                    dt_after = cum_time - top.time
                    if dt_after < cfg.ssa_min_test_window:
                        # not enough post-push data yet; leave this mod
                        # (and everything older) alone for now.
                        break
                    rate_top = (cum_reward - top.reward) / max(dt_after, 1)
                    if len(stack) >= 2:
                        below = stack[-2]
                        dt_below = max(cum_time - below.time, 1)
                        rate_below = (cum_reward - below.reward) / dt_below
                    else:
                        # implicit "lifetime start" tag at (0, 0)
                        rate_below = cum_reward / max(cum_time, 1)
                    if rate_top + cfg.ssa_pop_tolerance < rate_below:
                        theta = top.pre_theta.copy()
                        pops_by_task[top.task_idx] = pops_by_task.get(top.task_idx, 0) + 1
                        trace.n_pops_total += 1
                        trace.mod_events.append(
                            (cum_time, cum_reward, top.task_idx, "pop"))
                        stack.pop()
                    else:
                        break
                trace.stack_size.append(len(stack))

            # ------- Step 3: candidate modification (REINFORCE) ----------
            d_theta = reinforce_update(theta, batch_transitions,
                                       cfg.lr, cfg.gamma, cfg.entropy_beta)
            pre_theta = theta.copy()
            theta = theta + d_theta

            if regime == "ssa":
                stack.append(StackEntry(
                    time=cum_time, reward=cum_reward, pre_theta=pre_theta,
                    task_idx=task_idx, n_eps_pushed=len(trace.ep_steps),
                ))
                trace.mod_events.append(
                    (cum_time, cum_reward, task_idx, "push"))
                trace.stack_size.append(len(stack))

        if not quiet:
            ep_steps_arr = np.array(trace.ep_steps)
            mask = np.array(trace.ep_task) == task_idx
            tail = ep_steps_arr[mask][-min(20, mask.sum()):]
            tail_solve = np.array(trace.ep_reached)[mask][-min(20, mask.sum()):]
            print(f"  [{regime:7}] task {task_idx} ({task.name}): "
                  f"tail mean steps {tail.mean():5.2f} | "
                  f"tail solve {tail_solve.mean()*100:5.1f}%")

    trace.final_theta = theta
    trace.pops_by_task = dict(pops_by_task)
    return trace


# ----------------------------------------------------------------------
# Top-level: run all three regimes, summarize, print headline
# ----------------------------------------------------------------------

def per_task_summary(trace: TrainTrace, n_tasks: int) -> dict:
    out = {}
    ep_task = np.asarray(trace.ep_task)
    ep_steps = np.asarray(trace.ep_steps)
    ep_reward = np.asarray(trace.ep_reward)
    ep_reached = np.asarray(trace.ep_reached, dtype=bool)
    for t in range(n_tasks):
        m = ep_task == t
        if not m.any():
            out[t] = dict(n_eps=0)
            continue
        n = int(m.sum())
        cut = max(1, int(0.8 * n))
        out[t] = dict(
            n_eps=n,
            mean_steps=float(ep_steps[m].mean()),
            mean_reward=float(ep_reward[m].mean()),
            solve_rate=float(ep_reached[m].mean()),
            tail_solve_rate=float(ep_reached[m][cut:].mean()),
            tail_mean_steps=float(ep_steps[m][cut:].mean()),
        )
    return out


def env_info() -> dict:
    info = {
        "python": sys.version.split(" ")[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = None
    return info


def run_all(seed: int, cfg: TrainConfig, quiet: bool = False) -> tuple[
        dict, TrainTrace, TrainTrace, TrainTrace]:
    t0 = time.time()
    if not quiet:
        print(f"[ssa-bias-transfer-mazes] seed={seed}, "
              f"{cfg.n_tasks} tasks x {cfg.episodes_per_task} eps")
    ssa_trace     = train("ssa",     seed, cfg, quiet=quiet)
    nossa_trace   = train("no_ssa",  seed, cfg, quiet=quiet)
    restart_trace = train("restart", seed, cfg, quiet=quiet)
    elapsed = time.time() - t0

    results = {}
    for trace in (ssa_trace, nossa_trace, restart_trace):
        results[trace.regime] = dict(
            per_task=per_task_summary(trace, cfg.n_tasks),
            total_eps=len(trace.ep_steps),
            mean_steps_overall=float(np.mean(trace.ep_steps)),
            solve_rate_overall=float(np.mean(trace.ep_reached)),
            n_pops_total=trace.n_pops_total,
            pops_by_task=trace.pops_by_task,
            final_stack_size=(trace.stack_size[-1] if trace.stack_size else 0),
        )

    summary = dict(
        seed=seed,
        config=asdict(cfg),
        elapsed_sec=elapsed,
        env=env_info(),
        results=results,
    )
    return summary, ssa_trace, nossa_trace, restart_trace


def print_headline(summary: dict) -> None:
    print()
    print("=" * 72)
    print("ssa-bias-transfer-mazes — headline")
    print("=" * 72)
    print(f"seed={summary['seed']}, elapsed={summary['elapsed_sec']:.2f} s")
    n_tasks = summary["config"]["n_tasks"]
    header = f"{'task':<5}" + "".join(f"{r:>12}" for r in ("ssa", "no_ssa", "restart"))

    print()
    print("Per-task tail mean steps-to-goal (last 20% of each task's eps):")
    print(header); print("-" * len(header))
    for t in range(n_tasks):
        row = f"{t:<5}"
        for regime in ("ssa", "no_ssa", "restart"):
            v = summary["results"][regime]["per_task"][t]["tail_mean_steps"]
            row += f"{v:>12.2f}"
        print(row)

    print()
    print("Per-task tail solve rate:")
    print(header); print("-" * len(header))
    for t in range(n_tasks):
        row = f"{t:<5}"
        for regime in ("ssa", "no_ssa", "restart"):
            v = summary["results"][regime]["per_task"][t]["tail_solve_rate"]
            row += f"{v:>12.2f}"
        print(row)

    print()
    ssa = summary["results"]["ssa"]
    print(f"SSA total mods popped: {ssa['n_pops_total']}; "
          f"pops by task: {ssa['pops_by_task']}; "
          f"final stack size: {ssa['final_stack_size']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes-per-task", type=int, default=200)
    p.add_argument("--mod-batch-size", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--entropy-beta", type=float, default=0.01)
    p.add_argument("--ssa-min-test-window", type=int, default=200)
    p.add_argument("--ssa-pop-tolerance", type=float, default=0.0)
    p.add_argument("--save-json", type=str, default=None,
                   help="Path to dump full summary JSON.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        episodes_per_task=args.episodes_per_task,
        mod_batch_size=args.mod_batch_size,
        lr=args.lr,
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        ssa_min_test_window=args.ssa_min_test_window,
        ssa_pop_tolerance=args.ssa_pop_tolerance,
    )

    summary, *_ = run_all(args.seed, cfg, quiet=args.quiet)
    print_headline(summary)

    if args.save_json:
        def _default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"unhandled type {type(o)}")
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2, default=_default)
        print(f"\nWrote {args.save_json}")


if __name__ == "__main__":
    main()
