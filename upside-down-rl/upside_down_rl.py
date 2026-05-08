"""upside-down-rl -- Schmidhuber, *Reinforcement Learning Upside Down: Don't
Predict Rewards -- Just Map Them to Actions*, arXiv:1912.02875 (2019).
Companion: Srivastava, Shyam, Mutz, Jaskowski, Schmidhuber, *Training Agents
using Upside-Down Reinforcement Learning*, arXiv:1912.02877 (2019).

UDRL flips the standard RL formulation. Instead of training a value function
or a policy gradient that maximises expected return, it treats the policy as
a *supervised* mapping

    policy(state, desired_return, desired_horizon) -> action

learned by self-imitation: every (s, a) pair in a collected episode is labelled
with the return *actually realised from t onward* and the *remaining horizon*,
and the network is trained to predict a from (s, R_remaining, h_remaining) by
plain cross-entropy. At deployment the agent is *commanded* with a high desired
return, and -- if the buffer contains enough high-return trajectories -- the
network generalises to produce the action sequence that achieves that command.

Per SPEC issue #1 (cybertronai/schmidhuber-problems), v1 RL stubs use a numpy
mini-env, NOT LunarLander. This stub uses a deterministic chain MDP:

    states  : 0 .. N-1     (N = 9 by default)
    start   : floor(N/2) = 4
    actions : 0 = left, 1 = right
    rewards : -0.1 step cost; +1 at state 0 (left absorbing);
              +5 at state N-1 (right absorbing)
    horizon : t_max = 30 steps

A purely random policy gets the small left reward roughly as often as the big
right reward, so realised returns are bimodal around ~+0.5 and ~+4. UDRL,
trained on a buffer of these random rollouts plus its own self-imitated rollouts,
must learn that conditioning on R_desired = 5 implies "go right" and conditioning
on R_desired ~ 1 implies "go left". The headline check is whether the achieved
return at inference rises monotonically with the commanded return.

Architecture (Srivastava et al. 2019, fig. 1, scaled to chain MDP):

    behavior_fn:  (one-hot state | scalar dR | scalar dH)
                  -> tanh-MLP (hidden=64, 2 layers)
                  -> softmax over 2 actions

dR and dH are the *desired* remaining return and remaining horizon; both are
fed in raw units (the network learns its own scaling). A small Gaussian
exploration noise sigma is added to dR during behavior-phase rollouts.

Algorithm (paper Algorithm 1, with one practical knob):

    repeat n_iters times:
      1. sample top-K-return episodes from buffer; mean(return) and mean(length)
         of that slice define the command (desired_return, desired_horizon)
      2. roll out n_episodes_per_iter trajectories with the *current* policy,
         conditioned on that command + Gaussian exploration; add to buffer
      3. for n_grad_steps minibatches sampled uniformly over (s, a, t, T, R)
         from the buffer, train policy by cross-entropy on a given
         (state, R_realized_from_t, T-t)
      4. evict low-return episodes so |buffer| <= buffer_size

Eval (every eval_every iterations):
  Run greedy rollouts conditioned on a sweep of desired-return commands
  {1.0, 2.5, 4.0, 5.0}; record achieved return per command.

Determinism: a single --seed seeds numpy and the env. Two runs with the same
seed produce identical numbers (verified in §Results of README).

CLI:
    python3 upside_down_rl.py --seed 0
    python3 upside_down_rl.py --seed 0 --quick      # smaller, smoke test
    python3 upside_down_rl.py --seed 0 --save-json out/run.json
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
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Reproducibility metadata
# ----------------------------------------------------------------------

def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def env_metadata() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "git": git_hash(),
    }


# ----------------------------------------------------------------------
# Chain MDP environment
# ----------------------------------------------------------------------

@dataclass
class ChainMDP:
    """Deterministic 1-D chain with two absorbing terminals.

    Layout (N=9):
        +1                                                              +5
         0 <-- 1 <-- 2 <-- 3 <-- [S=4] --> 5 --> 6 --> 7 --> 8
        terminal                            start                    terminal

    Step cost -0.1 per non-terminal transition.
    """

    N: int = 9
    start: int = 4
    step_cost: float = -0.1
    left_reward: float = 1.0
    right_reward: float = 5.0
    t_max: int = 30

    def reset(self) -> int:
        self.state = self.start
        self.t = 0
        self.done = False
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        # action: 0 = left, 1 = right
        if self.done:
            raise RuntimeError("step() after done")
        new_state = self.state + (1 if action == 1 else -1)
        new_state = max(0, min(self.N - 1, new_state))
        self.state = new_state
        self.t += 1
        if new_state == 0:
            self.done = True
            return new_state, self.left_reward, True
        if new_state == self.N - 1:
            self.done = True
            return new_state, self.right_reward, True
        if self.t >= self.t_max:
            self.done = True
            return new_state, self.step_cost, True
        return new_state, self.step_cost, False


# ----------------------------------------------------------------------
# Policy network: tanh MLP, hand-coded forward + backward
# ----------------------------------------------------------------------

@dataclass
class MLP:
    """2-hidden-layer tanh MLP -> softmax. Hand-coded forward + backward."""

    in_dim: int
    hidden: int
    out_dim: int
    rng: np.random.Generator

    def __post_init__(self):
        # Xavier-ish init for tanh
        s1 = np.sqrt(1.0 / self.in_dim)
        s2 = np.sqrt(1.0 / self.hidden)
        s3 = np.sqrt(1.0 / self.hidden)
        self.W1 = self.rng.normal(0, s1, size=(self.in_dim, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0, s2, size=(self.hidden, self.hidden))
        self.b2 = np.zeros(self.hidden)
        self.W3 = self.rng.normal(0, s3, size=(self.hidden, self.out_dim))
        self.b3 = np.zeros(self.out_dim)
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        z3 = h2 @ self.W3 + self.b3
        # softmax
        z3 = z3 - z3.max(axis=-1, keepdims=True)
        exp = np.exp(z3)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        cache = {"x": x, "h1": h1, "h2": h2, "probs": probs}
        return probs, cache

    def cross_entropy_grad(
        self,
        cache: Dict,
        actions: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        x = cache["x"]
        h1 = cache["h1"]
        h2 = cache["h2"]
        probs = cache["probs"]
        B = x.shape[0]
        loss = -np.log(probs[np.arange(B), actions] + 1e-12).mean()
        dlogits = probs.copy()
        dlogits[np.arange(B), actions] -= 1.0
        dlogits /= B
        dW3 = h2.T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3.T
        dz2 = dh2 * (1.0 - h2 ** 2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (1.0 - h1 ** 2)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        return loss, grads

    def adam_step(
        self,
        grads: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 5.0,
    ):
        # Global-norm clip
        total = 0.0
        for g in grads.values():
            total += float((g ** 2).sum())
        norm = np.sqrt(total)
        if norm > clip:
            scale = clip / (norm + 1e-12)
            for k in grads:
                grads[k] = grads[k] * scale
        self.t_step += 1
        bc1 = 1.0 - beta1 ** self.t_step
        bc2 = 1.0 - beta2 ** self.t_step
        for k, p in self.params().items():
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------

def encode(
    states: np.ndarray,
    desired_return: np.ndarray,
    desired_horizon: np.ndarray,
    N: int,
    return_scale: float,
    horizon_scale: float,
) -> np.ndarray:
    """Concatenate one-hot state, scaled dR, scaled dH."""
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    desired_return = np.asarray(desired_return, dtype=np.float64).reshape(-1)
    desired_horizon = np.asarray(desired_horizon, dtype=np.float64).reshape(-1)
    B = states.shape[0]
    onehot = np.zeros((B, N))
    onehot[np.arange(B), states] = 1.0
    dr = (desired_return / return_scale).reshape(-1, 1)
    dh = (desired_horizon / horizon_scale).reshape(-1, 1)
    return np.concatenate([onehot, dr, dh], axis=1)


# ----------------------------------------------------------------------
# Replay buffer (sorted by episode return)
# ----------------------------------------------------------------------

@dataclass
class Episode:
    states: List[int] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def total_return(self) -> float:
        return float(sum(self.rewards))

    def length(self) -> int:
        return len(self.actions)


@dataclass
class Buffer:
    """FIFO replay buffer. Preserves episode-return diversity so the policy
    can be trained to condition on both low and high returns -- discarding
    by return (keeping only top-K) collapses the conditioning signal.

    Top-K-by-return is computed on demand for sampling exploration commands.
    """

    capacity: int
    episodes: List[Episode] = field(default_factory=list)

    def add(self, ep: Episode):
        self.episodes.append(ep)
        # FIFO: drop oldest when over capacity.
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def __len__(self):
        return len(self.episodes)

    def top_k_command(
        self, k: int
    ) -> Tuple[float, float]:
        """Mean return and mean length over the top-k highest-return episodes
        currently in the buffer (recomputed on demand)."""
        if not self.episodes:
            return 0.0, 0.0
        sorted_eps = sorted(
            self.episodes,
            key=lambda e: (-e.total_return(), e.length()),
        )
        k = min(k, len(sorted_eps))
        slice_ = sorted_eps[:k]
        mean_R = float(np.mean([e.total_return() for e in slice_]))
        mean_L = float(np.mean([e.length() for e in slice_]))
        return mean_R, mean_L

    def sample_transitions(
        self, rng: np.random.Generator, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample (s, a, R_remaining, h_remaining) tuples, where the
        episode is sampled uniformly and a step within it is sampled uniformly,
        as in Srivastava et al. 2019 alg 2.
        """
        states, actions, dRs, dHs = [], [], [], []
        ep_idx = rng.integers(0, len(self.episodes), size=n)
        for i in ep_idx:
            e = self.episodes[i]
            t1 = rng.integers(0, e.length())
            # second time-step picked from [t1+1, T] (paper's "between" view).
            # We use the simpler "remaining-from-t1" labelling from algorithm 1
            # of the paper, which is equivalent when t2 = T.
            R_rem = float(sum(e.rewards[t1:]))
            h_rem = float(e.length() - t1)
            states.append(e.states[t1])
            actions.append(e.actions[t1])
            dRs.append(R_rem)
            dHs.append(h_rem)
        return (
            np.array(states, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(dRs, dtype=np.float64),
            np.array(dHs, dtype=np.float64),
        )


# ----------------------------------------------------------------------
# Rollout helpers
# ----------------------------------------------------------------------

def random_rollout(env: ChainMDP, rng: np.random.Generator) -> Episode:
    s = env.reset()
    ep = Episode()
    while not env.done:
        a = int(rng.integers(0, 2))
        ep.states.append(s)
        ep.actions.append(a)
        s, r, done = env.step(a)
        ep.rewards.append(r)
    return ep


def policy_rollout(
    env: ChainMDP,
    pol: MLP,
    desired_return: float,
    desired_horizon: float,
    rng: np.random.Generator,
    return_scale: float,
    horizon_scale: float,
    greedy: bool = False,
    explore_sigma: float = 0.0,
) -> Episode:
    s = env.reset()
    ep = Episode()
    dR = float(desired_return)
    dH = float(desired_horizon)
    if explore_sigma > 0.0:
        dR = dR + rng.normal(0.0, explore_sigma)
    while not env.done:
        x = encode(
            np.array([s]),
            np.array([dR]),
            np.array([max(dH, 1.0)]),
            env.N,
            return_scale,
            horizon_scale,
        )
        probs, _ = pol.forward(x)
        if greedy:
            a = int(np.argmax(probs[0]))
        else:
            a = int(rng.choice(probs.shape[1], p=probs[0]))
        ep.states.append(s)
        ep.actions.append(a)
        s, r, done = env.step(a)
        ep.rewards.append(r)
        dR = dR - r
        dH = dH - 1.0
    return ep


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    N: int = 9
    t_max: int = 30
    hidden: int = 64
    n_iters: int = 80
    n_warmup_random: int = 100
    episodes_per_iter: int = 15
    grad_steps_per_iter: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    buffer_size: int = 400
    top_k: int = 50
    explore_sigma: float = 0.1
    eval_every: int = 5
    eval_episodes: int = 30
    eval_commands: Tuple[float, ...] = (1.0, 2.5, 4.0, 5.0)


def train(cfg: RunConfig, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    env = ChainMDP(N=cfg.N, t_max=cfg.t_max)
    return_scale = float(max(abs(env.right_reward), abs(env.left_reward)))
    horizon_scale = float(env.t_max)
    in_dim = env.N + 2  # one-hot state, dR, dH
    pol = MLP(
        in_dim=in_dim, hidden=cfg.hidden, out_dim=2,
        rng=np.random.default_rng(cfg.seed + 1),
    )
    buf = Buffer(capacity=cfg.buffer_size)

    # Warmup with random rollouts
    for _ in range(cfg.n_warmup_random):
        buf.add(random_rollout(env, rng))

    history = {
        "iter": [],
        "loss": [],
        "buffer_mean_return": [],
        "buffer_top_command_R": [],
        "buffer_top_command_H": [],
        "rollout_mean_return": [],
        "eval_iter": [],
        "eval_commands": list(cfg.eval_commands),
        "eval_achieved": [],  # list of [achieved_per_command] per eval iter
    }

    t0 = time.time()
    for it in range(cfg.n_iters):
        # Behaviour phase: pull command from top-k, roll out N episodes
        cmd_R, cmd_H = buf.top_k_command(cfg.top_k)
        rollout_returns = []
        for _ in range(cfg.episodes_per_iter):
            ep = policy_rollout(
                env, pol, cmd_R, cmd_H, rng,
                return_scale=return_scale,
                horizon_scale=horizon_scale,
                greedy=False,
                explore_sigma=cfg.explore_sigma,
            )
            buf.add(ep)
            rollout_returns.append(ep.total_return())

        # Train phase
        last_loss = 0.0
        for _ in range(cfg.grad_steps_per_iter):
            s_b, a_b, dR_b, dH_b = buf.sample_transitions(rng, cfg.batch_size)
            x_b = encode(s_b, dR_b, dH_b, env.N, return_scale, horizon_scale)
            _, cache = pol.forward(x_b)
            last_loss, grads = pol.cross_entropy_grad(cache, a_b)
            pol.adam_step(grads, lr=cfg.lr)

        buf_mean_R = float(np.mean([e.total_return() for e in buf.episodes]))
        history["iter"].append(it)
        history["loss"].append(float(last_loss))
        history["buffer_mean_return"].append(buf_mean_R)
        history["buffer_top_command_R"].append(float(cmd_R))
        history["buffer_top_command_H"].append(float(cmd_H))
        history["rollout_mean_return"].append(float(np.mean(rollout_returns)))

        if (it % cfg.eval_every) == 0 or it == cfg.n_iters - 1:
            # Use a sensible eval horizon: the mean length of top-K buffer
            # episodes (within-distribution for the trained policy). Per
            # Srivastava et al. 2019 §3.2 -- "command at deployment from the
            # same distribution as during training".
            _, eval_H = buf.top_k_command(cfg.top_k)
            eval_H = max(eval_H, 1.0)
            achieved = []
            for desired in cfg.eval_commands:
                returns = []
                for _ in range(cfg.eval_episodes):
                    ep = policy_rollout(
                        env, pol, desired, eval_H, rng,
                        return_scale=return_scale,
                        horizon_scale=horizon_scale,
                        greedy=True,
                        explore_sigma=0.0,
                    )
                    returns.append(ep.total_return())
                achieved.append(float(np.mean(returns)))
            history["eval_iter"].append(it)
            history["eval_achieved"].append(achieved)
            history.setdefault("eval_horizon", []).append(float(eval_H))
            if verbose:
                print(
                    f"iter {it:3d}  loss={last_loss:.4f}  "
                    f"buf_mean_R={buf_mean_R:.3f}  "
                    f"cmd_R={cmd_R:.3f} cmd_H={cmd_H:.2f}  "
                    f"eval(R*={cfg.eval_commands[-1]})={achieved[-1]:.3f}"
                )

    wall = time.time() - t0

    # Final greedy eval, dense command sweep for the figure.
    # Eval horizon = mean length of top-K buffer episodes (within-distribution).
    _, eval_H_final = buf.top_k_command(cfg.top_k)
    eval_H_final = max(eval_H_final, 1.0)
    sweep_commands = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    sweep_results = []
    sweep_episodes_seed = np.random.default_rng(cfg.seed + 2)
    for desired in sweep_commands:
        returns = []
        steps_dist = []
        end_states = []
        for _ in range(cfg.eval_episodes):
            ep = policy_rollout(
                env, pol, desired, eval_H_final, sweep_episodes_seed,
                return_scale=return_scale,
                horizon_scale=horizon_scale,
                greedy=True,
                explore_sigma=0.0,
            )
            returns.append(ep.total_return())
            steps_dist.append(ep.length())
            end_states.append(ep.states[-1])
        sweep_results.append({
            "desired_return": float(desired),
            "achieved_return_mean": float(np.mean(returns)),
            "achieved_return_std": float(np.std(returns)),
            "mean_steps": float(np.mean(steps_dist)),
            "end_states": end_states,
        })

    # Random baseline for the §Results table
    base_rng = np.random.default_rng(cfg.seed + 3)
    rand_returns = []
    for _ in range(cfg.eval_episodes):
        ep = random_rollout(env, base_rng)
        rand_returns.append(ep.total_return())
    rand_mean = float(np.mean(rand_returns))
    rand_std = float(np.std(rand_returns))

    # Action heatmap: P(right) over (state, desired_return) at horizon=eval_H_final
    h_grid = float(eval_H_final)
    state_grid = np.arange(env.N)
    R_grid = np.linspace(-1.0, env.right_reward, 25)
    p_right_grid = np.zeros((env.N, len(R_grid)))
    for i, s in enumerate(state_grid):
        for j, R in enumerate(R_grid):
            x = encode(
                np.array([s]), np.array([R]), np.array([h_grid]),
                env.N, return_scale, horizon_scale,
            )
            probs, _ = pol.forward(x)
            p_right_grid[i, j] = float(probs[0, 1])

    summary = {
        "config": asdict(cfg),
        "env": {
            "N": env.N,
            "start": env.start,
            "step_cost": env.step_cost,
            "left_reward": env.left_reward,
            "right_reward": env.right_reward,
            "t_max": env.t_max,
        },
        "env_metadata": env_metadata(),
        "history": history,
        "sweep_commands": sweep_commands,
        "sweep_results": sweep_results,
        "random_baseline": {"mean_return": rand_mean, "std_return": rand_std,
                            "n_episodes": cfg.eval_episodes},
        "p_right_grid": p_right_grid.tolist(),
        "p_right_grid_R_axis": R_grid.tolist(),
        "p_right_grid_state_axis": state_grid.tolist(),
        "p_right_grid_horizon": float(h_grid),
        "eval_horizon_final": float(eval_H_final),
        "wallclock_sec": wall,
    }
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Smaller / shorter run for smoke testing.")
    p.add_argument("--save-json", type=str, default=None,
                   help="Path to dump full summary JSON.")
    p.add_argument("--n-iters", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = RunConfig(seed=args.seed)
    if args.quick:
        cfg.n_iters = 20
        cfg.episodes_per_iter = 10
        cfg.grad_steps_per_iter = 30
        cfg.eval_episodes = 10
    if args.n_iters is not None:
        cfg.n_iters = args.n_iters

    summary = train(cfg, verbose=not args.quiet)

    print()
    print("=== Final command sweep (greedy eval) ===")
    print(f"{'desired R':>10}  {'achieved R':>12}  {'mean steps':>10}")
    for row in summary["sweep_results"]:
        print(
            f"{row['desired_return']:>10.2f}  "
            f"{row['achieved_return_mean']:>12.3f}  "
            f"{row['mean_steps']:>10.2f}"
        )
    print()
    print(
        f"Random-policy baseline mean return: "
        f"{summary['random_baseline']['mean_return']:.3f}  "
        f"(std {summary['random_baseline']['std_return']:.3f})"
    )
    print(f"Wallclock: {summary['wallclock_sec']:.1f}s   git={git_hash()}")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".",
                    exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.save_json}")


if __name__ == "__main__":
    main()
