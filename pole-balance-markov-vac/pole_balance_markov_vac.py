"""
pole-balance-markov-vac — Vector-valued Adaptive Critic on Markov cart-pole.

Schmidhuber, *Recurrent Networks Adjusted by Adaptive Critics*, IJCNN 1990
Washington DC (also FKI-129-90 and supplementary §6.1 of Schmidhuber 2015,
*Deep Learning in Neural Networks: An Overview*).

Problem
-------
Standard cart-pole, **Markov regime** — the controller observes the full
state s_t = (x, x_dot, theta, theta_dot). Goal: keep the pole upright and
the cart on the track for at least 1,000 steps.

Algorithm — Vector Adaptive Critic (VAC)
----------------------------------------
The 1990 paper generalises Barto/Sutton/Anderson's scalar Adaptive Heuristic
Critic (AHC) to a *vector-valued* critic that predicts several future-return
components in parallel. We implement the Markov special case:

  - Actor (controller)  pi_theta : s -> Bernoulli(p) over force +/- F_mag.
  - Critic V_phi       : s -> R^K. Component 0 predicts discounted
    pole-up return; component 1 predicts discounted cart-centred return.
  - Vector reward at each step:  r_t = [1.0,  1 - |x|/x_thresh].
  - Vector TD residual:           delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
  - Critic update (componentwise TD):  Wc <- Wc + alpha_c * delta_t (x) grad_W V.
  - Actor advantage (scalar mix):  A_t = w . delta_t  with mixing weights
    w = (w_pole, w_cart).
  - Actor REINFORCE step:          theta <- theta + alpha_a * A_t * grad_theta log pi.

Why "vector" matters: in our learned policy, components 0 and 1 disagree
during the early-balance phase (the critic predicts pole-up reward will
keep flowing while cart-centred reward will not), so the *mixed* advantage
A_t carries credit-assignment signal that a scalar critic would smear.
Re-mixing weights w trades off how aggressively the actor centres the
cart vs. keeps the pole vertical without retraining the critic.

Implementation: pure numpy. No torch/gym/scipy.

Reproducibility
---------------
    python3 pole_balance_markov_vac.py --seed 0

Default solves the Markov cart-pole in ~150-300 episodes / a few seconds
on an M-series laptop CPU; the eval phase reports mean balance time
across 20 greedy episodes.
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

import numpy as np


# ----------------------------------------------------------------------
# Cart-pole numpy mini-environment (no gym dependency)
# ----------------------------------------------------------------------

# Standard Sutton-Barto / Florian-corrected cart-pole equations of motion.
GRAVITY = 9.8
M_CART = 1.0
M_POLE = 0.1
M_TOTAL = M_CART + M_POLE
L_HALF = 0.5            # half-length of the pole
F_MAG = 10.0
DT = 0.02

X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * np.pi / 180.0  # 12 degrees, ~ 0.2094 rad


def reset_env(rng: np.random.Generator) -> np.ndarray:
    """Initialise cart-pole near upright, small uniform jitter (gym-equivalent)."""
    return rng.uniform(-0.05, 0.05, size=4).astype(np.float64)


def step_env(state: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
    """One physics step.

    action in {0, 1}: 0 -> push cart left (force = -F_MAG),
                       1 -> push cart right (force = +F_MAG).

    Returns (next_state, terminated).
    """
    x, x_dot, theta, theta_dot = state
    force = F_MAG if action == 1 else -F_MAG
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    temp = (force + M_POLE * L_HALF * theta_dot * theta_dot * sin_t) / M_TOTAL
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        L_HALF * (4.0 / 3.0 - M_POLE * cos_t * cos_t / M_TOTAL)
    )
    x_acc = temp - M_POLE * L_HALF * theta_acc * cos_t / M_TOTAL

    x = x + DT * x_dot
    x_dot = x_dot + DT * x_acc
    theta = theta + DT * theta_dot
    theta_dot = theta_dot + DT * theta_acc

    next_state = np.array([x, x_dot, theta, theta_dot])
    terminated = bool(abs(x) > X_THRESHOLD or abs(theta) > THETA_THRESHOLD)
    return next_state, terminated


# State normalisation (rough scale of each dimension at the threshold).
STATE_SCALE = np.array([X_THRESHOLD, 2.0, THETA_THRESHOLD, 3.0])


def normalize(state: np.ndarray) -> np.ndarray:
    return state / STATE_SCALE


# ----------------------------------------------------------------------
# Networks: actor (Bernoulli policy) + vector critic V: R^4 -> R^K
# ----------------------------------------------------------------------

@dataclass
class VACParams:
    # Actor: 4 -> tanh(H) -> sigmoid(1)
    Wa1: np.ndarray
    ba1: np.ndarray
    Wa2: np.ndarray
    ba2: np.ndarray
    # Critic: 4 -> tanh(H) -> linear(K)
    Wc1: np.ndarray
    bc1: np.ndarray
    Wc2: np.ndarray
    bc2: np.ndarray
    K: int = 2
    H: int = 16

    def copy(self) -> "VACParams":
        return VACParams(
            Wa1=self.Wa1.copy(), ba1=self.ba1.copy(),
            Wa2=self.Wa2.copy(), ba2=self.ba2.copy(),
            Wc1=self.Wc1.copy(), bc1=self.bc1.copy(),
            Wc2=self.Wc2.copy(), bc2=self.bc2.copy(),
            K=self.K, H=self.H,
        )


def init_params(rng: np.random.Generator, hidden: int = 16, K: int = 2,
                scale: float = 0.3) -> VACParams:
    return VACParams(
        Wa1=rng.standard_normal((hidden, 4)) * scale,
        ba1=np.zeros(hidden),
        Wa2=rng.standard_normal((1, hidden)) * scale,
        ba2=np.zeros(1),
        Wc1=rng.standard_normal((hidden, 4)) * scale,
        bc1=np.zeros(hidden),
        Wc2=rng.standard_normal((K, hidden)) * scale,
        bc2=np.zeros(K),
        K=K, H=hidden,
    )


def sigmoid(z: float) -> float:
    # numerically stable scalar sigmoid
    if z >= 0:
        e = np.exp(-z)
        return 1.0 / (1.0 + e)
    e = np.exp(z)
    return e / (1.0 + e)


def actor_forward(p: VACParams, s_norm: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Returns (P(action=1), tanh hidden activations, pre-sigmoid logit)."""
    pre_h = p.Wa1 @ s_norm + p.ba1
    h = np.tanh(pre_h)
    z = float((p.Wa2 @ h + p.ba2)[0])
    pr = sigmoid(z)
    return pr, h, z


def critic_forward(p: VACParams, s_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (V_vector of shape (K,), tanh hidden)."""
    pre_h = p.Wc1 @ s_norm + p.bc1
    h = np.tanh(pre_h)
    v = p.Wc2 @ h + p.bc2
    return v, h


# ----------------------------------------------------------------------
# Updates
# ----------------------------------------------------------------------

def critic_update_(p: VACParams, s_norm: np.ndarray, h_c: np.ndarray,
                   delta: np.ndarray, lr: float) -> None:
    """In-place critic ascent on TD residual delta (vector of shape (K,))."""
    Wc2_old = p.Wc2.copy()
    p.Wc2 += lr * np.outer(delta, h_c)
    p.bc2 += lr * delta
    dh = (Wc2_old.T @ delta) * (1.0 - h_c * h_c)
    p.Wc1 += lr * np.outer(dh, s_norm)
    p.bc1 += lr * dh


def actor_update_(p: VACParams, s_norm: np.ndarray, h_a: np.ndarray,
                  pr: float, action: int, advantage: float, lr: float,
                  entropy_coef: float = 0.01) -> None:
    """In-place actor ascent on advantage * grad log pi(a|s) + entropy bonus.

    For a Bernoulli with P(a=1)=pr, d/dz log pi(a|s) = a - pr.
    Entropy: H(pr) = -pr*log(pr) - (1-pr)*log(1-pr); dH/dz = (0.5 - pr) * 2*pr*(1-pr)
    -- actually simpler: dH/dz = (1 - 2pr) * pr*(1-pr)  ... let me just add a
    log-pi-of-uniform pull. Concrete form below uses gradient of -KL(pi || U)
    which for Bernoulli reduces to -(2pr-1)*pr*(1-pr).
    """
    dlogpi_dz = float(action) - pr  # scalar
    # Entropy gradient (push pr toward 0.5 when entropy_coef > 0).
    dent_dz = (1.0 - 2.0 * pr) * pr * (1.0 - pr)
    dz = lr * advantage * dlogpi_dz + lr * entropy_coef * dent_dz

    # Output layer
    Wa2_old = p.Wa2.copy()
    p.Wa2 += dz * h_a[None, :]
    p.ba2 += np.array([dz])
    # Backprop through tanh hidden: dh = dz * Wa2^T * (1 - h^2)
    dh = dz * Wa2_old[0] * (1.0 - h_a * h_a)
    p.Wa1 += np.outer(dh, s_norm)
    p.ba1 += dh


# ----------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------

def vector_reward(state: np.ndarray) -> np.ndarray:
    """Per-step vector reward.

    Component 0: pole-up reward, +1 each surviving step.
    Component 1: cart-centred reward, 1 - |x|/x_threshold (clipped to [0,1]).
    """
    cart_centred = max(0.0, 1.0 - abs(state[0]) / X_THRESHOLD)
    return np.array([1.0, cart_centred])


def run_episode(p: VACParams, rng: np.random.Generator, *,
                gamma: float, mix_w: np.ndarray,
                actor_lr: float, critic_lr: float,
                entropy_coef: float, max_steps: int,
                train: bool, greedy: bool = False
                ) -> dict:
    """Run one episode. If train=True, apply VAC updates online.

    Returns metrics for the episode.
    """
    state = reset_env(rng)
    total_v_reward = np.zeros(p.K)
    n_steps = 0
    log = {"v_pole": [], "v_cart": [], "actions": [], "states": []}

    for t in range(max_steps):
        s_norm = normalize(state)
        pr, h_a, _ = actor_forward(p, s_norm)
        if greedy:
            action = 1 if pr >= 0.5 else 0
        else:
            action = 1 if rng.random() < pr else 0

        v_t, h_c = critic_forward(p, s_norm)
        log["v_pole"].append(float(v_t[0]))
        log["v_cart"].append(float(v_t[1]))
        log["actions"].append(action)
        log["states"].append(state.copy())

        next_state, terminated = step_env(state, action)
        r_t = vector_reward(next_state)
        total_v_reward += (gamma ** t) * r_t
        n_steps += 1

        if train:
            if terminated:
                v_next = np.zeros(p.K)
            else:
                s_next_norm = normalize(next_state)
                v_next, _ = critic_forward(p, s_next_norm)
            delta = r_t + gamma * v_next - v_t

            critic_update_(p, s_norm, h_c, delta, critic_lr)
            advantage = float(mix_w @ delta)
            actor_update_(p, s_norm, h_a, pr, action, advantage,
                          actor_lr, entropy_coef)

        state = next_state
        if terminated:
            break

    return {
        "n_steps": n_steps,
        "discounted_v_return": total_v_reward,
        "log": log,
    }


# ----------------------------------------------------------------------
# Training driver
# ----------------------------------------------------------------------

@dataclass
class TrainHistory:
    balance_steps: list = field(default_factory=list)
    moving_avg: list = field(default_factory=list)
    snapshot_episode: list = field(default_factory=list)
    snapshot_params: list = field(default_factory=list)
    solve_episode: int | None = None


def train_vac(seed: int = 0, *, hidden: int = 16, K: int = 2,
              gamma: float = 0.99, actor_lr: float = 0.003,
              critic_lr: float = 0.015, entropy_coef: float = 0.005,
              mix_w: tuple = (1.0, 0.3), max_episodes: int = 1000,
              max_steps: int = 1000, solve_window: int = 20,
              solve_threshold: float = 950.0, snapshot_every: int = 50,
              verbose: bool = True) -> tuple[VACParams, TrainHistory]:
    """Train a Vector Adaptive Critic on Markov cart-pole.

    Returns (final params, training history).
    """
    rng = np.random.default_rng(seed)
    p = init_params(rng, hidden=hidden, K=K)
    w = np.asarray(mix_w, dtype=np.float64)
    hist = TrainHistory()
    hist.snapshot_episode.append(0)
    hist.snapshot_params.append(p.copy())

    for ep in range(1, max_episodes + 1):
        info = run_episode(p, rng,
                           gamma=gamma, mix_w=w,
                           actor_lr=actor_lr, critic_lr=critic_lr,
                           entropy_coef=entropy_coef,
                           max_steps=max_steps, train=True)
        hist.balance_steps.append(info["n_steps"])

        # Trailing-window mean
        window = hist.balance_steps[-solve_window:]
        avg = float(np.mean(window))
        hist.moving_avg.append(avg)

        if ep % snapshot_every == 0:
            hist.snapshot_episode.append(ep)
            hist.snapshot_params.append(p.copy())

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"  ep {ep:4d}  balance={info['n_steps']:4d}  "
                  f"trail-avg({len(window)})={avg:6.1f}")

        if (hist.solve_episode is None
                and len(window) >= solve_window
                and avg >= solve_threshold):
            hist.solve_episode = ep
            if verbose:
                print(f"  >>> SOLVED at episode {ep}: trail-avg "
                      f"{avg:.1f} >= {solve_threshold}")
            # Snapshot at solve.
            hist.snapshot_episode.append(ep)
            hist.snapshot_params.append(p.copy())
            break

    if hist.snapshot_episode[-1] != len(hist.balance_steps):
        hist.snapshot_episode.append(len(hist.balance_steps))
        hist.snapshot_params.append(p.copy())

    return p, hist


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(p: VACParams, *, seed: int, n_episodes: int = 20,
             max_steps: int = 1000, gamma: float = 0.99,
             mix_w: tuple = (1.0, 0.3), greedy: bool = True) -> dict:
    rng = np.random.default_rng(seed + 100_000)
    w = np.asarray(mix_w, dtype=np.float64)
    durations = []
    for _ in range(n_episodes):
        info = run_episode(p, rng, gamma=gamma, mix_w=w,
                           actor_lr=0.0, critic_lr=0.0,
                           entropy_coef=0.0, max_steps=max_steps,
                           train=False, greedy=greedy)
        durations.append(info["n_steps"])
    durations = np.asarray(durations)
    return {
        "n_episodes": n_episodes,
        "mean_balance": float(durations.mean()),
        "median_balance": float(np.median(durations)),
        "min_balance": int(durations.min()),
        "max_balance": int(durations.max()),
        "n_perfect": int((durations >= max_steps).sum()),
        "durations": durations.tolist(),
        "greedy": greedy,
    }


# ----------------------------------------------------------------------
# Environment metadata for reproducibility
# ----------------------------------------------------------------------

def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def env_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": _git_hash(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Vector Adaptive Critic on Markov cart-pole (Schmidhuber 1990)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--K", type=int, default=2,
                        help="critic vector dimensionality.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=0.003)
    parser.add_argument("--critic-lr", type=float, default=0.015)
    parser.add_argument("--entropy", type=float, default=0.005)
    parser.add_argument("--mix-w-pole", type=float, default=1.0)
    parser.add_argument("--mix-w-cart", type=float, default=0.3)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--solve-window", type=int, default=20)
    parser.add_argument("--solve-threshold", type=float, default=950.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="optional JSON results path.")
    args = parser.parse_args(argv)

    print(f"# pole-balance-markov-vac  seed={args.seed}")
    print(f"# config: hidden={args.hidden} K={args.K} gamma={args.gamma} "
          f"actor_lr={args.actor_lr} critic_lr={args.critic_lr} "
          f"mix_w=({args.mix_w_pole}, {args.mix_w_cart})")

    t0 = time.time()
    final_p, hist = train_vac(
        seed=args.seed,
        hidden=args.hidden, K=args.K,
        gamma=args.gamma, actor_lr=args.actor_lr,
        critic_lr=args.critic_lr, entropy_coef=args.entropy,
        mix_w=(args.mix_w_pole, args.mix_w_cart),
        max_episodes=args.max_episodes, max_steps=args.max_steps,
        solve_window=args.solve_window,
        solve_threshold=args.solve_threshold,
        verbose=not args.quiet,
    )
    train_time = time.time() - t0

    eval_t0 = time.time()
    eval_info = evaluate(final_p, seed=args.seed,
                         n_episodes=args.eval_episodes,
                         max_steps=args.max_steps, gamma=args.gamma,
                         mix_w=(args.mix_w_pole, args.mix_w_cart),
                         greedy=True)
    eval_time = time.time() - eval_t0

    print()
    print(f"# training: {len(hist.balance_steps)} episodes  "
          f"({train_time:.2f}s)")
    if hist.solve_episode is not None:
        print(f"# SOLVED at episode {hist.solve_episode}  "
              f"(trail-avg threshold = {args.solve_threshold})")
    else:
        last_avg = hist.moving_avg[-1] if hist.moving_avg else 0.0
        print(f"# unsolved at end  (trail-avg = {last_avg:.1f})")

    print(f"# eval ({args.eval_episodes} greedy eps, seed offset 100000):")
    print(f"#   mean balance:   {eval_info['mean_balance']:.1f}")
    print(f"#   median balance: {eval_info['median_balance']:.1f}")
    print(f"#   min/max:        {eval_info['min_balance']}/{eval_info['max_balance']}")
    print(f"#   perfect 1000:   {eval_info['n_perfect']}/{eval_info['n_episodes']}")
    print(f"# wallclock: train {train_time:.2f}s + eval {eval_time:.2f}s")

    if args.out:
        record = {
            "seed": args.seed,
            "config": vars(args),
            "history": {
                "balance_steps": hist.balance_steps,
                "moving_avg": hist.moving_avg,
                "solve_episode": hist.solve_episode,
            },
            "eval": eval_info,
            "wallclock": {"train_s": train_time, "eval_s": eval_time},
            "env": env_metadata(),
        }
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"# wrote {args.out}")

    return 0 if eval_info["mean_balance"] >= 500.0 else 1


if __name__ == "__main__":
    sys.exit(main())
