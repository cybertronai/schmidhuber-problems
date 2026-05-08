"""double-pole-no-velocity - Gomez & Schmidhuber 2005, "Co-evolving recurrent
neurons learn deep memory POMDPs" (GECCO).

Cart with TWO poles of different lengths. Observation = positions only:
(x, theta_1, theta_2). Velocities (x_dot, theta_1_dot, theta_2_dot) are
HIDDEN; the controller must infer them from the position history. A small
recurrent net (Elman, H = 5 hidden) is evolved by ESP (Enforced
Sub-Populations, Gomez 2003): one subpopulation per hidden neuron,
networks assembled by combining one neuron from each subpopulation,
fitness propagated back to the constituent neurons.

Pure numpy. No torch / no gym. CLI: python3 double_pole_no_velocity.py
--seed N.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Double cart-pole (Wieland 1991, Gomez 2003 thesis Appendix A)
# ----------------------------------------------------------------------
#
# State s = (x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot).
# Action u in [-1, 1], applied as force F = u * F_MAX.
#
# Equations of motion (g negative, theta = 0 means upright, +theta to the
# right). For each pole i in {1, 2}:
#
#   m_eff_i = m_i (1 - 3/4 cos^2 theta_i)
#   F_eff_i = m_i l_i theta_dot_i^2 sin theta_i
#             + (3/4) m_i cos theta_i (mu_p theta_dot_i / (m_i l_i)
#                                       + g sin theta_i)
#
#   x_ddot      = (F + sum F_eff_i - mu_c sgn(x_dot)) / (M + sum m_eff_i)
#   theta_ddot_i = -(3 / (4 l_i))
#                  * (x_ddot cos theta_i + g sin theta_i
#                     + mu_p theta_dot_i / (m_i l_i))
#
# We integrate with classical RK4 at dt = 0.01 s (10 ms), the standard
# choice for the double-pole task in Gomez 2003 / Wieland 1991.

GRAVITY = -9.8
M_CART = 1.0
M_POLE_1 = 0.1
M_POLE_2 = 0.01
L_HALF_1 = 0.5      # half-length of long pole
L_HALF_2 = 0.05     # half-length of short pole (1/10 of the long one)
MU_C = 0.0005
MU_P = 0.000002
F_MAX = 10.0
DT = 0.01

X_LIMIT = 2.4                       # m
THETA_LIMIT = 36.0 * np.pi / 180.0  # 0.6283 rad (Wieland 1991 spec)


def double_pole_deriv(state: np.ndarray, force: float) -> np.ndarray:
    """Continuous-time derivative ds/dt for the double cart-pole."""
    x, x_dot, t1, t1_dot, t2, t2_dot = state
    sin1, cos1 = np.sin(t1), np.cos(t1)
    sin2, cos2 = np.sin(t2), np.cos(t2)

    m_eff_1 = M_POLE_1 * (1.0 - 0.75 * cos1 * cos1)
    m_eff_2 = M_POLE_2 * (1.0 - 0.75 * cos2 * cos2)

    f_eff_1 = (M_POLE_1 * L_HALF_1 * t1_dot * t1_dot * sin1
               + 0.75 * M_POLE_1 * cos1
                 * (MU_P * t1_dot / (M_POLE_1 * L_HALF_1) + GRAVITY * sin1))
    f_eff_2 = (M_POLE_2 * L_HALF_2 * t2_dot * t2_dot * sin2
               + 0.75 * M_POLE_2 * cos2
                 * (MU_P * t2_dot / (M_POLE_2 * L_HALF_2) + GRAVITY * sin2))

    cart_friction = MU_C * np.sign(x_dot)
    x_ddot = ((force + f_eff_1 + f_eff_2 - cart_friction)
              / (M_CART + m_eff_1 + m_eff_2))

    t1_ddot = -(3.0 / (4.0 * L_HALF_1)) * (
        x_ddot * cos1 + GRAVITY * sin1 + MU_P * t1_dot / (M_POLE_1 * L_HALF_1)
    )
    t2_ddot = -(3.0 / (4.0 * L_HALF_2)) * (
        x_ddot * cos2 + GRAVITY * sin2 + MU_P * t2_dot / (M_POLE_2 * L_HALF_2)
    )
    return np.array([x_dot, x_ddot, t1_dot, t1_ddot, t2_dot, t2_ddot])


def double_pole_step(state: np.ndarray, force: float) -> np.ndarray:
    """One RK4 step of the double cart-pole."""
    k1 = double_pole_deriv(state, force)
    k2 = double_pole_deriv(state + 0.5 * DT * k1, force)
    k3 = double_pole_deriv(state + 0.5 * DT * k2, force)
    k4 = double_pole_deriv(state + DT * k3, force)
    return state + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def init_state(theta1_0: float = 4.5 * np.pi / 180.0) -> np.ndarray:
    """Initial state: long pole tilted by theta1_0, everything else zero."""
    return np.array([0.0, 0.0, theta1_0, 0.0, 0.0, 0.0])


def init_state_random(rng: np.random.Generator,
                      theta1_max: float = 4.5 * np.pi / 180.0) -> np.ndarray:
    """Initial state with random long-pole tilt in [-theta1_max, +theta1_max]."""
    return np.array([0.0, 0.0, rng.uniform(-theta1_max, theta1_max),
                     0.0, 0.0, 0.0])


def is_failed(state: np.ndarray) -> bool:
    return (abs(state[0]) > X_LIMIT
            or abs(state[2]) > THETA_LIMIT
            or abs(state[4]) > THETA_LIMIT)


# Normalize positions for RNN input (each component in roughly [-1, 1]).
def normalize_obs(state: np.ndarray) -> np.ndarray:
    return np.array([state[0] / X_LIMIT,
                     state[2] / THETA_LIMIT,
                     state[4] / THETA_LIMIT])


# ----------------------------------------------------------------------
# ESP-encoded recurrent network
# ----------------------------------------------------------------------
#
# Architecture: Elman RNN with tanh activations.
#
#   h_t = tanh(W_x x_t + W_h h_{t-1} + b)              (H hidden units)
#   u_t = tanh(V h_t + c)                              (1 output)
#
# For ESP, every individual encodes the parameters of one hidden neuron i:
#
#   genome_i = [W_x[i, :]  (IN_DIM),
#               W_h[i, :]  (H),
#               b[i]       (1),
#               V[0, i]    (1)]                # output bias c is shared/zero
#
# A network is assembled by stacking H individuals (one from each
# subpopulation) row-wise into the weight matrices.

IN_DIM = 3      # (x, theta_1, theta_2) normalized


def gene_size(hidden: int) -> int:
    return IN_DIM + hidden + 1 + 1   # W_x, W_h, b, V


def assemble_network(genomes: np.ndarray):
    """genomes: (H, gene_size) -> dict of weight matrices."""
    H = genomes.shape[0]
    W_x = genomes[:, :IN_DIM].copy()                 # (H, IN_DIM)
    W_h = genomes[:, IN_DIM:IN_DIM + H].copy()       # (H, H)
    b   = genomes[:, IN_DIM + H].copy()              # (H,)
    V   = genomes[:, IN_DIM + H + 1].reshape(1, H).copy()   # (1, H)
    c   = np.zeros(1)
    return {"W_x": W_x, "W_h": W_h, "b": b, "V": V, "c": c}


def network_step(net: dict, h: np.ndarray, x: np.ndarray):
    """One forward step. Returns (action u in [-1,1], new hidden h)."""
    pre = net["W_x"] @ x + net["W_h"] @ h + net["b"]
    h_new = np.tanh(pre)
    u_pre = (net["V"] @ h_new + net["c"]).item()
    u = float(np.tanh(u_pre))
    return u, h_new


def run_episode(net: dict, state0: np.ndarray, T_max: int,
                hidden_dim: int) -> int:
    """Roll out the network on the double cart-pole. Return balance time
    (capped at T_max)."""
    state = state0.copy()
    h = np.zeros(hidden_dim)
    for t in range(T_max):
        u, h = network_step(net, h, normalize_obs(state))
        state = double_pole_step(state, u * F_MAX)
        if is_failed(state):
            return t + 1
    return T_max


# ----------------------------------------------------------------------
# ESP population
# ----------------------------------------------------------------------

@dataclass
class ESPConfig:
    hidden: int = 5
    pop_size: int = 40
    trials_per_indiv: int = 4         # K participations per individual / gen
    max_generations: int = 200
    elite_frac: float = 0.25          # top fraction kept as parents
    mut_prob: float = 0.4             # per-gene mutation probability
    mut_sigma: float = 0.3            # Gaussian mutation std
    init_scale: float = 0.5           # std of initial Gaussian weights
    cauchy_frac: float = 0.0          # if > 0, fraction of mutations from Cauchy
    burst_after_stale: int = 20       # generations w/o improvement before burst
    solve_threshold: int = 1000       # balance steps to call it solved


class ESPPopulation:
    """H subpopulations, each of size pop_size; per-individual fitness stats."""

    def __init__(self, cfg: ESPConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.gene_dim = gene_size(cfg.hidden)
        # subpops shape: (H, pop_size, gene_dim)
        self.pop = (rng.standard_normal((cfg.hidden, cfg.pop_size, self.gene_dim))
                    * cfg.init_scale).astype(np.float64)
        # accumulated fitness sum and trial count per individual
        self.fit_sum = np.zeros((cfg.hidden, cfg.pop_size), dtype=np.float64)
        self.fit_cnt = np.zeros((cfg.hidden, cfg.pop_size), dtype=np.int64)

    def reset_stats(self):
        self.fit_sum[...] = 0.0
        self.fit_cnt[...] = 0

    def assemble_random(self):
        """Pick one random individual per subpop, return indices and assembled net."""
        idx = self.rng.integers(0, self.cfg.pop_size, size=self.cfg.hidden)
        rows = self.pop[np.arange(self.cfg.hidden), idx]
        return idx, assemble_network(rows)

    def assemble_with(self, fixed_subpop: int, fixed_idx: int):
        """Assemble a net with a specific individual in `fixed_subpop` and
        random partners elsewhere."""
        idx = self.rng.integers(0, self.cfg.pop_size, size=self.cfg.hidden)
        idx[fixed_subpop] = fixed_idx
        rows = self.pop[np.arange(self.cfg.hidden), idx]
        return idx, assemble_network(rows)

    def credit(self, idx: np.ndarray, fitness: float):
        """Add `fitness` to each constituent neuron's running sum."""
        for s in range(self.cfg.hidden):
            self.fit_sum[s, idx[s]] += fitness
            self.fit_cnt[s, idx[s]] += 1

    def mean_fitness(self) -> np.ndarray:
        cnt = np.maximum(self.fit_cnt, 1)
        return self.fit_sum / cnt

    def best_assembly(self):
        """Greedy assembly: from each subpop, pick the individual with the
        highest mean fitness."""
        mf = self.mean_fitness()
        idx = mf.argmax(axis=1)              # (H,)
        rows = self.pop[np.arange(self.cfg.hidden), idx]
        return idx, assemble_network(rows)

    # ---- evolution ----------------------------------------------------

    def evolve_step(self):
        """One generation of selection + crossover + mutation per subpop."""
        cfg = self.cfg
        n_elite = max(2, int(round(cfg.elite_frac * cfg.pop_size)))
        mf = self.mean_fitness()
        new_pop = np.empty_like(self.pop)

        for s in range(cfg.hidden):
            order = np.argsort(-mf[s])              # descending
            elites = self.pop[s, order[:n_elite]]   # (n_elite, gene_dim)
            new_pop[s, :n_elite] = elites           # elitism
            # children: one-point crossover + mutation, parents drawn uniformly
            n_children = cfg.pop_size - n_elite
            for j in range(n_children):
                pa, pb = self.rng.integers(0, n_elite, size=2)
                cut = self.rng.integers(1, self.gene_dim)
                child = np.concatenate([elites[pa, :cut], elites[pb, cut:]])
                # gaussian mutation
                mask = self.rng.random(self.gene_dim) < cfg.mut_prob
                if cfg.cauchy_frac > 0.0:
                    use_cauchy = self.rng.random(self.gene_dim) < cfg.cauchy_frac
                    noise_g = self.rng.standard_normal(self.gene_dim) * cfg.mut_sigma
                    noise_c = self.rng.standard_cauchy(self.gene_dim) * cfg.mut_sigma
                    noise = np.where(use_cauchy, noise_c, noise_g)
                else:
                    noise = self.rng.standard_normal(self.gene_dim) * cfg.mut_sigma
                child[mask] += noise[mask]
                new_pop[s, n_elite + j] = child
        self.pop = new_pop
        self.reset_stats()

    def burst(self, sigma: float = 0.3):
        """Burst mutation: keep top elite from each subpop, regenerate the
        rest with Gaussian noise around it. Used to escape stagnation."""
        cfg = self.cfg
        mf = self.mean_fitness()
        for s in range(cfg.hidden):
            best = self.pop[s, mf[s].argmax()].copy()
            # keep best at slot 0
            self.pop[s, 0] = best
            self.pop[s, 1:] = (best
                               + self.rng.standard_normal(
                                   (cfg.pop_size - 1, self.gene_dim)) * sigma)
        self.reset_stats()


# ----------------------------------------------------------------------
# Evolutionary outer loop
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 0
    hidden: int = 5
    pop_size: int = 40
    trials_per_indiv: int = 4
    max_generations: int = 200
    elite_frac: float = 0.25
    mut_prob: float = 0.4
    mut_sigma: float = 0.3
    init_scale: float = 0.5
    burst_after_stale: int = 25
    solve_threshold: int = 1000
    eval_T_max: int = 1000
    final_eval_episodes: int = 20
    final_eval_theta1_max: float = 4.5 * np.pi / 180.0
    init_theta1: float = 4.5 * np.pi / 180.0


def evolve(cfg: RunConfig, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    esp_cfg = ESPConfig(
        hidden=cfg.hidden, pop_size=cfg.pop_size,
        trials_per_indiv=cfg.trials_per_indiv,
        max_generations=cfg.max_generations,
        elite_frac=cfg.elite_frac, mut_prob=cfg.mut_prob,
        mut_sigma=cfg.mut_sigma, init_scale=cfg.init_scale,
        burst_after_stale=cfg.burst_after_stale,
        solve_threshold=cfg.solve_threshold,
    )
    pop = ESPPopulation(esp_cfg, rng)

    history = {
        "gen": [],
        "best_fitness": [],
        "mean_fitness": [],
        "trials_per_gen": [],
        "n_solved_in_gen": [],
        "burst": [],
    }
    best_overall = -1
    best_overall_genomes = None
    stale = 0
    state0 = init_state(cfg.init_theta1)
    total_trials = 0

    for gen in range(1, cfg.max_generations + 1):
        pop.reset_stats()
        # K trials per individual: iterate (subpop, individual, trial), assemble
        # with that individual fixed, partners random.
        n_trials_target = cfg.trials_per_indiv * cfg.pop_size * cfg.hidden
        # Simpler: do `trials_per_indiv` random assemblies per individual.
        n_trials = 0
        n_solved = 0
        for s in range(cfg.hidden):
            for j in range(cfg.pop_size):
                for _ in range(cfg.trials_per_indiv):
                    idx, net = pop.assemble_with(s, j)
                    fitness = run_episode(net, state0, cfg.eval_T_max,
                                          cfg.hidden)
                    pop.credit(idx, float(fitness))
                    n_trials += 1
                    if fitness >= cfg.solve_threshold:
                        n_solved += 1
        total_trials += n_trials

        mf = pop.mean_fitness()
        gen_best = float(mf.max())
        gen_mean = float(mf.mean())
        # snapshot best assembly under the *just-evaluated* fitness
        idx_best, net_best = pop.best_assembly()
        best_genomes = pop.pop[np.arange(cfg.hidden), idx_best].copy()
        # Confirm with a fresh eval (best_assembly is greedy across subpops, may
        # not have been assembled together).
        confirm_balance = run_episode(net_best, state0, cfg.eval_T_max,
                                      cfg.hidden)

        if confirm_balance > best_overall:
            best_overall = confirm_balance
            best_overall_genomes = best_genomes
            stale = 0
        else:
            stale += 1

        burst_now = False
        if (stale >= cfg.burst_after_stale
                and gen < cfg.max_generations
                and best_overall < cfg.solve_threshold):
            pop.burst(sigma=cfg.init_scale)
            burst_now = True
            stale = 0

        history["gen"].append(gen)
        history["best_fitness"].append(int(confirm_balance))
        history["mean_fitness"].append(gen_mean)
        history["trials_per_gen"].append(n_trials)
        history["n_solved_in_gen"].append(n_solved)
        history["burst"].append(burst_now)

        if verbose and (gen == 1 or gen % 5 == 0
                        or burst_now
                        or best_overall >= cfg.solve_threshold):
            print(f"  gen {gen:4d}  trials={n_trials}  "
                  f"best_assembly_eval={confirm_balance:5d}  "
                  f"mean_fit={gen_mean:7.1f}  n_solved/{n_trials}={n_solved}"
                  + ("  [burst]" if burst_now else ""))

        if best_overall >= cfg.solve_threshold:
            if verbose:
                print(f"  -> solved at gen {gen} "
                      f"(best assembly balanced {best_overall} steps).")
            # keep the best genomes; do not break early — actually break to
            # conserve budget, the user can extend with --max-generations.
            if confirm_balance >= cfg.solve_threshold:
                break

        # crossover + mutation step (skip on the very last gen so we keep stats)
        if gen < cfg.max_generations:
            pop.evolve_step()

    return {
        "history": history,
        "best_genomes": best_overall_genomes,
        "best_balance": int(best_overall),
        "total_trials": total_trials,
        "config": cfg,
    }


# ----------------------------------------------------------------------
# Final evaluation: run the best assembled net on K random initial conditions
# ----------------------------------------------------------------------

def final_eval(best_genomes: np.ndarray, cfg: RunConfig,
               rng: np.random.Generator) -> dict:
    net = assemble_network(best_genomes)
    times = []
    for _ in range(cfg.final_eval_episodes):
        s0 = init_state_random(rng, cfg.final_eval_theta1_max)
        t = run_episode(net, s0, cfg.eval_T_max, cfg.hidden)
        times.append(int(t))
    times = np.array(times)
    return {
        "times": times.tolist(),
        "mean": float(times.mean()),
        "median": int(np.median(times)),
        "min": int(times.min()),
        "max": int(times.max()),
        "n_solved": int((times >= cfg.solve_threshold).sum()),
        "n_episodes": int(len(times)),
    }


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


# ----------------------------------------------------------------------
# Top-level run
# ----------------------------------------------------------------------

def run(cfg: RunConfig, verbose: bool = True) -> dict:
    if verbose:
        print(f"=== ESP co-evolution: H={cfg.hidden}, pop={cfg.pop_size}, "
              f"trials/indiv={cfg.trials_per_indiv}, max_gen="
              f"{cfg.max_generations} ===")
    t0 = time.time()
    res = evolve(cfg, verbose=verbose)
    t_evolve = time.time() - t0
    if verbose:
        print(f"  Evolution done in {t_evolve:.1f}s.  "
              f"best balance under fixed init = {res['best_balance']} "
              f"(trials = {res['total_trials']:,d})")

    if verbose:
        print(f"\n=== Final eval ({cfg.final_eval_episodes} random inits, "
              f"|theta1_0| <= {np.degrees(cfg.final_eval_theta1_max):.2f} deg, "
              f"cap {cfg.eval_T_max}) ===")
    rng_eval = np.random.default_rng(cfg.seed + 1_000_000)
    final = final_eval(res["best_genomes"], cfg, rng_eval)
    if verbose:
        print(f"  balance: mean={final['mean']:.1f}  median={final['median']} "
              f"min={final['min']}  max={final['max']}  "
              f">= {cfg.solve_threshold}: "
              f"{final['n_solved']}/{final['n_episodes']}")

    return {
        "evolve": res,
        "final": final,
        "t_evolve": t_evolve,
        "env": env_info(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    d = RunConfig()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--hidden", type=int, default=d.hidden)
    p.add_argument("--pop", type=int, default=d.pop_size,
                   help="individuals per subpopulation")
    p.add_argument("--trials", type=int, default=d.trials_per_indiv,
                   help="trial assemblies per individual per generation")
    p.add_argument("--max-gen", type=int, default=d.max_generations)
    p.add_argument("--elite-frac", type=float, default=d.elite_frac)
    p.add_argument("--mut-prob", type=float, default=d.mut_prob)
    p.add_argument("--mut-sigma", type=float, default=d.mut_sigma)
    p.add_argument("--init-scale", type=float, default=d.init_scale)
    p.add_argument("--burst-after", type=int, default=d.burst_after_stale)
    p.add_argument("--solve", type=int, default=d.solve_threshold)
    p.add_argument("--eval-T", type=int, default=d.eval_T_max)
    p.add_argument("--final-eps", type=int, default=d.final_eval_episodes)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--save-json", type=str, default=None,
                   help="dump final summary to this path")
    args = p.parse_args()

    cfg = RunConfig(
        seed=args.seed,
        hidden=args.hidden,
        pop_size=args.pop,
        trials_per_indiv=args.trials,
        max_generations=args.max_gen,
        elite_frac=args.elite_frac,
        mut_prob=args.mut_prob,
        mut_sigma=args.mut_sigma,
        init_scale=args.init_scale,
        burst_after_stale=args.burst_after,
        solve_threshold=args.solve,
        eval_T_max=args.eval_T,
        final_eval_episodes=args.final_eps,
    )

    t0 = time.time()
    res = run(cfg, verbose=not args.quiet)
    wall = time.time() - t0

    summary = {
        "seed": cfg.seed,
        "config": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in cfg.__dict__.items()},
        "env": res["env"],
        "best_balance_during_evolution": res["evolve"]["best_balance"],
        "total_trials": res["evolve"]["total_trials"],
        "final": res["final"],
        "t_evolve_seconds": res["t_evolve"],
        "wallclock_seconds": wall,
    }
    print(f"\nWallclock: {wall:.1f}s "
          f"(evolution {res['t_evolve']:.1f}s)")
    print(f"Best balance during evolution (fixed init {np.degrees(cfg.init_theta1):.2f} deg): "
          f"{res['evolve']['best_balance']}")
    print(f"Final {cfg.final_eval_episodes}-init eval: mean = "
          f"{res['final']['mean']:.1f}, "
          f">= {cfg.solve_threshold}: "
          f"{res['final']['n_solved']}/{cfg.final_eval_episodes}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Wrote summary -> {args.save_json}")
    return summary


if __name__ == "__main__":
    main()
