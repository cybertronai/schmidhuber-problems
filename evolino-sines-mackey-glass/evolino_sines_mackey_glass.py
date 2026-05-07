"""evolino-sines-mackey-glass — Schmidhuber, Wierstra, Gomez 2005/2007.

Evolino = EVolution of recurrent systems with Optimal LINear Output.

Two time-series prediction problems:

  (a) Superimposed sines: y(t) = sum_k sin(omega_k * t) for incommensurate
      omega_k. The headline run uses three sines with frequencies
      (0.20, 0.311, 0.42), a non-rational mix the network cannot solve by
      memorising a single period.
  (b) Mackey-Glass with delay tau=17: a chaotic series widely used in
      time-series benchmarks since Lapedes & Farber 1987.

The Evolino idea: a small recurrent net (here an LSTM) has its hidden-layer
weights evolved by a population-based search; the linear readout from
hidden states to scalar prediction is solved per individual in closed form
by least-squares (Moore-Penrose pseudo-inverse). The closed-form readout
makes per-individual fitness evaluation cheap and removes a whole class of
local minima the evolutionary search would otherwise have to crawl over.

This file holds the entire pipeline: dataset generators, LSTM forward,
fitness via linear regression on the hidden-state matrix, an
evolutionary outer loop with elitism, mating, gaussian mutation, and burst
mutation on stagnation, free-running closed-loop evaluation, and a CLI
that reproduces both headline runs in well under five minutes on an
M-series laptop.

Pure numpy.
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def superimposed_sines(
    T: int,
    freqs: tuple[float, ...] = (0.20, 0.311, 0.42),
) -> np.ndarray:
    """Generate y(t) = (1/K) sum_k sin(omega_k * t) for t in 0..T-1.

    Frequencies are chosen incommensurate so the sum has no short period.
    Output normalised by 1/K so the range stays close to [-1, 1].
    """
    t = np.arange(T, dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    for omega in freqs:
        y += np.sin(omega * t)
    y /= len(freqs)
    return y


def mackey_glass(
    T: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    x0: float = 1.2,
    burn_in: int = 200,
) -> np.ndarray:
    """Numerically integrate the Mackey-Glass delay-differential equation.

      dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

    Constant initial-condition history x(t) = x0 for t < tau, Euler step
    of size dt, then drop the first `burn_in` samples to avoid the
    transient. Returns a length-T trajectory.
    """
    history_len = int(round(tau / dt))
    total = T + burn_in
    series = np.empty(total + history_len + 1, dtype=np.float64)
    series[: history_len + 1] = x0
    for i in range(history_len + 1, total + history_len + 1):
        x_tau = series[i - history_len - 1]
        x = series[i - 1]
        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x
        series[i] = x + dt * dx
    return series[history_len + 1 + burn_in : history_len + 1 + burn_in + T]


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LSTM:
    """Vanilla LSTM with input dim 1 and a configurable hidden width.

    The genome encodes the four gate weight blocks (z, i, f, o), each block
    a (input_dim + hidden + 1, hidden) matrix where the +1 row is a bias
    row. The output layer is *not* part of the genome; it is solved per
    individual by the caller via least-squares on the hidden-state matrix.
    """

    def __init__(self, hidden: int, input_dim: int = 1):
        self.h = hidden
        self.x_dim = input_dim
        self.row = input_dim + hidden + 1  # rows of [x | h_prev | bias]
        self.gene_per_gate = self.row * hidden
        self.gene_size = 4 * self.gene_per_gate

    def unflatten(self, genome: np.ndarray) -> dict[str, np.ndarray]:
        """Slice a flat genome vector into the four gate weight blocks."""
        out = {}
        offset = 0
        for name in ("z", "i", "f", "o"):
            block = genome[offset : offset + self.gene_per_gate]
            out[name] = block.reshape(self.row, self.h)
            offset += self.gene_per_gate
        return out

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.h), np.zeros(self.h)

    def step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        gates: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        xh1 = np.concatenate([x_t, h_prev, np.ones(1)])
        z = np.tanh(xh1 @ gates["z"])
        i = _sigmoid(xh1 @ gates["i"])
        f = _sigmoid(xh1 @ gates["f"] + 1.0)  # forget-gate bias +1 (Gers 2000)
        o = _sigmoid(xh1 @ gates["o"])
        c = f * c_prev + i * z
        h = o * np.tanh(c)
        return h, c

    def run(
        self,
        genome: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Run the LSTM over `inputs` (T, x_dim). Return hidden states (T, h)."""
        gates = self.unflatten(genome)
        T = inputs.shape[0]
        h_t, c_t = self.initial_state()
        H = np.empty((T, self.h), dtype=np.float64)
        for t in range(T):
            h_t, c_t = self.step(inputs[t], h_t, c_t, gates)
            H[t] = h_t
        return H

    def run_with_state(
        self,
        genome: np.ndarray,
        inputs: np.ndarray,
        h0: np.ndarray | None = None,
        c0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gates = self.unflatten(genome)
        T = inputs.shape[0]
        h_t = np.zeros(self.h) if h0 is None else h0.copy()
        c_t = np.zeros(self.h) if c0 is None else c0.copy()
        H = np.empty((T, self.h), dtype=np.float64)
        for t in range(T):
            h_t, c_t = self.step(inputs[t], h_t, c_t, gates)
            H[t] = h_t
        return H, h_t, c_t


# ---------------------------------------------------------------------------
# Evolino: linear-regression inner loop, evolutionary outer loop
# ---------------------------------------------------------------------------


@dataclass
class EvolinoConfig:
    hidden: int = 6
    pop_size: int = 40
    n_gens: int = 80
    elite: int = 4
    mutation_sigma: float = 0.20
    mutation_rate: float = 0.15
    init_sigma: float = 0.30
    burst_after: int = 15
    seed: int = 0
    ridge: float = 1e-6  # Tikhonov regularisation in the inner regression


def linear_readout(
    H: np.ndarray, Y: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the closed-form readout from hidden states to targets.

    Inputs:
      H: (T, h) hidden-state matrix
      Y: (T,)   scalar targets
      ridge: tikhonov term added to H^T H for numerical stability

    Returns (W, Y_pred) where W has shape (h+1,) — last entry is the bias.
    """
    T, h = H.shape
    X = np.concatenate([H, np.ones((T, 1))], axis=1)  # bias column
    A = X.T @ X + ridge * np.eye(h + 1)
    b = X.T @ Y
    W = np.linalg.solve(A, b)
    Y_pred = X @ W
    return W, Y_pred


def evaluate(
    genome: np.ndarray,
    lstm: LSTM,
    inputs: np.ndarray,
    targets: np.ndarray,
    washout: int,
    ridge: float,
    val_horizon: int = 0,
    val_targets: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run an LSTM, fit a linear readout, return fitness.

    The training window is teacher-forced over `inputs`. Hidden states from
    index `washout` onward are regressed against `targets[washout:]` to get
    the readout W.

    If `val_horizon > 0`, the network is then *free-run* (closed-loop:
    its previous prediction is fed back as the next input) for `val_horizon`
    steps starting from the trained terminal state, and fitness is
    `-MSE` on that closed-loop validation window. This is the Evolino
    scoring rule from §3 of Schmidhuber, Wierstra & Gomez 2007: the
    evolutionary search must drive the network to be a useful predictor
    of itself, not merely a teacher-forced fit.

    If `val_horizon == 0`, fitness falls back to teacher-forced training MSE.

    Returns (fitness, W_readout, hidden_states).
    """
    H = lstm.run(genome, inputs)
    H_train = H[washout:]
    Y_train = targets[washout:]
    W, Y_pred = linear_readout(H_train, Y_train, ridge)
    train_mse = float(np.mean((Y_pred - Y_train) ** 2))

    if val_horizon == 0 or val_targets is None:
        return -train_mse, W, H

    # closed-loop free run from the post-training terminal state
    gates = lstm.unflatten(genome)
    h_t, c_t = lstm.initial_state()
    for t in range(inputs.shape[0]):
        h_t, c_t = lstm.step(inputs[t], h_t, c_t, gates)
    feat = np.concatenate([h_t, np.ones(1)])
    y_pred = float(feat @ W)
    preds = np.empty(val_horizon, dtype=np.float64)
    preds[0] = y_pred
    x_t = np.array([y_pred])
    for t in range(1, val_horizon):
        h_t, c_t = lstm.step(x_t, h_t, c_t, gates)
        feat = np.concatenate([h_t, np.ones(1)])
        y_pred = float(feat @ W)
        preds[t] = y_pred
        x_t = np.array([y_pred])

    val_mse = float(np.mean((preds - val_targets[:val_horizon]) ** 2))
    # Penalty if the closed-loop run blows up (hidden saturation -> diverges)
    if not np.isfinite(val_mse):
        val_mse = 1e6
    # Combine train and val MSE with weight on val (closed-loop is the goal)
    fit = -(0.1 * train_mse + val_mse)
    return fit, W, H


def evolve(
    lstm: LSTM,
    inputs: np.ndarray,
    targets: np.ndarray,
    washout: int,
    cfg: EvolinoConfig,
    log_every: int = 5,
    verbose: bool = False,
    val_horizon: int = 0,
    val_targets: np.ndarray | None = None,
) -> dict:
    """Run the evolutionary outer loop.

    Returns a dict with keys 'best_genome', 'best_W', 'best_fit', 'history'.
    """
    rng = np.random.default_rng(cfg.seed)
    pop = rng.normal(0.0, cfg.init_sigma, (cfg.pop_size, lstm.gene_size))
    fits = np.full(cfg.pop_size, -np.inf)

    best_genome = None
    best_W = None
    best_fit = -np.inf
    history: list[dict] = []
    stagnation = 0
    last_best = -np.inf

    for gen in range(cfg.n_gens):
        for i in range(cfg.pop_size):
            f, W, _ = evaluate(
                pop[i], lstm, inputs, targets, washout, cfg.ridge,
                val_horizon=val_horizon, val_targets=val_targets,
            )
            fits[i] = f
            if f > best_fit:
                best_fit = f
                best_genome = pop[i].copy()
                best_W = W

        order = np.argsort(-fits)
        pop = pop[order]
        fits = fits[order]

        history.append(
            {
                "gen": gen,
                "best_fit": float(fits[0]),
                "mean_fit": float(np.mean(fits)),
                "best_mse": float(-fits[0]),
            }
        )

        if verbose and (gen % log_every == 0 or gen == cfg.n_gens - 1):
            print(
                f"  gen {gen:4d}  best_mse={-fits[0]:.6f}  "
                f"mean_mse={-np.mean(fits):.6f}",
                flush=True,
            )

        # reproduction
        n_keep = cfg.elite
        n_parents = cfg.pop_size // 2
        n_children = cfg.pop_size - n_keep
        children = np.empty((n_children, lstm.gene_size))
        for k in range(n_children):
            p1 = pop[rng.integers(n_parents)]
            p2 = pop[rng.integers(n_parents)]
            mask = rng.random(lstm.gene_size) < 0.5
            child = np.where(mask, p1, p2)
            mut_mask = rng.random(lstm.gene_size) < cfg.mutation_rate
            child = child + mut_mask * rng.normal(
                0.0, cfg.mutation_sigma, lstm.gene_size
            )
            children[k] = child
        pop[n_keep:] = children

        # burst mutation on stagnation: re-spray bottom half around best
        if best_fit <= last_best + 1e-12:
            stagnation += 1
        else:
            stagnation = 0
            last_best = best_fit
        if stagnation > cfg.burst_after:
            half = cfg.pop_size // 2
            jitter = rng.normal(0.0, cfg.init_sigma, (half, lstm.gene_size))
            pop[half:] = best_genome[None, :] + jitter
            stagnation = 0

    return {
        "best_genome": best_genome,
        "best_W": best_W,
        "best_fit": float(best_fit),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Closed-loop free-running prediction
# ---------------------------------------------------------------------------


def free_run(
    lstm: LSTM,
    genome: np.ndarray,
    W_readout: np.ndarray,
    warmup_inputs: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Warm up the LSTM on `warmup_inputs`, then free-run for `horizon` steps.

    During warmup we drive with the true sequence; for free-running we feed
    the network's previous output back in as the next input. The first
    output is computed at the last warmup step.

    Returns the predicted sequence of length `horizon`.
    """
    gates = lstm.unflatten(genome)
    h_t, c_t = lstm.initial_state()
    for t in range(warmup_inputs.shape[0]):
        h_t, c_t = lstm.step(warmup_inputs[t], h_t, c_t, gates)

    # First prediction: take the most recent hidden, run through W_readout.
    preds = np.empty(horizon, dtype=np.float64)
    x_t = warmup_inputs[-1]  # last true input; will be overwritten by output
    # produce one prediction at the END of warmup, then iterate
    feat = np.concatenate([h_t, np.ones(1)])
    y_pred = float(feat @ W_readout)
    preds[0] = y_pred
    x_t = np.array([y_pred])
    for t in range(1, horizon):
        h_t, c_t = lstm.step(x_t, h_t, c_t, gates)
        feat = np.concatenate([h_t, np.ones(1)])
        y_pred = float(feat @ W_readout)
        preds[t] = y_pred
        x_t = np.array([y_pred])
    return preds


def teacher_forced_predict(
    lstm: LSTM,
    genome: np.ndarray,
    W_readout: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """Run the LSTM teacher-forced over `inputs`, return readout predictions.

    Useful for the training-window plot and for sanity-checking that the
    inner-loop linear regression actually fits the training targets.
    """
    H = lstm.run(genome, inputs)
    feat = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
    return feat @ W_readout


# ---------------------------------------------------------------------------
# Task wrappers
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    name: str
    train_mse: float
    free_run_mse: float
    free_run_horizon: int
    nrmse_84: float | None  # standard MG metric: NRMSE at horizon 84
    history: list
    best_genome: np.ndarray
    best_W: np.ndarray
    series: np.ndarray
    washout: int
    train_end: int
    free_run_pred: np.ndarray


def _normalised_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    std = float(np.std(target))
    if std < 1e-12:
        return rmse
    return rmse / std


def run_sines(cfg: EvolinoConfig, verbose: bool = True) -> TaskResult:
    """Headline sines run: 3 incommensurate sines, predict next value."""
    T_total = 700
    washout = 100
    train_end = 400  # train fits next-step on indices [washout..train_end)
    val_end = 500    # closed-loop validation: train_end..val_end
    series = superimposed_sines(T_total, freqs=(0.20, 0.311, 0.42))
    # next-step prediction: input at t is y[t-1], target at t is y[t]
    inputs = series[:-1].reshape(-1, 1)
    targets = series[1:]

    train_inputs = inputs[:train_end]
    train_targets = targets[:train_end]
    val_targets = targets[train_end:val_end]

    lstm = LSTM(hidden=cfg.hidden, input_dim=1)
    if verbose:
        print(f"sines: pop={cfg.pop_size}, gens={cfg.n_gens}, hidden={cfg.hidden}, "
              f"genome={lstm.gene_size}")
    result = evolve(
        lstm, train_inputs, train_targets, washout, cfg, verbose=verbose,
        val_horizon=val_end - train_end, val_targets=val_targets,
    )

    train_pred = teacher_forced_predict(
        lstm, result["best_genome"], result["best_W"], train_inputs
    )
    train_mse = float(np.mean((train_pred[washout:] - train_targets[washout:]) ** 2))

    horizon = T_total - 1 - train_end  # predict the rest free-running
    free_pred = free_run(
        lstm,
        result["best_genome"],
        result["best_W"],
        warmup_inputs=train_inputs,
        horizon=horizon,
    )
    target_free = targets[train_end : train_end + horizon]
    free_mse = float(np.mean((free_pred - target_free) ** 2))

    return TaskResult(
        name="sines",
        train_mse=train_mse,
        free_run_mse=free_mse,
        free_run_horizon=horizon,
        nrmse_84=None,
        history=result["history"],
        best_genome=result["best_genome"],
        best_W=result["best_W"],
        series=series,
        washout=washout,
        train_end=train_end,
        free_run_pred=free_pred,
    )


def run_mackey_glass(cfg: EvolinoConfig, verbose: bool = True) -> TaskResult:
    """Headline Mackey-Glass run: tau=17, predict next sample free-running."""
    T_total = 1000
    washout = 100
    train_end = 600
    val_end = 700  # closed-loop validation horizon during evolution
    series = mackey_glass(T_total, tau=17)
    # rescale to roughly [-1, 1] like the paper does (subtract 1, scale by 1.0)
    # MG natural range is roughly [0.4, 1.4], so subtract mean for stability.
    series = series - np.mean(series)
    series = series / (np.std(series) + 1e-12)

    inputs = series[:-1].reshape(-1, 1)
    targets = series[1:]
    train_inputs = inputs[:train_end]
    train_targets = targets[:train_end]
    val_targets = targets[train_end:val_end]

    lstm = LSTM(hidden=cfg.hidden, input_dim=1)
    if verbose:
        print(f"mackey-glass: pop={cfg.pop_size}, gens={cfg.n_gens}, "
              f"hidden={cfg.hidden}, genome={lstm.gene_size}")
    result = evolve(
        lstm, train_inputs, train_targets, washout, cfg, verbose=verbose,
        val_horizon=val_end - train_end, val_targets=val_targets,
    )

    train_pred = teacher_forced_predict(
        lstm, result["best_genome"], result["best_W"], train_inputs
    )
    train_mse = float(np.mean((train_pred[washout:] - train_targets[washout:]) ** 2))

    horizon = T_total - 1 - train_end
    free_pred = free_run(
        lstm,
        result["best_genome"],
        result["best_W"],
        warmup_inputs=train_inputs,
        horizon=horizon,
    )
    target_free = targets[train_end : train_end + horizon]
    free_mse = float(np.mean((free_pred - target_free) ** 2))

    # 84-step NRMSE: classical MG benchmark metric
    horizon_84 = min(84, horizon)
    nrmse_84 = _normalised_rmse(
        free_pred[:horizon_84], target_free[:horizon_84]
    )

    return TaskResult(
        name="mackey_glass",
        train_mse=train_mse,
        free_run_mse=free_mse,
        free_run_horizon=horizon,
        nrmse_84=nrmse_84,
        history=result["history"],
        best_genome=result["best_genome"],
        best_W=result["best_W"],
        series=series,
        washout=washout,
        train_end=train_end,
        free_run_pred=free_pred,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "unknown"


def _env_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": _git_hash(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--task", choices=("sines", "mackey", "both"), default="both")
    p.add_argument("--hidden", type=int, default=6)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=str, default="results.json")
    args = p.parse_args()

    cfg = EvolinoConfig(
        hidden=args.hidden,
        pop_size=args.pop,
        n_gens=args.gens,
        seed=args.seed,
    )

    summary: dict = {
        "args": vars(args),
        "config": asdict(cfg),
        "env": _env_record(),
        "results": {},
    }

    overall_t0 = time.time()

    if args.task in ("sines", "both"):
        t0 = time.time()
        sines = run_sines(cfg, verbose=not args.quiet)
        wall = time.time() - t0
        summary["results"]["sines"] = {
            "train_mse": sines.train_mse,
            "free_run_mse": sines.free_run_mse,
            "free_run_horizon": sines.free_run_horizon,
            "wallclock_s": wall,
            "best_fit_per_gen": [h["best_fit"] for h in sines.history],
        }
        print(
            f"[sines] train_mse={sines.train_mse:.6f}  "
            f"free_run_mse={sines.free_run_mse:.6f} over {sines.free_run_horizon} steps  "
            f"wall={wall:.1f}s"
        )

    if args.task in ("mackey", "both"):
        # Different seed offset per task to keep determinism while making
        # them independent; the seed is recorded in summary.
        cfg_mg = EvolinoConfig(
            hidden=args.hidden,
            pop_size=args.pop,
            n_gens=args.gens,
            seed=args.seed + 1000,
        )
        t0 = time.time()
        mg = run_mackey_glass(cfg_mg, verbose=not args.quiet)
        wall = time.time() - t0
        summary["results"]["mackey_glass"] = {
            "train_mse": mg.train_mse,
            "free_run_mse": mg.free_run_mse,
            "free_run_horizon": mg.free_run_horizon,
            "nrmse_84": mg.nrmse_84,
            "wallclock_s": wall,
            "best_fit_per_gen": [h["best_fit"] for h in mg.history],
        }
        print(
            f"[mackey-glass] train_mse={mg.train_mse:.6f}  "
            f"free_run_mse={mg.free_run_mse:.6f}  NRMSE@84={mg.nrmse_84:.4f}  "
            f"wall={wall:.1f}s"
        )

    summary["total_wallclock_s"] = time.time() - overall_t0

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}  total_wall={summary['total_wallclock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
