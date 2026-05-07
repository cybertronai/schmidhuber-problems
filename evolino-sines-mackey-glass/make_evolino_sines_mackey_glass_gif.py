"""Make evolino_sines_mackey_glass.gif.

Animate the closed-loop free-running prediction of the elite individual at
each generation, side by side: sines on the left, Mackey-Glass on the
right. The GIF makes the convergence of the population visible — early
generations show flat or oscillatory predictions, late generations show
the network locking onto the target.

Usage:

  python3 make_evolino_sines_mackey_glass_gif.py --seed 1
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from evolino_sines_mackey_glass import (
    LSTM,
    EvolinoConfig,
    evaluate,
    free_run,
    mackey_glass,
    superimposed_sines,
)


def _evolve_with_snapshots(
    lstm: LSTM,
    inputs: np.ndarray,
    targets: np.ndarray,
    washout: int,
    cfg: EvolinoConfig,
    val_horizon: int,
    val_targets: np.ndarray,
    snapshot_every: int,
) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    """Run the evolutionary loop and snapshot the elite predictor periodically.

    Returns a list of (gen, best_mse, best_genome, best_W) tuples — one per
    snapshot generation, including the final.
    """
    rng = np.random.default_rng(cfg.seed)
    pop = rng.normal(0.0, cfg.init_sigma, (cfg.pop_size, lstm.gene_size))
    fits = np.full(cfg.pop_size, -np.inf)

    best_genome = None
    best_W = None
    best_fit = -np.inf
    snapshots: list[tuple[int, float, np.ndarray, np.ndarray]] = []
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

        if gen % snapshot_every == 0 or gen == cfg.n_gens - 1:
            snapshots.append((gen, float(-fits[0]), best_genome.copy(), best_W.copy()))

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

    return snapshots


def _frames_for_task(
    snaps,
    lstm,
    train_inputs,
    series,
    train_end,
    horizon,
):
    targets = series[1:]
    free_targets = targets[train_end : train_end + horizon]
    frames = []
    for gen, mse, genome, W in snaps:
        free_pred = free_run(lstm, genome, W, train_inputs, horizon)
        frames.append((gen, mse, free_pred, free_targets))
    return frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hidden", type=int, default=6)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--snap-every", type=int, default=2)
    p.add_argument("--out", type=str, default="evolino_sines_mackey_glass.gif")
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    cfg = EvolinoConfig(
        hidden=args.hidden, pop_size=args.pop, n_gens=args.gens, seed=args.seed
    )
    cfg_mg = EvolinoConfig(
        hidden=args.hidden,
        pop_size=args.pop,
        n_gens=args.gens,
        seed=args.seed + 1000,
    )
    lstm = LSTM(hidden=args.hidden, input_dim=1)

    # ---------- sines ----------
    T_total = 700
    washout = 100
    train_end = 400
    val_end = 500
    sines_series = superimposed_sines(T_total, freqs=(0.20, 0.311, 0.42))
    inputs = sines_series[:-1].reshape(-1, 1)
    targets = sines_series[1:]
    print("evolving (sines, with snapshots)...")
    sines_snaps = _evolve_with_snapshots(
        lstm,
        inputs[:train_end],
        targets[:train_end],
        washout,
        cfg,
        val_horizon=val_end - train_end,
        val_targets=targets[train_end:val_end],
        snapshot_every=args.snap_every,
    )
    sines_horizon = T_total - 1 - train_end
    sines_frames = _frames_for_task(
        sines_snaps, lstm, inputs[:train_end], sines_series, train_end, sines_horizon
    )

    # ---------- mackey-glass ----------
    T_total_mg = 1000
    washout_mg = 100
    train_end_mg = 600
    val_end_mg = 700
    mg_series = mackey_glass(T_total_mg, tau=17)
    mg_series = mg_series - np.mean(mg_series)
    mg_series = mg_series / (np.std(mg_series) + 1e-12)
    inputs_mg = mg_series[:-1].reshape(-1, 1)
    targets_mg = mg_series[1:]
    print("evolving (mackey-glass, with snapshots)...")
    mg_snaps = _evolve_with_snapshots(
        lstm,
        inputs_mg[:train_end_mg],
        targets_mg[:train_end_mg],
        washout_mg,
        cfg_mg,
        val_horizon=val_end_mg - train_end_mg,
        val_targets=targets_mg[train_end_mg:val_end_mg],
        snapshot_every=args.snap_every,
    )
    mg_horizon = T_total_mg - 1 - train_end_mg
    mg_frames = _frames_for_task(
        mg_snaps,
        lstm,
        inputs_mg[:train_end_mg],
        mg_series,
        train_end_mg,
        mg_horizon,
    )

    # ---------- animate ----------
    n_frames = min(len(sines_frames), len(mg_frames))
    print(f"composing GIF ({n_frames} frames)...")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5))

    line_t_l, = ax_l.plot([], [], "k-", lw=1.0, label="ground truth")
    line_p_l, = ax_l.plot([], [], "C2-", lw=1.4, label="evolino prediction")
    title_l = ax_l.set_title("")
    ax_l.set_xlabel("time step (free-running, after train end)")
    ax_l.set_ylabel("y(t)")
    ax_l.set_ylim(-1.2, 1.2)
    ax_l.set_xlim(0, sines_horizon)
    ax_l.legend(loc="lower right")
    ax_l.grid(True, alpha=0.3)

    line_t_r, = ax_r.plot([], [], "k-", lw=1.0)
    line_p_r, = ax_r.plot([], [], "C3-", lw=1.4)
    title_r = ax_r.set_title("")
    ax_r.set_xlabel("time step (free-running, after train end)")
    ax_r.set_ylabel("y(t)")
    ax_r.set_xlim(0, mg_horizon)
    ax_r.set_ylim(min(targets_mg) - 0.2, max(targets_mg) + 0.2)
    ax_r.grid(True, alpha=0.3)

    def init():
        line_t_l.set_data([], [])
        line_p_l.set_data([], [])
        line_t_r.set_data([], [])
        line_p_r.set_data([], [])
        return line_t_l, line_p_l, line_t_r, line_p_r, title_l, title_r

    def update(i):
        gen_s, mse_s, pred_s, true_s = sines_frames[i]
        gen_m, mse_m, pred_m, true_m = mg_frames[i]
        x_s = np.arange(len(true_s))
        line_t_l.set_data(x_s, true_s)
        line_p_l.set_data(x_s, pred_s)
        title_l.set_text(f"sines  gen={gen_s}  free-run-MSE={mse_s:.4f}")
        x_m = np.arange(len(true_m))
        line_t_r.set_data(x_m, true_m)
        line_p_r.set_data(x_m, pred_m)
        title_r.set_text(f"mackey-glass  gen={gen_m}  free-run-MSE={mse_m:.4f}")
        return line_t_l, line_p_l, line_t_r, line_p_r, title_l, title_r

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1000 // args.fps,
        blit=False,
    )
    anim.save(args.out, writer="pillow", fps=args.fps)
    plt.close(fig)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({size_kb:.0f} KB, {n_frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
