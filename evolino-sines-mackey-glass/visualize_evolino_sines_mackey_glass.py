"""Static PNGs for evolino-sines-mackey-glass.

Reproduces the same train+evolve pipeline as the headline run, then writes
five plots into ./viz/ :

  fitness_curve.png       best/mean MSE per generation, both tasks
  sines_prediction.png    ground truth vs. predicted, train + free-run window
  mackey_prediction.png   ground truth vs. predicted, train + free-run window
  hidden_states.png       trace of every LSTM hidden unit, sines task
  weight_blocks.png       heatmaps of evolved gate weight matrices, both tasks

Usage:

  python3 visualize_evolino_sines_mackey_glass.py --seed 1
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evolino_sines_mackey_glass import (
    LSTM,
    EvolinoConfig,
    run_mackey_glass,
    run_sines,
    teacher_forced_predict,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fitness_curve(sines, mg, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, task, title in [
        (axes[0], sines, "Sines (3 incommensurate)"),
        (axes[1], mg, "Mackey-Glass tau=17"),
    ]:
        gens = [h["gen"] for h in task.history]
        best = [h["best_mse"] for h in task.history]
        mean = [-h["mean_fit"] for h in task.history]
        ax.plot(gens, best, "C0-", label="best individual MSE")
        ax.plot(gens, mean, "C1-", alpha=0.5, label="population mean MSE")
        ax.set_yscale("log")
        ax.set_xlabel("generation")
        ax.set_ylabel("MSE  (log scale)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def prediction_plot(task, out: str, title: str) -> None:
    """Plot ground truth, teacher-forced training fit, and free-running test."""
    lstm = LSTM(hidden=task.best_genome.shape[0] // (4 * (1 + 1)))
    # Recompute hidden dim from genome size: gene_size = 4 * (1 + h + 1) * h
    h = _hidden_from_genome(task.best_genome.shape[0])
    lstm = LSTM(hidden=h, input_dim=1)

    targets = task.series[1:]
    inputs = task.series[:-1].reshape(-1, 1)
    train_inputs = inputs[: task.train_end]

    train_pred = teacher_forced_predict(
        lstm, task.best_genome, task.best_W, train_inputs
    )

    train_targets = targets[: task.train_end]
    free_pred = task.free_run_pred
    horizon = task.free_run_horizon
    free_targets = targets[task.train_end : task.train_end + horizon]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(np.arange(task.train_end), train_targets, "k-", lw=1.0, label="ground truth (train)")
    ax.plot(
        np.arange(task.washout, task.train_end),
        train_pred[task.washout :],
        "C0-",
        lw=0.8,
        alpha=0.85,
        label="teacher-forced fit",
    )
    ax.axvspan(0, task.washout, color="grey", alpha=0.10, label="washout")
    ax.axvline(task.train_end, color="C3", ls="--", lw=0.8, label="train / free-run boundary")
    ax.plot(
        np.arange(task.train_end, task.train_end + horizon),
        free_targets,
        "k-",
        lw=1.0,
        alpha=0.6,
    )
    ax.plot(
        np.arange(task.train_end, task.train_end + horizon),
        free_pred,
        "C2-",
        lw=1.2,
        label="free-running prediction",
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("y(t)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _hidden_from_genome(gene_size: int) -> int:
    """Invert gene_size = 4*(1 + h + 1)*h = 4*(h^2 + 2h) -> solve for h."""
    # 4 h^2 + 8 h - gene_size = 0  ->  h = (-8 + sqrt(64 + 16*gene_size)) / 8
    h = (-8 + np.sqrt(64 + 16 * gene_size)) / 8
    return int(round(h))


def hidden_states_plot(task, out: str) -> None:
    h = _hidden_from_genome(task.best_genome.shape[0])
    lstm = LSTM(hidden=h, input_dim=1)
    inputs = task.series[:-1].reshape(-1, 1)
    H = lstm.run(task.best_genome, inputs)
    T = H.shape[0]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cmap = plt.get_cmap("tab10")
    for j in range(h):
        ax.plot(H[:, j], color=cmap(j % 10), lw=0.7, label=f"h{j}")
    ax.axvline(task.train_end, color="k", ls="--", lw=0.6)
    ax.set_xlabel("time step")
    ax.set_ylabel("hidden activation")
    ax.set_title(f"LSTM hidden traces — {task.name}")
    ax.legend(ncol=h, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def weight_block_plot(task, out: str) -> None:
    h = _hidden_from_genome(task.best_genome.shape[0])
    lstm = LSTM(hidden=h, input_dim=1)
    gates = lstm.unflatten(task.best_genome)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    names = ["z (cell input)", "i (input gate)", "f (forget gate)", "o (output gate)"]
    for ax, key, name in zip(axes, ["z", "i", "f", "o"], names):
        block = gates[key]
        vmax = float(np.max(np.abs(block)))
        ax.imshow(block.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks([0] + list(range(1, 1 + h)) + [1 + h])
        ax.set_xticklabels(
            ["x"] + [f"h{j}" for j in range(h)] + ["b"], fontsize=7, rotation=90
        )
        ax.set_yticks(range(h))
        ax.set_yticklabels([f"u{j}" for j in range(h)], fontsize=7)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"Evolved LSTM gate weights — {task.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hidden", type=int, default=6)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--out", type=str, default="viz")
    args = p.parse_args()

    _ensure_dir(args.out)

    cfg = EvolinoConfig(
        hidden=args.hidden, pop_size=args.pop, n_gens=args.gens, seed=args.seed
    )
    cfg_mg = EvolinoConfig(
        hidden=args.hidden,
        pop_size=args.pop,
        n_gens=args.gens,
        seed=args.seed + 1000,
    )

    print("running sines...")
    sines = run_sines(cfg, verbose=False)
    print(f"  train_mse={sines.train_mse:.6f}  free_run_mse={sines.free_run_mse:.6f}")

    print("running mackey-glass...")
    mg = run_mackey_glass(cfg_mg, verbose=False)
    print(f"  train_mse={mg.train_mse:.6f}  free_run_mse={mg.free_run_mse:.6f}  "
          f"NRMSE@84={mg.nrmse_84:.4f}")

    print("plotting...")
    fitness_curve(sines, mg, os.path.join(args.out, "fitness_curve.png"))
    prediction_plot(
        sines,
        os.path.join(args.out, "sines_prediction.png"),
        "Evolino on three superimposed sines",
    )
    prediction_plot(
        mg,
        os.path.join(args.out, "mackey_prediction.png"),
        "Evolino on Mackey-Glass tau=17",
    )
    hidden_states_plot(sines, os.path.join(args.out, "hidden_states.png"))
    weight_block_plot(sines, os.path.join(args.out, "weight_blocks_sines.png"))
    weight_block_plot(mg, os.path.join(args.out, "weight_blocks_mackey.png"))

    print(f"wrote PNGs to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
