"""Static visualizations for self-referential weight matrix.

Outputs:
    viz/learning_curves.png   training loss + eval accuracy + per-task acc
    viz/W_per_task.png        W_fast at end of demo phase, averaged per task
    viz/W_fast_trace.png      W_fast at each step of one episode (XOR task)
    viz/write_attention.png   row/col attention + write gate over one episode
    viz/W_slow.png            the trained slow weight matrix
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from self_referential_weight_matrix import (
    SRWM, TASK_NAMES, make_episode, train,
)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_learning_curves(history, out):
    eps = history["episode"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(eps, history["train_loss"], label="train BCE (single ep)", color="C7", alpha=0.6)
    ax.set_xlabel("episode"); ax.set_ylabel("BCE")
    ax.set_title("training loss"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(eps, history["eval_acc"], label="overall", color="black", lw=2)
    pt = np.array(history["per_task_acc"])
    for i, name in enumerate(TASK_NAMES):
        ax.plot(eps, pt[:, i], label=name, lw=1, alpha=0.8)
    ax.axhline(0.5, color="grey", ls="--", lw=0.7, label="chance")
    ax.set_xlabel("episode"); ax.set_ylabel("query accuracy")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("eval accuracy")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("Self-referential weight matrix: meta-learning curves", y=1.02)
    _save(fig, out)


def plot_W_per_task(model: SRWM, seed: int, out: str, n_avg: int = 50):
    """For each task, average W_fast at end of demo phase across n_avg episodes."""
    rng = np.random.default_rng(seed + 5_000)
    avg = np.zeros((4, model.n_h, model.n_h))
    last = np.zeros((4, model.n_h, model.n_h))   # final W_fast at end of episode
    for task_id in range(4):
        for k in range(n_avg):
            inputs, _, _ = make_episode(rng, task_id)
            model.episode(inputs)
            # Demo phase = first 4 steps; W_fast at index 4 is post-demo.
            avg[task_id] += model.fast_history[4]
            last[task_id] += model.fast_history[-1]
        avg[task_id] /= n_avg
        last[task_id] /= n_avg

    vmax = float(np.max(np.abs(avg)))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for i, name in enumerate(TASK_NAMES):
        ax = axes[0, i]
        im = ax.imshow(avg[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}\nW_fast after demo (avg of {n_avg})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    vmax2 = float(np.max(np.abs(last)))
    for i, name in enumerate(TASK_NAMES):
        ax = axes[1, i]
        im = ax.imshow(last[i], cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
        ax.set_title(f"{name}\nW_fast end of episode")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "Task-conditional W_fast. Different tasks drive the network to write\n"
        "different patterns into its own weight matrix.", y=1.02,
    )
    _save(fig, out)
    return avg, last


def plot_W_fast_trace(model: SRWM, seed: int, out: str, task_id: int = 2):
    """W_fast at every step of one episode."""
    rng = np.random.default_rng(seed + 7_777)
    inputs, targets, is_query = make_episode(rng, task_id)
    ys = model.episode(inputs)
    history = model.fast_history    # length T+1, contains pre and post each step
    T = len(history) - 1
    vmax = float(np.max(np.abs(history))) + 1e-8
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    axes = axes.ravel()
    # 9 frames: index 0..8 (pre-step 0 .. post-step 7); show 0,1,..,8
    for t in range(T + 1):
        ax = axes[t]
        im = ax.imshow(history[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        if t == 0:
            ax.set_title("before step 0")
        else:
            phase = "demo" if t <= 4 else "query"
            x = inputs[t - 1]
            label = (
                f"step {t-1} ({phase})\n"
                f"x=({x[0]:+.0f},{x[1]:+.0f}) "
                f"y_label={x[2]:+.0f} "
            )
            if is_query[t - 1]:
                label += f"\npred={ys[t-1]:.2f} target={targets[t-1]:.0f}"
            ax.set_title(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].set_visible(False)  # we have 9 subplots needed, 10 axes
    plt.colorbar(im, ax=axes[-2], fraction=0.046)
    fig.suptitle(
        f"W_fast trace, task = {TASK_NAMES[task_id]}.\n"
        "Demo phase (t=0..3) writes into the matrix; query phase reads "
        "implicitly via W_eff = W_slow + W_fast.",
        y=1.02,
    )
    _save(fig, out)


def plot_write_attention(model: SRWM, seed: int, out: str, task_id: int = 2):
    """Per-step row attention, col attention, write value, write gate."""
    rng = np.random.default_rng(seed + 8_888)
    inputs, targets, is_query = make_episode(rng, task_id)
    model.episode(inputs)
    T = len(model.tape)
    rows = np.stack([tp["row"] for tp in model.tape], axis=0)         # (T, n_h)
    cols = np.stack([tp["col"] for tp in model.tape], axis=0)         # (T, n_h)
    vals = np.array([float(tp["val"][0]) for tp in model.tape])
    gates = np.array([float(tp["gate"][0]) for tp in model.tape])

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    ax = axes[0, 0]
    im = ax.imshow(rows.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("row_attn over time"); ax.set_xlabel("t"); ax.set_ylabel("row idx")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 1]
    im = ax.imshow(cols.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("col_attn over time"); ax.set_xlabel("t"); ax.set_ylabel("col idx")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 0]
    ax.bar(range(T), gates, color="C2", label="write_gate")
    ax.bar(range(T), vals, color="C3", alpha=0.7, label="write_value")
    ax.axvspan(-0.5, 3.5, color="grey", alpha=0.1, label="demo phase")
    ax.set_xlabel("t"); ax.set_ylim(-1.05, 1.05); ax.set_title("scalar write controls")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    write_strength = gates * np.abs(vals)
    ax.plot(range(T), write_strength, "o-", color="C0", label="|gate * val|")
    ax.axvspan(-0.5, 3.5, color="grey", alpha=0.1)
    ax.set_xlabel("t"); ax.set_ylim(0, 1.05); ax.set_title("effective write strength")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Write controls during episode, task = {TASK_NAMES[task_id]}.", y=1.02)
    _save(fig, out)


def plot_W_slow(model: SRWM, out: str):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, name, mat in zip(
        axes,
        ["W_slow", "W_xh", "A_row", "A_col", "A_val (1xH)"],
        [model.W_slow, model.W_xh, model.A_row, model.A_col, np.tile(model.A_val, (model.n_h, 1))],
    ):
        v = float(np.max(np.abs(mat))) + 1e-8
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Trained slow parameters (after BPTT across episodes).", y=1.05)
    _save(fig, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=3000)
    parser.add_argument("--n-h", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    viz = os.path.join(here, "viz")

    print("training (visualization wrapper) ...")
    model, history, summary = train(
        seed=args.seed,
        n_episodes=args.n_episodes,
        n_h=args.n_h,
        eta=args.eta,
        lr=args.lr,
        quick=args.quick,
        verbose=False,
    )
    print(f"  final acc {summary['final_overall_acc']:.3f} "
          f"per-task {summary['final_per_task_acc']}")

    plot_learning_curves(history, os.path.join(viz, "learning_curves.png"))
    plot_W_per_task(model, args.seed, os.path.join(viz, "W_per_task.png"))
    plot_W_fast_trace(model, args.seed, os.path.join(viz, "W_fast_trace.png"), task_id=2)
    plot_write_attention(model, args.seed, os.path.join(viz, "write_attention.png"), task_id=2)
    plot_W_slow(model, os.path.join(viz, "W_slow.png"))


if __name__ == "__main__":
    main()
