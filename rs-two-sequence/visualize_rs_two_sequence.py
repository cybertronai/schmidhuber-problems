"""
Static visualizations for rs-two-sequence.

Outputs (in `viz/`):
  search_curve.png     — best-train-accuracy-so-far vs trial; accepted trials marked
  weight_dist.png      — distribution of solution weights vs. uniform prior
  rollout.png          — solution net's hidden-state trajectory on train sequences
                         (latch behavior: hidden state separates by class at t=0
                          and stays separated through 99 distractor noise steps)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from rs_two_sequence import (
    forward_rnn, make_two_sequence_data, run, sigmoid,
)


def plot_search_curve(result: dict, out_path: str) -> None:
    """Best-train-acc-so-far vs trial. Mark accepted trials and the solving trial."""
    trace_trial = np.asarray(result["trace_trial"], dtype=np.int64)
    trace_train = np.asarray(result["trace_train"])
    trace_test = np.asarray(result["trace_test"])
    accepted_trial = np.asarray(result["accepted_trial"], dtype=np.int64)
    accepted_test = np.asarray(result["accepted_test"])
    final_trial = result["trial"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    # Step plot of best-so-far
    if len(trace_trial) > 0:
        # Extend to final trial for visibility
        x = np.concatenate([[1], trace_trial, [final_trial]])
        # Best-so-far is monotone non-decreasing on trace_train
        running = np.maximum.accumulate(trace_train)
        y = np.concatenate([[0.5], running, [running[-1]]])
        ax.step(x, y, where="post", color="#1f77b4", lw=1.6,
                label="best train acc so far")

    # Mark every accepted trial (train_acc >= threshold)
    if len(accepted_trial) > 0:
        ax.scatter(accepted_trial, accepted_test, color="#d62728", s=40,
                   zorder=5, label="accepted trial — test acc")

    # Highlight the solving trial
    ax.axvline(final_trial, color="black", lw=0.8, ls=":")
    ax.text(final_trial, 0.51, f" solved @ trial {final_trial:,}",
            color="black", fontsize=9, va="bottom")

    ax.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.set_xscale("log")
    ax.set_xlim(1, max(final_trial, 10))
    ax.grid(alpha=0.3)
    cfg = result["config"]
    ax.set_title(
        f"random-weight-guessing on Bengio-94 latch  "
        f"(T={cfg['lag']}, H={cfg['hidden']}, r={cfg['weight_range']}, "
        f"seed={cfg['seed']})"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_distribution(result: dict, out_path: str) -> None:
    """Distribution of solution weights vs the uniform prior they were sampled from."""
    theta = result["theta"]
    cfg = result["config"]
    r = cfg["weight_range"]

    fig, axes = plt.subplots(1, 5, figsize=(13, 2.6), dpi=140, sharey=True)
    names = ["W_xh (1×H)", "W_hh (H×H)", "b_h (H)", "W_hy (H×1)", "b_y (1)"]
    keys = ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    bins = np.linspace(-r, r, 9)
    for ax, name, key, c in zip(axes, names, keys, colors):
        w = theta[key].flatten()
        ax.hist(w, bins=bins, color=c, alpha=0.85, edgecolor="white")
        ax.axvspan(-r, r, color="gray", alpha=0.06,
                   label=f"prior U[-{r}, {r}]")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("weight value")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-r * 1.05, r * 1.05)
    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Solution weights — accepted trial {result['trial']:,}  "
        f"(small parameter set: 1+25+5+5+1 = {1 + cfg['hidden']**2 + 3*cfg['hidden'] + 1} scalars)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rollout(result: dict, out_path: str) -> None:
    """Hidden-state trajectory on a few train sequences. Show that h_t separates by
    class at t=0 and stays separated through T-1 distractor noise steps — this is
    the latch behavior the network must implement.
    """
    cfg = result["config"]
    seed_seq = np.random.SeedSequence(cfg["seed"])
    data_seed, _ = seed_seq.spawn(2)
    rng = np.random.default_rng(data_seed)
    n_per_class = 4
    X, y = make_two_sequence_data(2 * n_per_class, cfg["lag"], cfg["noise_std"], rng)
    # Force balanced classes
    pos_idx = np.where(y == 1)[0][:n_per_class]
    neg_idx = np.where(y == 0)[0][:n_per_class]
    if len(pos_idx) < n_per_class or len(neg_idx) < n_per_class:
        # Re-sample with explicit labels if needed
        X = np.zeros((2 * n_per_class, cfg["lag"]), dtype=np.float32)
        y = np.zeros(2 * n_per_class, dtype=np.int32)
        for i in range(n_per_class):
            X[i, 0] = 1.0
            X[n_per_class + i, 0] = -1.0
            y[i] = 1
            y[n_per_class + i] = 0
        X[:, 1:] = rng.normal(0, cfg["noise_std"],
                              size=(2 * n_per_class, cfg["lag"] - 1))
        pos_idx = np.arange(n_per_class)
        neg_idx = np.arange(n_per_class, 2 * n_per_class)

    theta = result["theta"]

    # Re-run forward but record the hidden state trajectory
    H = theta["W_hh"].shape[0]
    B, T = X.shape
    h = np.zeros((B, H), dtype=np.float32)
    h_traj = np.zeros((T + 1, B, H), dtype=np.float32)
    for t in range(T):
        x_t = X[:, t:t + 1]
        h = np.tanh(x_t @ theta["W_xh"] + h @ theta["W_hh"] + theta["b_h"])
        h_traj[t + 1] = h
    z = (h @ theta["W_hy"] + theta["b_y"]).reshape(-1)
    yhat = sigmoid(z)

    # Project to "latch axis": find the hidden-unit dim with the largest
    # absolute readout weight; that's the dimension carrying the latch signal.
    readout = theta["W_hy"].reshape(-1)
    latch_dim = int(np.argmax(np.abs(readout)))
    sign = float(np.sign(readout[latch_dim]))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), dpi=140, sharex=True)
    ax = axes[0]
    ts = np.arange(T + 1)
    for b in pos_idx:
        ax.plot(ts, sign * h_traj[:, b, latch_dim], color="#d62728",
                alpha=0.85, lw=1.0)
    for b in neg_idx:
        ax.plot(ts, sign * h_traj[:, b, latch_dim], color="#1f77b4",
                alpha=0.85, lw=1.0)
    ax.axvline(0.5, color="gray", lw=0.6, ls="--",
               label="t=1: signal seen")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel(f"hidden unit {latch_dim} (sign-aligned to readout)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(
        f"Latch behavior — solution from trial {result['trial']:,}  "
        f"(red: class +1, blue: class -1; predicted ŷ shown below)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for b in pos_idx:
        ax.plot([0, T], [yhat[b], yhat[b]], color="#d62728", alpha=0.6)
    for b in neg_idx:
        ax.plot([0, T], [yhat[b], yhat[b]], color="#1f77b4", alpha=0.6)
    # Plot a separate marker at the right edge for clarity
    for b in pos_idx:
        ax.scatter([T], [yhat[b]], color="#d62728", s=30, zorder=5)
    for b in neg_idx:
        ax.scatter([T], [yhat[b]], color="#1f77b4", s=30, zorder=5)
    ax.axhline(0.5, color="gray", lw=0.6, ls="--", label="decision boundary")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("timestep")
    ax.set_ylabel("ŷ at final step")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lag", type=int, default=100)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--weight-range", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--max-trials", type=int, default=200_000)
    p.add_argument("--outdir", type=str, default="viz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Re-running RS to capture trace for visualizations ...")
    result = run(
        seed=args.seed, lag=args.lag, hidden=args.hidden,
        noise_std=args.noise_std,
        n_train=args.n_train, n_test=args.n_test,
        weight_range=args.weight_range,
        threshold=args.threshold, max_trials=args.max_trials,
        verbose=True,
    )
    if not result["solved"]:
        print(f"WARNING: did not solve within {args.max_trials:,} trials. "
              f"Visualizations will use the best-found theta.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nWriting visualizations to {args.outdir}/ ...")
    plot_search_curve(result, os.path.join(args.outdir, "search_curve.png"))
    plot_weight_distribution(result, os.path.join(args.outdir, "weight_dist.png"))
    plot_rollout(result, os.path.join(args.outdir, "rollout.png"))


if __name__ == "__main__":
    main()
