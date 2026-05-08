"""
Static visualizations for rs-parity.

Outputs (in `viz/`):
  search_curve.png   - best-acc-so-far + per-trial acc vs trial number
  weights.png        - the winning RNN's weight matrices as Hinton diagrams
  hidden_dynamics.png - h(t) trajectories of the winning RNN on a few patterns
  trial_acc_hist.png - histogram of per-trial accuracies (most are near chance)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rs_parity import (
    RNNParams,
    forward,
    make_parity_dataset,
    random_search,
    sample_parity_dataset,
)


# ----------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------

def plot_search_curve(history: dict, out_path: str):
    """Best-acc-so-far (step plot) plus subsampled per-trial acc."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    # subsampled raw per-trial accuracies (lots of dots, nearly all ~50%)
    trials = np.array(history["all_trial"])
    accs = np.array(history["all_acc"]) * 100
    ax.scatter(trials, accs, s=4, color="#888", alpha=0.35,
               label="random trials (subsampled)", rasterized=True)

    # best-so-far step
    bt = np.array(history["best_trial"])
    ba = np.array(history["best_acc"]) * 100
    if len(bt) > 0:
        # extend the step out to the final trial
        bt_ext = np.append(bt, history["n_trials"])
        ba_ext = np.append(ba, ba[-1])
        ax.step(bt_ext, ba_ext, where="post", color="#cc0000",
                linewidth=2.0, label="best so far")

    if history.get("found_trial") is not None:
        ax.axvline(history["found_trial"], color="#cc0000", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.text(history["found_trial"], 50,
                f" solved @ {history['found_trial']:,}",
                color="#cc0000", fontsize=9, va="bottom")

    ax.axhline(50, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(trials[-1] if len(trials) else 1, 50.5,
            "chance", fontsize=8, color="black", ha="right", alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlim(1, max(history["n_trials"], 10))
    ax.set_ylim(40, 105)
    ax.set_xlabel("trial number")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Random-weight guessing on N-bit sequence parity")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(params: RNNParams, out_path: str):
    """Hinton-diagram view of the winning RNN's weights."""
    H = params.b_h.shape[0]

    # Build a single combined weight panel:
    #   row 0:        input bias-into-hidden line and hidden bias  (1 + 1 = 2 cols)
    #   rows 0..H-1:  W_xh[:, j]   |   W_hh[:, j]       j = each hidden unit
    # Easier: 2-row figure with W_hh on left, W_xh+b_h on middle, W_hy+b_y on right.

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), dpi=140,
                             gridspec_kw={"width_ratios": [H, 1.5, 1.5]})

    def hinton(ax, M, row_labels, col_labels, title):
        max_abs = max(abs(M).max(), 1e-3)
        nr, nc = M.shape
        for i in range(nr):
            for j in range(nc):
                w = M[i, j]
                sz = 0.85 * (abs(w) / max_abs) ** 0.5
                color = "#cc0000" if w > 0 else "#003366"
                ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                       facecolor=color, edgecolor="black",
                                       linewidth=0.3))
        ax.set_xlim(-0.6, nc - 0.4)
        ax.set_ylim(-0.6, nr - 0.4)
        ax.invert_yaxis()
        ax.set_xticks(range(nc))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    hinton(axes[0], params.W_hh,
           [f"h[{i}](t-1)" for i in range(H)],
           [f"h[{j}](t)" for j in range(H)],
           f"$W_{{hh}}$  (||·|| = {np.linalg.norm(params.W_hh):.2f})")

    # W_xh + b_h side by side
    M_xb = np.concatenate([params.W_xh.reshape(1, H).T,
                           params.b_h.reshape(H, 1)], axis=1)
    hinton(axes[1], M_xb,
           [f"h[{i}]" for i in range(H)],
           ["x", "b"],
           "input + bias")

    # W_hy + b_y side by side
    M_yb = np.concatenate([params.W_hy, params.b_y.reshape(1, 1)], axis=0)
    hinton(axes[2], M_yb,
           [f"h[{i}]" for i in range(H)] + ["b"],
           ["y"],
           "readout + bias")

    fig.suptitle("Winning RNN weights (Hinton diagram)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_dynamics(params: RNNParams, N: int, out_path: str,
                         seed: int = 7, n_seqs: int = 6):
    """Plot h(t) trajectories on a small set of test sequences, color-coded
    by whether the running parity is 0 or 1."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_seqs, N))
    X = (bits * 2 - 1).astype(np.float32)
    y_true = X.prod(axis=1)
    _, traj = forward(params, X, return_states=True)        # (n_seqs, N+1, H)
    H = traj.shape[-1]

    # running parity (target y_t at each step)
    running_parity = np.cumprod(X, axis=1)                  # (n_seqs, N)

    fig, axes = plt.subplots(n_seqs, 1, figsize=(9, 1.4 * n_seqs),
                             dpi=140, sharex=True)
    if n_seqs == 1:
        axes = [axes]
    cmap = plt.cm.tab10
    for s, ax in enumerate(axes):
        for j in range(H):
            ax.plot(range(N + 1), traj[s, :, j],
                    color=cmap(j), linewidth=1.4, label=f"h[{j}]")
        # background shading: green = even parity (target +1), red = odd (-1)
        for t in range(N):
            color = "#d4edda" if running_parity[s, t] > 0 else "#f8d7da"
            ax.axvspan(t + 0.5, t + 1.5, color=color, alpha=0.5, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlim(0, N)
        verdict = "OK" if np.sign(traj[s, -1] @ params.W_hy + params.b_y) == y_true[s] else "FAIL"
        ax.set_ylabel(f"seq {s}\nparity={int(y_true[s])} [{verdict}]",
                      fontsize=8)
        if s == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=H)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("timestep t")
    fig.suptitle(f"Hidden-unit trajectories on {n_seqs} test sequences "
                 f"(green=running parity +1, red=-1)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_trial_accuracy_hist(history: dict, out_path: str):
    """How are random-weight trials distributed in accuracy?"""
    accs = np.array(history["all_acc"]) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=140)
    ax.hist(accs, bins=np.arange(20, 102, 2), color="#1f77b4",
            edgecolor="black", linewidth=0.4)
    ax.axvline(50, color="black", linestyle=":", linewidth=0.8,
               label="chance (50%)")
    if history.get("found_trial") is not None:
        ax.axvline(100, color="#cc0000", linewidth=1.2,
                   label="solver found")
    ax.set_xlabel("per-trial accuracy (%)")
    ax.set_ylabel("number of trials (logged subsample)")
    ax.set_title("Distribution of random-weight scores: most trials "
                 "are near chance,\nthe solver basin is rare but exists")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--hidden", type=int, default=2)
    p.add_argument("--weight-scale", type=float, default=30.0)
    p.add_argument("--max-trials", type=int, default=100_000)
    p.add_argument("--sample-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Re-run RS so the visualization is reproducible from scratch
    best_params, history = random_search(
        N=args.n, H=args.hidden, weight_scale=args.weight_scale,
        max_trials=args.max_trials, sample_size=args.sample_size,
        log_every=200,                  # finer subsampling for the histogram
        seed=args.seed, verbose=True,
    )

    if best_params is None:
        raise SystemExit("no parameters returned (max_trials exhausted "
                         "without finding any non-zero accuracy)")

    plot_search_curve(history, os.path.join(args.outdir, "search_curve.png"))
    plot_trial_accuracy_hist(history,
                             os.path.join(args.outdir, "trial_acc_hist.png"))
    plot_weights(best_params, os.path.join(args.outdir, "weights.png"))
    plot_hidden_dynamics(best_params, args.n,
                         os.path.join(args.outdir, "hidden_dynamics.png"))


if __name__ == "__main__":
    main()
