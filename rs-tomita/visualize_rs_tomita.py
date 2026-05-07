"""
Static visualizations for the rs-tomita random-search baseline.

Outputs (in `viz/`):
  search_curves.png           - train-acc vs trial number for all 3 grammars
  hidden_trajectories.png     - hidden-state trajectories on accepted vs rejected strings
  weight_matrices.png         - weight matrices of solved networks
  weight_distributions.png    - histograms of per-trial train accuracy

Run:
  python3 visualize_rs_tomita.py --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rs_tomita import (
    GRAMMARS,
    forward,
    encode_batch,
    hidden_trajectory,
    sample_weights,
    run_grammar,
)


GRAMMAR_LABELS = {
    1: "#1: $a^*$",
    2: "#2: $(ab)^*$",
    4: "#4: no $aaa$",
}

GRAMMAR_COLORS = {1: "#1f77b4", 2: "#2ca02c", 4: "#d62728"}


def load_npz(path: Path):
    """Load saved results into a dict keyed by grammar number."""
    npz = np.load(path, allow_pickle=True)
    out = {}
    for g in [1, 2, 4]:
        weights = {}
        for k in ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]:
            key = f"g{g}_W_{k}"
            if key in npz.files:
                weights[k] = npz[key]
        out[g] = {
            "history": npz[f"g{g}_history"],
            "solved_at": int(npz[f"g{g}_solved_at"][0]),
            "best_train": float(npz[f"g{g}_best_train"][0]),
            "best_test": float(npz[f"g{g}_best_test"][0]),
            "wallclock": float(npz[f"g{g}_wallclock"][0]),
            "train_strings": npz[f"g{g}_train_strings"].tolist(),
            "train_y": npz[f"g{g}_train_y"],
            "test_strings": npz[f"g{g}_test_strings"].tolist(),
            "test_y": npz[f"g{g}_test_y"],
            "weights": weights if weights else None,
        }
    return out


def plot_search_curves(results: dict, out_path: Path) -> None:
    """One subplot per grammar: trial number on x-axis, accuracy on y-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
    for ax, g in zip(axes, [1, 2, 4]):
        h = results[g]["history"]
        if len(h) == 0:
            continue
        trials, train, test = h[:, 0], h[:, 1], h[:, 2]
        ax.plot(trials, train, marker="o", color=GRAMMAR_COLORS[g],
                label="train", linewidth=1.8, markersize=4)
        ax.plot(trials, test, marker="s", color=GRAMMAR_COLORS[g],
                alpha=0.5, linestyle="--", label="test", linewidth=1.2,
                markersize=3)
        sa = results[g]["solved_at"]
        if sa > 0:
            ax.axvline(sa, color="black", linewidth=0.8, linestyle=":",
                       alpha=0.6)
            ax.text(sa, 0.05, f"solved\n@ {sa}", fontsize=8, ha="right",
                    va="bottom")
        ax.set_xscale("symlog")
        ax.set_xlabel("trial (log)")
        ax.set_ylabel("running best accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(f"Tomita {GRAMMAR_LABELS[g]}")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("rs-tomita: best-so-far accuracy across random-weight trials",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_trajectories(results: dict, out_path: Path,
                              n_per_class: int = 3) -> None:
    """For each solved grammar, plot hidden-state trajectories on a few
    accepted and a few rejected test strings."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=120)
    for row, g in enumerate([1, 2, 4]):
        W = results[g]["weights"]
        if W is None:
            for col in range(2):
                axes[row, col].axis("off")
            continue
        test_strings = results[g]["test_strings"]
        test_y = results[g]["test_y"]
        accept = GRAMMARS[g]
        pos_strings = [s for s, y in zip(test_strings, test_y) if y == 1][:n_per_class]
        neg_strings = [s for s, y in zip(test_strings, test_y) if y == 0][:n_per_class]

        for col, (label, strings) in enumerate(
            [("accepted", pos_strings), ("rejected", neg_strings)]
        ):
            ax = axes[row, col]
            for s in strings:
                if not s:
                    continue
                traj = hidden_trajectory(W, s)
                t = np.arange(len(traj))
                # Plot each hidden unit as a separate line
                for h_idx in range(traj.shape[1]):
                    ax.plot(t, traj[:, h_idx], alpha=0.7, linewidth=1.0)
            ax.set_xlabel("step")
            ax.set_ylabel("hidden activation")
            ax.set_ylim(-1.1, 1.1)
            ax.grid(alpha=0.3)
            ax.set_title(f"Tomita {GRAMMAR_LABELS[g]} -- {label} strings")
            # Add a tiny tick label legend showing which strings
            ax.text(0.02, 0.98,
                    "\n".join(s[:14] + ("..." if len(s) > 14 else "") for s in strings),
                    transform=ax.transAxes, fontsize=7,
                    family="monospace", va="top",
                    bbox={"boxstyle": "round", "facecolor": "white",
                          "alpha": 0.7, "edgecolor": "lightgray"})
    fig.suptitle("Hidden-state trajectories of the solved networks "
                 "(5 hidden units, tanh)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_matrices(results: dict, out_path: Path) -> None:
    """Heatmap of the recurrent weight matrices for each solved grammar."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), dpi=120,
                              gridspec_kw={"width_ratios": [1, 1.6, 1, 1.6]})
    for row, g in enumerate([1, 2, 4]):
        W = results[g]["weights"]
        if W is None:
            for col in range(4):
                axes[row, col].axis("off")
            continue
        items = [
            ("$W_{xh}$ (in)", W["W_xh"]),
            ("$W_{hh}$ (recurrent)", W["W_hh"]),
            ("$W_{hy}$ (out)", W["W_hy"]),
            ("$b_h$ (bias)", W["b_h"][:, None]),
        ]
        for col, (title, mat) in enumerate(items):
            ax = axes[row, col]
            mx = max(abs(mat).max(), 1e-3)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-mx, vmax=mx, aspect="auto")
            ax.set_title(f"Tomita {GRAMMAR_LABELS[g]}\n{title}",
                         fontsize=9 if row == 0 else 9)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_yticks(range(mat.shape[0]))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(mat[i, j]) > mx * 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Solved-network weight matrices", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_accuracy_distribution(results: dict, out_path: Path,
                                seed: int, n_samples: int = 5000) -> None:
    """Histogram of per-trial train accuracy across n_samples random networks.

    Shows that for grammars #1, #2, the chance of fitting all train strings
    is appreciable; for #4 it drops to a long-tail event.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120, sharey=True)
    for ax, g in zip(axes, [1, 2, 4]):
        train_strings = results[g]["train_strings"]
        train_y = results[g]["train_y"]
        max_len = max(len(s) for s in train_strings)
        # Also include test for batch padding (but we evaluate on train only)
        X, lens = encode_batch(train_strings, max_len=max_len)
        rng = np.random.default_rng(seed + g * 1000)
        accs = []
        for _ in range(n_samples):
            W = sample_weights(rng, hidden=5, scale=2.0)
            pred, _ = forward(W, X, lens)
            accs.append(np.mean(pred == train_y))
        accs = np.array(accs)
        ax.hist(accs, bins=20, color=GRAMMAR_COLORS[g], alpha=0.85,
                edgecolor="black", linewidth=0.4)
        frac_perfect = float((accs >= 0.999).mean())
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.text(0.05, 0.95,
                f"P(train_acc = 1) ≈ {frac_perfect:.4f}\n"
                f"E[trials to solve] ≈ {1 / max(frac_perfect, 1e-6):.0f}",
                transform=ax.transAxes, fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white",
                      "alpha": 0.85, "edgecolor": "lightgray"},
                va="top")
        ax.set_xlabel("training accuracy of a single random net")
        ax.set_ylabel("count" if g == 1 else "")
        ax.set_title(f"Tomita {GRAMMAR_LABELS[g]}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Distribution of training accuracy over {n_samples} "
                 f"random-weight networks (uniform[-2, 2], 5 hidden)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def maybe_run(seed: int, npz_path: Path) -> dict:
    """Load results from npz_path, or run the search to populate it."""
    if not npz_path.exists():
        print(f"results file {npz_path} missing -- running search now ...")
        from rs_tomita import DEFAULT_MAX_TRIALS, save_results
        rs_results = []
        for g in [1, 2, 4]:
            print(f"  Tomita #{g} ...")
            r = run_grammar(g, seed, DEFAULT_MAX_TRIALS[g], scale=2.0, hidden=5)
            rs_results.append(r)
        save_results(npz_path, rs_results, seed)
    return load_npz(npz_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results", type=str, default="results/rs_tomita_seed{seed}.npz",
        help="Path to NPZ produced by rs_tomita.py.",
    )
    parser.add_argument("--outdir", type=str, default="viz")
    parser.add_argument("--n-acc-samples", type=int, default=5000,
                        help="Random-net samples for accuracy histogram.")
    args = parser.parse_args()

    npz_path = Path(args.results.format(seed=args.seed))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = maybe_run(args.seed, npz_path)

    plot_search_curves(results, outdir / "search_curves.png")
    plot_hidden_trajectories(results, outdir / "hidden_trajectories.png")
    plot_weight_matrices(results, outdir / "weight_matrices.png")
    plot_accuracy_distribution(
        results, outdir / "weight_distributions.png",
        seed=args.seed, n_samples=args.n_acc_samples,
    )


if __name__ == "__main__":
    main()
