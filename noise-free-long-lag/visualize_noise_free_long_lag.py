"""Static-PNG visualisations for noise-free-long-lag.

Generates four panels in `viz/`:

  1. training_curves.png  -- per-eval loss + accuracy over training
  2. cell_state_trace.png -- one cell of `c[t]` across a full sequence
                              for both x-key and y-key inputs, showing
                              the constant-error-carousel (cell flat
                              through distractors)
  3. gate_activations.png -- input/forget/output gate values across a
                              sequence (rows are gates, x-axis is t)
  4. last_step_probs.png  -- bar chart of softmax probabilities for the
                              last step on a y-key vs x-key example
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import noise_free_long_lag as nfl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=4000)
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out = nfl.train(
        p=args.p,
        hidden=args.hidden,
        seed=args.seed,
        max_seq=args.max_seq,
        verbose=False,
    )
    model = out["model"]
    log = out["log"]
    report = out["report"]

    # --- Panel 1: training curves ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(log["step"], log["loss_total"], label="total CE", color="C0")
    axes[0].plot(log["step"], log["loss_last"], label="last-step CE",
                 color="C3", linestyle="--")
    axes[0].set_xlabel("training sequence")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("loss")

    axes[1].plot(log["step"], log["rolling_acc_last"],
                 label="rolling-256 last-step acc", color="C2")
    axes[1].plot(log["step"], log["acc_last"],
                 label="held-out last-step acc", color="C0", linestyle="--")
    axes[1].plot(log["step"], log["acc_per_step"],
                 label="held-out per-step acc", color="C1", linestyle=":")
    axes[1].axhline(0.95, color="k", lw=0.5, linestyle=":", alpha=0.4)
    axes[1].set_xlabel("training sequence")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].set_title(f"accuracy (solved at seq {report['solved_at_seq']})")

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=110)
    plt.close(fig)

    # --- Build two test sequences (one per key) -------------------------
    p = args.p
    V = nfl.alphabet_size(p)

    def fixed_seq(start_key: int):
        """Build a sequence that starts/ends with the given key index."""
        seq = [start_key] + list(range(p - 1)) + [start_key]
        inputs = np.asarray(seq[:-1], dtype=np.int64)
        targets = np.asarray(seq[1:], dtype=np.int64)
        T = inputs.shape[0]
        X = np.zeros((T, V))
        X[np.arange(T), inputs] = 1.0
        return X, targets

    Xy, Ty = fixed_seq(nfl.y_index(p))
    Xx, Tx = fixed_seq(nfl.x_index(p))
    _, cy = model.forward(Xy)
    _, cx = model.forward(Xx)

    # --- Panel 2: cell-state trace -------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    H = model.H
    # Pick the cell that diverges the most between the two keys
    diff = np.abs(cy["c"][1:] - cx["c"][1:]).sum(axis=0)
    cell_id = int(np.argmax(diff))
    ax.plot(cy["c"][1:, cell_id], label=f"y-key sequence (cell #{cell_id})", color="C0")
    ax.plot(cx["c"][1:, cell_id], label=f"x-key sequence (cell #{cell_id})", color="C3")
    ax.axvline(0, color="k", lw=0.5, linestyle=":")
    ax.axvline(p - 1, color="k", lw=0.5, linestyle=":")
    ax.set_xlabel("time step t")
    ax.set_ylabel("cell state c[t]")
    ax.legend()
    ax.set_title("constant error carousel: cell state preserves the key across distractors")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cell_state_trace.png"), dpi=110)
    plt.close(fig)

    # --- Panel 3: gate activations -------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, gate_name, key in zip(axes, ["input gate", "forget gate", "output gate"],
                                  ["i", "f", "o"]):
        # Mean over cells for clarity
        ax.plot(cy[key].mean(axis=1), label="y-key", color="C0")
        ax.plot(cx[key].mean(axis=1), label="x-key", color="C3")
        ax.set_ylabel(gate_name)
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(0, color="k", lw=0.5, linestyle=":")
        ax.axvline(p - 1, color="k", lw=0.5, linestyle=":")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time step t")
    fig.suptitle("gate activations averaged across cells (key at t=0 and t=p)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "gate_activations.png"), dpi=110)
    plt.close(fig)

    # --- Panel 4: last-step probability bars ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, cache, label in zip(axes, [cy, cx], ["y-key sequence", "x-key sequence"]):
        probs_last = cache["probs"][-1]
        colors = ["#aaaaaa"] * V
        colors[nfl.x_index(p)] = "#3070C0"
        colors[nfl.y_index(p)] = "#C03030"
        ax.bar(np.arange(V), probs_last, color=colors)
        ax.set_xlabel("symbol index")
        ax.set_ylabel("p(symbol | sequence)")
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
    fig.suptitle("predicted distribution at the final step")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "last_step_probs.png"), dpi=110)
    plt.close(fig)

    print(f"Wrote 4 PNGs to {args.outdir}/")
    print(f"  - training_curves.png")
    print(f"  - cell_state_trace.png   (cell #{cell_id})")
    print(f"  - gate_activations.png")
    print(f"  - last_step_probs.png")


if __name__ == "__main__":
    main()
