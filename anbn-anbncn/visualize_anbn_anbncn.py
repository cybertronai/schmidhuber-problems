"""Static visualisations for anbn-anbncn.

Produces six PNGs under viz/:

* training_loss.png        — per-step BCE loss for both languages
* generalization.png       — per-n accept/reject for n=1..N_TEST, both languages
* generalization_curve.png — max accepted-run-from-n=1 over training step
* cell_state_anbn.png      — cell trajectory on a^15 b^15
* cell_state_anbncn.png    — cell trajectory on a^15 b^15 c^15
* gates.png                — gate activations on the same long sequence

Run: python3 visualize_anbn_anbncn.py --seed 1
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anbn_anbncn import (
    train, lstm_forward, dict_to_lstm, lstm_to_dict,
    make_anbn, make_anbncn, ANBN_VOCAB, ANBNCN_VOCAB,
    eval_generalisation,
)


HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)


def smooth(x, k=50):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--steps-anbn", type=int, default=4000)
    ap.add_argument("--steps-anbncn", type=int, default=10000)
    args = ap.parse_args()

    # Train both, capturing param snapshots for the gif script.
    print("Training anbn...")
    p_a, stats_a = train(
        lang="anbn", hidden=2, n_train_max=args.n_train, n_test=args.n_test,
        n_steps=args.steps_anbn, lr=0.01, seed=args.seed, log_every=200,
        early_stop_target=2 * args.n_train,
    )
    print(f"  anbn final max_run={stats_a['final_eval']['max_run']}  steps={stats_a['steps_run']}")

    print("Training anbncn...")
    p_b, stats_b = train(
        lang="anbncn", hidden=3, n_train_max=args.n_train, n_test=args.n_test,
        n_steps=args.steps_anbncn, lr=0.01, seed=args.seed, log_every=200,
    )
    print(f"  anbncn final max_run={stats_b['final_eval']['max_run']}  steps={stats_b['steps_run']}")

    # ---- Training loss
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    la = np.array(stats_a["loss_hist"]); lb = np.array(stats_b["loss_hist"])
    ax.plot(smooth(la), label=f"a^n b^n (h=2, ends step {stats_a['steps_run']})", color="tab:blue")
    ax.plot(smooth(lb), label=f"a^n b^n c^n (h=3, ends step {stats_b['steps_run']})", color="tab:red")
    ax.set_xlabel("training step")
    ax.set_ylabel("per-symbol BCE (50-step moving avg)")
    ax.set_yscale("log")
    ax.set_title(f"Training loss (seed={args.seed}, n trained ≤ {args.n_train})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "training_loss.png"), dpi=120)
    plt.close(fig)

    # ---- Per-n accept
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    for ax, p, lang, hidden, name, color in (
        (axes[0], p_a, "anbn", 2, "a^n b^n", "tab:blue"),
        (axes[1], p_b, "anbncn", 3, "a^n b^n c^n", "tab:red"),
    ):
        ev = eval_generalisation(p, lang, args.n_test)
        ok = np.array(ev["per_n"], dtype=int)
        ax.bar(np.arange(1, args.n_test + 1), ok, color=color)
        ax.axvspan(0.5, args.n_train + 0.5, alpha=0.15, color="grey",
                   label=f"trained n=1..{args.n_train}")
        ax.set_ylabel("accepted")
        ax.set_yticks([0, 1])
        ax.set_xlim(0.5, args.n_test + 0.5)
        ax.set_title(f"{name}: contiguous accept-run from n=1 = {ev['max_run']}; "
                     f"total accepted = {len(ev['accepted'])} of {args.n_test}")
        ax.legend(loc="lower right")
    axes[1].set_xlabel("n")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "generalization.png"), dpi=120)
    plt.close(fig)

    # ---- Generalisation max-run curve over training
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if stats_a["gen_hist"]:
        s_a, r_a = zip(*stats_a["gen_hist"])
        ax.plot(s_a, r_a, "o-", color="tab:blue", label="a^n b^n max accept run")
    if stats_b["gen_hist"]:
        s_b, r_b = zip(*stats_b["gen_hist"])
        ax.plot(s_b, r_b, "s-", color="tab:red", label="a^n b^n c^n max accept run")
    ax.axhline(args.n_train, color="grey", linestyle=":", label=f"end of training range n={args.n_train}")
    ax.axhline(2 * args.n_train, color="black", linestyle=":", label=f"2× training (n={2*args.n_train})")
    ax.set_xlabel("training step")
    ax.set_ylabel("max contiguous accepted n (from 1)")
    ax.set_title("Generalisation expands with training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "generalization_curve.png"), dpi=120)
    plt.close(fig)

    # ---- Cell state on a long sequence (anbn)
    inp_a, _ = make_anbn(15)
    cache_a = lstm_forward(p_a, inp_a)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    T = inp_a.shape[0]
    for h in range(p_a.bi.shape[0]):
        ax.plot(cache_a["c"][:, h], "-o", label=f"cell {h}", markersize=3)
    # Mark the segment boundaries.
    ax.axvspan(0, 0.5, alpha=0.15, color="grey")
    ax.axvspan(0.5, 15.5, alpha=0.10, color="tab:green")     # a-block
    ax.axvspan(15.5, 30.5, alpha=0.10, color="tab:orange")  # b-block
    ax.set_xlabel("time step")
    ax.set_ylabel("cell value c_t")
    ax.set_title(f"a^n b^n cell state on n=15 (trained on n≤{args.n_train})")
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels([ANBN_VOCAB[int(np.argmax(inp_a[t]))] for t in range(T)])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "cell_state_anbn.png"), dpi=120)
    plt.close(fig)

    # ---- Cell state (anbncn)
    inp_b, _ = make_anbncn(15)
    cache_b = lstm_forward(p_b, inp_b)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    T = inp_b.shape[0]
    for h in range(p_b.bi.shape[0]):
        ax.plot(cache_b["c"][:, h], "-o", label=f"cell {h}", markersize=3)
    ax.axvspan(0, 0.5, alpha=0.15, color="grey")
    ax.axvspan(0.5, 15.5, alpha=0.10, color="tab:green")     # a
    ax.axvspan(15.5, 30.5, alpha=0.10, color="tab:orange")  # b
    ax.axvspan(30.5, 45.5, alpha=0.10, color="tab:purple")  # c
    ax.set_xlabel("time step")
    ax.set_ylabel("cell value c_t")
    ax.set_title(f"a^n b^n c^n cell state on n=15 (trained on n≤{args.n_train})")
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels([ANBNCN_VOCAB[int(np.argmax(inp_b[t]))] for t in range(T)])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "cell_state_anbncn.png"), dpi=120)
    plt.close(fig)

    # ---- Gate activations: 3 gates × 2 languages
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col")
    columns = (
        (cache_a, ANBN_VOCAB, "a^n b^n  n=15"),
        (cache_b, ANBNCN_VOCAB, "a^n b^n c^n  n=15"),
    )
    gate_names = (("input gate i", "i"), ("forget gate f", "f"), ("output gate o", "o"))
    for col, (cache, vocab, name) in enumerate(columns):
        Tlen = cache["X"].shape[0]
        xticks = [vocab[int(np.argmax(cache["X"][t]))] for t in range(Tlen)]
        for r, (gname, key) in enumerate(gate_names):
            ax = axes[r, col]
            gact = cache[key]
            for h in range(gact.shape[1]):
                ax.plot(gact[:, h], "-o", markersize=3, label=f"unit {h}")
            ax.set_ylabel(gname)
            if r == 0:
                ax.set_title(name)
            ax.set_xticks(np.arange(Tlen))
            ax.set_xticklabels(xticks, fontsize=6)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if col == 0 and r == 0:
                ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Gate activations on a long sequence (peephole LSTM)")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "gates.png"), dpi=120)
    plt.close(fig)

    print(f"Wrote 6 PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
