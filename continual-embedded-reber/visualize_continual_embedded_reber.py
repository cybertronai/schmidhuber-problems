"""
Static visualizations for continual-embedded-reber.

Trains both architectures (LSTMForget, LSTMNoForget) on the same
continual stream, then writes the following PNGs under viz/:

    training_curves.png     loss + outer T/P accuracy across training,
                            overlaid for both networks
    cell_state_trace.png    cell-state norm along a long fresh stream;
                            forget-LSTM resets at 'E', no-forget grows
    forget_gate_at_E.png    mean forget-gate activation in a window
                            around end-of-string markers (forget LSTM)
    sample_rollout.png      side-by-side next-symbol heatmaps for a
                            short fresh stream (~3 strings)
    outer_acc_by_string.png outer T/P accuracy as a function of the
                            string's position in the continual stream
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continual_embedded_reber import (
    ALPHABET, N_SYM, SYM2IDX,
    LSTMForget, LSTMNoForget,
    gen_continual_stream, make_io,
    outer_acc_by_position, train,
    _legal_next_in_substring,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=2000)
    ap.add_argument("--chunk-strings", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--outdir", default="viz")
    ap.add_argument("--eval-strings", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("training LSTMForget...")
    out_f = train(
        LSTMForget,
        seed=args.seed,
        n_hidden=args.hidden,
        n_chunks=args.n_chunks,
        chunk_strings=args.chunk_strings,
        eval_every=200,
        eval_strings=args.eval_strings,
        verbose=False,
    )
    net_f = out_f["net"]
    print(f"  forget   final outer={out_f['final_outer']:.3f}  legal={out_f['final_legal']:.3f}")

    print("training LSTMNoForget...")
    out_n = train(
        LSTMNoForget,
        seed=args.seed,
        n_hidden=args.hidden,
        n_chunks=args.n_chunks,
        chunk_strings=args.chunk_strings,
        eval_every=200,
        eval_strings=args.eval_strings,
        verbose=False,
    )
    net_n = out_n["net"]
    print(f"  noforget final outer={out_n['final_outer']:.3f}  legal={out_n['final_legal']:.3f}")

    # ----------------------------------------------------------------
    # 1. training curves
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    win = 30
    for out, label, color in [
        (out_f, "forget", "C2"),
        (out_n, "no-forget", "C3"),
    ]:
        L = np.asarray(out["losses"])
        if len(L) >= win:
            kern = np.ones(win) / win
            smooth = np.convolve(L, kern, mode="valid")
            x = np.arange(len(smooth)) + win
            axes[0].plot(x, smooth, color=color, label=label)
        else:
            axes[0].plot(L, color=color, label=label)
    axes[0].set_xlabel("training chunks")
    axes[0].set_ylabel("loss / step (smoothed)")
    axes[0].set_title("Training loss")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(out_f["chunk_index"], out_f["outer_curve"],
                 color="C2", marker="o", ms=4, label="forget: outer T/P")
    axes[1].plot(out_n["chunk_index"], out_n["outer_curve"],
                 color="C3", marker="o", ms=4, label="no-forget: outer T/P")
    axes[1].plot(out_f["chunk_index"], out_f["legal_curve"],
                 color="C2", linestyle="--", alpha=0.6, label="forget: legal")
    axes[1].plot(out_n["chunk_index"], out_n["legal_curve"],
                 color="C3", linestyle="--", alpha=0.6, label="no-forget: legal")
    axes[1].axhline(0.5, color="grey", linestyle=":", lw=1, label="outer chance (0.5)")
    axes[1].set_xlabel("training chunks")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title(f"Eval on continual stream of {args.eval_strings} strings")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=130)
    plt.close(fig)
    print("  wrote training_curves.png")

    # ----------------------------------------------------------------
    # 2. cell state norm along a long fresh stream
    # ----------------------------------------------------------------
    long_rng_seed = args.seed + 12345
    long_n_strings = 60
    stats_f = outer_acc_by_position(net_f, long_n_strings, np.random.default_rng(long_rng_seed))
    stats_n = outer_acc_by_position(net_n, long_n_strings, np.random.default_rng(long_rng_seed))

    fig, ax = plt.subplots(figsize=(11, 3.6))
    T_total = stats_f["cell_norm"].shape[0]
    ax.plot(np.arange(T_total), stats_n["cell_norm"], color="C3",
            label="no-forget   |s_t|", lw=1.0)
    ax.plot(np.arange(T_total), stats_f["cell_norm"], color="C2",
            label="forget       |s_t|", lw=1.0)
    # mark string boundaries
    for (start, end) in stats_f["bounds"][:25]:
        ax.axvline(end - 1, color="grey", alpha=0.18, lw=0.5)
    ax.set_xlabel("step in continual stream")
    ax.set_ylabel("‖cell state‖₂")
    ax.set_yscale("log")
    ax.set_title("Cell-state magnitude along a continual Reber stream\n"
                 "(grey verticals = string boundaries 'E')")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cell_state_trace.png"), dpi=130)
    plt.close(fig)
    print("  wrote cell_state_trace.png")

    # ----------------------------------------------------------------
    # 3. forget gate activation around end-of-string markers
    # ----------------------------------------------------------------
    f_trace = stats_f["cache"]["f"]   # (T, H)
    bounds = stats_f["bounds"]
    pre = 6
    post = 6
    aligned = []
    for (start, end) in bounds[1:-1]:
        # the symbol AT position (end-1) is 'E'; gate at that step is f_trace[end-1]
        center = end - 1
        if center - pre >= 0 and center + post < f_trace.shape[0]:
            aligned.append(f_trace[center - pre:center + post + 1])
    aligned = np.array(aligned)            # (n_strings, pre+post+1, H)
    mean_per_unit = aligned.mean(axis=0)   # (pre+post+1, H)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    xs = np.arange(-pre, post + 1)
    for u in range(mean_per_unit.shape[1]):
        ax.plot(xs, mean_per_unit[:, u], lw=0.8, alpha=0.7)
    ax.plot(xs, mean_per_unit.mean(axis=1), color="black", lw=2.0,
            label="mean across units")
    ax.axvline(0, color="C3", linestyle="--", lw=1.0, label="step at 'E'")
    ax.axhline(0.5, color="grey", linestyle=":", lw=0.8)
    ax.set_xlabel("offset from end-of-string marker 'E'")
    ax.set_ylabel("forget gate activation f_t")
    ax.set_title("Forget-gate activation aligned on string boundaries\n"
                 "(values close to 0 = cell state is reset)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "forget_gate_at_E.png"), dpi=130)
    plt.close(fig)
    print("  wrote forget_gate_at_E.png")

    # ----------------------------------------------------------------
    # 4. side-by-side rollout heatmap on the same short fresh stream
    # ----------------------------------------------------------------
    rng_short = np.random.default_rng(args.seed + 2025)
    stream_short, bounds_short = gen_continual_stream(rng_short, 3)
    Xs, _ = make_io(stream_short)
    probs_f, _ = net_f.predict(Xs)
    probs_n, _ = net_n.predict(Xs)

    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.4 * len(stream_short)), 5.4))
    for ax, probs, title in [(axes[0], probs_f, "Forget LSTM"),
                             (axes[1], probs_n, "No-forget LSTM")]:
        im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(N_SYM))
        ax.set_yticklabels(ALPHABET)
        ax.set_xticks(range(len(stream_short) - 1))
        ax.set_xticklabels(
            [f"{stream_short[i]}" for i in range(len(stream_short) - 1)],
            fontsize=7,
        )
        # legal-symbol red boxes + outer-position yellow box
        for (start, end) in bounds_short:
            t_outer = end - 3
            if 0 <= t_outer < probs.shape[0]:
                ax.add_patch(plt.Rectangle((t_outer - 0.5, -0.5), 1, N_SYM,
                                           fill=False, edgecolor="yellow", lw=1.6))
            ax.axvline(end - 1.5, color="white", alpha=0.5, lw=0.8)
        for t in range(probs.shape[0]):
            for s in _legal_next_in_substring(stream_short, bounds_short, t):
                r = SYM2IDX[s]
                ax.add_patch(plt.Rectangle((t - 0.5, r - 0.5), 1, 1,
                                           fill=False, edgecolor="red", lw=0.7))
        ax.set_title(f"{title} -- next-symbol distribution on continual stream",
                     fontsize=10)
        ax.set_ylabel("predicted next symbol")
    axes[1].set_xlabel(
        "step (symbol shown). Red = Reber-legal. Yellow = outer-T/P column. White = string boundaries.")
    plt.colorbar(im, ax=axes.ravel().tolist(), pad=0.01).set_label("p(next = sym)")
    fig.savefig(os.path.join(args.outdir, "sample_rollout.png"), dpi=130)
    plt.close(fig)
    print("  wrote sample_rollout.png")

    # ----------------------------------------------------------------
    # 5. outer T/P accuracy as a function of string index in the stream
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.4))
    n_seeds = 5
    bin_size = 5
    forget_acc_runs = []
    noforget_acc_runs = []
    for s_off in range(n_seeds):
        s_rng = np.random.default_rng(args.seed + 7777 + s_off)
        stats_f_run = outer_acc_by_position(net_f, long_n_strings, s_rng)
        s_rng = np.random.default_rng(args.seed + 7777 + s_off)
        stats_n_run = outer_acc_by_position(net_n, long_n_strings, s_rng)
        forget_acc_runs.append(stats_f_run["outer_hits"])
        noforget_acc_runs.append(stats_n_run["outer_hits"])
    forget_acc_arr = np.array(forget_acc_runs).mean(axis=0)
    noforget_acc_arr = np.array(noforget_acc_runs).mean(axis=0)

    def bin_mean(a, b):
        n = (len(a) // b) * b
        return a[:n].reshape(-1, b).mean(axis=1)

    ax.plot(np.arange(len(forget_acc_arr)) + 1,
            forget_acc_arr, color="C2", alpha=0.3, lw=0.8)
    ax.plot(np.arange(len(noforget_acc_arr)) + 1,
            noforget_acc_arr, color="C3", alpha=0.3, lw=0.8)
    fb = bin_mean(forget_acc_arr, bin_size)
    nb = bin_mean(noforget_acc_arr, bin_size)
    ax.plot((np.arange(len(fb)) + 0.5) * bin_size, fb,
            color="C2", marker="o", lw=1.5, label=f"forget (bin={bin_size})")
    ax.plot((np.arange(len(nb)) + 0.5) * bin_size, nb,
            color="C3", marker="s", lw=1.5, label=f"no-forget (bin={bin_size})")
    ax.axhline(0.5, color="grey", linestyle=":", lw=1, label="chance (0.5)")
    ax.set_xlabel("string position in continual stream")
    ax.set_ylabel("outer T/P accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Outer T/P accuracy along the stream "
                 f"(mean of {n_seeds} fresh streams)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "outer_acc_by_string.png"), dpi=130)
    plt.close(fig)
    print("  wrote outer_acc_by_string.png")


if __name__ == "__main__":
    main()
