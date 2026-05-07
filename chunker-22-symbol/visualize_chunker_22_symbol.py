"""
Static visualisations for the trained chunker-22-symbol pair.

Outputs (in `viz/`):

    training_curves.png    --- rolling label and symbol accuracy over training
                               for A-alone vs Chunker, plus surprise count.
    surprise_pattern.png   --- heatmap of which symbols (a, x, b1..b20) trigger
                               A-surprises over training, demonstrating that
                               surprises taper to the choice-bits as A learns.
    network_weights.png    --- Hinton diagrams of A's W_xh / W_hh and C's W_xh
                               + label head, showing A spreads attention over
                               the 22 symbols while C concentrates on a/x.
    test_episode.png       --- one fresh test block: events, A's per-step
                               predicted next-symbol probs, surprise marker,
                               C's label readout.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from chunker_22_symbol import (
    train, RNN, gen_block, gen_stream, rnn_forward,
    softmax, sigmoid,
    ALPHABET, A_IDX, X_IDX, B_START, BLOCK_LEN,
)


# ----------------------------------------------------------------------
# Symbol naming helpers
# ----------------------------------------------------------------------

def sym_name(idx: int) -> str:
    if idx == A_IDX:
        return "a"
    if idx == X_IDX:
        return "x"
    return f"b{idx - 1}"   # b1..b20


SYM_LABELS = [sym_name(i) for i in range(ALPHABET)]


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(hist_alone: dict, hist_chunker: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), dpi=120)

    ax = axes[0]
    ax.plot(hist_alone["block"], np.array(hist_alone["label_acc"]) * 100,
            color="#d62728", linewidth=1.0, label="A-alone")
    ax.plot(hist_chunker["block"], np.array(hist_chunker["label_acc"]) * 100,
            color="#1f77b4", linewidth=1.0, label="Chunker (A + C)")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("training block")
    ax.set_ylabel("label accuracy (rolling 200, %)")
    ax.set_ylim(0, 105)
    ax.set_title("20-step-lag label task")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hist_alone["block"], np.array(hist_alone["sym_acc"]) * 100,
            color="#d62728", linewidth=1.0, label="A-alone")
    ax.plot(hist_chunker["block"], np.array(hist_chunker["sym_acc"]) * 100,
            color="#1f77b4", linewidth=1.0, label="Chunker")
    ax.set_xlabel("training block")
    ax.set_ylabel("next-symbol accuracy (rolling 200, %)")
    ax.set_ylim(0, 105)
    ax.set_title("Next-symbol prediction (A)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    blocks = np.array(hist_chunker["block"])
    n_surp = np.array(hist_chunker["n_surprises"])
    ax.plot(blocks, n_surp / np.maximum(blocks + 1, 1),
            color="#1f77b4", linewidth=1.0)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":",
               label="ideal (1 boundary surprise / block)")
    ax.set_xlabel("training block")
    ax.set_ylabel("running surprises / block")
    ax.set_ylim(0, 22)
    ax.set_title("Chunker: surprises tapering as A learns")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("chunker-22-symbol  (vanilla RNN, hidden=32, threshold=0.95)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Surprise heatmap (which symbols trigger surprises over training)
# ----------------------------------------------------------------------

def plot_surprise_pattern(seed: int, n_blocks: int, threshold: float,
                          out_path: str):
    """Re-run training while logging the per-block surprise pattern."""
    surprise_log = []  # list of arrays of shape (BLOCK_LEN,) — bool

    rng = np.random.default_rng(seed)
    blocks, _ = gen_stream(n_blocks + 1, rng)
    stream = np.concatenate([np.array(b, dtype=np.int64) for b in blocks])
    eye = np.eye(ALPHABET)

    # Re-run a stripped training of A only (we don't need C for this plot)
    seed_seq = np.random.SeedSequence(seed)
    _, rng_a, _ = (np.random.default_rng(s) for s in seed_seq.spawn(3))
    A = RNN(ALPHABET, 32, ALPHABET, rng_a)
    from chunker_22_symbol import Adam, rnn_backward
    opt_A = Adam(A.params(), lr=1e-2)
    h_a = np.zeros(A.n_hidden)

    for chunk_i in range(n_blocks):
        start = chunk_i * BLOCK_LEN
        in_idx = stream[start:start + BLOCK_LEN]
        target_idx = stream[start + 1:start + 1 + BLOCK_LEN]
        inputs = eye[in_idx]
        traj = rnn_forward(A, inputs, h_a)

        # detect surprise on each step
        mask = np.zeros(BLOCK_LEN, dtype=bool)
        for t in range(BLOCK_LEN):
            p_target = softmax(traj["sym_logits"][t])[int(target_idx[t])]
            mask[t] = (p_target < threshold)
        surprise_log.append(mask)

        # train A on the deterministic transitions only (same as main loop)
        sym_targets = [(t, int(target_idx[t])) for t in range(BLOCK_LEN - 1)]
        grads = rnn_backward(A, traj, sym_targets, [])
        opt_A.step([grads[n] for n in A.param_names()])
        h_a = traj["h"][-1].copy()

    arr = np.stack(surprise_log).astype(float)  # (n_blocks, BLOCK_LEN)

    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=120)
    im = ax.imshow(arr.T, aspect="auto", cmap="Blues", interpolation="nearest",
                   origin="lower",
                   extent=(0, n_blocks, -0.5, BLOCK_LEN - 0.5))
    ax.set_xlabel("training block")
    ax.set_ylabel("position within 21-symbol block")
    ax.set_yticks(range(BLOCK_LEN))
    yticklabels = ["choice (a or x)"] + [f"b{i+1}" for i in range(20)]
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_title("A's surprise pattern over training\n"
                 "(blue = A's prob of actual next symbol < 0.95)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("surprise indicator")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Hinton-diagram weight matrix
# ----------------------------------------------------------------------

def hinton(ax, W: np.ndarray, row_labels=None, col_labels=None, title=""):
    n_row, n_col = W.shape
    max_abs = max(abs(W).max(), 1e-3)
    ax.set_xlim(-0.6, n_col - 0.4)
    ax.set_ylim(-0.6, n_row - 0.4)
    ax.invert_yaxis()
    for i in range(n_row):
        for j in range(n_col):
            w = W[i, j]
            sz = 0.85 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.3))
    if col_labels is not None:
        ax.set_xticks(range(n_col))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=90)
    if row_labels is not None:
        ax.set_yticks(range(n_row))
        ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def plot_network_weights(A: RNN, C: RNN, out_path: str):
    fig = plt.figure(figsize=(13.5, 5.5), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # A's W_xh: (n_hidden, n_in)  --  show W_xh.T so rows = symbols
    axA1 = fig.add_subplot(gs[0, 0])
    h_labels = [f"h{i}" for i in range(A.n_hidden)]
    hinton(axA1, A.W_xh.T, row_labels=SYM_LABELS, col_labels=h_labels,
           title=r"A: $W_{xh}^T$  (input -> hidden)")

    # A's W_hh
    axA2 = fig.add_subplot(gs[0, 1])
    hinton(axA2, A.W_hh, row_labels=h_labels, col_labels=h_labels,
           title=r"A: $W_{hh}$  (hidden -> hidden)")

    # A's W_hy.T (rows = symbols)
    axA3 = fig.add_subplot(gs[0, 2])
    hinton(axA3, A.W_hy, row_labels=SYM_LABELS, col_labels=h_labels,
           title=r"A: $W_{hy}$  (hidden -> next-symbol)")

    # C's W_xh.T
    axC1 = fig.add_subplot(gs[1, 0])
    hC_labels = [f"h{i}" for i in range(C.n_hidden)]
    hinton(axC1, C.W_xh.T, row_labels=SYM_LABELS, col_labels=hC_labels,
           title=r"C: $W_{xh}^T$  (input -> hidden)")

    # C's W_hl.T  (one column = the label head)
    axC2 = fig.add_subplot(gs[1, 1])
    hinton(axC2, C.W_hl.T, row_labels=hC_labels,
           col_labels=["label"], title=r"C: $W_{hl}^T$  (hidden -> label)")

    # C's b_h reshaped as a column for visual sanity
    axC3 = fig.add_subplot(gs[1, 2])
    hinton(axC3, C.W_hh, row_labels=hC_labels, col_labels=hC_labels,
           title=r"C: $W_{hh}$  (unused on this clean stream; see Deviations)")

    fig.suptitle("Trained network weights (A on raw stream, C on surprises)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Test-episode rollout
# ----------------------------------------------------------------------

def plot_test_episode(A: RNN, C: RNN, out_path: str,
                      seed: int = 12345, n_blocks: int = 4,
                      threshold: float = 0.95):
    rng = np.random.default_rng(seed)
    blocks, labels = gen_stream(n_blocks + 1, rng)
    stream = np.concatenate([np.array(b, dtype=np.int64) for b in blocks])
    T_total = n_blocks * BLOCK_LEN
    eye = np.eye(ALPHABET)

    h_a = np.zeros(A.n_hidden)
    last_surprise_sym = None

    p_actual = np.zeros(T_total)            # A's predicted prob of actual next
    surprise_mark = np.zeros(T_total, dtype=bool)
    label_pred = np.zeros(n_blocks)
    label_targets = np.zeros(n_blocks)

    for chunk_i in range(n_blocks):
        start = chunk_i * BLOCK_LEN
        in_idx = stream[start:start + BLOCK_LEN]
        target_idx = stream[start + 1:start + 1 + BLOCK_LEN]
        inputs = eye[in_idx]
        traj = rnn_forward(A, inputs, h_a)

        # label readout (BEFORE this chunk's surprises)
        if last_surprise_sym is not None:
            c_in = eye[last_surprise_sym][None, :]
            traj_q = rnn_forward(C, c_in, np.zeros(C.n_hidden))
            label_pred[chunk_i] = float(sigmoid(traj_q["label_pre"][0]))
        else:
            label_pred[chunk_i] = 0.5
        label_targets[chunk_i] = labels[chunk_i]

        for t in range(BLOCK_LEN):
            p_t = float(softmax(traj["sym_logits"][t])[int(target_idx[t])])
            p_actual[start + t] = p_t
            if p_t < threshold:
                surprise_mark[start + t] = True
                if int(target_idx[t]) in (A_IDX, X_IDX):
                    last_surprise_sym = int(target_idx[t])

        h_a = traj["h"][-1].copy()

    fig = plt.figure(figsize=(13.5, 6.5), dpi=120)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.0, 1.0])

    # Top: stream as colored squares
    ax0 = fig.add_subplot(gs[0])
    colors = []
    for s in stream[:T_total]:
        if s == A_IDX:
            colors.append("#d62728")
        elif s == X_IDX:
            colors.append("#1f77b4")
        else:
            colors.append("#cccccc")
    for t, c in enumerate(colors):
        ax0.add_patch(Rectangle((t, 0), 1, 1, facecolor=c, edgecolor="black",
                                linewidth=0.3))
    for chunk_i in range(n_blocks):
        ax0.axvline(chunk_i * BLOCK_LEN, color="black", linewidth=1.5)
    ax0.set_xlim(0, T_total)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title("Stream (red = a, blue = x, gray = b1..b20)")

    # 2nd: A's predicted prob of actual next symbol
    ax1 = fig.add_subplot(gs[1])
    t_axis = np.arange(T_total)
    ax1.plot(t_axis, p_actual, color="#1f77b4", linewidth=1.0)
    ax1.axhline(threshold, color="#d62728", linestyle=":",
                linewidth=1.0, label=f"surprise threshold = {threshold}")
    surprise_idx = np.flatnonzero(surprise_mark)
    ax1.scatter(surprise_idx, p_actual[surprise_idx], s=18,
                color="#d62728", marker="x", label="surprise")
    for chunk_i in range(n_blocks):
        ax1.axvline(chunk_i * BLOCK_LEN, color="gray", linewidth=0.5,
                    linestyle="--")
    ax1.set_xlim(0, T_total)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("P(actual next)")
    ax1.set_title("A's predicted probability of the actual next symbol")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 3rd: per-block label readout from C  (bars centred at 0.5 so an
    # "x" prediction (P close to 0) is just as visible as an "a" one)
    ax2 = fig.add_subplot(gs[2])
    block_centers = np.arange(n_blocks) * BLOCK_LEN + BLOCK_LEN / 2.0
    ax2.bar(block_centers, label_pred - 0.5,
            width=BLOCK_LEN * 0.85, bottom=0.5,
            color=["#d62728" if lp >= 0.5 else "#1f77b4" for lp in label_pred],
            edgecolor="black", linewidth=0.5)
    for chunk_i in range(n_blocks):
        x = block_centers[chunk_i]
        tgt = "a" if label_targets[chunk_i] == 1 else "x"
        ax2.text(x, 1.06, "target=" + tgt, ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 color="#d62728" if label_targets[chunk_i] == 1 else "#1f77b4")
    ax2.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlim(0, T_total)
    ax2.set_ylim(-0.05, 1.20)
    ax2.set_ylabel("C's P(label='a')")
    ax2.set_title("Chunker label readout per block "
                  "(red bar = predict 'a', blue bar = predict 'x')")
    ax2.grid(True, alpha=0.3)

    # 4th: cumulative-correct strip
    ax3 = fig.add_subplot(gs[3])
    correct = ((label_pred > 0.5).astype(int) == label_targets.astype(int))
    cum = np.cumsum(correct) / np.arange(1, n_blocks + 1)
    ax3.step(np.arange(1, n_blocks + 1), cum * 100, where="mid",
             color="#2ca02c", linewidth=1.5)
    ax3.set_xlim(0.5, n_blocks + 0.5)
    ax3.set_ylim(0, 105)
    ax3.set_xlabel("block index")
    ax3.set_ylabel("cumulative label acc (%)")
    ax3.set_title(f"Cumulative label accuracy over {n_blocks} test blocks "
                  f"({int(correct.sum())}/{n_blocks} correct)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training A-alone (seed={args.seed}, blocks={args.blocks})...")
    A_only, _, hist_alone = train(seed=args.seed, n_blocks=args.blocks,
                                  mode="a_alone", surprise_threshold=args.threshold,
                                  verbose=False)
    print(f"  A-alone final label_acc={hist_alone['label_acc'][-1]*100:.1f}%")

    print(f"Training Chunker (seed={args.seed}, blocks={args.blocks})...")
    A, C, hist_chunker = train(seed=args.seed, n_blocks=args.blocks,
                               mode="chunker",
                               surprise_threshold=args.threshold,
                               verbose=False)
    print(f"  Chunker final label_acc={hist_chunker['label_acc'][-1]*100:.1f}%")

    plot_training_curves(hist_alone, hist_chunker,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_surprise_pattern(args.seed, args.blocks, args.threshold,
                          os.path.join(args.outdir, "surprise_pattern.png"))
    plot_network_weights(A, C, os.path.join(args.outdir, "network_weights.png"))
    plot_test_episode(A, C, os.path.join(args.outdir, "test_episode.png"),
                      seed=12345, n_blocks=8, threshold=args.threshold)


if __name__ == "__main__":
    main()
