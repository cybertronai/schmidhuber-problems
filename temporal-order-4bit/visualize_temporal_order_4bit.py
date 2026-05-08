"""Static visualisations for temporal-order-4bit.

Reads `results.json` (training log) and `snapshots.npz` (hidden-state trace
captured during training) and writes PNGs into `viz/`.

Run after `temporal_order_4bit.py --record_hidden`.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ("XXX", "XXY", "XYX", "XYY", "YXX", "YXY", "YYX", "YYY")
SYMBOLS = ["a", "b", "c", "d", "X", "Y", "B", "E"]
N_CLASSES = 8


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_curves(results: dict, outdir: str) -> str:
    steps = np.asarray(results["steps"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, results["lstm_loss"], label="LSTM", color="C0")
    axes[0].plot(steps, results["rnn_loss"], label="RNN", color="C1")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("training loss (cross-entropy)")
    axes[0].set_title("Training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, results["lstm_acc"], label="LSTM", color="C0")
    axes[1].plot(steps, results["rnn_acc"], label="RNN", color="C1")
    axes[1].axhline(1.0 / N_CLASSES, color="grey", linestyle=":", label="chance (1/8)")
    axes[1].axhline(1.0, color="grey", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title(
        f"8-class temporal-order accuracy "
        f"(LSTM final {results['lstm_final_acc']:.3f}, "
        f"RNN final {results['rnn_final_acc']:.3f})"
    )
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(results: dict, outdir: str) -> str:
    cm = np.asarray(results["confusion_lstm"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("LSTM confusion matrix on validation set (8 classes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_example_sequences(snap: np.lib.npyio.NpzFile, outdir: str) -> str:
    record_X = snap["record_X"]
    record_y = snap["record_y"]
    n = record_X.shape[0]
    T = record_X.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.4 * n))
    if n == 1:
        axes = [axes]
    for ax, seq, lbl in zip(axes, record_X, record_y):
        ax.imshow(seq.T, aspect="auto", cmap="Greys", origin="lower")
        ax.set_yticks(range(8))
        ax.set_yticklabels(SYMBOLS)
        ax.set_xlabel("time step")
        ax.set_title(f"sample sequence — class {CLASS_NAMES[int(lbl)]}")
        for t in range(T):
            tok = np.argmax(seq[t])
            if tok == 4:
                ax.axvline(t, color="C3", alpha=0.5, linewidth=0.7)
            if tok == 5:
                ax.axvline(t, color="C0", alpha=0.5, linewidth=0.7)
    fig.tight_layout()
    path = os.path.join(outdir, "example_sequences.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_hidden_trajectories(snap: np.lib.npyio.NpzFile, outdir: str) -> str:
    """For the FINAL snapshot, plot c_t and h_t through time on the recorded examples."""
    c = snap["c"][-1]  # (n_examples, T, H)
    h = snap["h"][-1]
    record_y = snap["record_y"]
    n_ex, T, H = c.shape
    # 2 rows, n_ex cols
    fig, axes = plt.subplots(2, n_ex, figsize=(2.2 * n_ex, 5.5), sharex=True)
    if n_ex == 1:
        axes = axes.reshape(2, 1)
    for j in range(n_ex):
        for k in range(H):
            axes[0, j].plot(range(T), c[j, :, k], label=f"cell {k}", linewidth=0.9)
            axes[1, j].plot(range(T), h[j, :, k], label=f"h {k}", linewidth=0.9)
        axes[0, j].set_title(f"{CLASS_NAMES[int(record_y[j])]}", fontsize=10)
        axes[0, j].set_ylabel("c_t" if j == 0 else "")
        axes[1, j].set_ylabel("h_t" if j == 0 else "")
        axes[1, j].set_xlabel("t")
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Cell state c_t (top) and hidden h_t (bottom) — trained LSTM, all 8 classes")
    fig.tight_layout()
    path = os.path.join(outdir, "hidden_trajectories.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_input_gate_activity(snap: np.lib.npyio.NpzFile, outdir: str) -> str:
    """Show that the input gate fires only on the X/Y positions."""
    i_seq = snap["i"][-1]  # (n_examples, T, H)
    record_X = snap["record_X"]
    record_y = snap["record_y"]
    n_ex, T, H = i_seq.shape
    fig, axes = plt.subplots(n_ex, 1, figsize=(11, 1.3 * n_ex), sharex=True)
    if n_ex == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        gate_max = i_seq[j].max(axis=-1)
        ax.bar(range(T), gate_max, color="C0", alpha=0.7)
        seq = record_X[j]
        for t in range(T):
            tok = np.argmax(seq[t])
            if tok == 4:
                ax.axvline(t, color="C3", alpha=0.7, linewidth=1.5)
            if tok == 5:
                ax.axvline(t, color="C2", alpha=0.7, linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("max i_t")
        ax.set_title(f"class {CLASS_NAMES[int(record_y[j])]}", fontsize=9)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Input gate selects the three X/Y positions and ignores distractors")
    fig.tight_layout()
    path = os.path.join(outdir, "input_gate_activity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cell_state_heatmap(snap: np.lib.npyio.NpzFile, outdir: str) -> str:
    """Heatmap of the cell-state matrix at the final snapshot."""
    c = snap["c"][-1]  # (n, T, H)
    n_ex, T, H = c.shape
    fig, axes = plt.subplots(2, n_ex // 2, figsize=(2.2 * (n_ex // 2), 5.5),
                              sharey=True)
    axes = axes.reshape(-1)
    cmax = float(np.abs(c).max())
    for j, ax in enumerate(axes):
        im = ax.imshow(c[j].T, aspect="auto", cmap="RdBu_r",
                       vmin=-cmax, vmax=cmax)
        ax.set_xlabel("t")
        ax.set_title(f"{CLASS_NAMES[int(snap['record_y'][j])]}", fontsize=10)
        if j % (n_ex // 2) == 0:
            ax.set_ylabel("cell index")
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04, label="c_t")
    fig.suptitle("Cell state heatmap at end of training (all 8 classes)")
    path = os.path.join(outdir, "cell_state_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results.json")
    ap.add_argument("--snap", type=str, default="snapshots.npz")
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir)
    with open(args.results) as f:
        results = json.load(f)

    paths = []
    paths.append(plot_training_curves(results, outdir))
    paths.append(plot_confusion(results, outdir))

    if os.path.exists(args.snap):
        snap = np.load(args.snap)
        paths.append(plot_example_sequences(snap, outdir))
        paths.append(plot_hidden_trajectories(snap, outdir))
        paths.append(plot_input_gate_activity(snap, outdir))
        paths.append(plot_cell_state_heatmap(snap, outdir))
    else:
        print(f"warning: {args.snap} not found, skipping snapshot plots")

    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
