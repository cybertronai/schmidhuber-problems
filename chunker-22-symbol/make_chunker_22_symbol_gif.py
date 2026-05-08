"""
Animated GIF showing the chunker learning to solve the 20-step lag.

Each frame is one snapshot during training.  At each snapshot we replay a
*fixed* test stream of 6 blocks and visualise:

    Top:    the stream (red = a, blue = x, grey = b1..b20)
    Mid:    A's predicted probability of the actual next symbol
            (drops to ~0.05 at every block boundary -> surprises fire there)
    Bottom: C's per-block label readout, with the target letter annotated

The headline number in the title is the cumulative label accuracy across
the 6 test blocks at this point in training.

Usage:
    python3 make_chunker_22_symbol_gif.py --seed 0
    python3 make_chunker_22_symbol_gif.py --seed 0 --max-frames 40 --fps 8
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from chunker_22_symbol import (
    train, RNN, gen_stream, rnn_forward, softmax, sigmoid,
    ALPHABET, A_IDX, X_IDX, BLOCK_LEN,
)


def replay(A: RNN, C: RNN, stream: np.ndarray, labels: list,
           n_blocks: int, threshold: float):
    """Run A through the test stream, track surprises and C's per-block
    label readout (with h_c reset each query, matching training protocol)."""
    eye = np.eye(ALPHABET)
    h_a = np.zeros(A.n_hidden)
    last_surprise_sym = None
    T_total = n_blocks * BLOCK_LEN

    p_actual = np.zeros(T_total)
    surprise_mark = np.zeros(T_total, dtype=bool)
    label_pred = np.zeros(n_blocks)

    for chunk_i in range(n_blocks):
        start = chunk_i * BLOCK_LEN
        in_idx = stream[start:start + BLOCK_LEN]
        target_idx = stream[start + 1:start + 1 + BLOCK_LEN]
        inputs = eye[in_idx]
        traj = rnn_forward(A, inputs, h_a)

        if last_surprise_sym is not None:
            c_in = eye[last_surprise_sym][None, :]
            traj_q = rnn_forward(C, c_in, np.zeros(C.n_hidden))
            label_pred[chunk_i] = float(sigmoid(traj_q["label_pre"][0]))
        else:
            label_pred[chunk_i] = 0.5

        for t in range(BLOCK_LEN):
            p_t = float(softmax(traj["sym_logits"][t])[int(target_idx[t])])
            p_actual[start + t] = p_t
            if p_t < threshold:
                surprise_mark[start + t] = True
                if int(target_idx[t]) in (A_IDX, X_IDX):
                    last_surprise_sym = int(target_idx[t])

        h_a = traj["h"][-1].copy()

    return p_actual, surprise_mark, label_pred


def render_frame(A: RNN, C: RNN, stream: np.ndarray, labels: list,
                 n_blocks: int, threshold: float,
                 step: int, max_step: int) -> Image.Image:
    p_actual, surprise_mark, label_pred = replay(A, C, stream, labels,
                                                  n_blocks, threshold)
    T_total = n_blocks * BLOCK_LEN
    correct = ((label_pred > 0.5).astype(int) == np.array(labels[:n_blocks]))
    label_acc = float(correct.mean())

    fig = plt.figure(figsize=(9.5, 5.5), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.3, 1.3])

    # Stream
    ax0 = fig.add_subplot(gs[0])
    for t in range(T_total):
        s = int(stream[t])
        if s == A_IDX:
            c = "#d62728"
        elif s == X_IDX:
            c = "#1f77b4"
        else:
            c = "#cccccc"
        ax0.add_patch(Rectangle((t, 0), 1, 1, facecolor=c,
                                edgecolor="black", linewidth=0.3))
    for chunk_i in range(n_blocks):
        ax0.axvline(chunk_i * BLOCK_LEN, color="black", linewidth=1.2)
    ax0.set_xlim(0, T_total)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title("Stream (red = a, blue = x, grey = b1..b20)", fontsize=9)

    # A's prob of actual next
    ax1 = fig.add_subplot(gs[1])
    t_axis = np.arange(T_total)
    ax1.plot(t_axis, p_actual, color="#1f77b4", linewidth=1.0)
    ax1.axhline(threshold, color="#d62728", linestyle=":",
                linewidth=1.0, label=f"surprise threshold={threshold}")
    surp = np.flatnonzero(surprise_mark)
    ax1.scatter(surp, p_actual[surp], s=22, color="#d62728",
                marker="x", label="surprise")
    for chunk_i in range(n_blocks):
        ax1.axvline(chunk_i * BLOCK_LEN, color="gray", linewidth=0.4,
                    linestyle="--")
    ax1.set_xlim(0, T_total)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("P(actual next)")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.set_title("A: predicted probability of actual next symbol",
                  fontsize=9)
    ax1.grid(True, alpha=0.3)

    # C's per-block label
    ax2 = fig.add_subplot(gs[2])
    block_centers = np.arange(n_blocks) * BLOCK_LEN + BLOCK_LEN / 2.0
    ax2.bar(block_centers, label_pred - 0.5,
            width=BLOCK_LEN * 0.85, bottom=0.5,
            color=["#d62728" if lp >= 0.5 else "#1f77b4" for lp in label_pred],
            edgecolor="black", linewidth=0.5)
    for chunk_i in range(n_blocks):
        x = block_centers[chunk_i]
        tgt = "a" if labels[chunk_i] == 1 else "x"
        ax2.text(x, 1.06, "target=" + tgt, ha="center", va="bottom",
                 fontsize=8, fontweight="bold",
                 color="#d62728" if labels[chunk_i] == 1 else "#1f77b4")
    ax2.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlim(0, T_total)
    ax2.set_ylim(-0.05, 1.22)
    ax2.set_ylabel("C's P(label='a')")
    ax2.set_xlabel("time step")
    ax2.set_title("Chunker per-block label readout "
                  "(red = predict 'a', blue = predict 'x')",
                  fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"chunker-22-symbol  --  outer block "
                 f"{step+1}/{max_step}   "
                 f"label acc on this fixed stream = {label_acc*100:.0f}%",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--snapshot-every", type=int, default=30)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--hold-final", type=int, default=12)
    p.add_argument("--out", type=str, default="chunker_22_symbol.gif")
    p.add_argument("--test-blocks", type=int, default=6)
    p.add_argument("--test-seed", type=int, default=12345)
    p.add_argument("--max-frames", type=int, default=50,
                   help="Hard cap on frames; reduces gif size.")
    args = p.parse_args()

    # Pre-generate the fixed test stream so every frame compares apples
    # to apples.
    rng_test = np.random.default_rng(args.test_seed)
    test_blocks, test_labels = gen_stream(args.test_blocks + 1, rng_test)
    test_stream = np.concatenate([np.array(b, dtype=np.int64)
                                  for b in test_blocks])

    frames = []

    def cb(step: int, A, C, history):
        # We only render the chunker run; A is partially trained, C is
        # partially trained.  At step=-1 (initial), C is None for a_alone
        # mode but here we always run mode='chunker' so C is defined.
        if C is None:
            return
        frame = render_frame(A, C, test_stream, test_labels,
                             args.test_blocks, args.threshold,
                             step if step >= 0 else 0, args.blocks)
        frames.append(frame)
        if step >= 0:
            print(f"  frame {len(frames):3d}  outer block {step+1:5d}")

    # Tune snapshot_every so we don't blow past max_frames
    eff_every = max(args.snapshot_every,
                    max(args.blocks // max(args.max_frames - 2, 1), 1))

    print(f"Training {args.blocks} blocks (chunker mode), "
          f"snapshot every {eff_every}...")
    A, C, history = train(
        seed=args.seed,
        n_blocks=args.blocks,
        mode="chunker",
        surprise_threshold=args.threshold,
        snapshot_every=eff_every,
        snapshot_callback=cb,
        verbose=False,
    )
    print(f"  final training label_acc={history['label_acc'][-1]*100:.1f}%")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
