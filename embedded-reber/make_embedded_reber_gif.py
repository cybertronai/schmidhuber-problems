"""
Train an embedded-Reber LSTM while snapshotting weights, then render
embedded_reber.gif: each frame shows the model's next-symbol distribution
on a fixed test string at that point in training. The viewer watches the
distribution sharpen onto the legal continuations and -- critically --
onto the matching outer T/P at the second-to-last position.
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio

from embedded_reber import (
    ALPHABET, N_SYM, SYM2IDX,
    LSTM1997, gen_embedded_reber, make_io, legal_next, evaluate, train as full_train,
)


def render_frame(net: LSTM1997, sample: str, step: int, outer_acc: float):
    X, _ = make_io(sample)
    probs = net.predict(X)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(sample)), 4.0))
    im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(N_SYM))
    ax.set_yticklabels(ALPHABET)
    ax.set_xticks(range(len(sample) - 1))
    ax.set_xticklabels([f"{i}\n{sample[i]}" for i in range(len(sample) - 1)],
                       fontsize=8)
    ax.set_xlabel("step (input symbol shown below index)")
    ax.set_ylabel("predicted next symbol")
    ax.set_title(f"step {step:5d}    outer T/P acc = {outer_acc:.3f}\n"
                 f"string: {sample}", fontsize=10)
    # legal-symbol boxes
    for t in range(len(sample) - 1):
        for s in legal_next(sample, t):
            r = SYM2IDX[s]
            ax.add_patch(plt.Rectangle((t - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=1.0))
    # outer column
    t_outer = len(sample) - 3
    ax.add_patch(plt.Rectangle((t_outer - 0.5, -0.5), 1, N_SYM,
                               fill=False, edgecolor="yellow", lw=2.0))

    plt.colorbar(im, ax=ax, pad=0.01).set_label("p(next = sym)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=8000)
    ap.add_argument("--snapshot-every", type=int, default=200)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default="embedded_reber.gif")
    args = ap.parse_args()

    print("training with snapshots...")
    out = full_train(
        seed=args.seed,
        max_seqs=args.max_seqs,
        eval_every=200,
        eval_n=120,
        verbose=False,
        snapshot_every=args.snapshot_every,
        snapshot_dir=".",   # any non-empty value enables snapshots
    )
    snaps = out["snapshots"]
    if not snaps:
        # if training stopped before any snapshot was captured, fall back
        # to an additional one of the final state.
        snaps = [{"step": out["total_sequences"], "params":
                  [p.copy() for p in out["net"].params()]}]

    # Always include a final-state snapshot.
    final_step = out["total_sequences"]
    snaps.append({"step": final_step, "params":
                  [p.copy() for p in out["net"].params()]})

    fixed_rng = np.random.default_rng(2025)
    sample = gen_embedded_reber(fixed_rng)
    print(f"  fixed test string: {sample}")

    # We need to be able to run forward at each snapshot. Make a temp net
    # we mutate by copying parameters in.
    temp = LSTM1997(rng=np.random.default_rng(0))

    # Cap to ~50 frames so the GIF stays under 2 MB.
    max_frames = 50
    if len(snaps) > max_frames:
        idxs = np.linspace(0, len(snaps) - 1, max_frames).astype(int)
        snaps = [snaps[i] for i in idxs]

    frames = []
    for snap in snaps:
        temp.set_params([p.copy() for p in snap["params"]])
        _, outer = evaluate(temp, n=120, rng=np.random.default_rng(99999))
        frames.append(render_frame(temp, sample, snap["step"], outer))

    # Hold the final frame for an extra second.
    tail = [frames[-1]] * args.fps
    imageio.mimsave(args.out, frames + tail, fps=args.fps)
    sz = os.path.getsize(args.out) / 1024.0
    print(f"  wrote {args.out}  ({len(frames)} frames, {sz:.0f} KB)")


if __name__ == "__main__":
    main()
