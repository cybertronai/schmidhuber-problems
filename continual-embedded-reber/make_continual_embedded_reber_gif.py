"""
Train Forget-LSTM and No-forget-LSTM in parallel while snapshotting
weights, then render continual_embedded_reber.gif.

Each frame shows both networks' next-symbol distributions on the same
fixed continual stream of three embedded-Reber strings, side by side.
The viewer watches the forget LSTM lock onto the legal-symbol structure
and onto the matching outer T/P at every string, while the no-forget
LSTM stays diffuse in its outer-T/P columns.
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

from continual_embedded_reber import (
    ALPHABET, N_SYM, SYM2IDX,
    LSTMForget, LSTMNoForget,
    gen_continual_stream, make_io,
    train as full_train,
    _legal_next_in_substring,
)


def render_frame(temp_f, temp_n, stream, bounds, chunk, outer_f, outer_n):
    X, _ = make_io(stream)
    probs_f, _ = temp_f.predict(X)
    probs_n, _ = temp_n.predict(X)

    width = max(7.0, 0.32 * len(stream))
    fig, axes = plt.subplots(2, 1, figsize=(width, 5.4))
    for ax, probs, title, outer in [
        (axes[0], probs_f, f"Forget LSTM    outer T/P = {outer_f:.2f}",   outer_f),
        (axes[1], probs_n, f"No-forget LSTM outer T/P = {outer_n:.2f}",   outer_n),
    ]:
        im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(N_SYM))
        ax.set_yticklabels(ALPHABET)
        ax.set_xticks(range(len(stream) - 1))
        ax.set_xticklabels([stream[i] for i in range(len(stream) - 1)],
                           fontsize=7)
        for (start, end) in bounds:
            t_outer = end - 3
            if 0 <= t_outer < probs.shape[0]:
                ax.add_patch(plt.Rectangle((t_outer - 0.5, -0.5), 1, N_SYM,
                                           fill=False, edgecolor="yellow", lw=1.5))
            ax.axvline(end - 1.5, color="white", alpha=0.5, lw=0.7)
        for t in range(probs.shape[0]):
            for s in _legal_next_in_substring(stream, bounds, t):
                r = SYM2IDX[s]
                ax.add_patch(plt.Rectangle((t - 0.5, r - 0.5), 1, 1,
                                           fill=False, edgecolor="red", lw=0.6))
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("predicted next sym")
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])
    axes[1].set_xlabel("step (input symbol shown)")
    fig.suptitle(f"chunk {chunk}", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def evaluate_outer(net, n_strings: int, rng_seed: int) -> float:
    """Outer-T/P accuracy on a fresh continual stream."""
    rng = np.random.default_rng(rng_seed)
    from continual_embedded_reber import outer_acc_by_position
    stats = outer_acc_by_position(net, n_strings, rng)
    return float(stats["outer_hits"].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=2000)
    ap.add_argument("--chunk-strings", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--snapshot-every", type=int, default=80)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default="continual_embedded_reber.gif")
    args = ap.parse_args()

    print("training Forget-LSTM with snapshots...")
    out_f = full_train(
        LSTMForget,
        seed=args.seed,
        n_hidden=args.hidden,
        n_chunks=args.n_chunks,
        chunk_strings=args.chunk_strings,
        eval_every=200,
        eval_strings=40,
        verbose=False,
        snapshot_every=args.snapshot_every,
    )
    print(f"  forget   final outer={out_f['final_outer']:.3f}")

    print("training No-forget-LSTM with snapshots...")
    out_n = full_train(
        LSTMNoForget,
        seed=args.seed,
        n_hidden=args.hidden,
        n_chunks=args.n_chunks,
        chunk_strings=args.chunk_strings,
        eval_every=200,
        eval_strings=40,
        verbose=False,
        snapshot_every=args.snapshot_every,
    )
    print(f"  noforget final outer={out_n['final_outer']:.3f}")

    snaps_f = out_f["snapshots"]
    snaps_n = out_n["snapshots"]
    # match snapshots by chunk index
    common = sorted(set(s["chunk"] for s in snaps_f) & set(s["chunk"] for s in snaps_n))
    f_by_c = {s["chunk"]: s for s in snaps_f}
    n_by_c = {s["chunk"]: s for s in snaps_n}

    # cap to 40 frames so the GIF stays reasonable
    max_frames = 40
    if len(common) > max_frames:
        idxs = np.linspace(0, len(common) - 1, max_frames).astype(int)
        common = [common[i] for i in idxs]

    # fixed test stream -- 3 embedded Reber strings
    fixed_rng = np.random.default_rng(args.seed + 2025)
    stream, bounds = gen_continual_stream(fixed_rng, 3)
    print(f"  fixed test stream: {stream}")

    temp_f = LSTMForget(n_hidden=args.hidden, rng=np.random.default_rng(0))
    temp_n = LSTMNoForget(n_hidden=args.hidden, rng=np.random.default_rng(0))

    frames = []
    for c in common:
        temp_f.set_params([p.copy() for p in f_by_c[c]["params"]])
        temp_n.set_params([p.copy() for p in n_by_c[c]["params"]])
        outer_f = evaluate_outer(temp_f, 30, args.seed + 99999)
        outer_n = evaluate_outer(temp_n, 30, args.seed + 99999)
        frames.append(render_frame(temp_f, temp_n, stream, bounds, c, outer_f, outer_n))

    # final-state frame from the actual trained nets
    outer_f = evaluate_outer(out_f["net"], 30, args.seed + 99999)
    outer_n = evaluate_outer(out_n["net"], 30, args.seed + 99999)
    frames.append(render_frame(out_f["net"], out_n["net"], stream, bounds,
                               args.n_chunks, outer_f, outer_n))

    # hold final frame
    tail = [frames[-1]] * args.fps
    imageio.mimsave(args.out, frames + tail, fps=args.fps)
    sz = os.path.getsize(args.out) / 1024.0
    print(f"  wrote {args.out}  ({len(frames)} frames, {sz:.0f} KB)")


if __name__ == "__main__":
    main()
