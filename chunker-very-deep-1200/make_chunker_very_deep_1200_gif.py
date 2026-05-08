"""
Animated GIF for chunker-very-deep-1200.

Visualises the credit-assignment story.  Each frame shows the same fresh
test sequence with:

    Top   : the input symbol stream colour-coded by class (trigger / filler /
            target).
    Middle: per-step gradient norm  ||d L_terminal / d h_t||  flowing
            backward in time, frame by frame, for the single-network baseline.
            The frame index is the reverse-time distance the gradient has
            travelled so far.  After ~5 steps the curve has vanished into the
            log-floor.
    Bottom: the chunker's view -- the same sequence collapsed to just the
            surprise tokens (trigger and target).  The chunker only ever
            backprops through these k = 2 steps, so its credit-assignment
            channel is "always full".

Frames step from t = T - 1 (the terminal step) down to t = 0; the title
counts the layers crossed and reports the effective depth ratio.

Usage
-----

    python3 make_chunker_very_deep_1200_gif.py --seed 0 --T 1200
    python3 make_chunker_very_deep_1200_gif.py --seed 0 --T 500 --max-frames 60
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from chunker_very_deep_1200 import (
    NUM_SYMBOLS, RNN, SYMBOL_NAMES, FILLER_BASE, TRIG_A, TRIG_B,
    make_sequence, train_automatizer, train_baseline, detect_surprises,
)


def category(idx: int):
    if idx in (TRIG_A, TRIG_B):
        return "trigger"
    return "filler"


def render_frame(x_idx, surprise_mask, grad_norms, t_cursor, T, ratio):
    """Render one frame.  Returns a PIL Image."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.0), dpi=110,
                             gridspec_kw={"height_ratios": [1.1, 2.2, 1.1]})

    # --- Top: input stream
    ax = axes[0]
    Tlen = len(x_idx)
    pos = np.arange(Tlen)
    # Colour by category; dim everything to the right of the cursor too.
    is_trig = np.isin(x_idx, [TRIG_A, TRIG_B])
    ax.scatter(pos[is_trig], x_idx[is_trig], color="#d62728", s=18,
               label="trigger / target")
    ax.scatter(pos[~is_trig], x_idx[~is_trig], color="#7f7f7f", s=4,
               alpha=0.6, label="filler")
    surp_idx = np.where(surprise_mask)[0]
    for i in surp_idx:
        ax.axvline(i, color="#d62728", linewidth=0.4, alpha=0.5)
    ax.axvline(t_cursor, color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.set_yticks(range(NUM_SYMBOLS))
    ax.set_yticklabels(SYMBOL_NAMES, fontsize=8)
    ax.set_xlim(-2, Tlen + 2)
    ax.set_title(f"Input sequence (T = {T});  "
                 f"recall-target loss back-propagating from step T - 1")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- Middle: gradient norm (log) backward through time, revealed up to cursor.
    ax = axes[1]
    base = max(grad_norms[-1], 1e-12)
    g_normed = grad_norms / base
    revealed = np.full_like(g_normed, np.nan, dtype=float)
    revealed[t_cursor:] = g_normed[t_cursor:]
    ax.semilogy(pos, g_normed, color="#cccccc", linewidth=0.8,
                label="full curve (faint)")
    ax.semilogy(pos, revealed, color="#1f77b4", linewidth=1.6,
                label="gradient that has propagated so far")
    ax.axhline(0.01, color="#d62728", linewidth=0.6, linestyle="--",
               label="1% of terminal (vanishing cutoff)")
    ax.axvline(t_cursor, color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.set_ylim(1e-25, 5)
    ax.set_xlim(-2, Tlen + 2)
    ax.set_ylabel(r"$\|\partial L_{\mathrm{terminal}} / \partial h_t\|$  "
                  "(normalised)")
    layers_crossed = (Tlen - 1) - t_cursor
    ax.set_title(
        f"Baseline RNN: gradient has crossed {layers_crossed} virtual layers "
        f"of {Tlen - 1};  depth-reduction ratio {ratio:.1f}x"
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # --- Bottom: chunker compressed view
    ax = axes[2]
    ax.scatter(np.where(surprise_mask)[0],
               x_idx[surprise_mask],
               color="#2ca02c", s=70, edgecolor="black", zorder=3,
               label="surprises (chunker input)")
    ax.set_yticks(range(NUM_SYMBOLS))
    ax.set_yticklabels(SYMBOL_NAMES, fontsize=8)
    ax.set_xlim(-2, Tlen + 2)
    ax.set_xlabel("sequence position t")
    n_surp = int(surprise_mask.sum())
    ax.set_title(
        f"Chunker view: the same sequence compressed to {n_surp} events;  "
        f"BPTT only spans those {n_surp} steps"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--auto-epochs", type=int, default=80)
    p.add_argument("--baseline-epochs", type=int, default=10)
    p.add_argument("--out", type=str, default="chunker_very_deep_1200.gif")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)

    rng = np.random.default_rng(args.seed)
    print("[gif] training automatizer (level 0)...")
    A, _ = train_automatizer(args.T, rng, hidden=16,
                             epochs=args.auto_epochs, lr=0.05,
                             truncate=6, verbose=False)
    # Threshold: midpoint between filler and trigger/target loss
    probe_filler, probe_surprise = [], []
    for _ in range(8):
        x, y, _ = make_sequence(args.T, rng)
        ls = A.per_step_loss(x, y)
        probe_surprise.extend([ls[0], ls[-1]])
        probe_filler.extend(ls[1:-1].tolist())
    threshold = 0.5 * (np.median(probe_filler) + np.median(probe_surprise))

    print("[gif] training baseline RNN (full BPTT)...")
    R, _ = train_baseline(rng, args.T, hidden=16,
                          epochs=args.baseline_epochs, lr=0.05,
                          verbose=False)

    # Pick one fresh probe sequence and use it for the whole animation.
    x, y, trig = make_sequence(args.T, rng)
    mask, _ = detect_surprises(A, x, y, threshold)
    grad_norms = R.terminal_target_grad_norms(x, y)

    # Effective depths
    base = max(grad_norms[-1], 1e-12)
    eff_baseline = 0
    for t in range(len(grad_norms) - 1, -1, -1):
        if grad_norms[t] < 0.01 * base:
            break
        eff_baseline = (len(grad_norms) - 1) - t
    n_surp = max(2, int(mask.sum()))
    ratio = (args.T - 1) / n_surp

    # Frame indices: cursor steps from T-1 down to 0; sample max_frames evenly.
    Tlen = len(x)
    cursor_positions = np.linspace(Tlen - 1, 0, args.max_frames).astype(int)

    print(f"[gif] rendering {len(cursor_positions)} frames "
          f"(T={args.T}, eff_baseline={eff_baseline}, k_chunker={n_surp})")
    frames = []
    for i, t_cur in enumerate(cursor_positions):
        frame = render_frame(x, mask, grad_norms, int(t_cur),
                             args.T, ratio)
        frames.append(frame)

    # Hold the final frame for emphasis
    for _ in range(args.fps):
        frames.append(frames[-1])

    duration_ms = int(1000 / args.fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    sz = os.path.getsize(out_path) / 1024
    print(f"[gif] wrote {out_path}  ({sz:.0f} KB)")


if __name__ == "__main__":
    main()
