"""Render clockwork_rnn.gif: a frame-by-frame animation of the CW-RNN
*learning* the multi-rate waveform.

Each frame corresponds to a training-progress checkpoint. The frame
shows three panels:

    (top)    target waveform vs current CW-RNN output and current
             vanilla-RNN output.
    (middle) per-group active-step heatmap with a vertical cursor at
             the current timestep.
    (bottom) per-group hidden-state mean over time, colour-coded by
             clock period.

The clockwork structure is highlighted: slow groups update only at
sparse boundaries, fast groups update every step.
"""

from __future__ import annotations

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from clockwork_rnn import (
    ClockworkRNN,
    VanillaRNN,
    fixed_target,
    memorisation_inputs,
    train_memorise,
    vanilla_hidden_dim_to_match,
)


def imageio_or_pillow():
    try:
        import imageio.v2 as imageio
        return ("imageio", imageio)
    except Exception:
        from PIL import Image
        return ("pillow", Image)


def render_frame(target, cw_out, vn_out, active, group_means, periods,
                 epoch, total_epochs):
    G, T = active.shape
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.6], hspace=0.45)

    # --- Top: target vs predictions
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(target, color="black", linewidth=1.5, label="target")
    ax0.plot(cw_out, color="C0", linewidth=1.2, label="CW-RNN")
    ax0.plot(vn_out, color="C3", linewidth=1.2, alpha=0.85,
             label="vanilla (matched)")
    ax0.set_xlim(0, T)
    ax0.set_ylim(-1.6, 1.6)
    ax0.set_ylabel("y(t)")
    ax0.set_title(f"Memorising a multi-rate waveform — epoch {epoch}/{total_epochs}")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, linestyle=":", alpha=0.4)

    # --- Middle: clock schedule heatmap
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(active.astype(float), aspect="auto", cmap="Greys",
               interpolation="nearest", vmin=0, vmax=1)
    ax1.set_yticks(range(G))
    ax1.set_yticklabels([f"T={periods[g]}" for g in range(G)], fontsize=7)
    ax1.set_xlabel("t")
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_title("active-step schedule  (black = group updates)")

    # --- Bottom: per-group activations
    ax2 = fig.add_subplot(gs[2])
    for g in range(G):
        ax2.plot(group_means[:, g], color=plt.cm.viridis(g / max(1, G - 1)),
                 label=f"T={periods[g]}", linewidth=1.0)
    ax2.set_xlim(0, T)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel("t")
    ax2.set_ylabel("group hidden mean")
    ax2.set_title("per-group hidden activations  (slow → blue, fast → yellow)")
    ax2.legend(fontsize=7, ncol=4, loc="lower center")
    ax2.grid(True, linestyle=":", alpha=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--T", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--out", type=str, default="clockwork_rnn.gif")
    args = parser.parse_args()

    periods = tuple(2 ** g for g in range(args.groups))
    signal_periods = (8, 32, 80, 160)
    target = fixed_target(args.T, signal_periods, seed=args.seed)
    X = memorisation_inputs(args.T)

    cw = ClockworkRNN(in_dim=1, hidden_dim=args.hidden, out_dim=1,
                      n_groups=args.groups, periods=periods, seed=args.seed)
    nv = vanilla_hidden_dim_to_match(cw)
    vanilla = VanillaRNN(in_dim=1, hidden_dim=nv, out_dim=1, seed=args.seed + 1)
    active = cw.active_groups(args.T).T  # (G, T)

    # Logarithmic checkpoint schedule so early dynamics are visible.
    if args.n_frames < 2:
        checkpoints = [args.epochs]
    else:
        checkpoints = sorted(set(
            int(round(c)) for c in np.geomspace(1, args.epochs, args.n_frames)
        ))
        checkpoints[0] = max(0, checkpoints[0])

    frames = []
    last = 0
    for cp in checkpoints:
        delta = max(1, cp - last)
        if delta > 0 and last < args.epochs:
            train_memorise(cw, target, n_epochs=delta, lr=args.lr)
            train_memorise(vanilla, target, n_epochs=delta, lr=args.lr)
            last = cp
        cw_out, cache = cw.forward(X)
        vn_out, _ = vanilla.forward(X)
        H = cache["h"][1:]  # (T, N)
        # Per-group mean over the block, shape (T, G).
        group_means = np.stack(
            [H[:, g * cw.M:(g + 1) * cw.M].mean(axis=1) for g in range(args.groups)],
            axis=1,
        )
        frames.append(render_frame(
            target, cw_out[:, 0], vn_out[:, 0], active, group_means,
            cw.periods, last, args.epochs,
        ))

    backend, lib = imageio_or_pillow()
    if backend == "imageio":
        images = [lib.imread(f) for f in frames]
        lib.mimsave(args.out, images, duration=0.45, loop=0)
    else:
        Image = lib
        pil_frames = [Image.open(f).convert("RGBA") for f in frames]
        pil_frames[0].save(args.out, save_all=True, append_images=pil_frames[1:],
                           duration=450, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
