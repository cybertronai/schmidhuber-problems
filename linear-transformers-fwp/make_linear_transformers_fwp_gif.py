"""Render linear_transformers_fwp.gif: a frame-by-frame animation of the
2021 equivalence on a *fixed* test episode.

Each frame adds one more (k_t, v_t) pair to the stored memory and shows:

    (left)   Schedule A bar chart of inner products <k_t, q> (linear-attention).
    (middle) Schedule B running W_fast = sum_t v_t k_t^T (1992 FWP).
    (right)  Retrieved y so far, target v_q, and max|y_attn - y_fwp|.

The same numpy code path is taken in the title to make the equivalence
visually unambiguous: the bar chart's value-weighted sum *is* the matrix-
vector multiply.
"""

from __future__ import annotations

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from linear_transformers_fwp import (
    fwp_outer_product_write,
    fwp_read,
    generate_episode,
    linear_attention,
    train,
)


def imageio_or_pillow():
    try:
        import imageio.v2 as imageio
        return ("imageio", imageio)
    except Exception:
        from PIL import Image
        return ("pillow", Image)


def fig_to_rgb(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    try:
        import imageio.v2 as imageio
        return imageio.imread(buf)
    except Exception:
        from PIL import Image
        img = Image.open(buf).convert("RGB")
        return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--d-key", type=int, default=8)
    parser.add_argument("--d-val", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--out", type=str, default="linear_transformers_fwp.gif")
    args = parser.parse_args()

    # Train a slow projector so the demo is readable (otherwise biased keys
    # dominate W_fast). Use the same recipe as the headline numbers.
    W_K, _ = train(seed=args.seed, n_pairs=5, d_key=args.d_key,
                   d_val=args.d_val, n_steps=args.n_steps, lr=args.lr)

    # Fixed test episode with `n_pairs` slots; reveal one slot per frame.
    rng = np.random.default_rng(args.seed + 12345 + 7)
    keys, values, q_idx = generate_episode(rng, args.n_pairs, args.d_key, args.d_val)
    q_key = keys[q_idx]
    target_v = values[q_idx]
    K_all = keys @ W_K.T
    k_q = W_K @ q_key
    v_target = values[q_idx]

    name, lib = imageio_or_pillow()
    frames = []
    # Pre-compute once so y-axes stay stable across frames.
    W_full = fwp_outer_product_write(K_all, values)
    vmax_W = float(np.max(np.abs(W_full))) * 1.05
    score_lim = float(np.max(np.abs(K_all @ k_q))) * 1.2 + 1e-6
    val_lim = float(np.max(np.abs(np.concatenate([target_v, W_full @ k_q])))) * 1.2

    # Frame 0: empty memory. Then write one new pair per frame.
    for t in range(args.n_pairs + 1):
        K_t = K_all[:t]
        V_t = values[:t]
        scores = K_t @ k_q if t > 0 else np.zeros(0)
        # Schedule A: linear attention re-fetches every key
        y_attn = linear_attention(K_t, V_t, k_q) if t > 0 else np.zeros(args.d_val)
        # Schedule B: 1992 FWP build matrix and read once
        W_fast = fwp_outer_product_write(K_t, V_t) if t > 0 else np.zeros((args.d_val, args.d_key))
        y_fwp = fwp_read(W_fast, k_q)
        diff = float(np.max(np.abs(y_attn - y_fwp))) if t > 0 else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                                 gridspec_kw={"width_ratios": [1.1, 1.1, 1.1]})

        # left: linear-attention scores per stored pair
        bar_x = np.arange(args.n_pairs)
        bar_y = np.full(args.n_pairs, np.nan)
        bar_y[:t] = scores
        # Plot only filled-in ones; use NaN for unwritten slots.
        axes[0].bar(bar_x[:t], bar_y[:t], color="#3366cc")
        # Mark query slot with hatched outline
        if q_idx < args.n_pairs:
            axes[0].axvline(q_idx, color="green", lw=1, ls="--",
                            alpha=0.7, label=f"query = pair {q_idx}")
            axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_ylim(-score_lim, score_lim)
        axes[0].set_xticks(bar_x)
        axes[0].set_xlabel("stored pair index t")
        axes[0].set_ylabel(r"$\langle W_K k_t,\ W_K q\rangle$")
        axes[0].set_title(f"Schedule A: linear attention\n"
                          rf"$y = \sum_{{t=1}}^{{{t}}} v_t \langle k_t, q\rangle$",
                          fontsize=10)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].grid(True, alpha=0.3, axis="y")

        # middle: 1992 FWP scratchpad after t writes
        im = axes[1].imshow(W_fast, cmap="RdBu_r", vmin=-vmax_W, vmax=vmax_W)
        axes[1].set_title(rf"Schedule B: $W_\mathrm{{fast}} = \sum_{{t=1}}^{{{t}}} v_t k_t^\top$"
                          "\n(1992 FWP scratchpad)",
                          fontsize=10)
        axes[1].set_xlabel("key dim")
        axes[1].set_ylabel("value dim")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # right: retrieved y vs target
        x = np.arange(args.d_val)
        w = 0.32
        axes[2].bar(x - w, v_target, width=w, label=r"target $v_q$", color="#222222")
        axes[2].bar(x, y_attn, width=w, label=r"$y$ via A", color="#3366cc", alpha=0.85)
        axes[2].bar(x + w, y_fwp, width=w, label=r"$y$ via B", color="#cc6633", alpha=0.85)
        axes[2].set_ylim(-val_lim, val_lim)
        axes[2].set_xlabel("value dim")
        axes[2].set_ylabel("activation")
        axes[2].set_title(f"after {t} writes  |  max |A - B| = {diff:.1e}",
                          fontsize=10)
        axes[2].legend(fontsize=8, loc="upper right")
        axes[2].grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "linear-transformers-fwp  |  Schlag, Irie, Schmidhuber 2021:  "
            r"$V^\top(Kq) \equiv (V^\top K)\,q$",
            fontsize=11
        )
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    # Pad: hold the last frame for emphasis.
    for _ in range(3):
        frames.append(frames[-1])

    if name == "imageio":
        lib.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    else:
        imgs = [lib.fromarray(f) for f in frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out}  ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
