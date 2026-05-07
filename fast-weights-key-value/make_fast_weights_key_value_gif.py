"""Render fast_weights_key_value.gif: training animation showing how the
slow projector W_K transforms raw biased keys into a near-orthogonal
address space and unlocks correct retrieval.

Frames are sampled at log-spaced training steps. Each frame shows:

    (left)   cos(W_K k_i, W_K k_j) for a fixed test episode -- starts off-
             diagonal heavy because of the shared bias direction, becomes
             diagonal as W_K projects the bias out.
    (right)  retrieved value y vs target v_q on the same fixed episode.
"""

from __future__ import annotations

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from fast_weights_key_value import (
    fast_weight_backward,
    fast_weight_forward,
    generate_episode,
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


def cos_norm(a, b):
    n = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--d-key", type=int, default=8)
    parser.add_argument("--d-val", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=str, default="fast_weights_key_value.gif")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Fixed test episode (same one every frame).
    test_rng = np.random.default_rng(args.seed + 12345 + 7)
    test_keys, test_values, test_q_idx = generate_episode(
        test_rng, args.n_pairs, args.d_key, args.d_val
    )
    test_q_key = test_keys[test_q_idx]
    test_target_v = test_values[test_q_idx]

    # Snapshot schedule: log-spaced steps (visible at the start, sparser later).
    snap_steps = sorted(set(
        [0]
        + list(np.unique(np.round(
            np.logspace(0, np.log10(args.n_steps - 1), args.max_frames - 1)
        ).astype(int)))
        + [args.n_steps - 1]
    ))

    # Init W_K close to identity (matches train()'s init).
    W_K = np.eye(args.d_key) + 0.05 * rng.standard_normal((args.d_key, args.d_key))
    snapshots = {}  # step -> W_K copy

    for step in range(args.n_steps):
        keys, values, q_idx = generate_episode(rng, args.n_pairs, args.d_key, args.d_val)
        q_key = keys[q_idx]
        target_v = values[q_idx]
        y, W_fast, K, k_q = fast_weight_forward(W_K, keys, values, q_key)
        dW_K = fast_weight_backward(W_K, keys, values, q_key, target_v,
                                    y, W_fast, K, k_q)
        gnorm = float(np.linalg.norm(dW_K))
        if gnorm > 1.0:
            dW_K = dW_K / gnorm
        W_K -= args.lr * dW_K
        if step in snap_steps:
            snapshots[step] = W_K.copy()

    name, lib = imageio_or_pillow()
    frames = []
    for step in snap_steps:
        Wk = snapshots[step]
        K = test_keys @ Wk.T
        norms = np.linalg.norm(K, axis=1, keepdims=True) + 1e-12
        Kn = K / norms
        cos_mat = Kn @ Kn.T
        y, *_ = fast_weight_forward(Wk, test_keys, test_values, test_q_key)
        cos_y = cos_norm(y, test_target_v)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        im = axes[0].imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title("cos(W_K k_i, W_K k_j)\n(fixed test episode)")
        axes[0].set_xlabel("key index")
        axes[0].set_ylabel("key index")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        x = np.arange(args.d_val)
        w = 0.4
        axes[1].bar(x - w / 2, test_target_v, width=w, label="target v_q",
                    color="#222222")
        axes[1].bar(x + w / 2, y, width=w, label="retrieved y",
                    color="#3366cc")
        axes[1].set_xlabel("value dim")
        axes[1].set_ylabel("activation")
        axes[1].set_title(f"step {step:>5d}    cos(y, v_q) = {cos_y:+.3f}")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].legend(loc="upper right", fontsize=9)
        # Lock y-limits to a sensible global range so bars don't scale-shift.
        ylim = max(1.5, float(np.max(np.abs(test_target_v))) * 1.5)
        axes[1].set_ylim(-ylim, ylim)

        fig.suptitle(
            f"fast-weights-key-value  |  N = {args.n_pairs}, d = {args.d_key}",
            fontsize=11
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    if name == "imageio":
        lib.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    else:
        imgs = [lib.fromarray(f) for f in frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out}  ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
