"""Static visualisations for fast-weights-key-value.

Outputs to viz/:
    training_curves.png   - per-step loss + cosine similarity
    capacity_curve.png    - retrieval cosine vs N (pre vs post training)
    W_K_heatmap.png       - learned slow-projector matrix
    W_fast_heatmap.png    - one episode's fast-weight scratchpad after training
    key_cosine_pre.png    - cosine matrix of projected keys, pre-training
    key_cosine_post.png   - cosine matrix of projected keys, post-training
    retrieval_bars.png    - bar chart: target value vs retrieved y, one episode
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from fast_weights_key_value import (
    capacity_sweep,
    fast_weight_forward,
    fixed_bias_direction,
    generate_episode,
    train,
)


def smooth(x: np.ndarray, k: int = 25) -> np.ndarray:
    if len(x) < k:
        return x
    pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k - k // 2 - 1, x[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def projected_key_cosine_matrix(W_K, keys):
    K = keys @ W_K.T
    norms = np.linalg.norm(K, axis=1, keepdims=True) + 1e-12
    Kn = K / norms
    return Kn @ Kn.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--d-key", type=int, default=8)
    parser.add_argument("--d-val", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Train
    W_K, history, _ = train(seed=args.seed, n_pairs=args.n_pairs,
                            d_key=args.d_key, d_val=args.d_val,
                            n_steps=args.n_steps, lr=args.lr)
    pre_W_K = np.eye(args.d_key)

    # ----- training curves -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.array(history["step"])
    loss = np.array(history["loss"])
    cos = np.array(history["cos"])
    axes[0].plot(steps, loss, color="#bbbbbb", lw=0.5, label="raw")
    axes[0].plot(steps, smooth(loss, 51), color="#cc3333", lw=1.5, label="smoothed (51)")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("0.5 ||y - target||²")
    axes[0].set_title("Episodic retrieval loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(steps, cos, color="#bbbbbb", lw=0.5, label="raw")
    axes[1].plot(steps, smooth(cos, 51), color="#3366cc", lw=1.5, label="smoothed (51)")
    axes[1].axhline(0.9, color="green", lw=1, ls="--", alpha=0.7, label="cos = 0.9")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("cosine(y, target)")
    axes[1].set_title("Retrieval cosine")
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=120)
    plt.close()

    # ----- capacity curve -----
    eval_seed = args.seed + 12345
    pre_cap = capacity_sweep(pre_W_K, eval_seed, args.d_key, args.d_val,
                             max_pairs=12, n_test=100)
    post_cap = capacity_sweep(W_K, eval_seed, args.d_key, args.d_val,
                              max_pairs=12, n_test=100)
    Ns = [r["n_pairs"] for r in pre_cap]
    pre_y = [r["mean_cos"] for r in pre_cap]
    post_y = [r["mean_cos"] for r in post_cap]
    plt.figure(figsize=(7, 4))
    plt.plot(Ns, pre_y, "o-", color="#bb5555", label="pre-training (W_K = I)")
    plt.plot(Ns, post_y, "o-", color="#3366cc", label="post-training (W_K learned)")
    plt.axvline(args.n_pairs, color="gray", ls=":", lw=1,
                label=f"trained at N = {args.n_pairs}")
    plt.axhline(0.9, color="green", lw=1, ls="--", alpha=0.5)
    plt.xlabel("N (stored key/value pairs in episode)")
    plt.ylabel("mean retrieval cosine (100 test episodes)")
    plt.title(f"Capacity curve (d_key = {args.d_key})")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "capacity_curve.png"), dpi=120)
    plt.close()

    # ----- W_K heatmap -----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax = max(np.max(np.abs(pre_W_K)), np.max(np.abs(W_K)))
    im0 = axes[0].imshow(pre_W_K, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("W_K pre-training (= I)")
    axes[0].set_xlabel("input key dim")
    axes[0].set_ylabel("projected key dim")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(W_K, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("W_K post-training (slow projector)")
    axes[1].set_xlabel("input key dim")
    axes[1].set_ylabel("projected key dim")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "W_K_heatmap.png"), dpi=120)
    plt.close()

    # ----- one fixed test episode: cosine matrix of projected keys + retrieval -----
    rng = np.random.default_rng(eval_seed + 7)
    keys, values, q_idx = generate_episode(rng, args.n_pairs, args.d_key, args.d_val)
    q_key = keys[q_idx]
    target_v = values[q_idx]

    # Pre
    cos_pre = projected_key_cosine_matrix(pre_W_K, keys)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cos_pre, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("pre-training: cos(W_K k_i, W_K k_j)")
    plt.xlabel("key index j")
    plt.ylabel("key index i")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "key_cosine_pre.png"), dpi=120)
    plt.close()

    # Post
    cos_post = projected_key_cosine_matrix(W_K, keys)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cos_post, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("post-training: cos(W_K k_i, W_K k_j)")
    plt.xlabel("key index j")
    plt.ylabel("key index i")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "key_cosine_post.png"), dpi=120)
    plt.close()

    # ----- W_fast heatmap (post-training, one episode) -----
    y_post, W_fast_post, _, _ = fast_weight_forward(W_K, keys, values, q_key)
    plt.figure(figsize=(5, 4))
    vmax_w = float(np.max(np.abs(W_fast_post)))
    plt.imshow(W_fast_post, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    plt.title(f"W_fast (one episode, post-training)\n"
              f"= sum_t v_t (W_K k_t)^T,  N = {args.n_pairs}")
    plt.xlabel("key dim")
    plt.ylabel("value dim")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "W_fast_heatmap.png"), dpi=120)
    plt.close()

    # ----- retrieval bar chart: target vs retrieved -----
    y_pre, _, _, _ = fast_weight_forward(pre_W_K, keys, values, q_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(args.d_val)
    w = 0.27
    ax.bar(x - w, target_v, width=w, label="target value v_q", color="#222222")
    ax.bar(x, y_pre, width=w, label="y (pre-training)", color="#bb5555")
    ax.bar(x + w, y_post, width=w, label="y (post-training)", color="#3366cc")
    ax.set_xlabel("value dimension")
    ax.set_ylabel("activation")
    ax.set_title(
        f"Retrieval on a single episode\n"
        f"pre cos = {float(np.dot(target_v, y_pre) / (np.linalg.norm(target_v) * np.linalg.norm(y_pre) + 1e-12)):.3f},  "
        f"post cos = {float(np.dot(target_v, y_post) / (np.linalg.norm(target_v) * np.linalg.norm(y_post) + 1e-12)):.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "retrieval_bars.png"), dpi=120)
    plt.close()

    # ----- bias direction visualization -----
    b = fixed_bias_direction(args.d_key)
    plt.figure(figsize=(5, 3))
    plt.bar(np.arange(args.d_key), b, color="#444444")
    plt.title(f"Fixed bias direction b (every raw key contains alpha * b)")
    plt.xlabel("key dimension")
    plt.ylabel("b component")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "bias_direction.png"), dpi=120)
    plt.close()

    print(f"Saved 8 PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
