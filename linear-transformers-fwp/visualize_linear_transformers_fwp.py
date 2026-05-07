"""Static visualisations for linear-transformers-fwp.

Outputs to viz/:
    equivalence_panel.png   - HEADLINE: same numpy code does both reads
    training_curves.png     - per-step loss + retrieval cosine
    capacity_curve.png      - sum rule (1992 FWP) vs delta rule (2021)
    W_K_heatmap.png         - learned slow projector
    W_fast_heatmap.png      - one episode's scratchpad after training
    key_cosine_pre.png      - cosine matrix of projected keys, pre-training
    key_cosine_post.png     - cosine matrix of projected keys, post-training
    retrieval_bars.png      - target value vs retrieved y, two schedules side by side
    schedule_diff_bar.png   - max abs diff between linear-attn and FWP reads
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from linear_transformers_fwp import (
    capacity_sweep_rules,
    delta_rule_write,
    equivalence_check,
    fwp_outer_product_write,
    fwp_read,
    generate_episode,
    linear_attention,
    linear_attention_via_fwp,
    train,
)


def smooth(x: np.ndarray, k: int = 51) -> np.ndarray:
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

    # Train.
    W_K, history = train(seed=args.seed, n_pairs=args.n_pairs,
                         d_key=args.d_key, d_val=args.d_val,
                         n_steps=args.n_steps, lr=args.lr)
    pre_W_K = np.eye(args.d_key)

    # ----- HEADLINE: equivalence panel -----
    # On the same fixed test episode, run BOTH schedules and show the
    # numbers are bit-identical. The visual is "same arrows, same answer".
    eval_seed = args.seed + 12345
    rng = np.random.default_rng(eval_seed + 7)
    keys, values, q_idx = generate_episode(rng, args.n_pairs, args.d_key, args.d_val)
    q_key = keys[q_idx]
    target_v = values[q_idx]
    K = keys @ W_K.T
    k_q = W_K @ q_key

    y_attn = linear_attention(K, values, k_q)        # schedule A
    y_fwp = linear_attention_via_fwp(K, values, k_q) # schedule B
    diff_panel = float(np.max(np.abs(y_attn - y_fwp)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})

    # left: schedule A (linear attention), values bar with scores annotated
    scores = K @ k_q
    axes[0].bar(np.arange(args.n_pairs), scores, color="#3366cc")
    for i, s in enumerate(scores):
        axes[0].text(i, s, f"{s:+.2f}", ha="center",
                     va="bottom" if s >= 0 else "top", fontsize=8)
    axes[0].set_title("Schedule A: linear attention\n"
                      r"$y = \sum_t v_t \langle k_t, q\rangle = V^\top (K q)$",
                      fontsize=10)
    axes[0].set_xlabel("stored pair index t")
    axes[0].set_ylabel(r"$\langle k_t, q \rangle$")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")

    # middle: schedule B (1992 FWP outer-product write -> single matvec read)
    W_fast = fwp_outer_product_write(K, values)
    vmax = float(np.max(np.abs(W_fast)))
    im = axes[1].imshow(W_fast, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Schedule B: 1992 FWP outer-product write\n"
                      r"$W_\mathrm{fast} = \sum_t v_t k_t^\top = V^\top K$,  read $y = W_\mathrm{fast} q$",
                      fontsize=10)
    axes[1].set_xlabel("key dim")
    axes[1].set_ylabel("value dim")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # right: y comparison
    x = np.arange(args.d_val)
    w = 0.32
    axes[2].bar(x - w, target_v, width=w, label=r"target $v_q$", color="#222222")
    axes[2].bar(x, y_attn, width=w, label=r"$y$ via schedule A",
                color="#3366cc", alpha=0.85)
    axes[2].bar(x + w, y_fwp, width=w, label=r"$y$ via schedule B",
                color="#cc6633", alpha=0.85)
    axes[2].set_title(
        f"Same answer (max |A - B| = {diff_panel:.1e})",
        fontsize=10)
    axes[2].set_xlabel("value dim")
    axes[2].set_ylabel("activation")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=8, loc="best")

    fig.suptitle("Linear-attention IS the 1992 fast-weight programmer  "
                 r"(Schlag, Irie, Schmidhuber 2021):  $V^\top (K q) \equiv (V^\top K) q$",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(os.path.join(args.outdir, "equivalence_panel.png"), dpi=120)
    plt.close()

    # ----- training curves -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.array(history["step"])
    loss = np.array(history["loss"])
    cos = np.array(history["cos"])
    axes[0].plot(steps, loss, color="#bbbbbb", lw=0.5, label="raw")
    axes[0].plot(steps, smooth(loss, 51), color="#cc3333", lw=1.5, label="smoothed (51)")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$0.5 \|y - v_q\|^2$")
    axes[0].set_title("Episodic retrieval loss (sum-rule write)")
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

    # ----- capacity curve: sum rule vs delta rule -----
    sweep = capacity_sweep_rules(W_K, eval_seed, args.d_key, args.d_val,
                                 max_pairs=16, n_test=100)
    Ns = [r["n_pairs"] for r in sweep]
    sum_y = [r["mean_cos_sum_rule"] for r in sweep]
    delta_y = [r["mean_cos_delta_rule"] for r in sweep]
    plt.figure(figsize=(7, 4))
    plt.plot(Ns, sum_y, "o-", color="#cc6633",
             label="sum-rule write (1992 FWP / linear attention)")
    plt.plot(Ns, delta_y, "s-", color="#3366cc",
             label="delta-rule write (Schlag et al. 2021)")
    plt.axvline(args.n_pairs, color="gray", ls=":", lw=1,
                label=f"trained at N = {args.n_pairs}")
    plt.axhline(0.9, color="green", lw=1, ls="--", alpha=0.5)
    plt.xlabel("N stored key/value pairs in episode")
    plt.ylabel("mean retrieval cosine (100 test episodes)")
    plt.title(f"Sum-rule vs delta-rule capacity (d_key = {args.d_key}, post-training W_K)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "capacity_curve.png"), dpi=120)
    plt.close()

    # ----- W_K heatmap -----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax_w = max(float(np.max(np.abs(pre_W_K))), float(np.max(np.abs(W_K))))
    im0 = axes[0].imshow(pre_W_K, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    axes[0].set_title("$W_K$ pre-training (= I)")
    axes[0].set_xlabel("input key dim")
    axes[0].set_ylabel("projected key dim")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(W_K, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    axes[1].set_title("$W_K$ post-training (slow projector)")
    axes[1].set_xlabel("input key dim")
    axes[1].set_ylabel("projected key dim")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "W_K_heatmap.png"), dpi=120)
    plt.close()

    # ----- W_fast heatmap (one episode, post-training, sum vs delta) -----
    W_sum = fwp_outer_product_write(K, values)
    W_delta = delta_rule_write(K, values)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax_wf = max(float(np.max(np.abs(W_sum))), float(np.max(np.abs(W_delta))))
    im0 = axes[0].imshow(W_sum, cmap="RdBu_r", vmin=-vmax_wf, vmax=vmax_wf)
    axes[0].set_title(r"$W_\mathrm{fast}$ sum rule = $\sum_t v_t k_t^\top$")
    axes[0].set_xlabel("key dim")
    axes[0].set_ylabel("value dim")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(W_delta, cmap="RdBu_r", vmin=-vmax_wf, vmax=vmax_wf)
    axes[1].set_title(r"$W_\mathrm{fast}$ delta rule (Schlag 2021)")
    axes[1].set_xlabel("key dim")
    axes[1].set_ylabel("value dim")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "W_fast_heatmap.png"), dpi=120)
    plt.close()

    # ----- projected-key cosine matrices -----
    cos_pre = projected_key_cosine_matrix(pre_W_K, keys)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cos_pre, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("pre-training: cos($W_K k_i$, $W_K k_j$)")
    plt.xlabel("key index j")
    plt.ylabel("key index i")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "key_cosine_pre.png"), dpi=120)
    plt.close()

    cos_post = projected_key_cosine_matrix(W_K, keys)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cos_post, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("post-training: cos($W_K k_i$, $W_K k_j$)")
    plt.xlabel("key index j")
    plt.ylabel("key index i")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "key_cosine_post.png"), dpi=120)
    plt.close()

    # ----- retrieval bar chart: target vs both schedules, post-training -----
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(args.d_val)
    w = 0.27
    ax.bar(x - w, target_v, width=w, label=r"target $v_q$", color="#222222")
    ax.bar(x, y_attn, width=w, label=r"linear attention $V^\top(Kq)$",
           color="#3366cc", alpha=0.85)
    ax.bar(x + w, y_fwp, width=w, label=r"FWP $(V^\top K)q$",
           color="#cc6633", alpha=0.85)
    cos_pre_y_attn = float(np.dot(target_v, linear_attention(keys, values, q_key))
                           / (np.linalg.norm(target_v)
                              * np.linalg.norm(linear_attention(keys, values, q_key))
                              + 1e-12))
    cos_post_y = float(np.dot(target_v, y_attn)
                       / (np.linalg.norm(target_v) * np.linalg.norm(y_attn) + 1e-12))
    ax.set_title(f"Retrieval, one episode  |  pre-train cos = {cos_pre_y_attn:.3f},  "
                 f"post-train cos = {cos_post_y:.3f}\n"
                 f"max |linear-attn - FWP| = {diff_panel:.1e}",
                 fontsize=10)
    ax.set_xlabel("value dim")
    ax.set_ylabel("activation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "retrieval_bars.png"), dpi=120)
    plt.close()

    # ----- schedule diff bar (random inputs) -----
    eq = equivalence_check(seed=args.seed, n_trials=20, d_key=16, d_val=16)
    plt.figure(figsize=(6, 3))
    plt.bar(["max abs diff", "mean abs diff", "machine epsilon (float64)"],
            [eq["max_diff"], eq["mean_diff"], np.finfo(np.float64).eps],
            color=["#3366cc", "#3366cc", "#bbbbbb"])
    plt.yscale("log")
    plt.ylabel("|y_linear_attention - y_FWP|")
    plt.title(f"20 random inputs, d_key=d_val=16, N drawn in [1, 32]\n"
              f"linear-attention and 1992-FWP outputs agree to round-off.",
              fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "schedule_diff_bar.png"), dpi=120)
    plt.close()

    print(f"Saved 9 PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
