"""Generate static figures for neural-data-router from run.json.

Produces, in viz/:
  learning_curves.png       — train loss + train/test acc over steps
  per_depth_final.png       — per-depth accuracy bar chart, NDR vs vanilla
  length_generalization.png — train/test acc vs depth, with train/test split
  geometric_attention.png   — geometric attention map for one NDR layer
  vanilla_attention.png     — softmax attention map for the same input
  copy_gate_evolution.png   — gate openness vs layer & position
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)


def load_run(path: str = "run.json") -> Dict:
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


# ----------------------------------------------------------------------
def plot_learning_curves(s: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    for kind, color in [("ndr", "C0"), ("vanilla", "C3")]:
        ax.plot(s[kind]["steps"], s[kind]["train_loss"],
                label=kind.upper(), color=color, alpha=0.85, lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("train batch loss")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for kind, color in [("ndr", "C0"), ("vanilla", "C3")]:
        steps_tr = [t for (t, _) in s[kind]["eval_train_acc"]]
        acc_tr = [a for (_, a) in s[kind]["eval_train_acc"]]
        steps_te = [t for (t, _) in s[kind]["eval_test_acc"]]
        acc_te = [a for (_, a) in s[kind]["eval_test_acc"]]
        ax.plot(steps_tr, acc_tr, "-", color=color,
                label=f"{kind.upper()} train (d=1..4)")
        ax.plot(steps_te, acc_te, "--", color=color,
                label=f"{kind.upper()} test  (d=5..7)")
    ax.axhline(0.25, color="grey", ls=":", alpha=0.5, label="chance (4 classes)")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by partition")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(VIZ, "learning_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_per_depth_final(s: Dict):
    ndr_pd = s["ndr"]["final_eval"]["per_depth"]
    van_pd = s["vanilla"]["final_eval"]["per_depth"]
    depths = sorted(int(k) for k in ndr_pd.keys())
    ndr_vals = [ndr_pd[str(d)] for d in depths]
    van_vals = [van_pd[str(d)] for d in depths]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    x = np.arange(len(depths))
    w = 0.4
    bars1 = ax.bar(x - w / 2, ndr_vals, w, label="NDR (geometric + gate)", color="C0")
    bars2 = ax.bar(x + w / 2, van_vals, w, label="Vanilla (softmax, no gate)", color="C3")
    ax.axhline(0.25, color="grey", ls=":", alpha=0.6, label="chance (4 classes)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in depths])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-depth accuracy (final)")
    # Mark train vs test region
    train_max = 4
    test_min = 5
    train_x = [i for i, d in enumerate(depths) if d <= train_max]
    test_x = [i for i, d in enumerate(depths) if d >= test_min]
    if train_x:
        ax.axvspan(min(train_x) - 0.5, max(train_x) + 0.5, color="green",
                    alpha=0.05, label="train depths (1..4)")
    if test_x:
        ax.axvspan(min(test_x) - 0.5, max(test_x) + 0.5, color="orange",
                    alpha=0.08, label="test depths (5..7)")

    for bars, vals in [(bars1, ndr_vals), (bars2, van_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(VIZ, "per_depth_final.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_length_generalization(s: Dict):
    """Per-depth accuracy curves over training (NDR vs vanilla)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind, color in [(axes[0], "ndr", "C0"), (axes[1], "vanilla", "C3")]:
        per_depth = s[kind]["per_depth"]
        depths = sorted(int(k) for k in per_depth[0][1].keys())
        for d in depths:
            line = "-" if d <= 4 else "--"
            steps = [t for (t, _) in per_depth]
            vals = [pd[str(d)] for (_, pd) in per_depth]
            ax.plot(steps, vals, line, label=f"d={d}", alpha=0.85)
        ax.axhline(0.25, color="grey", ls=":", alpha=0.5)
        ax.set_xlabel("step")
        ax.set_title(kind.upper())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Length-generalization curves: solid = train depths, dashed = test depths")
    fig.tight_layout()
    out = os.path.join(VIZ, "length_generalization.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def _pretty_token_label(tid: int, n_values: int = 4, n_funcs: int = 4) -> str:
    if tid < n_values:
        return f"v{tid}"
    if tid < n_values + n_funcs:
        return f"f{tid - n_values}"
    return "<pad>"


def plot_attention_maps(s: Dict):
    sample = s["attn_sample"]
    x_ids = sample["x_ids"][0]
    L = sample["length"]
    target = sample["target"]
    labels = [_pretty_token_label(int(t)) for t in x_ids]

    # NDR -- show attention from each layer averaged over heads
    ndr_attn = np.array(sample["ndr_attn"])      # (n_layers, n_heads, L, L)
    van_attn = np.array(sample["van_attn"])

    n_layers = ndr_attn.shape[0]

    fig, axes = plt.subplots(2, n_layers, figsize=(2.4 * n_layers, 5.2))
    for li in range(n_layers):
        # Average across heads to get a single per-layer view
        ndr_a = ndr_attn[li].mean(axis=0)         # (L, L)
        van_a = van_attn[li].mean(axis=0)
        ax = axes[0, li]
        im = ax.imshow(ndr_a[:L, :L], cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"NDR L{li}")
        ax.set_xticks(range(L))
        ax.set_xticklabels(labels[:L], rotation=45, fontsize=7)
        if li == 0:
            ax.set_yticks(range(L))
            ax.set_yticklabels(labels[:L], fontsize=7)
            ax.set_ylabel("query")
        else:
            ax.set_yticks([])
        ax = axes[1, li]
        im = ax.imshow(van_a[:L, :L], cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Van L{li}")
        ax.set_xticks(range(L))
        ax.set_xticklabels(labels[:L], rotation=45, fontsize=7)
        if li == 0:
            ax.set_yticks(range(L))
            ax.set_yticklabels(labels[:L], fontsize=7)
            ax.set_ylabel("query")
        else:
            ax.set_yticks([])

    fig.suptitle(f"Attention maps (head-mean) for input {labels[:L]} → target v{target}",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(VIZ, "attention_maps.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_copy_gate(s: Dict):
    sample = s["attn_sample"]
    L = sample["length"]
    labels = [_pretty_token_label(int(t)) for t in sample["x_ids"][0]]
    gates = sample["ndr_gates"]
    if gates is None or all(g is None for g in gates):
        return
    G = np.array(gates)                          # (n_layers, L)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(G[:, :L], cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(L))
    ax.set_xticklabels(labels[:L], rotation=0)
    ax.set_yticks(range(G.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(G.shape[0])])
    ax.set_xlabel("position")
    ax.set_ylabel("layer")
    ax.set_title("NDR copy gate openness  (1 = update, 0 = copy)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for li in range(G.shape[0]):
        for j in range(L):
            ax.text(j, li, f"{G[li, j]:.2f}", ha="center", va="center",
                    color="white" if G[li, j] < 0.5 else "black", fontsize=7)
    fig.tight_layout()
    out = os.path.join(VIZ, "copy_gate.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def main():
    s = load_run()
    plot_learning_curves(s)
    plot_per_depth_final(s)
    plot_length_generalization(s)
    plot_attention_maps(s)
    plot_copy_gate(s)


if __name__ == "__main__":
    main()
