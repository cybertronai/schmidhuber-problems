"""Static visualizations for predictable-stereo.

Outputs:
    viz/learning_curves.png     IMAX I + recovery accuracy + agreement vs epoch
    viz/code_scatter.png        (yL, yR) scatter, colored by hidden depth z
    viz/weight_maps.png         input-layer weight magnitudes per view, with
                                a marker showing which input dims are shared
    viz/agreement_hist.png      histogram of (yL - yR) before/after training
    viz/baseline_compare.png    real-stereo vs shuffled-stereo recovery curves
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from predictable_stereo import train, ViewNet, make_stereo_dataset


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_learning_curves(history, out):
    eps = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(eps, history["I_nats"], color="C0", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("I(yL; yR) [nats]")
    ax.set_title("IMAX mutual-information estimate")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eps, history["recovery_acc_train"], color="C1", lw=2, label="train")
    ax.plot(eps, history["recovery_acc_eval"], color="C0", lw=2, label="eval (held-out)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary recovery accuracy")
    ax.set_title("hidden depth recovery (max(acc, 1-acc))")
    ax.set_ylim(0.45, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.plot(eps, history["agreement_train"], color="C1", lw=2, label="train")
    ax.plot(eps, history["agreement_eval"], color="C0", lw=2, label="eval (held-out)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary agreement (sign(yL) == sign(yR))")
    ax.set_title("L/R code agreement")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle("predictable-stereo learning curves (IMAX, two-net cooperative)")
    fig.tight_layout()
    _save(fig, out)


def plot_code_scatter(netL, netR, data, out, title_prefix=""):
    yL = netL.forward(data["x_L"])
    yR = netR.forward(data["x_R"])
    z = data["z"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for k, (ax, label) in enumerate(zip(axes, ["before training (random init)",
                                                "after training"])):
        pass  # left axis populated by caller

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pos = z > 0
    neg = ~pos
    ax.scatter(yL[pos], yR[pos], s=14, c="C3", alpha=0.6, label="z = +1")
    ax.scatter(yL[neg], yR[neg], s=14, c="C0", alpha=0.6, label="z = -1")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.plot([-1, 1], [-1, 1], color="grey", lw=0.5, ls="--", label="y_L = y_R")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("y_L (left network code)")
    ax.set_ylabel("y_R (right network code)")
    ax.set_title(f"{title_prefix}code pair (yL, yR), colored by hidden depth z")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_code_scatter_before_after(seed, data, out,
                                    n_epochs=400, **train_kwargs):
    """Side-by-side: random-init scatter vs trained scatter."""
    rng = np.random.default_rng(seed + 31_337)
    d_in = data["x_L"].shape[1]
    netL_init = ViewNet(d_in, train_kwargs.get("d_hidden", 16), rng)
    netR_init = ViewNet(d_in, train_kwargs.get("d_hidden", 16), rng)
    yL0 = netL_init.forward(data["x_L"]); yR0 = netR_init.forward(data["x_R"])

    out_t = train(seed=seed, n_epochs=n_epochs, verbose=False, **train_kwargs)
    netL = out_t["netL"]; netR = out_t["netR"]
    data_tr = out_t["data"]
    yL = netL.forward(data_tr["x_L"]); yR = netR.forward(data_tr["x_R"])
    z_tr = data_tr["z"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (yL_p, yR_p, z_p, ttl) in zip(
        axes,
        [(yL0, yR0, data["z"], "before training (random init)"),
         (yL, yR, z_tr, "after IMAX training")]):
        pos = z_p > 0
        neg = ~pos
        ax.scatter(yL_p[pos], yR_p[pos], s=12, c="C3", alpha=0.55, label="z = +1")
        ax.scatter(yL_p[neg], yR_p[neg], s=12, c="C0", alpha=0.55, label="z = -1")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.plot([-1, 1], [-1, 1], color="grey", lw=0.5, ls="--")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("y_L"); ax.set_ylabel("y_R")
        ax.set_title(ttl)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("predictable-stereo: scalar codes (yL, yR) before vs after training")
    fig.tight_layout()
    _save(fig, out)
    return out_t


def plot_weight_maps(netL, netR, d_shared, out):
    """Weight magnitudes from input dims into the first hidden layer.

    The shared dims (0..d_shared) should pick up larger weights than the
    view-specific dims. This is the visualization of which dims the network
    learned to attend to."""
    d_h, d_in = netL.W1.shape

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    for ax, W, name in zip(axes, [netL.W1, netR.W1], ["left net (W1)", "right net (W1)"]):
        # Per-input-dim importance: L2 norm across hidden units.
        importance = np.linalg.norm(W, axis=0)
        cols = ["C2" if i < d_shared else "C7" for i in range(d_in)]
        ax.bar(np.arange(d_in), importance, color=cols)
        ax.axvline(d_shared - 0.5, color="black", lw=1.0)
        ax.set_ylabel(f"||W1[:, i]||_2\n({name})")
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xlabel("input dim index")
    axes[0].set_title(
        f"Input-dim importance after IMAX training. Green = shared dims (0..{d_shared-1}), "
        f"grey = view-specific distractors.")
    fig.tight_layout()
    _save(fig, out)


def plot_agreement_hist(yL_init, yR_init, yL_trained, yR_trained, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(yL_init - yR_init, bins=40, color="C7", alpha=0.85)
    axes[0].set_title("yL - yR  (random init)")
    axes[0].set_xlim(-2.05, 2.05)
    axes[0].grid(alpha=0.3)
    axes[1].hist(yL_trained - yR_trained, bins=40, color="C2", alpha=0.85)
    axes[1].set_title("yL - yR  (after training)")
    axes[1].set_xlim(-2.05, 2.05)
    axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("yL - yR")
    axes[0].set_ylabel("count")
    fig.suptitle("disagreement distribution: training collapses (yL - yR) to 0")
    fig.tight_layout()
    _save(fig, out)


def plot_baseline_compare(history_real, history_shuffled, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history_real["epoch"], history_real["recovery_acc_eval"],
            color="C2", lw=2, label="real stereo (shared depth)")
    ax.plot(history_shuffled["epoch"], history_shuffled["recovery_acc_eval"],
            color="C3", lw=2, label="shuffled (no shared depth)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("epoch"); ax.set_ylabel("eval recovery accuracy")
    ax.set_title("recovery vs negative control")
    ax.set_ylim(0.4, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(history_real["epoch"], history_real["agreement_eval"],
            color="C2", lw=2, label="real stereo")
    ax.plot(history_shuffled["epoch"], history_shuffled["agreement_eval"],
            color="C3", lw=2, label="shuffled")
    ax.set_xlabel("epoch"); ax.set_ylabel("eval agreement")
    ax.set_title("L/R agreement (note: shuffled hits 1.0 by output collapse)")
    ax.set_ylim(0.0, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower right")

    fig.suptitle("predictable-stereo: real vs shuffled stereo")
    fig.tight_layout()
    _save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--out-dir", default="viz")
    args = p.parse_args()

    print("training real-stereo run for visualizations ...")
    rng = np.random.default_rng(args.seed + 31_337)
    train_kwargs = dict(
        n_samples=1024, n_eval=1024, d_shared=8, d_view=8,
        flip_p=0.10, d_hidden=16, lr=0.03,
        n_epochs=args.n_epochs, eval_every=20, shuffled=False, quick=False,
    )

    # Capture random-init outputs first (using same seed schedule as train()).
    data_init = make_stereo_dataset(
        n_samples=1024, d_shared=8, d_view=8, flip_p=0.10,
        seed=args.seed, shuffled=False,
    )
    rng_init = np.random.default_rng(args.seed + 31_337)
    netL_init = ViewNet(16, 16, rng_init)
    netR_init = ViewNet(16, 16, rng_init)
    yL_init = netL_init.forward(data_init["x_L"])
    yR_init = netR_init.forward(data_init["x_R"])

    out_real = train(seed=args.seed, verbose=False, **train_kwargs)
    history_real = out_real["history"]
    netL = out_real["netL"]; netR = out_real["netR"]
    data_tr = out_real["data"]
    yL_t = netL.forward(data_tr["x_L"])
    yR_t = netR.forward(data_tr["x_R"])

    print("training shuffled-stereo baseline run ...")
    train_kwargs_s = dict(train_kwargs); train_kwargs_s["shuffled"] = True
    out_shuffled = train(seed=args.seed, verbose=False, **train_kwargs_s)
    history_shuffled = out_shuffled["history"]

    plot_learning_curves(history_real, os.path.join(args.out_dir, "learning_curves.png"))

    # Code scatter: before/after side-by-side using fresh-train re-do for the
    # "after" panel so we share state across the function.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (yL_p, yR_p, z_p, ttl) in zip(
        axes,
        [(yL_init, yR_init, data_init["z"], "before training (random init)"),
         (yL_t, yR_t, data_tr["z"], "after IMAX training")]):
        pos = z_p > 0; neg = ~pos
        ax.scatter(yL_p[pos], yR_p[pos], s=12, c="C3", alpha=0.55, label="z = +1")
        ax.scatter(yL_p[neg], yR_p[neg], s=12, c="C0", alpha=0.55, label="z = -1")
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.plot([-1, 1], [-1, 1], color="grey", lw=0.5, ls="--")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("y_L"); ax.set_ylabel("y_R")
        ax.set_title(ttl)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("predictable-stereo: scalar codes (yL, yR) before vs after training")
    fig.tight_layout()
    _save(fig, os.path.join(args.out_dir, "code_scatter.png"))

    plot_weight_maps(netL, netR, data_tr["d_shared"],
                     os.path.join(args.out_dir, "weight_maps.png"))
    plot_agreement_hist(yL_init, yR_init, yL_t, yR_t,
                        os.path.join(args.out_dir, "agreement_hist.png"))
    plot_baseline_compare(history_real, history_shuffled,
                          os.path.join(args.out_dir, "baseline_compare.png"))


if __name__ == "__main__":
    main()
