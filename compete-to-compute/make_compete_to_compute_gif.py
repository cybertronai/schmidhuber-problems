"""Animate the training-time forgetting curve.

Re-trains a small ReLU vs LWTA pair, recording per-epoch test accuracy on
Task1 and Task2, and dumps an animation that reveals where the LWTA curve
(blue) preserves Task1 better than ReLU (red) when Task2 starts.

Output: ``compete_to_compute.gif`` (~1 MB, 8 frames + a hold).
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# allow `import compete_to_compute` from this script's folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compete_to_compute as ctc


def _frame(history, n_ep_per_task, frame_idx, total_frames):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5), dpi=110)
    colors = {"relu": "tab:red", "lwta": "tab:blue"}
    for m in ("relu", "lwta"):
        x = np.arange(1, len(history[m]["t1"]) + 1)
        ax.plot(x, history[m]["t1"], color=colors[m], lw=2,
                label=f"{m.upper()} : Task1 acc")
        ax.plot(x, history[m]["t2"], color=colors[m], lw=2, ls="--",
                label=f"{m.upper()} : Task2 acc")
    ax.axvline(n_ep_per_task + 0.5, color="k", lw=1, alpha=0.5)
    ax.text(n_ep_per_task / 2, 0.05, "Task1 (digits 0-4)",
            ha="center", color="grey")
    ax.text(n_ep_per_task * 1.5, 0.05, "Task2 (digits 5-9)",
            ha="center", color="grey")
    ax.set_xlim(0.5, 2 * n_ep_per_task + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epoch (across both tasks, in order)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"compete-to-compute  --  step {frame_idx}/{total_frames}")
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


class Args:
    seed = 0
    n_seeds = 1
    hidden = 400
    k = 2
    depth = 2
    n_train_per_class = 500
    n_test = 1000
    task1_classes = [0, 1, 2, 3, 4]
    task2_classes = [5, 6, 7, 8, 9]
    epochs = 5
    batch = 64
    lr = 0.05
    momentum = 0.9
    weight_decay = 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", default="compete_to_compute.gif")
    cli = p.parse_args()

    args = Args()
    args.seed = cli.seed
    args.epochs = cli.epochs

    print("loading MNIST + retraining for animation...")
    mn = ctc.load_mnist()
    rng = np.random.default_rng(args.seed)
    X_tr_full = mn["train_images"].reshape(-1, 28 * 28)
    y_tr_full = mn["train_labels"]
    X_te_full = mn["test_images"].reshape(-1, 28 * 28)
    y_te_full = mn["test_labels"]
    X_tr_t1, y_tr_t1 = ctc.balanced_subsample(
        *ctc.split_by_classes(X_tr_full, y_tr_full, args.task1_classes),
        n_per_class=args.n_train_per_class, rng=rng)
    X_tr_t2, y_tr_t2 = ctc.balanced_subsample(
        *ctc.split_by_classes(X_tr_full, y_tr_full, args.task2_classes),
        n_per_class=args.n_train_per_class, rng=rng)
    X_te_t1, y_te_t1 = ctc.split_by_classes(X_te_full, y_te_full,
                                            args.task1_classes)
    X_te_t2, y_te_t2 = ctc.split_by_classes(X_te_full, y_te_full,
                                            args.task2_classes)
    X_te_t1, y_te_t1 = X_te_t1[:args.n_test], y_te_t1[:args.n_test]
    X_te_t2, y_te_t2 = X_te_t2[:args.n_test], y_te_t2[:args.n_test]
    Y_tr_t1 = ctc.one_hot(y_tr_t1)
    Y_tr_t2 = ctc.one_hot(y_tr_t2)
    head1 = np.zeros(10, dtype=np.float32); head1[args.task1_classes] = 1.0
    head2 = np.zeros(10, dtype=np.float32); head2[args.task2_classes] = 1.0

    sizes = [28 * 28] + [args.hidden] * args.depth + [10]
    nets = {
        "relu": ctc.MLP(sizes, "relu", args.k,
                        np.random.default_rng(args.seed + 1)),
        "lwta": ctc.MLP(sizes, "lwta", args.k,
                        np.random.default_rng(args.seed + 1)),
    }
    history = {m: {"t1": [], "t2": []} for m in nets}

    frames = []
    total_epochs = 2 * args.epochs

    # one frame per epoch interleaved across both nets
    rng_train = {m: np.random.default_rng(args.seed + (100 if m == "relu"
                                                       else 200))
                 for m in nets}
    for task_idx, (X, Y, head) in enumerate(
            [(X_tr_t1, Y_tr_t1, head1), (X_tr_t2, Y_tr_t2, head2)]):
        for ep in range(args.epochs):
            for m, net in nets.items():
                ctc.train_one_task(
                    net, X, Y, epochs=1, batch=args.batch,
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    rng=rng_train[m], eval_fns={}, head_mask=head)
                t1_acc = ctc.evaluate(net, X_te_t1, y_te_t1,
                                      head_mask=head1)["acc"]
                t2_acc = ctc.evaluate(net, X_te_t2, y_te_t2,
                                      head_mask=head2)["acc"]
                history[m]["t1"].append(t1_acc)
                history[m]["t2"].append(t2_acc)
            step = task_idx * args.epochs + ep + 1
            frames.append(_frame(history, args.epochs, step, total_epochs))
    # hold the final frame
    for _ in range(4):
        frames.append(frames[-1])

    print(f"writing {cli.out} ({len(frames)} frames)")
    imageio.mimsave(cli.out, frames, duration=0.55, loop=0)


if __name__ == "__main__":
    main()
