"""Static training-time visualisations for compete-to-compute.

Reads ``viz/snapshots.npz`` produced by ``compete_to_compute.py`` and writes:

* ``viz/training_curves.png``     ReLU and LWTA test-set accuracy on
                                   Task1 / Task2 across all training epochs.
* ``viz/forgetting_bar.png``      summary bar chart of T1 acc before / after
                                   T2 training for each model.
* ``viz/W1_relu.png``             first-layer receptive fields, 100 units.
* ``viz/W1_lwta.png``             same for LWTA, 100 units.
* ``viz/winner_freq.png``         per-unit activation frequency on Task1
                                   vs Task2 inputs (specialisation diagnostic).
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _grid_image(W: np.ndarray, n_rows: int = 10, n_cols: int = 10,
                cell: int = 28) -> np.ndarray:
    """Tile the first ``n_rows*n_cols`` columns of W (each a 28*28 vector)
    into one big image."""
    n = n_rows * n_cols
    img = np.zeros((n_rows * cell, n_cols * cell))
    for k in range(min(n, W.shape[1])):
        r, c = divmod(k, n_cols)
        f = W[:, k].reshape(cell, cell)
        m = max(abs(f.min()), abs(f.max()), 1e-6)
        img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = f / m
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", default="viz/snapshots.npz")
    p.add_argument("--out", default="viz")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    s = np.load(args.snapshots, allow_pickle=False)
    epochs = s["epochs"]
    tasks = s["tasks"]
    models = s["models"]
    t1 = s["t1_test_acc"]
    t2 = s["t2_test_acc"]

    n_ep = epochs.max()
    # Re-index into a 2D table: row=model in {relu, lwta}, col = global step
    # global step = (task-1)*n_ep + epoch
    def series(model: str, key: str):
        idx = np.where(models == model)[0]
        order = np.array([(t - 1) * n_ep + e for t, e in
                          zip(tasks[idx], epochs[idx])])
        ord_ = np.argsort(order)
        return order[ord_] + 1, (t1 if key == "t1" else t2)[idx][ord_]

    # ------------------------------- training curves ----------
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    colors = {"relu": "tab:red", "lwta": "tab:blue"}
    for m in ("relu", "lwta"):
        x, y1 = series(m, "t1")
        _, y2 = series(m, "t2")
        ax.plot(x, y1, color=colors[m], lw=2,
                label=f"{m.upper()} : Task1 test acc")
        ax.plot(x, y2, color=colors[m], lw=2, ls="--",
                label=f"{m.upper()} : Task2 test acc")
    ax.axvline(n_ep + 0.5, color="k", lw=1, alpha=0.5)
    ax.text(n_ep / 2, 0.05, "Task1 training", ha="center", color="grey")
    ax.text(n_ep * 1.5, 0.05, "Task2 training", ha="center", color="grey")
    ax.set_xlabel("Epoch (across both tasks, in order)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sequential 2-task MNIST split: ReLU vs LWTA")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=120)
    plt.close(fig)

    # ------------------------------- forgetting bar ----------
    # Pick T1 acc at end of Task1 vs end of Task2 for each model
    def pick(model: str, task: int, ep: int) -> float:
        m = (models == model) & (tasks == task) & (epochs == ep)
        return float(t1[m][0])

    pre = {m: pick(m, 1, n_ep) for m in ("relu", "lwta")}
    post = {m: pick(m, 2, n_ep) for m in ("relu", "lwta")}
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))
    width = 0.35
    x = np.array([0, 1])
    ax.bar(x - width / 2, [pre["relu"], pre["lwta"]], width,
           color="lightgrey", label="T1 acc, after T1 training")
    ax.bar(x + width / 2, [post["relu"], post["lwta"]], width,
           color=["tab:red", "tab:blue"],
           label="T1 acc, after T2 training")
    ax.set_xticks(x)
    ax.set_xticklabels(["ReLU MLP", "LWTA MLP"])
    ax.set_ylabel("Task1 test accuracy")
    ax.set_ylim(0, 1.0)
    for i, m in enumerate(("relu", "lwta")):
        d = pre[m] - post[m]
        ax.text(i, post[m] + 0.02, f"forget = {d:.3f}",
                ha="center", fontsize=10)
    ax.set_title("Catastrophic forgetting on Task1")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "forgetting_bar.png"), dpi=120)
    plt.close(fig)

    # ------------------------------- W1 receptive fields -----
    for tag in ("relu", "lwta"):
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))
        img = _grid_image(s[f"W1_{tag}"], 10, 10, 28)
        ax.imshow(img, cmap="seismic", vmin=-1, vmax=1)
        ax.set_title(f"{tag.upper()} : first-layer receptive fields "
                     f"(100 units)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"W1_{tag}.png"), dpi=120)
        plt.close(fig)

    # ------------------------------- specialisation -----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, tag, title in [
            (axes[0], "relu", "ReLU MLP : units active per task"),
            (axes[1], "lwta", "LWTA MLP : units active per task")]:
        f1 = s[f"winner_freq_{tag}_t1"]
        f2 = s[f"winner_freq_{tag}_t2"]
        order = np.argsort(f1 - f2)
        ax.plot(f1[order], color="tab:red", label="Task1 inputs",
                lw=1)
        ax.plot(f2[order], color="tab:blue", label="Task2 inputs",
                lw=1)
        ax.set_xlabel("Hidden unit (sorted by Task1-Task2 activation gap)")
        ax.set_ylabel("Activation frequency")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "winner_freq.png"), dpi=120)
    plt.close(fig)

    print(f"wrote curves + bar + 2 weight grids + winner_freq under {args.out}/")


if __name__ == "__main__":
    main()
