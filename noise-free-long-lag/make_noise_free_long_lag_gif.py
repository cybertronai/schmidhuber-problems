"""Build noise_free_long_lag.gif from training snapshots.

Each frame shows: (a) the held-out last-step accuracy curve up to the
current snapshot, (b) the predicted softmax over the alphabet at the final
step for a fixed y-key example and a fixed x-key example. As training
proceeds the bars at the x-index and y-index light up correctly while the
distractor mass evaporates.

Generates in-memory frames and stitches via Pillow (no imageio dependency).
"""

from __future__ import annotations

import argparse
import os
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import noise_free_long_lag as nfl

try:
    from PIL import Image
except ImportError:
    Image = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2000)
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", type=str, default="noise_free_long_lag.gif")
    args = ap.parse_args()

    if Image is None:
        print("Pillow is not installed; skipping GIF render.")
        print("`pip install pillow` to produce the animation.")
        return

    out = nfl.train(
        p=args.p,
        hidden=args.hidden,
        seed=args.seed,
        max_seq=args.max_seq,
        snapshots=args.n_frames,
        verbose=False,
    )
    snaps = out["snapshots"]
    log = out["log"]
    if not snaps:
        print("No snapshots captured.")
        return

    # Fixed test sequences
    p = args.p
    V = nfl.alphabet_size(p)

    def fixed_seq(start_key: int):
        seq = [start_key] + list(range(p - 1)) + [start_key]
        inputs = np.asarray(seq[:-1], dtype=np.int64)
        targets = np.asarray(seq[1:], dtype=np.int64)
        T = inputs.shape[0]
        X = np.zeros((T, V))
        X[np.arange(T), inputs] = 1.0
        return X, targets

    Xy, Ty = fixed_seq(nfl.y_index(p))
    Xx, Tx = fixed_seq(nfl.x_index(p))

    # Curve x-axis is per-eval log; convert step -> rolling_acc_last
    eval_steps = np.array(log["step"])
    eval_acc = np.array(log["rolling_acc_last"])

    images = []
    for snap in snaps:
        # Reconstruct an LSTM with these parameters
        m = nfl.LSTM(V_in=V, hidden=args.hidden, V_out=V, seed=args.seed)
        m.W = snap["W"].copy()
        m.b = snap["b"].copy()
        m.Wy = snap["Wy"].copy()
        m.by = snap["by"].copy()

        _, cy = m.forward(Xy)
        _, cx = m.forward(Xx)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

        # Curve panel
        mask = eval_steps <= snap["step"]
        axes[0].plot(eval_steps[mask], eval_acc[mask], color="C2")
        axes[0].set_xlim(0, args.max_seq)
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(0.95, color="k", lw=0.5, linestyle=":", alpha=0.4)
        axes[0].set_xlabel("training sequence")
        axes[0].set_ylabel("rolling-256 last-step acc")
        axes[0].set_title(f"step {snap['step']}")

        for ax, cache, label in zip(axes[1:], [cy, cx],
                                    ["y-key sequence", "x-key sequence"]):
            probs_last = cache["probs"][-1]
            colors = ["#aaaaaa"] * V
            colors[nfl.x_index(p)] = "#3070C0"
            colors[nfl.y_index(p)] = "#C03030"
            ax.bar(np.arange(V), probs_last, color=colors)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("symbol idx")
            ax.set_title(label)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())

    duration = int(1000 / args.fps)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {args.out}  ({len(images)} frames, {duration} ms each)")


if __name__ == "__main__":
    main()
