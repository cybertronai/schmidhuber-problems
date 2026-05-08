"""Make iam_handwriting.gif: animate the BLSTM+CTC reading a synthetic
handwritten word frame by frame.

Each frame shows:
  - top:    the pen trajectory drawn so far (up to time t)
  - middle: BLSTM softmax heatmap up to time t (full sequence is needed for
            true output -- we use the trained model on the *whole* sequence
            and reveal the heatmap progressively to keep the visual honest)
  - bottom: the running greedy CTC decode at time t (collapse repeats, drop
            blanks)

Output: iam_handwriting.gif (target <= 2 MB).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from iam_handwriting import (  # noqa: E402
    ALPHABET, ID2CHAR, BLANK, CHAR2ID, render_word, greedy_decode,
    BLSTMCTC, RunConfig, train as run_train,
)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _retrain_quick(seed: int = 0):
    """Retrain model from scratch with the same seed to recover weights."""
    cfg = RunConfig(seed=seed)
    summary = run_train(cfg, verbose=False)
    return summary, cfg


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        import subprocess
        subprocess.run(
            ["python3", os.path.join(HERE, "iam_handwriting.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )

    summary = _load(json_path)
    # Use the saved long alignment for the GIF
    a = summary["long_alignment"]
    word = a["word"]
    abs_xy = np.array(a["abs_xy"])
    traj = np.array(a["traj"])
    log_probs = np.array(a["log_probs"])
    probs = np.exp(log_probs)
    target = a["label_chars"]
    T, K = log_probs.shape

    fig = plt.figure(figsize=(10.0, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 2.4, 0.5], hspace=0.25)
    ax_traj = fig.add_subplot(gs[0])
    ax_hm = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])

    # Static trajectory background bounds
    pad = 0.2
    ax_traj.set_xlim(abs_xy[:, 0].min() - pad, abs_xy[:, 0].max() + pad)
    ax_traj.set_ylim(abs_xy[:, 1].min() - pad, abs_xy[:, 1].max() + pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_axis_off()
    ax_traj.set_title(f"BLSTM + CTC reads synthetic handwriting "
                      f"(target = {word!r})", fontsize=11)

    # Stroke segments (split where pen-up flag = 1 marks a NEW stroke)
    pen_up = traj[:, 2].astype(bool)
    stroke_starts = list(np.where(pen_up)[0]) + [T]
    n_strokes = len(stroke_starts) - 1

    # Build one Line2D per stroke; we update its xdata/ydata each frame.
    stroke_lines = [ax_traj.plot([], [], color="#222", linewidth=1.8)[0]
                    for _ in range(n_strokes)]
    pen_dot, = ax_traj.plot([], [], marker="o", color="#d4694e",
                            markersize=8, zorder=5)

    # Heatmap canvas: fixed size (K, T); we mask out future timesteps.
    hm_data = np.full((K, T), np.nan)
    im = ax_hm.imshow(hm_data, aspect="auto", origin="lower",
                      cmap="magma", vmin=0.0, vmax=1.0)
    ax_hm.set_yticks(range(K))
    ax_hm.set_yticklabels(["-"] + ALPHABET, fontsize=9)
    ax_hm.set_xlabel("timestep")
    ax_hm.set_ylabel("class")
    ax_hm.set_title("BLSTM softmax (revealed up to current frame)",
                    fontsize=10)

    ax_text.set_axis_off()
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    decode_text = ax_text.text(
        0.02, 0.5,
        "running greedy decode: ''",
        ha="left", va="center", fontsize=12,
        family="monospace",
    )

    def _update_strokes(t_now: int):
        for si in range(n_strokes):
            a0 = stroke_starts[si]
            b0 = stroke_starts[si + 1]
            up_to = min(t_now + 1, b0)
            if up_to <= a0:
                stroke_lines[si].set_data([], [])
            else:
                xs = abs_xy[a0:up_to, 0]
                ys = abs_xy[a0:up_to, 1]
                stroke_lines[si].set_data(xs, ys)
        cur_t = min(t_now, T - 1)
        pen_dot.set_data([abs_xy[cur_t, 0]], [abs_xy[cur_t, 1]])

    def update(frame):
        # Frame 0..T-1 reveal time t = frame; T..T+5 hold final state.
        t = min(frame, T - 1)
        # Trajectory
        _update_strokes(t)
        # Heatmap: reveal columns 0..t
        d = np.full((K, T), np.nan)
        d[:, :t + 1] = probs.T[:, :t + 1]
        im.set_array(d)
        # Decoded so far
        partial_argmax = log_probs[:t + 1].argmax(axis=1).tolist()
        out, prev = [], -1
        for k in partial_argmax:
            if k != prev and k != BLANK:
                out.append(ID2CHAR[int(k)])
            prev = k
        decoded_str = "".join(out)
        decode_text.set_text(
            f"running greedy decode @ t = {t:>3d}/{T - 1}:  "
            f"{decoded_str!r}    target = {word!r}"
        )
        return stroke_lines + [pen_dot, im, decode_text]

    n_frames = T + 6  # tail-pause for a beat at the end
    anim = FuncAnimation(fig, update, frames=n_frames, interval=120,
                         blit=False)
    out_path = os.path.join(HERE, "iam_handwriting.gif")
    writer = PillowWriter(fps=8)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    sz = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({sz/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
