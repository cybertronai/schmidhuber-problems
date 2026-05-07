"""
Animated GIF showing random-weight-guessing in progress on a Tomita grammar.

Layout per frame:
  Top:    bar chart of best train_acc and test_acc for each of the 3 grammars
  Bottom: trial-vs-best-train-acc trace, log x-axis

Each frame corresponds to one new "best" event (i.e., a trial whose train
accuracy strictly improves on what came before). The number of frames is
therefore short (typically 5-15 per grammar), keeping the GIF small.

Usage:
  python3 make_rs_tomita_gif.py --seed 0 --out rs_tomita.gif
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from rs_tomita import DEFAULT_MAX_TRIALS, run_grammar


GRAMMAR_LABELS = {
    1: "#1: a*",
    2: "#2: (ab)*",
    4: "#4: no aaa",
}
GRAMMAR_COLORS = {1: "#1f77b4", 2: "#2ca02c", 4: "#d62728"}


def render_frame(
    histories: dict[int, np.ndarray],
    upto_trial: dict[int, int],
    n_grammars_done: int,
    title_extra: str = "",
) -> Image.Image:
    fig = plt.figure(figsize=(10, 5.5), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.45)

    # --- top: bar chart of current best train and test acc ---
    ax_bar = fig.add_subplot(gs[0])
    grammars = [1, 2, 4]
    x = np.arange(len(grammars))
    width = 0.35
    train_vals = []
    test_vals = []
    for g in grammars:
        h = histories[g]
        upto = upto_trial[g]
        if h is not None and len(h) > 0:
            mask = h[:, 0] <= upto
            if mask.any():
                row = h[mask][-1]
                train_vals.append(row[1])
                test_vals.append(row[2])
            else:
                train_vals.append(0.0)
                test_vals.append(0.0)
        else:
            train_vals.append(0.0)
            test_vals.append(0.0)
    ax_bar.bar(x - width / 2, train_vals, width, color=[GRAMMAR_COLORS[g] for g in grammars],
                label="train", alpha=0.95, edgecolor="black", linewidth=0.5)
    ax_bar.bar(x + width / 2, test_vals, width, color=[GRAMMAR_COLORS[g] for g in grammars],
                label="test", alpha=0.45, edgecolor="black", linewidth=0.5)
    ax_bar.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    for xi, (tr, te) in enumerate(zip(train_vals, test_vals)):
        ax_bar.text(xi - width / 2, tr + 0.02, f"{tr:.2f}",
                     ha="center", fontsize=8)
        ax_bar.text(xi + width / 2, te + 0.02, f"{te:.2f}",
                     ha="center", fontsize=8, alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Tomita {GRAMMAR_LABELS[g]}" for g in grammars])
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel("running best accuracy")
    ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.set_title(f"Random-weight guessing on Tomita grammars{title_extra}",
                      fontsize=11)

    # --- bottom: trial vs train_acc (running max) for all grammars ---
    ax_curve = fig.add_subplot(gs[1])
    for g in grammars:
        h = histories[g]
        upto = upto_trial[g]
        if h is None or len(h) == 0:
            continue
        mask = h[:, 0] <= upto
        if mask.sum() < 1:
            continue
        sub = h[mask]
        # Step plot: train acc as a function of trial
        ax_curve.step(sub[:, 0], sub[:, 1], where="post",
                       color=GRAMMAR_COLORS[g],
                       label=f"Tomita {GRAMMAR_LABELS[g]}",
                       linewidth=2)
        # Mark current point
        ax_curve.scatter([sub[-1, 0]], [sub[-1, 1]],
                          color=GRAMMAR_COLORS[g], s=40, zorder=5,
                          edgecolor="black", linewidth=0.6)
    ax_curve.set_xscale("symlog")
    ax_curve.set_xlabel("trial (log scale)")
    ax_curve.set_ylabel("running best train accuracy")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def build_frame_schedule(histories: dict[int, np.ndarray]) -> list[tuple[dict[int, int], str]]:
    """Plan a frame at each new "best" event, plus a final hold frame.

    Each frame is keyed by upto_trial[g] = highest trial seen so far for grammar g.
    """
    # Combined ordering of events across grammars by trial number.
    events = []
    for g in [1, 2, 4]:
        h = histories[g]
        for row in h:
            trial = int(row[0])
            events.append((trial, g))
    events.sort()

    frames: list[tuple[dict[int, int], str]] = []
    upto = {1: -1, 2: -1, 4: -1}
    for trial, g in events:
        upto[g] = trial
        frames.append((dict(upto), f"  ·  trial {trial}, advanced #{g}"))
    # Hold the last frame for a few extra slots.
    if frames:
        last = frames[-1]
        for _ in range(6):
            frames.append(last)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=None,
                        help="cap per grammar (default 5k/50k/300k for #1/#2/#4)")
    parser.add_argument("--out", type=str, default="rs_tomita.gif")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    print(f"Running RS for seed={args.seed} ...")
    histories: dict[int, np.ndarray] = {}
    for g in [1, 2, 4]:
        max_trials = args.max_trials if args.max_trials else DEFAULT_MAX_TRIALS[g]
        r = run_grammar(g, args.seed, max_trials, args.scale, args.hidden)
        h = np.array(r["history"], dtype=np.float64) if r["history"] else np.zeros((0, 3))
        histories[g] = h
        print(f"  #{g}: solved_at={r['solved_at']} | "
              f"train={r['best_train']:.3f} test={r['best_test']:.3f}")

    print("Rendering frames ...")
    frame_specs = build_frame_schedule(histories)
    frames = []
    for upto, extra in frame_specs:
        frames.append(render_frame(histories, upto, n_grammars_done=0,
                                    title_extra=extra))
    print(f"  rendered {len(frames)} frames")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB, {len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
