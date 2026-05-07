"""Static visualisations for blues-improvisation.

Generates the following PNGs into viz/:
  training_curves.png           total / chord / pitch loss + accuracies
  weight_matrices.png           layer-1 and layer-2 LSTM weight panels
  generated_pianoroll.png       deterministic-chord chorus rendered as piano roll
  corpus_pianoroll.png          one ground-truth training chorus for comparison

Usage:
    python3 visualize_blues_improvisation.py --seed 0 --epochs 200
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blues_improvisation import (
    BLUES_PROGRESSION, BARS_PER_CHORUS, CHORDS, INPUT_DIM,
    N_CHORDS, N_PITCHES, PITCH_NAMES, REST, STEPS_PER_BAR, STEPS_PER_CHORUS,
    bar_onset_chord_match, chord_progression_match, chord_tone_rate,
    generate, on_beat_note_rate, synth_corpus, train,
)


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    eps = history.epochs

    ax = axes[0]
    ax.plot(eps, history.loss, "C0-", lw=1.6, label="total")
    ax.plot(eps, history.loss_c, "C1-", lw=1.2, label="chord head")
    ax.plot(eps, history.loss_p, "C2-", lw=1.2, label="pitch head")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eps, history.chord_acc, "C1-", lw=1.6, label="chord")
    ax.plot(eps, history.pitch_acc, "C2-", lw=1.6, label="pitch")
    ax.axhline(1.0 / N_CHORDS, color="C1", ls=":", lw=0.9,
               label=f"chord chance = {1.0/N_CHORDS:.2f}")
    ax.axhline(1.0 / N_PITCHES, color="C2", ls=":", lw=0.9,
               label=f"pitch chance = {1.0/N_PITCHES:.2f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("teacher-forced argmax accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-step prediction accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("blues-improvisation  —  training dynamics",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _imshow_weight(ax, W, title):
    a = float(np.abs(W).max())
    a = max(a, 1e-6)
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-a, vmax=a)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_weight_matrices(params, out_path):
    fig = plt.figure(figsize=(11, 6.5), dpi=110)
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.30,
                          top=0.91, bottom=0.06, left=0.04, right=0.97)
    H1 = params.W1h.shape[0]
    H2 = params.W2h.shape[0]

    # Reshape gate-stacked weights (D, 4H) into (D, H, 4) for clarity.
    def split_gates(W, H):
        return [W[:, 0:H], W[:, H:2*H], W[:, 2*H:3*H], W[:, 3*H:4*H]]

    gate_names = ["i (in)", "f (forget)", "g (cell)", "o (out)"]

    # Layer-1 input weights (per gate)
    W1x_gates = split_gates(params.W1x, H1)
    for j, (g, name) in enumerate(zip(W1x_gates, gate_names)):
        ax = fig.add_subplot(gs[0, j])
        im = _imshow_weight(ax, g.T, f"L1 W1x[{name}]")
        if j == 0:
            ax.set_ylabel(f"H1 ({H1})", fontsize=8)
            ax.set_xlabel(f"input ({INPUT_DIM})", fontsize=8)
            ax.set_xticks(range(0, INPUT_DIM, 2))
            ax.set_xticklabels([str(x) for x in range(0, INPUT_DIM, 2)],
                               fontsize=6)
            ax.set_yticks(range(0, H1, max(1, H1 // 6)))
            ax.set_yticklabels([str(x) for x in range(0, H1, max(1, H1 // 6))],
                               fontsize=6)

    # Layer-2 recurrent weights (per gate)
    W2h_gates = split_gates(params.W2h, H2)
    for j, (g, name) in enumerate(zip(W2h_gates, gate_names)):
        ax = fig.add_subplot(gs[1, j])
        _imshow_weight(ax, g.T, f"L2 W2h[{name}]")
        if j == 0:
            ax.set_ylabel(f"H2 ({H2})", fontsize=8)
            ax.set_xlabel(f"prev H2 ({H2})", fontsize=8)

    fig.suptitle("LSTM weight panels (red = positive, blue = negative)",
                 fontsize=11, y=0.98)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pianoroll(chords, pitches, out_path, title=""):
    """Render a (96,) chord/pitch sequence as a piano-roll PNG.

    Top strip: chord cells (color-coded).
    Main grid: pitch on y-axis (rest at top), time on x-axis.
    """
    n = len(chords)
    fig = plt.figure(figsize=(11, 4.2), dpi=110)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.6, 4.0], hspace=0.18,
                          left=0.10, right=0.98, top=0.90, bottom=0.13)
    ax_c = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    chord_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]    # C7, F7, G7
    for t in range(n):
        ax_c.add_patch(plt.Rectangle(
            (t - 0.5, 0), 1, 1, facecolor=chord_colors[chords[t]],
            edgecolor="none"))
    ax_c.set_xlim(-0.5, n - 0.5)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.set_xticks([])
    ax_c.set_ylabel("chord", fontsize=9)
    # Bar boundary lines
    for b in range(BARS_PER_CHORUS + 1):
        ax_c.axvline(b * STEPS_PER_BAR - 0.5, color="white", lw=1.2)

    # Bar labels above
    for b in range(BARS_PER_CHORUS):
        cx = b * STEPS_PER_BAR + STEPS_PER_BAR / 2 - 0.5
        ax_c.text(cx, 1.4, BLUES_PROGRESSION[b],
                  ha="center", fontsize=8, color="black")

    # Pitch grid
    for t in range(n):
        ax_p.add_patch(plt.Rectangle(
            (t - 0.5, pitches[t] - 0.4), 1, 0.8,
            facecolor="0.25" if pitches[t] != REST else "0.85",
            edgecolor="white", lw=0.4))
    ax_p.set_xlim(-0.5, n - 0.5)
    ax_p.set_ylim(-0.6, N_PITCHES - 0.4)
    ax_p.set_yticks(range(N_PITCHES))
    ax_p.set_yticklabels(PITCH_NAMES, fontsize=8)
    ax_p.set_xlabel("time step (8th notes)", fontsize=9)
    ax_p.set_ylabel("pitch", fontsize=9)
    for b in range(BARS_PER_CHORUS + 1):
        ax_p.axvline(b * STEPS_PER_BAR - 0.5, color="0.5", lw=0.6, ls=":")
    for s in range(BARS_PER_CHORUS * STEPS_PER_BAR):
        if s % STEPS_PER_BAR in (0, 4):
            ax_p.axvline(s - 0.5, color="0.7", lw=0.4, alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--h1", type=int, default=20)
    ap.add_argument("--h2", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--n-pieces", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--lr-decay-every", type=int, default=80)
    ap.add_argument("--out-dir", type=str, default="viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[viz] training (seed={args.seed}, epochs={args.epochs})...")
    params, history, _, (chords, pitches) = train(
        seed=args.seed, h1=args.h1, h2=args.h2,
        n_pieces=args.n_pieces, epochs=args.epochs,
        batch_size=args.batch, lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        eval_every=max(1, args.epochs // 25),
        save_snapshots=False,
        verbose=False,
    )

    # ----- training curves -----
    p1 = os.path.join(args.out_dir, "training_curves.png")
    plot_training_curves(history, p1)
    print(f"  wrote {p1}")

    # ----- weight matrices -----
    p2 = os.path.join(args.out_dir, "weight_matrices.png")
    plot_weight_matrices(params, p2)
    print(f"  wrote {p2}")

    # ----- ground-truth chorus 0 piano roll -----
    p3 = os.path.join(args.out_dir, "corpus_pianoroll.png")
    plot_pianoroll(chords[0], pitches[0], p3,
                   title="Training corpus — chorus 1 (ground truth)")
    print(f"  wrote {p3}")

    # ----- generated chorus piano roll (deterministic chord, sampled pitch) -----
    gen_c, gen_p = generate(
        params, n_steps=STEPS_PER_CHORUS,
        seed=args.seed + 999,
        temperature=args.temperature, chord_temperature=0.0,
    )
    bom = bar_onset_chord_match(gen_c)
    cm = chord_progression_match(gen_c)
    obr = on_beat_note_rate(gen_p)
    ctr = chord_tone_rate(gen_c, gen_p)
    p4 = os.path.join(args.out_dir, "generated_pianoroll.png")
    title = (f"Free-running generation  —  bar-onset chord match {bom:.2f}, "
             f"step chord match {cm:.2f}, on-beat note rate {obr:.2f}, "
             f"chord-tone rate {ctr:.2f}")
    plot_pianoroll(gen_c, gen_p, p4, title=title)
    print(f"  wrote {p4}")

    print(f"[viz] done. final epoch-{history.epochs[-1]} "
          f"chord_acc={history.chord_acc[-1]:.3f}, "
          f"pitch_acc={history.pitch_acc[-1]:.3f}")


if __name__ == "__main__":
    main()
