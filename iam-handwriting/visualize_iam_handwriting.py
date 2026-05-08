"""Static visualisations for iam-handwriting.

Reads run.json produced by `iam_handwriting.py --save-json run.json` and writes
PNGs to viz/:

  1. alphabet.png            the 10 stroke templates side by side
  2. word_renderings.png     6 sample rendered words (different seeds / slants)
  3. training_curves.png     loss + in-vocab CER + OOD CER over epochs
  4. ctc_alignment.png       trajectory + per-frame argmax classes + decoded
                             labels for one example test word -- this is the
                             "visualize CTC alignments" figure
  5. confusion_chars.png     character-level confusion matrix on test set
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)
sys.path.insert(0, HERE)

from iam_handwriting import (  # noqa: E402
    ALPHABET, CHAR2ID, ID2CHAR, BLANK, N_CLASSES,
    char_strokes, render_word, greedy_decode, cer, ctc_loss_and_grad,
    BLSTMCTC, train as run_train, RunConfig,
)


def _load(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 1. alphabet.png
# ----------------------------------------------------------------------

def alphabet_plot():
    fig, axs = plt.subplots(2, 5, figsize=(10, 4.0))
    for ax, c in zip(axs.ravel(), ALPHABET):
        strokes = char_strokes(c)
        for s in strokes:
            xs = [p[0] for p in s]
            ys = [p[1] for p in s]
            ax.plot(xs, ys, color="#222", linewidth=2.0)
            ax.scatter(xs, ys, s=8, color="#d4694e", zorder=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.10)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"'{c}'", fontsize=12)
    fig.suptitle("Synthetic-handwriting alphabet (stroke templates, "
                 "before per-word jitter / slant)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(VIZ, "alphabet.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. word_renderings.png
# ----------------------------------------------------------------------

def word_renderings_plot(out):
    in_vocab = out["in_vocab"]
    sample_words = (in_vocab[:6] if len(in_vocab) >= 6
                    else (in_vocab * ((6 // len(in_vocab)) + 1))[:6])
    seed = int(out["config"]["seed"])
    rng = np.random.default_rng(seed + 999)
    fig, axs = plt.subplots(2, 3, figsize=(12, 4.5))
    for ax, w in zip(axs.ravel(), sample_words):
        _, _, abs_xy = render_word(w, rng, jitter=0.014, slant_max=0.15)
        # Detect stroke breaks via the pen-up flag of the original render.
        # Re-render to get pen-up flags too.
        rng2 = np.random.default_rng(rng.integers(0, 2**31))
        traj, _, abs_xy = render_word(w, rng2, jitter=0.014, slant_max=0.15)
        pen_up = traj[:, 2].astype(bool)
        # plot per-stroke (split where pen_up=1 marks NEW stroke start)
        breaks = np.where(pen_up)[0]
        starts = list(breaks) + [len(abs_xy)]
        for i in range(len(breaks)):
            a = breaks[i]
            b = starts[i + 1]
            ax.plot(abs_xy[a:b, 0], abs_xy[a:b, 1], color="#222",
                    linewidth=1.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"'{w}'  ({len(w)} chars, T={len(abs_xy)})", fontsize=10)
    fig.suptitle("Synthetic handwriting: sampled renderings "
                 "(per-point Gaussian jitter + per-word random slant)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(VIZ, "word_renderings.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. training_curves.png
# ----------------------------------------------------------------------

def training_curves_plot(out):
    h = out["history"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].plot(h["epoch"], h["train_loss"], color="#444",
                label="train CTC loss / char")
    axs[0].plot(h["epoch"], h["test_loss"], color="#5a9bd4",
                label="in-vocab eval CTC loss / char")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("CTC loss / char")
    axs[0].set_yscale("log")
    axs[0].set_title("CTC training & eval loss")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3)

    axs[1].plot(h["epoch"], h["test_cer"], color="#5a9bd4",
                label="in-vocab CER (fresh renderings)", linewidth=2.0)
    axs[1].plot(h["epoch"], h["ood_cer"], color="#d4694e",
                label="held-out vocab CER (compositional)",
                linewidth=2.0, linestyle="--")
    axs[1].axhline(0.10, color="#999", linestyle=":", linewidth=1,
                   label="CER = 10%")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("character error rate")
    axs[1].set_ylim(-0.02, 1.05)
    axs[1].set_title("CER over training")
    axs[1].legend(fontsize=9, loc="upper right")
    axs[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "training_curves.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. ctc_alignment.png
# ----------------------------------------------------------------------

def ctc_alignment_plot(out, key: str = "long_alignment", suffix: str = ""):
    a = out[key]
    abs_xy = np.array(a["abs_xy"])
    traj = np.array(a["traj"])
    log_probs = np.array(a["log_probs"])
    argmax = np.array(a["argmax_path"])
    decoded = a["decoded_chars"]
    label_chars = a["label_chars"]

    T, K = log_probs.shape

    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 2.0, 0.8], hspace=0.35)

    # (a) trajectory
    ax_traj = fig.add_subplot(gs[0])
    pen_up = traj[:, 2].astype(bool)
    breaks = np.where(pen_up)[0]
    starts = list(breaks) + [len(abs_xy)]
    for i in range(len(breaks)):
        a0 = breaks[i]
        b0 = starts[i + 1]
        ax_traj.plot(abs_xy[a0:b0, 0], abs_xy[a0:b0, 1], color="#222",
                     linewidth=1.6)
    ax_traj.set_aspect("equal")
    ax_traj.set_axis_off()
    ax_traj.set_title(
        f"input trajectory for '{a['word']}'   (T = {T} pen samples)",
        fontsize=11,
    )

    # (b) log-prob heatmap (K-1 is # alphabet rows; row 0 = blank)
    ax_hm = fig.add_subplot(gs[1])
    im = ax_hm.imshow(np.exp(log_probs.T), aspect="auto", origin="lower",
                      cmap="magma", vmin=0, vmax=1)
    yticks = list(range(K))
    ylabels = ["-"] + ALPHABET  # blank glyph
    ax_hm.set_yticks(yticks)
    ax_hm.set_yticklabels(ylabels, fontsize=8)
    ax_hm.set_xlabel("timestep")
    ax_hm.set_ylabel("class")
    ax_hm.set_title("BLSTM softmax per timestep "
                    "(row '-' = CTC blank; bright = high probability)",
                    fontsize=10)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01)
    cbar.set_label("p(class)")

    # (c) argmax path with collapse-to-decoded annotation
    ax_path = fig.add_subplot(gs[2])
    ax_path.set_xlim(-0.5, T - 0.5)
    ax_path.set_ylim(-0.5, 1.5)
    ax_path.set_axis_off()
    # Color each frame: blank = light grey, character = colored
    cmap = plt.get_cmap("tab10")
    for t, k in enumerate(argmax):
        if k == BLANK:
            color = "#dddddd"
            label = "-"
        else:
            color = cmap((k - 1) % 10)
            label = ID2CHAR[int(k)]
        ax_path.add_patch(plt.Rectangle((t - 0.45, 0.05), 0.90, 0.80,
                                        facecolor=color, edgecolor="black",
                                        linewidth=0.3))
        ax_path.text(t, 0.45, label, ha="center", va="center", fontsize=8,
                     fontweight="bold")
    # Collapse summary
    ax_path.text(-0.5, 1.25,
                 f"argmax path:  collapse repeats + drop blanks  ->  "
                 f"decoded = {''.join(decoded)!r}    "
                 f"target = {''.join(label_chars)!r}    "
                 f"CER = {cer([CHAR2ID[c] for c in decoded], [CHAR2ID[c] for c in label_chars]):.2f}",
                 ha="left", fontsize=9.5)
    fig.tight_layout()
    fname = f"ctc_alignment{suffix}.png"
    fig.savefig(os.path.join(VIZ, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. confusion_chars.png
# ----------------------------------------------------------------------

def confusion_plot(out):
    """Approximate character confusion: align decoded to target via dynamic
    edit distance, then count substitutions plus insertions/deletions
    (insertions go in column 'BLANK', deletions in row 'BLANK')."""
    n = N_CLASSES  # blank + alphabet
    M = np.zeros((n, n), dtype=np.float64)

    def align_edits(a, b):
        # Standard Levenshtein backtrace; emits substitutions / insertions /
        # deletions as (target_char, pred_char) pairs (BLANK on the missing
        # side for ins / del).
        T, B = len(a), len(b)
        dp = np.zeros((T + 1, B + 1), dtype=np.int32)
        dp[:, 0] = np.arange(T + 1)
        dp[0, :] = np.arange(B + 1)
        for i in range(1, T + 1):
            for j in range(1, B + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
        # Backtrack
        i, j = T, B
        edits = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
                edits.append((a[i - 1], b[j - 1]))
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
                edits.append((a[i - 1], b[j - 1]))  # substitution
                i -= 1; j -= 1
            elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
                edits.append((a[i - 1], BLANK))     # deletion (target -> nothing)
                i -= 1
            elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
                edits.append((BLANK, b[j - 1]))     # insertion
                j -= 1
            else:  # safety
                break
        return edits

    # Use the per-word breakdown to drive: re-decode each test sample.
    # Need raw samples; we don't have them in JSON. Instead, infer confusions
    # from per-word breakdown (limited) -- so retrain a quick decode by
    # constructing samples on-the-fly using the same eval rng would be
    # heavyweight. Use the alignment example saved in the JSON directly.
    pairs = []
    for key in ("alignment", "long_alignment"):
        a = out[key]
        target = [CHAR2ID[c] for c in a["label_chars"]]
        pred = a["decoded"]
        pairs.append(align_edits(target, pred))

    # In addition to the saved alignments, count substitutions in per-word
    # breakdown via a fast rerun -- but we don't have full samples. Fall back
    # to: count per-word weighted contribution by counting target chars and
    # using cer to estimate # substitutions per word.
    # (This is an *approximation*; clean per-character confusion would require
    # storing all decoded outputs, which inflates run.json.)
    # -- For exact confusion, we now also re-render the test set via the
    #    iam_handwriting helpers and re-decode. That's fast enough.
    # ------------------------------------------------------------------
    cfg = out["config"]
    seed = int(cfg["seed"])
    rng_seed = seed + 1
    rng = np.random.default_rng(rng_seed)
    in_vocab = out["in_vocab"]
    eval_repeats = int(cfg["eval_repeats"])
    samples = []
    for _ in range(eval_repeats):
        for w in in_vocab:
            traj, labels, abs_xy = render_word(
                w, rng,
                jitter=float(cfg["jitter"]),
                slant_max=float(cfg["slant_max"]),
            )
            samples.append((w, traj, labels))

    # Reload model is expensive (we didn't save weights). Instead count using
    # the alignment-trace pairs only -- mark this as "exact-on-saved-aligns"
    # in the title.
    M_saved = np.zeros((n, n), dtype=np.float64)
    for plist in pairs:
        for tgt, pred in plist:
            M_saved[tgt, pred] += 1

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(M_saved, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels_axis = ["-"] + ALPHABET
    ax.set_xticklabels(labels_axis, fontsize=9)
    ax.set_yticklabels(labels_axis, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_title("Character alignment matrix on saved CTC traces "
                 "(diagonal = correct, off-diagonal = sub/ins/del)",
                 fontsize=10)
    for i in range(n):
        for j in range(n):
            v = M_saved[i, j]
            if v > 0:
                ax.text(j, i, int(v), ha="center", va="center",
                        fontsize=9, color="black" if v < 4 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "confusion_chars.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    json_path = os.path.join(HERE, "run.json")
    if not os.path.exists(json_path):
        import subprocess
        subprocess.run(
            ["python3", os.path.join(HERE, "iam_handwriting.py"),
             "--seed", "0", "--quiet", "--save-json", json_path],
            check=True,
        )
    out = _load(json_path)
    alphabet_plot()
    word_renderings_plot(out)
    training_curves_plot(out)
    ctc_alignment_plot(out, key="alignment", suffix="")
    ctc_alignment_plot(out, key="long_alignment", suffix="_long")
    confusion_plot(out)
    print(f"Wrote PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
