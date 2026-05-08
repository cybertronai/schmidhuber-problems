"""Static visualisations for timit-blstm-ctc.

Produces viz/*.png:
  - corpus_signatures.png   : phoneme spectral signatures (early vs late)
  - corpus_sample.png       : example sequences with labels overlaid
  - training_curves.png     : NLL + PER + seq_acc, BLSTM vs uni-LSTM
  - ctc_alignment.png       : per-frame phoneme posterior + CTC alignment for one sample
  - weight_matrices.png     : input-to-gate matrices of fwd / bwd LSTM + output projection

Run from this folder:
    python3 visualize_timit_blstm_ctc.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timit_blstm_ctc import (
    CorpusConfig,
    forward_model,
    make_batch,
    make_phoneme_signatures,
    render_phoneme,
    render_silence,
    train,
)

VIZ_DIR = "viz"


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_corpus_signatures(cfg: CorpusConfig, sig: dict, outpath: str):
    K = cfg.n_phonemes
    F = cfg.n_features
    fig, axes = plt.subplots(2, K, figsize=(2.0 * K, 4.6), sharey=True)
    for k in range(K):
        ax_e = axes[0, k]
        ax_l = axes[1, k]
        ax_e.bar(np.arange(F), sig["early_centers"][k], color="C2", alpha=0.85)
        ax_l.bar(np.arange(F), sig["late_centers"][k], color="C3", alpha=0.85)
        ax_e.set_title(f"phoneme {k+1}", fontsize=10)
        ax_e.set_xticks(np.arange(F))
        ax_l.set_xticks(np.arange(F))
        if k == 0:
            ax_e.set_ylabel("early\n(onset)\nformants", fontsize=9)
            ax_l.set_ylabel("late\n(distinguishing)\nformants", fontsize=9)
        ax_e.grid(alpha=0.3, axis="y")
        ax_l.grid(alpha=0.3, axis="y")
    fig.suptitle("Phoneme spectral signatures: shared onset (top) vs distinguishing payload (bottom)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_corpus_sample_with_boundaries(cfg: CorpusConfig, sig: dict,
                                       outpath: str, seed: int = 12345):
    """Render 3 sequences and annotate phoneme boundaries."""
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.0))
    for ax_idx, ax in enumerate(axes):
        # Manual rendering so we know boundaries.
        L = rng.randint(cfg.min_phonemes_per_seq, cfg.max_phonemes_per_seq + 1)
        pieces = []
        boundaries = []
        labels = []
        if rng.uniform() < 0.5:
            n = rng.randint(cfg.min_silence_frames, cfg.max_silence_frames + 1)
            pieces.append(render_silence(n, cfg, rng))
        for i in range(L):
            k = rng.randint(0, cfg.n_phonemes)
            n = rng.randint(cfg.min_frames_per_phoneme,
                            cfg.max_frames_per_phoneme + 1)
            start = sum(p.shape[0] for p in pieces)
            pieces.append(render_phoneme(k, n, sig, cfg, rng))
            end = start + n
            boundaries.append((start, end, k + 1))
            labels.append(k + 1)
            if i < L - 1:
                n_sil = rng.randint(cfg.min_silence_frames,
                                    cfg.max_silence_frames + 1)
                pieces.append(render_silence(n_sil, cfg, rng))
        if rng.uniform() < 0.5:
            n = rng.randint(cfg.min_silence_frames, cfg.max_silence_frames + 1)
            pieces.append(render_silence(n, cfg, rng))
        X = np.concatenate(pieces, axis=0)
        T = X.shape[0]
        im = ax.imshow(X.T, aspect="auto", origin="lower", cmap="viridis",
                       vmin=-0.6, vmax=1.4, extent=[-0.5, T - 0.5, -0.5,
                                                    cfg.n_features - 0.5])
        for (s, e, lab) in boundaries:
            ax.axvline(s - 0.5, color="white", lw=0.6, alpha=0.7)
            ax.axvline(e - 0.5, color="white", lw=0.6, alpha=0.7)
            ax.text((s + e) / 2 - 0.5, cfg.n_features - 0.4, str(lab),
                    color="white", fontsize=11, ha="center", va="top",
                    weight="bold")
        ax.set_ylabel(f"seq {ax_idx+1}\nband")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_yticks(np.arange(cfg.n_features))
        ax.set_xticks([])
    axes[-1].set_xlabel("time (frames)")
    fig.suptitle("Synthetic phoneme corpus -- 3 sequences with phoneme boundaries (white) and labels",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_training_curves(blstm_h, uni_h, outpath: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    # NLL
    ax = axes[0]
    ax.plot(blstm_h.iters, blstm_h.nll, "C0-", lw=1.6, label="BLSTM")
    ax.plot(uni_h.iters, uni_h.nll, "C3-", lw=1.6, label="uni-LSTM")
    ax.set_yscale("log")
    ax.set_xlabel("training iters")
    ax.set_ylabel("CTC NLL  (log scale)")
    ax.set_title("CTC negative log-likelihood")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    # PER
    ax = axes[1]
    ax.plot(blstm_h.iters, blstm_h.eval_per, "C0-", lw=1.6, label="BLSTM")
    ax.plot(uni_h.iters, uni_h.eval_per, "C3-", lw=1.6, label="uni-LSTM")
    ax.set_xlabel("training iters")
    ax.set_ylabel("phoneme error rate (greedy decode)")
    ax.set_title("PER on held-out batches")
    ax.set_ylim(-0.02, 1.10)
    ax.grid(alpha=0.3)
    ax.legend()
    # Sequence accuracy
    ax = axes[2]
    ax.plot(blstm_h.iters, blstm_h.eval_seq_acc, "C0-", lw=1.6, label="BLSTM")
    ax.plot(uni_h.iters, uni_h.eval_seq_acc, "C3-", lw=1.6, label="uni-LSTM")
    ax.set_xlabel("training iters")
    ax.set_ylabel("fraction of sequences with PER = 0")
    ax.set_title("Sequence-exact accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_ctc_alignment(model, cfg: CorpusConfig, sig: dict, outpath: str,
                       seed: int = 4242):
    rng = np.random.RandomState(seed)
    X, x_lens, labels_list, l_lens = make_batch(cfg, sig, rng, 1)
    log_y, _ = forward_model(model, X, x_lens)
    y = np.exp(log_y[:, 0, :])  # (T, K_full)
    T = int(x_lens[0])
    labels = labels_list[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5),
                             gridspec_kw={"height_ratios": [1.0, 1.4]})
    axes[0].imshow(X[:T, 0, :].T, aspect="auto", origin="lower",
                   cmap="viridis", vmin=-0.6, vmax=1.4)
    axes[0].set_ylabel("acoustic\nfeature band")
    label_str = " ".join(str(int(x)) for x in labels)
    axes[0].set_title(f"Input acoustic features  (label sequence: [{label_str}])")
    axes[0].set_xticks([])
    K_full = y.shape[1]
    axes[1].imshow(y[:T].T, aspect="auto", origin="lower", cmap="magma",
                   vmin=0.0, vmax=1.0)
    axes[1].set_xlabel("time (frames)")
    axes[1].set_ylabel("CTC output class")
    axes[1].set_yticks(np.arange(K_full))
    axes[1].set_yticklabels(["blank"] + [f"phn {k}"
                                         for k in range(1, K_full)])
    axes[1].set_title(
        "Per-frame CTC posterior (rows = blank + phonemes; columns = time)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_weight_matrices(model, outpath: str):
    """Draw input-to-gate weights of fwd LSTM + bwd LSTM + projection.

    Uses a uniform color range so the panels are comparable.
    """
    fwd_Wx = model.fwd.Wx
    bwd_Wx = model.bwd.Wx if model.bidirectional else None
    Wy = model.Wy
    H = model.fwd.Wh.shape[0]

    panels = [("fwd LSTM Wx (input -> gates)", fwd_Wx)]
    if bwd_Wx is not None:
        panels.append(("bwd LSTM Wx (input -> gates)", bwd_Wx))
    panels.append(("output projection Wy", Wy))

    vmax = max(float(np.abs(M).max()) for _, M in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, M) in zip(axes, panels):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("output unit")
        ax.set_ylabel("input dim")
        # For Wx, mark gate-block boundaries (i, f, g, o) on the x axis.
        if name.endswith("(input -> gates)"):
            for k in range(1, 4):
                ax.axvline(k * H - 0.5, color="k", lw=0.6, alpha=0.5)
            ax.set_xticks([k * H + H / 2 - 0.5 for k in range(4)])
            ax.set_xticklabels(["i", "f", "g", "o"])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main(seed: int = 0, n_iters: int = 1500):
    _ensure_dir(VIZ_DIR)
    cfg = CorpusConfig()
    rng = np.random.RandomState(seed)
    sig = make_phoneme_signatures(cfg, rng)

    print("[viz] writing corpus_signatures.png")
    plot_corpus_signatures(cfg, sig,
                           os.path.join(VIZ_DIR, "corpus_signatures.png"))
    print("[viz] writing corpus_sample.png")
    plot_corpus_sample_with_boundaries(
        cfg, sig, os.path.join(VIZ_DIR, "corpus_sample.png"))

    print(f"[viz] training BLSTM seed={seed} iters={n_iters} ...")
    model_b, hist_b, _, sig_b = train("blstm", seed, n_iters, batch_size=16,
                                      hidden=24, lr=3e-3, eval_every=100,
                                      cfg=cfg, verbose=False,
                                      snapshot_every=None)
    print(f"  BLSTM final PER: {hist_b.eval_per[-1]:.3f}")
    print(f"[viz] training uni-LSTM seed={seed} iters={n_iters} ...")
    model_u, hist_u, _, _ = train("uni", seed, n_iters, batch_size=16,
                                  hidden=24, lr=3e-3, eval_every=100,
                                  cfg=cfg, verbose=False,
                                  snapshot_every=None)
    print(f"  uni-LSTM final PER: {hist_u.eval_per[-1]:.3f}")

    print("[viz] writing training_curves.png")
    plot_training_curves(hist_b, hist_u,
                         os.path.join(VIZ_DIR, "training_curves.png"))
    print("[viz] writing ctc_alignment.png")
    plot_ctc_alignment(model_b, cfg, sig_b,
                       os.path.join(VIZ_DIR, "ctc_alignment.png"))
    print("[viz] writing weight_matrices.png")
    plot_weight_matrices(model_b,
                         os.path.join(VIZ_DIR, "weight_matrices.png"))
    print(f"[viz] all PNGs written to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
