"""Build timit_blstm_ctc.gif: CTC alignment of a fixed sample as the BLSTM
trains.

Each frame shows:
  - top: input acoustic features (constant across frames)
  - middle: per-frame CTC posterior at the current iter (the alignment
            the network is choosing)
  - bottom: training-curve panel (PER on held-out batches over training,
            BLSTM vs uni-LSTM, with a vertical marker at the current iter)

Run from this folder:
    python3 make_timit_blstm_ctc_gif.py
"""

from __future__ import annotations

import io
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timit_blstm_ctc import (
    CorpusConfig,
    forward_model,
    init_model,
    make_batch,
    make_phoneme_signatures,
    train,
)


GIF_PATH = "timit_blstm_ctc.gif"


def render_frame(Xs, x_lens, labels, log_y, hist_b, hist_u,
                 cur_iter, vmin_logy=-5.0):
    """Compose one PIL/numpy frame."""
    T = int(x_lens[0])
    y = np.exp(log_y[:, 0, :])
    K_full = y.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.0),
                             gridspec_kw={"height_ratios": [1.0, 1.2, 1.0]})

    axes[0].imshow(Xs[:T, 0, :].T, aspect="auto", origin="lower",
                   cmap="viridis", vmin=-0.6, vmax=1.4)
    label_str = " ".join(str(int(x)) for x in labels[0])
    axes[0].set_title(f"Input features  (target labels: [{label_str}])",
                      fontsize=10)
    axes[0].set_ylabel("band")
    axes[0].set_xticks([])

    axes[1].imshow(y[:T].T, aspect="auto", origin="lower",
                   cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_ylabel("CTC class")
    axes[1].set_yticks(np.arange(K_full))
    axes[1].set_yticklabels(["blank"] + [f"phn {k}"
                                         for k in range(1, K_full)],
                             fontsize=8)
    axes[1].set_title(f"Per-frame CTC posterior at iter {cur_iter}",
                      fontsize=10)
    axes[1].set_xticks([])

    ax = axes[2]
    ax.plot(hist_b.iters, hist_b.eval_per, "C0-", lw=1.5, label="BLSTM")
    ax.plot(hist_u.iters, hist_u.eval_per, "C3-", lw=1.5, label="uni-LSTM")
    ax.axvline(cur_iter, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("training iters")
    ax.set_ylabel("PER")
    ax.set_ylim(-0.02, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=92)
    plt.close(fig)
    buf.seek(0)
    arr = imageio.imread(buf)
    return arr


def main(seed: int = 0, n_iters: int = 800, snapshot_every: int = 40):
    cfg = CorpusConfig()

    print(f"[gif] training BLSTM seed={seed} iters={n_iters} "
          f"snapshot_every={snapshot_every} ...")
    _, hist_b, snaps_b, _ = train(
        "blstm", seed, n_iters, batch_size=16, hidden=24, lr=3e-3,
        eval_every=snapshot_every, cfg=cfg, verbose=False,
        snapshot_every=snapshot_every,
    )
    print(f"  BLSTM final PER {hist_b.eval_per[-1]:.3f}, "
          f"{len(snaps_b)} snapshots")
    print(f"[gif] training uni-LSTM (for reference curve) ...")
    _, hist_u, _, _ = train(
        "uni", seed, n_iters, batch_size=16, hidden=24, lr=3e-3,
        eval_every=snapshot_every, cfg=cfg, verbose=False,
        snapshot_every=None,
    )
    print(f"  uni-LSTM final PER {hist_u.eval_per[-1]:.3f}")

    # Snapshots use a fixed 4-sample batch (same Xs across iters); we render
    # only sample 0 in each frame.
    frames = []
    for s in snaps_b:
        # Each snap stores Xs (T_max, 4, F), x_lens (4,), labels (list len 4),
        # log_y (T_max, 4, K_full).
        Xs = s["Xs"]
        x_lens = s["x_lens"]
        labels = s["labels"]
        log_y = s["log_y"]
        cur_iter = s["iter"]
        frames.append(render_frame(Xs, x_lens, labels, log_y,
                                   hist_b, hist_u, cur_iter))

    # Hold the last frame for ~1.5s.
    if frames:
        frames.extend([frames[-1]] * 6)

    print(f"[gif] writing {GIF_PATH}  ({len(frames)} frames)")
    # ~5 fps -> 0.2 s per frame.
    imageio.mimsave(GIF_PATH, frames, duration=0.20, loop=0)
    sz = os.path.getsize(GIF_PATH)
    print(f"  size = {sz/1024:.1f} KiB  ({sz/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
