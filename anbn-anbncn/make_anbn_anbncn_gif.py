"""Animation: cell state on a^15 b^15 forming as training proceeds.

Shows how the LSTM's two cells gradually learn to act as a counter — one cell
charges up while reading a's, and discharges while reading b's, crossing the
"start of T-window" threshold exactly when n equals the count of a's. This
is the operational picture behind the headline.

The GIF uses the anbn (context-free) language with hidden=2 because two
cells are enough and the picture is the cleanest. Frames are taken at
geometrically spaced training checkpoints, with the final frame showing the
solved counter behaviour.

Run: python3 make_anbn_anbncn_gif.py --seed 1
Output: anbn_anbncn.gif (in the script directory)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from anbn_anbncn import (
    train, lstm_forward, dict_to_lstm,
    make_anbn, ANBN_VOCAB,
)


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=15,
                    help="length n of the test sequence we animate")
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--out", default="anbn_anbncn.gif")
    args = ap.parse_args()

    history: list = []
    p_final, stats = train(
        lang="anbn", hidden=2, n_train_max=args.n_train, n_test=args.n_eval + 5,
        n_steps=args.steps, lr=0.01, seed=args.seed,
        log_every=200, history=history,
    )
    print(f"  final max_run = {stats['final_eval']['max_run']}, snapshots = {len(history)}")

    # Pick ~12 frames roughly geometrically spaced across the training history.
    n_frames = min(12, len(history))
    if n_frames == 0:
        raise SystemExit("no training history captured")
    if len(history) > n_frames:
        idx = np.linspace(0, len(history) - 1, n_frames).astype(int)
        snaps = [history[i] for i in idx]
    else:
        snaps = history

    inp, _ = make_anbn(args.n_eval)
    Tlen = inp.shape[0]
    xticks = [ANBN_VOCAB[int(np.argmax(inp[t]))] for t in range(Tlen)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    def draw(idx_frame: int):
        ax.clear()
        snap = snaps[idx_frame]
        from anbn_anbncn import LSTMParams
        p = dict_to_lstm(snap["params"])
        cache = lstm_forward(p, inp)
        c_seq = cache["c"]
        for h in range(c_seq.shape[1]):
            ax.plot(c_seq[:, h], "-o", markersize=4, label=f"cell {h}")
        ax.axvspan(0, 0.5, alpha=0.15, color="grey")
        ax.axvspan(0.5, args.n_eval + 0.5, alpha=0.10, color="tab:green")     # a's
        ax.axvspan(args.n_eval + 0.5, 2 * args.n_eval + 0.5, alpha=0.10, color="tab:orange")  # b's
        ax.set_xticks(np.arange(Tlen))
        ax.set_xticklabels(xticks, fontsize=8)
        ax.set_xlabel(f"input symbols (a^{args.n_eval} b^{args.n_eval})")
        ax.set_ylabel("cell value c_t")
        ax.set_title(
            f"step {snap['step']} / {stats['steps_run']}  •  "
            f"max accept-run from n=1: {snap['max_run']}  "
            f"(trained on n≤{args.n_train})"
        )
        ax.set_ylim(-2.5, 2.5)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    ani = anim.FuncAnimation(fig, draw, frames=len(snaps), interval=600, blit=False)
    out = os.path.join(HERE, args.out)
    ani.save(out, writer=anim.PillowWriter(fps=2))
    plt.close(fig)
    sz = os.path.getsize(out)
    print(f"Wrote {out} ({sz/1024:.0f} KB, {len(snaps)} frames)")


if __name__ == "__main__":
    main()
