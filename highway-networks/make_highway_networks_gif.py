"""Make highway_networks.gif: training-dynamics animation for the headline
30-layer run.

Each frame is a snapshot at a particular epoch and shows:
  * top  : test accuracy (highway vs plain) so far
  * bot. : per-layer mean(T_gate) for the highway net at that epoch

We re-run the headline experiment briefly to get per-epoch T snapshots
(the saved run.json already has them in `layer_T_history`). If
`run.json` is missing, we run main() with default settings first.

Output: highway_networks.gif (target ≤ 2 MB).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_run():
    p = os.path.join(HERE, "run.json")
    if not os.path.exists(p):
        print("run.json missing; running headline ...", flush=True)
        subprocess.check_call([sys.executable, os.path.join(HERE, "highway_networks.py"),
                               "--seed", "0"])
    with open(p) as f:
        return json.load(f)


def main():
    run = _ensure_run()
    headline = run["runs"][-1]
    depth = headline["depth"]
    hw = headline["highway"]
    pl = headline["plain"]

    epochs = hw["history"]["epoch"]
    hw_acc = hw["history"]["test_acc"]
    pl_acc = pl["history"]["test_acc"]
    layer_T_history = hw["layer_T_history"]  # [(ep, [T per layer])]
    if not layer_T_history:
        raise RuntimeError("run.json missing layer_T_history; rerun training")
    Tarr = np.array([T for _, T in layer_T_history])  # (n_ep, depth)

    n_frames = len(epochs)

    fig, (ax_acc, ax_T) = plt.subplots(2, 1, figsize=(7.4, 6.0),
                                        gridspec_kw={"height_ratios": [1.0, 1.1]})
    fig.suptitle(f"highway vs plain @ depth={depth} on MNIST",
                 fontsize=12)

    # Top: accuracy curves grow
    ax_acc.set_xlim(0.5, max(epochs) + 0.5)
    ax_acc.set_ylim(0, 1)
    ax_acc.axhline(0.10, color="grey", linestyle=":", linewidth=0.8,
                   label="chance")
    line_hw, = ax_acc.plot([], [], "o-", color="#1f77b4", label="highway")
    line_pl, = ax_acc.plot([], [], "s--", color="#d62728", label="plain")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(alpha=0.3)
    txt = ax_acc.text(0.02, 0.95, "", transform=ax_acc.transAxes,
                      verticalalignment="top", fontsize=10,
                      family="monospace")

    # Bottom: per-layer T-gate mean
    ax_T.set_xlim(-0.5, depth - 0.5)
    ax_T.set_ylim(0, 1)
    bars = ax_T.bar(np.arange(depth), np.zeros(depth), color="#1f77b4")
    init_T = 1.0 / (1.0 + np.exp(2.0))
    ax_T.axhline(init_T, color="red", linestyle=":", linewidth=0.8,
                 label=f"init T ≈ {init_T:.3f}")
    ax_T.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax_T.set_xlabel("layer index (input → output)")
    ax_T.set_ylabel("mean(T_gate)")
    ax_T.legend(loc="upper right")
    ax_T.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i):
        # Frame i corresponds to epoch i+1
        line_hw.set_data(epochs[: i + 1], hw_acc[: i + 1])
        line_pl.set_data(epochs[: i + 1], pl_acc[: i + 1])
        for bar, h in zip(bars, Tarr[i]):
            bar.set_height(float(h))
        txt.set_text(
            f"epoch {epochs[i]:>2d}\n"
            f"highway test  {hw_acc[i]:.3f}\n"
            f"plain   test  {pl_acc[i]:.3f}"
        )
        return [line_hw, line_pl, txt, *bars]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=400, blit=False)
    out = os.path.join(HERE, "highway_networks.gif")
    ani.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    size = os.path.getsize(out) / 1024.0
    print(f"wrote {out} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
