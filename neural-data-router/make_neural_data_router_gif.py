"""Build neural_data_router.gif from per-eval snapshots in run.json.

Each frame shows, side by side, the per-depth accuracy for NDR and the
vanilla Transformer at one evaluation step. The headline visual is the
two distributions diverging on the test depths (5..7) as training
progresses.
"""
from __future__ import annotations

import io
import json
import os
from typing import Dict, List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def load_run(path: str = "run.json") -> Dict:
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


def main():
    s = load_run()
    snapshots: List[Dict] = s["snapshots"]
    # Group by step: each step has one NDR snap and one vanilla snap.
    by_step: Dict[int, Dict[str, Dict]] = {}
    for snap in snapshots:
        by_step.setdefault(snap["step"], {})[snap["kind"]] = snap

    steps = sorted(by_step.keys())
    depths = sorted(int(k) for k in next(iter(snapshots))["per_depth"].keys())

    frames = []
    for step in steps:
        cell = by_step[step]
        if "ndr" not in cell or "vanilla" not in cell:
            continue
        ndr_pd = cell["ndr"]["per_depth"]
        van_pd = cell["vanilla"]["per_depth"]
        ndr_vals = [ndr_pd[str(d)] for d in depths]
        van_vals = [van_pd[str(d)] for d in depths]

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        x = np.arange(len(depths))
        w = 0.4
        ax.bar(x - w / 2, ndr_vals, w, label="NDR", color="C0")
        ax.bar(x + w / 2, van_vals, w, label="Vanilla", color="C3")
        ax.axhline(0.25, color="grey", ls=":", alpha=0.6, label="chance")
        train_x = [i for i, d in enumerate(depths) if d <= 4]
        test_x = [i for i, d in enumerate(depths) if d >= 5]
        if train_x:
            ax.axvspan(min(train_x) - 0.5, max(train_x) + 0.5,
                       color="green", alpha=0.06)
        if test_x:
            ax.axvspan(min(test_x) - 0.5, max(test_x) + 0.5,
                       color="orange", alpha=0.10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in depths])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(f"Per-depth accuracy at step {step}\n"
                     f"NDR test = {cell['ndr']['test_acc']:.2f}    "
                     f"Vanilla test = {cell['vanilla']['test_acc']:.2f}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    if not frames:
        print("No paired snapshots found")
        return
    out = os.path.join(HERE, "neural_data_router.gif")
    # Hold the final frame for a couple of seconds.
    durations = [0.6] * (len(frames) - 1) + [2.0]
    imageio.mimsave(out, frames, duration=durations, loop=0)
    size_kb = os.path.getsize(out) // 1024
    print(f"saved {out}  ({len(frames)} frames, {size_kb} KB)")


if __name__ == "__main__":
    main()
