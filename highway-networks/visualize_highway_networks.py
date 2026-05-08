"""Static visualisations for highway-networks.

Reads run.json (single depth, with layer-T history) and run_sweep.json
(multiple depths) and writes PNGs to viz/:

  1. learning_curves.png         test accuracy over epochs, headline depth
  2. depth_sweep.png             final test acc vs depth, highway vs plain
  3. plain_loss_collapse.png     plain-net train loss flat at log(10)
                                 vs highway descent, headline depth
  4. T_gate_evolution.png        per-layer T-gate mean over training,
                                 headline depth
  5. T_gate_final.png            final per-layer T-gate mean, headline depth
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(run, out):
    h = run["runs"][-1]
    epochs = h["highway"]["history"]["epoch"]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(epochs, h["highway"]["history"]["test_acc"],
             "o-", color="#1f77b4", label=f"highway (depth {h['depth']})")
    plt.plot(epochs, h["plain"]["history"]["test_acc"],
             "s--", color="#d62728", label=f"plain (depth {h['depth']})")
    plt.axhline(0.10, color="grey", linewidth=0.8, linestyle=":", label="chance (1/10)")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.ylim(0, 1)
    plt.title(f"highway vs plain MLP at depth {h['depth']} (MNIST)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_loss_collapse(run, out):
    h = run["runs"][-1]
    epochs = h["highway"]["history"]["epoch"]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(epochs, h["highway"]["history"]["train_loss"],
             "o-", color="#1f77b4", label=f"highway (depth {h['depth']})")
    plt.plot(epochs, h["plain"]["history"]["train_loss"],
             "s--", color="#d62728", label=f"plain (depth {h['depth']})")
    plt.axhline(np.log(10), color="grey", linewidth=0.8, linestyle=":",
                label="log(10) ≈ chance loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("plain net stuck at chance loss; highway descends")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_depth_sweep(sweep, out):
    runs = sweep["runs"]
    depths = [r["depth"] for r in runs]
    hw = [r["highway"]["final_test_acc"] for r in runs]
    pl = [r["plain"]["final_test_acc"] for r in runs]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(depths, hw, "o-", color="#1f77b4", label="highway")
    plt.plot(depths, pl, "s--", color="#d62728", label="plain")
    plt.axhline(0.10, color="grey", linewidth=0.8, linestyle=":", label="chance")
    plt.xlabel("depth (hidden blocks)")
    plt.ylabel("final test accuracy")
    plt.ylim(0, 1)
    plt.xticks(depths)
    plt.title("highway nets train at every depth; plain nets fail past ~5-10")
    plt.legend(loc="center right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_T_gate_evolution(run, out):
    h = run["runs"][-1]["highway"]
    history = h["layer_T_history"]  # list of (epoch, [T per layer])
    if not history:
        return
    epochs = [ep for ep, _ in history]
    arr = np.array([T for _, T in history])  # (n_epochs, depth)
    n_ep, depth = arr.shape
    plt.figure(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    for li in range(depth):
        c = cmap(li / max(1, depth - 1))
        plt.plot(epochs, arr[:, li], color=c, alpha=0.85,
                 linewidth=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=depth - 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=plt.gca(), label="layer index (input → output)")
    plt.xlabel("epoch")
    plt.ylabel("mean(T_gate) over test batch")
    plt.title(f"T-gate per-layer evolution, depth {run['runs'][-1]['depth']}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_T_gate_final(run, out):
    h = run["runs"][-1]["highway"]
    final = h.get("final_T_per_layer") or []
    if not final:
        return
    plt.figure(figsize=(7.0, 4.0))
    layers = np.arange(len(final))
    plt.bar(layers, final, color="#1f77b4")
    plt.axhline(0.5, color="grey", linewidth=0.8, linestyle="--",
                label="T = 0.5 (50/50 mix)")
    init_T = 1.0 / (1.0 + np.exp(2.0))  # sigmoid(-2)
    plt.axhline(init_T, color="red", linewidth=0.8, linestyle=":",
                label=f"init T = sigmoid(-2) ≈ {init_T:.3f}")
    plt.xlabel("layer index (input → output)")
    plt.ylabel("mean(T_gate) on 1000 test inputs")
    plt.ylim(0, 1)
    plt.title(f"learned T-gate per layer, depth {run['runs'][-1]['depth']}")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    run = _load(os.path.join(HERE, "run.json"))
    sweep_path = os.path.join(HERE, "run_sweep.json")
    sweep = _load(sweep_path) if os.path.exists(sweep_path) else None

    plot_learning_curves(run, os.path.join(VIZ, "learning_curves.png"))
    plot_loss_collapse(run, os.path.join(VIZ, "plain_loss_collapse.png"))
    plot_T_gate_evolution(run, os.path.join(VIZ, "T_gate_evolution.png"))
    plot_T_gate_final(run, os.path.join(VIZ, "T_gate_final.png"))
    if sweep is not None:
        plot_depth_sweep(sweep, os.path.join(VIZ, "depth_sweep.png"))

    print(f"wrote PNGs to {VIZ}/")


if __name__ == "__main__":
    main()
