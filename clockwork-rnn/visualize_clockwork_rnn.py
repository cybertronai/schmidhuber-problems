"""Static visualisations for clockwork-rnn.

Outputs to viz/:
    clock_schedule.png      - HEADLINE: per-group active-step heatmap
    target_vs_predicted.png - target waveform vs CW-RNN and vanilla output
    training_curves.png     - per-epoch MSE for CW-RNN and matched vanilla
    recurrent_mask.png      - block-lower-triangular structure of W_h
    group_activations.png   - hidden state per group over time (post-train)
    group_spectra.png       - FFT of each group's hidden state, slow-to-fast
    capacity_curve.png      - vanilla MSE plateau vs CW-RNN MSE across seeds
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from clockwork_rnn import (
    ClockworkRNN,
    VanillaRNN,
    eval_memorise,
    fixed_target,
    memorisation_inputs,
    run_headline,
    train_memorise,
    vanilla_hidden_dim_to_match,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--T", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    periods = tuple(2 ** g for g in range(args.groups))
    signal_periods = (8, 32, 80, 160)
    target = fixed_target(args.T, signal_periods, seed=args.seed)

    # Build, train both models.
    cw = ClockworkRNN(in_dim=1, hidden_dim=args.hidden, out_dim=1,
                      n_groups=args.groups, periods=periods, seed=args.seed)
    nv = vanilla_hidden_dim_to_match(cw)
    vanilla = VanillaRNN(in_dim=1, hidden_dim=nv, out_dim=1, seed=args.seed + 1)
    cw_losses = train_memorise(cw, target, n_epochs=args.epochs, lr=args.lr)
    vanilla_losses = train_memorise(vanilla, target, n_epochs=args.epochs, lr=args.lr)

    # ----- 1. Clock schedule heatmap (HEADLINE) ----------------------------
    active = cw.active_groups(args.T).T  # (G, T)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.imshow(active.astype(float), aspect="auto", cmap="Greys",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(args.groups))
    ax.set_yticklabels([f"g={g} T={cw.periods[g]}" for g in range(args.groups)])
    ax.set_xlabel("timestep t")
    ax.set_title("Clockwork RNN — per-group active timesteps "
                 "(black = group updates, white = carries previous state)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "clock_schedule.png"), dpi=120)
    plt.close()

    # ----- 2. Target vs predicted ------------------------------------------
    X = memorisation_inputs(args.T)
    cw_out, _ = cw.forward(X)
    vn_out, _ = vanilla.forward(X)
    cw_mse = eval_memorise(cw, target)
    vn_mse = eval_memorise(vanilla, target)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    t = np.arange(args.T)
    ax.plot(t, target, color="black", label="target", linewidth=1.5)
    ax.plot(t, cw_out[:, 0], color="C0", label=f"CW-RNN (MSE {cw_mse:.4f})")
    ax.plot(t, vn_out[:, 0], color="C3",
            label=f"vanilla RNN, matched params (MSE {vn_mse:.4f})")
    ax.set_xlabel("timestep")
    ax.set_ylabel("y(t)")
    ax.set_title(f"Memorised waveform: CW-RNN ({cw.n_params()} params) vs "
                 f"vanilla RNN ({vanilla.n_params()} params)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "target_vs_predicted.png"), dpi=120)
    plt.close()

    # ----- 3. Training curves ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cw_losses, label="CW-RNN", color="C0")
    ax.plot(vanilla_losses, label="vanilla RNN (matched params)", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Training curves on multi-rate waveform memorisation")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=120)
    plt.close()

    # ----- 4. Recurrent mask -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].imshow(cw.mask_h, cmap="Greys", interpolation="nearest")
    axes[0].set_title(f"CW-RNN W_h non-zero pattern\n"
                      f"({int(cw.mask_h.sum())} of {cw.N * cw.N} entries, "
                      f"{int(cw.mask_h.sum()) * 100 // (cw.N * cw.N)}%)")
    for g in range(args.groups + 1):
        axes[0].axhline(g * cw.M - 0.5, color="C0", linewidth=0.5)
        axes[0].axvline(g * cw.M - 0.5, color="C0", linewidth=0.5)
    axes[0].set_xlabel("col group (read from)")
    axes[0].set_ylabel("row group (write to)")

    axes[1].imshow(cw.W_h, cmap="RdBu_r", interpolation="nearest",
                   vmin=-np.abs(cw.W_h).max(), vmax=np.abs(cw.W_h).max())
    axes[1].set_title("Learned W_h (CW-RNN, post-training)")
    for g in range(args.groups + 1):
        axes[1].axhline(g * cw.M - 0.5, color="black", linewidth=0.5, alpha=0.3)
        axes[1].axvline(g * cw.M - 0.5, color="black", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "recurrent_mask.png"), dpi=120)
    plt.close()

    # ----- 5. Per-group hidden activations ---------------------------------
    _, cache = cw.forward(X)
    H = cache["h"][1:]  # (T, N)
    fig, axes = plt.subplots(args.groups, 1, figsize=(10, 1.0 * args.groups),
                             sharex=True)
    if args.groups == 1:
        axes = [axes]
    for g in range(args.groups):
        block = H[:, g * cw.M:(g + 1) * cw.M]
        # Plot mean activation across the block, plus a shaded band of std.
        m = block.mean(axis=1)
        s = block.std(axis=1)
        axes[g].fill_between(np.arange(args.T), m - s, m + s,
                             color=f"C{g % 10}", alpha=0.25)
        axes[g].plot(m, color=f"C{g % 10}", linewidth=1.0)
        axes[g].set_ylabel(f"T={cw.periods[g]}", rotation=0, ha="right",
                           va="center")
        axes[g].set_yticks([])
        axes[g].axhline(0, color="grey", linewidth=0.4, alpha=0.5)
    axes[-1].set_xlabel("timestep t")
    axes[0].set_title("Per-group hidden activations over time "
                      "(mean ± std across the M=8 units in each group)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "group_activations.png"), dpi=120)
    plt.close()

    # ----- 6. Per-group power spectra --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for g in range(args.groups):
        block = H[:, g * cw.M:(g + 1) * cw.M].mean(axis=1)
        f = np.fft.rfftfreq(args.T)
        P = np.abs(np.fft.rfft(block - block.mean())) ** 2
        # Skip the DC bin for visualisation.
        ax.semilogy(f[1:], P[1:] + 1e-12, label=f"g={g} T={cw.periods[g]}",
                    color=f"C{g % 10}")
    ax.set_xlabel("frequency (cycles per timestep)")
    ax.set_ylabel("power spectral density")
    ax.set_title("Per-group hidden-state spectra "
                 "(slow clocks concentrate power at low frequencies)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "group_spectra.png"), dpi=120)
    plt.close()

    # ----- 7. Multi-seed capacity curve ------------------------------------
    cw_mses, vn_mses = [], []
    for s in range(5):
        r = run_headline(seed=s, n_epochs=args.epochs, T=args.T,
                         hidden_dim=args.hidden, n_groups=args.groups,
                         lr=args.lr)
        cw_mses.append(r["cw_mse"])
        vn_mses.append(r["vanilla_mse"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    seeds = np.arange(5)
    ax.bar(seeds - width / 2, cw_mses, width, color="C0", label="CW-RNN")
    ax.bar(seeds + width / 2, vn_mses, width, color="C3",
           label="vanilla RNN (matched)")
    for i, (a, b) in enumerate(zip(cw_mses, vn_mses)):
        ax.text(i, max(a, b) + 0.005, f"{b / a:.1f}×",
                ha="center", fontsize=9)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("final MSE on memorised waveform")
    ax.set_title("Multi-seed: CW-RNN consistently below the vanilla plateau")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "capacity_curve.png"), dpi=120)
    plt.close()

    print(f"wrote {len(os.listdir(args.outdir))} PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
