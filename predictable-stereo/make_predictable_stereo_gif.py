"""Build predictable_stereo.gif.

Animation of training: at each frame, show
  * left  : (yL, yR) scatter with each point colored by hidden depth z;
            two clusters should emerge as IMAX training proceeds
  * right : the IMAX I(yL; yR) curve up to this frame, plus the held-out
            recovery accuracy

We re-run training but capture model state at a logarithmic schedule of
epochs so the early dynamics (where most change happens) are dense and the
late dynamics (where everything is converged) are sparse.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from predictable_stereo import (
    ViewNet, make_stereo_dataset, imax_loss_and_grads,
    recovery_accuracy, agreement,
)


def make_gif(seed: int, out_path: str, n_epochs: int = 200, fps: int = 6):
    print(f"training (seed={seed}, n_epochs={n_epochs}) for GIF ...")
    n_samples = 1024
    n_eval = 1024
    d_shared = 8
    d_view = 8
    flip_p = 0.10
    d_hidden = 16
    lr = 0.03

    # Build train + eval (eval has fresh draws under the same world templates).
    data = make_stereo_dataset(
        n_samples=n_samples, d_shared=d_shared, d_view=d_view,
        flip_p=flip_p, seed=seed, shuffled=False,
    )
    rng_eval = np.random.default_rng(seed + 1_234_567)
    z_evalL = rng_eval.choice([-1.0, 1.0], size=n_eval)
    base_L = np.outer(z_evalL, data["template_L"])
    base_R = np.outer(z_evalL, data["template_R"])
    flips_L = rng_eval.random((n_eval, d_shared)) < flip_p
    flips_R = rng_eval.random((n_eval, d_shared)) < flip_p
    shared_L = base_L * np.where(flips_L, -1.0, 1.0)
    shared_R = base_R * np.where(flips_R, -1.0, 1.0)
    view_L = rng_eval.choice([-1.0, 1.0], size=(n_eval, d_view))
    view_R = rng_eval.choice([-1.0, 1.0], size=(n_eval, d_view))
    eval_x_L = np.concatenate([shared_L, view_L], axis=1)
    eval_x_R = np.concatenate([shared_R, view_R], axis=1)

    rng = np.random.default_rng(seed + 31_337)
    netL = ViewNet(d_shared + d_view, d_hidden, rng)
    netR = ViewNet(d_shared + d_view, d_hidden, rng)

    # Frame schedule: log-spaced for early dynamics + a few late frames.
    frame_epochs = sorted(set(
        list(range(0, 21))                                  # 0..20 every step
        + list(range(22, 41, 2))                             # 22..40 every 2
        + list(range(45, 81, 5))                             # 45..80 every 5
        + list(range(90, n_epochs + 1, 10))                  # 90.. every 10
    ))
    if 0 not in frame_epochs:
        frame_epochs = [0] + frame_epochs

    frames = []
    history_I = []
    history_acc = []
    history_ep = []

    # Frame 0 = before any updates.
    yL = netL.forward(data["x_L"]); yR = netR.forward(data["x_R"])
    yL_ev = netL.forward(eval_x_L); yR_ev = netR.forward(eval_x_R)
    _, _, _, info = imax_loss_and_grads(yL, yR)
    acc_ev = recovery_accuracy(yL_ev, z_evalL)
    history_ep.append(0); history_I.append(info["I_nats"]); history_acc.append(acc_ev)
    frames.append({
        "epoch": 0,
        "yL": yL.copy(), "yR": yR.copy(), "z": data["z"].copy(),
        "I": info["I_nats"], "acc_ev": acc_ev,
        "agree_ev": agreement(yL_ev, yR_ev),
    })

    next_frame_idx = 1
    for ep in range(1, n_epochs + 1):
        yL = netL.forward(data["x_L"]); yR = netR.forward(data["x_R"])
        loss, dyL, dyR, info = imax_loss_and_grads(yL, yR)
        gL = netL.backward(dyL); gR = netR.backward(dyR)
        netL.step_adam(gL, lr); netR.step_adam(gR, lr)

        if (next_frame_idx < len(frame_epochs)
                and ep == frame_epochs[next_frame_idx]):
            yL_post = netL.forward(data["x_L"]); yR_post = netR.forward(data["x_R"])
            yL_ev = netL.forward(eval_x_L); yR_ev = netR.forward(eval_x_R)
            _, _, _, info_post = imax_loss_and_grads(yL_post, yR_post)
            acc_ev = recovery_accuracy(yL_ev, z_evalL)
            history_ep.append(ep)
            history_I.append(info_post["I_nats"])
            history_acc.append(acc_ev)
            frames.append({
                "epoch": ep,
                "yL": yL_post.copy(), "yR": yR_post.copy(), "z": data["z"].copy(),
                "I": info_post["I_nats"], "acc_ev": acc_ev,
                "agree_ev": agreement(yL_ev, yR_ev),
            })
            next_frame_idx += 1

    print(f"  captured {len(frames)} frames")

    # Build the figure.
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_curves = fig.add_subplot(gs[0, 1])

    # Animation frames.
    def draw(i):
        f = frames[i]
        ax_scatter.clear()
        pos = f["z"] > 0; neg = ~pos
        ax_scatter.scatter(f["yL"][pos], f["yR"][pos], s=10, c="C3", alpha=0.55, label="z = +1")
        ax_scatter.scatter(f["yL"][neg], f["yR"][neg], s=10, c="C0", alpha=0.55, label="z = -1")
        ax_scatter.axhline(0, color="grey", lw=0.5); ax_scatter.axvline(0, color="grey", lw=0.5)
        ax_scatter.plot([-1, 1], [-1, 1], color="grey", lw=0.5, ls="--")
        ax_scatter.set_xlim(-1.08, 1.08); ax_scatter.set_ylim(-1.08, 1.08)
        ax_scatter.set_xlabel("y_L"); ax_scatter.set_ylabel("y_R")
        ax_scatter.set_title(f"epoch {f['epoch']:>3}: codes (yL, yR) by depth z")
        ax_scatter.legend(loc="upper left", fontsize=8)
        ax_scatter.grid(alpha=0.3)

        ax_curves.clear()
        eps = history_ep[: i + 1]
        Is = history_I[: i + 1]
        accs = history_acc[: i + 1]
        # Two y-axes: left = I, right = recovery acc. We'll just draw both
        # series on the same axis with a shared 0..max scale by normalizing.
        ax_curves.plot(eps, Is, color="C0", lw=2, label="I(yL;yR) [nats]")
        ax_curves.set_xlabel("epoch")
        ax_curves.set_ylabel("I(yL;yR) [nats]", color="C0")
        ax_curves.set_ylim(-0.5, max(8.0, max(Is) * 1.1) if Is else 8.0)
        ax_curves.tick_params(axis="y", labelcolor="C0")
        ax_curves.set_xlim(0, frame_epochs[-1] + 1)

        ax2 = ax_curves.twinx()
        ax2.plot(eps, accs, color="C2", lw=2, label="eval recovery acc")
        ax2.axhline(0.5, color="grey", ls="--", lw=0.7)
        ax2.set_ylim(0.4, 1.05)
        ax2.set_ylabel("eval recovery accuracy", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")

        ax_curves.set_title(
            f"I = {f['I']:.3f} nats  |  recov(eval) = {f['acc_ev']:.3f}  |  "
            f"agree(eval) = {f['agree_ev']:.3f}")
        ax_curves.grid(alpha=0.3)

        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000.0 / fps)
    writer = PillowWriter(fps=fps)
    fig.tight_layout()
    anim.save(out_path, writer=writer)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"wrote {out_path} ({size_kb:.0f} KB, {len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--out", default="predictable_stereo.gif")
    args = p.parse_args()
    make_gif(args.seed, args.out, n_epochs=args.n_epochs, fps=args.fps)


if __name__ == "__main__":
    main()
