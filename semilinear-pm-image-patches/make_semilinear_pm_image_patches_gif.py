"""Render semilinear_pm_image_patches.gif: filter atlas evolving over PM training.

Frames are sampled at log-spaced training steps. Each frame shows the
encoder's M filter rows reshaped as patch-shaped atlases and rendered
side-by-side with the per-step predictability loss curve.
"""

from __future__ import annotations

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from semilinear_pm_image_patches import train


def imageio_or_pillow():
    try:
        import imageio.v2 as imageio
        return ("imageio", imageio)
    except Exception:
        from PIL import Image
        return ("pillow", Image)


def fig_to_rgb(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert("RGB")
    return np.asarray(img)


def filter_atlas(W: np.ndarray, patch_size: int, ncols: int = None) -> np.ndarray:
    M = W.shape[0]
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(M)))
    nrows = int(np.ceil(M / ncols))
    pad = 1
    cell = patch_size + pad
    out = np.full((nrows * cell + pad, ncols * cell + pad), 0.5)
    for i in range(M):
        f = W[i].reshape(patch_size, patch_size)
        f = f / (np.max(np.abs(f)) + 1e-12)
        f = (f + 1.0) / 2.0
        r, c = divmod(i, ncols)
        y0 = pad + r * cell
        x0 = pad + c * cell
        out[y0:y0 + patch_size, x0:x0 + patch_size] = f
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hidden", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--n-patches", type=int, default=30000)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr-e", type=float, default=0.05)
    parser.add_argument("--lr-p", type=float, default=0.05)
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--n-bars", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=str, default="semilinear_pm_image_patches.gif")
    args = parser.parse_args()

    snap_every = max(1, args.n_steps // args.max_frames)
    res = train(
        seed=args.seed, n_hidden=args.n_hidden, patch_size=args.patch_size,
        n_patches=args.n_patches, n_steps=args.n_steps, batch=args.batch,
        lr_e=args.lr_e, lr_p=args.lr_p, n_images=args.n_images,
        n_bars=args.n_bars, snap_every=snap_every,
    )
    snapshots = res["snapshots"]
    history = res["history"]

    frames = []
    for step, W_snap in snapshots:
        fig = plt.figure(figsize=(9, 4))
        ax_a = fig.add_subplot(1, 2, 1)
        ax_a.imshow(filter_atlas(W_snap, args.patch_size), cmap="gray", vmin=0, vmax=1)
        ax_a.set_title(f"encoder filters @ step {step}")
        ax_a.axis("off")

        ax_b = fig.add_subplot(1, 2, 2)
        # Plot training curve up to this step.
        upto = step + 1
        ax_b.plot(history["step"][:upto], history["L_pred"][:upto], lw=0.8)
        ax_b.set_xlim(0, args.n_steps)
        ax_b.set_xlabel("step")
        ax_b.set_ylabel("L_pred")
        ax_b.set_title("predictability loss")
        ax_b.grid(alpha=0.3)
        # marker at current step
        ax_b.axvline(step, color="r", lw=0.5, ls="--")

        fig.tight_layout()
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    backend, mod = imageio_or_pillow()
    if backend == "imageio":
        mod.mimsave(args.out, frames, fps=args.fps)
    else:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out} with {len(frames)} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
