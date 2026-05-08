"""
Animate the random-weight-guessing search for rs-two-sequence.

Produces `rs_two_sequence.gif` (≤ 2 MB target). Each frame shows:
  • Top: every trial so far as a dot (train_acc vs trial). Accepted trials
    (train_acc >= threshold) are highlighted; the running best is overlaid.
  • Bottom: the current best network's hidden-state trajectory on a few train
    sequences. Watch the latch snap into place when a successful weight
    sample is accepted.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from rs_two_sequence import (
    accuracy, forward_rnn, make_two_sequence_data, sample_weights, sigmoid,
)


def collect_trace(seed: int, lag: int, hidden: int, noise_std: float,
                  n_train: int, n_test: int, weight_range: float,
                  threshold: float, max_trials: int):
    """Run RS, recording every trial's train-acc and snapshotting theta whenever
    best-so-far improves.
    """
    seed_seq = np.random.SeedSequence(seed)
    data_seed, search_seed = seed_seq.spawn(2)
    data_rng = np.random.default_rng(data_seed)
    search_rng = np.random.default_rng(search_seed)

    X_tr, y_tr = make_two_sequence_data(n_train, lag, noise_std, data_rng)
    X_te, y_te = make_two_sequence_data(n_test, lag, noise_std, data_rng)

    all_trial = np.zeros(max_trials, dtype=np.int64)
    all_train = np.zeros(max_trials, dtype=np.float32)
    snapshots = []   # list of (trial, train_acc, test_acc, theta)

    best = -1.0
    solved_at = None
    for trial in range(1, max_trials + 1):
        theta = sample_weights(search_rng, hidden, weight_range)
        a_tr = accuracy(X_tr, y_tr, theta)
        all_trial[trial - 1] = trial
        all_train[trial - 1] = a_tr
        if a_tr > best:
            best = a_tr
            a_te = accuracy(X_te, y_te, theta)
            snapshots.append({"trial": trial, "train": a_tr, "test": a_te,
                              "theta": theta})
        if a_tr >= threshold:
            a_te = accuracy(X_te, y_te, theta)
            if a_te >= threshold:
                solved_at = trial
                break

    n = trial
    return {
        "all_trial": all_trial[:n],
        "all_train": all_train[:n],
        "snapshots": snapshots,
        "solved_at": solved_at,
        "X_tr": X_tr, "y_tr": y_tr,
        "config": {"seed": seed, "lag": lag, "hidden": hidden,
                   "noise_std": noise_std, "n_train": n_train,
                   "n_test": n_test, "weight_range": weight_range,
                   "threshold": threshold},
    }


def _hidden_traj(X: np.ndarray, theta: dict) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return (h_traj over time, yhat at final step, latch_dim, sign-aligned)."""
    H = theta["W_hh"].shape[0]
    B, T = X.shape
    h = np.zeros((B, H), dtype=np.float32)
    h_traj = np.zeros((T + 1, B, H), dtype=np.float32)
    for t in range(T):
        x_t = X[:, t:t + 1]
        h = np.tanh(x_t @ theta["W_xh"] + h @ theta["W_hh"] + theta["b_h"])
        h_traj[t + 1] = h
    z = (h @ theta["W_hy"] + theta["b_y"]).reshape(-1)
    yhat = sigmoid(z)
    readout = theta["W_hy"].reshape(-1)
    latch_dim = int(np.argmax(np.abs(readout)))
    sign = float(np.sign(readout[latch_dim]))
    return h_traj, yhat, latch_dim, sign


def render_frame(trace: dict, frame_trial: int, snapshot: dict,
                 demo_idx: np.ndarray, accepted_so_far: list[int]) -> np.ndarray:
    """Render a single frame as an RGB array."""
    cfg = trace["config"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6), dpi=85,
                             gridspec_kw={"height_ratios": [1, 1.1]})

    # ---- top: search progression up to frame_trial ----
    ax = axes[0]
    mask = trace["all_trial"] <= frame_trial
    trials = trace["all_trial"][mask]
    accs = trace["all_train"][mask]
    rejected = accs < cfg["threshold"]
    accepted = ~rejected
    ax.scatter(trials[rejected], accs[rejected], s=6, c="#999999",
               alpha=0.5, edgecolors="none", label="rejected trial")
    if accepted.any():
        ax.scatter(trials[accepted], accs[accepted], s=40, c="#d62728",
                   zorder=5, label="accepted trial (train acc ≥ thr)")
    # running best
    running_best = np.maximum.accumulate(accs)
    ax.step(trials, running_best, color="#1f77b4", lw=1.4,
            where="post", label="best so far", alpha=0.9)

    ax.axhline(cfg["threshold"], color="black", lw=0.6, ls="--",
               alpha=0.5, label=f"threshold={cfg['threshold']}")
    ax.set_xlabel("trial")
    ax.set_ylabel("train accuracy")
    ax.set_xlim(1, max(trace["solved_at"] or len(trace["all_trial"]), 10))
    ax.set_ylim(0.45, 1.05)
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"random-weight-guessing on Bengio-94 latch | "
        f"trial {frame_trial:,} | best train {snapshot['train']:.3f} "
        f"(found at trial {snapshot['trial']:,})"
    )

    # ---- bottom: hidden-state trajectory of best-so-far snapshot ----
    ax = axes[1]
    X_demo = trace["X_tr"][demo_idx]
    y_demo = trace["y_tr"][demo_idx]
    h_traj, yhat, latch_dim, sign = _hidden_traj(X_demo, snapshot["theta"])
    ts = np.arange(X_demo.shape[1] + 1)
    for i, b in enumerate(range(X_demo.shape[0])):
        color = "#d62728" if y_demo[b] == 1 else "#1f77b4"
        ax.plot(ts, sign * h_traj[:, b, latch_dim], color=color, lw=0.9,
                alpha=0.85)
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax.axvline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"hidden unit {latch_dim} (sign-aligned)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, X_demo.shape[1])
    ax.grid(alpha=0.3)
    ax.set_title(
        f"latch behavior of best-so-far net  "
        f"(red: class +1, blue: class -1) — train {snapshot['train']:.2f} "
        f"test {snapshot['test']:.2f}"
    )

    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def build_gif(trace: dict, out_path: str, n_frames: int, fps: int) -> None:
    """Produce a GIF with n_frames frames, log-spaced over trials."""
    end_trial = trace["solved_at"] or int(trace["all_trial"][-1])
    frame_trials = np.unique(np.round(np.geomspace(1, end_trial, n_frames)).astype(int))

    # Demo sequences: 4 of each class
    rng = np.random.default_rng(0)
    pos_idx = np.where(trace["y_tr"] == 1)[0]
    neg_idx = np.where(trace["y_tr"] == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    demo_idx = np.concatenate([pos_idx[:4], neg_idx[:4]])

    snapshots = trace["snapshots"]
    snap_trials = np.array([s["trial"] for s in snapshots])

    frames = []
    for ft in frame_trials:
        # Most recent snapshot at or before ft
        idx = int(np.searchsorted(snap_trials, ft, side="right") - 1)
        idx = max(idx, 0)
        snap = snapshots[idx]
        accepted_so_far = [s["trial"] for s in snapshots[:idx + 1]
                           if s["train"] >= trace["config"]["threshold"]]
        frames.append(render_frame(trace, int(ft), snap, demo_idx, accepted_so_far))

    # Hold the final frame for emphasis
    for _ in range(max(fps, 6)):
        frames.append(frames[-1])

    print(f"  rendering {len(frames)} frames @ {fps} fps -> {out_path}")
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    sz = os.path.getsize(out_path) / 1024
    print(f"  wrote {out_path} ({sz:.1f} KB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lag", type=int, default=100)
    p.add_argument("--hidden", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--weight-range", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--max-trials", type=int, default=200_000)
    p.add_argument("--n-frames", type=int, default=30,
                   help="Number of frames in the GIF (log-spaced over trials).")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="rs_two_sequence.gif")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Collecting trace ...")
    trace = collect_trace(
        seed=args.seed, lag=args.lag, hidden=args.hidden,
        noise_std=args.noise_std,
        n_train=args.n_train, n_test=args.n_test,
        weight_range=args.weight_range, threshold=args.threshold,
        max_trials=args.max_trials,
    )
    print(f"  solved at trial {trace['solved_at']} "
          f"(snapshots taken: {len(trace['snapshots'])})")
    build_gif(trace, args.out, args.n_frames, args.fps)


if __name__ == "__main__":
    main()
