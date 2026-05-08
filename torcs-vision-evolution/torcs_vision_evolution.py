"""torcs-vision-evolution — DCT-compressed neuroevolution on a numpy racing env.

Reference: Koutník, Cuccu, Schmidhuber, Gomez (GECCO 2013),
*Evolving Large-Scale Neural Networks for Vision-Based Reinforcement Learning*.

The original paper trained a TORCS controller from raw 64x64 pixels using a
network whose first-layer weights (>1M parameters) were parameterised by a
small set of low-frequency 2-D DCT coefficients evolved with CMA-ES. v1 of
the schmidhuber-problems catalog forbids the TORCS install, so this stub
captures the algorithmic claim under a numpy-only synthetic-data substitute:

  * 2-D oval racing track (closed loop, hand-drawn analytic centre line).
  * Top-down 16x16 grayscale rendering, car-centred and rotated-to-heading.
  * MLP controller (256 -> 16 -> 1, tanh) with the W1 weight matrix
    parameterised in DCT space (default K=4 -> 16 coefficients per hidden
    unit) and reconstructed at evaluation time via a precomputed orthonormal
    IDCT-II matrix.
  * Natural Evolution Strategy (OpenAI-ES style: antithetic sampling +
    rank-shaped fitness) on the DCT coefficients (plus the small remaining
    raw parameters: the hidden bias, the output weights, and the output
    bias).

CLI: `python3 torcs_vision_evolution.py --seed 0` reproduces the headline
result (DCT controller completes >100% of one lap on the eval track from
the default initial pose) in well under five minutes on an M-series CPU.

Determinism: every stochastic step uses np.random.default_rng(seed); the
same --seed produces the same final fitness on the same machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Track / environment                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class TrackConfig:
    """Closed-loop centre line, half-width width/2.

    Default shape: an oval modulated by sin(2t), so curvature varies around
    the loop and a constant-action policy cannot stay on it.
        cx(t) = (ax + bx sin 2t) cos t
        cy(t) = (ay + by sin 2t) sin t
    """

    ax: float = 4.0
    ay: float = 2.0
    bx: float = 0.55
    by: float = 0.40
    width: float = 0.55             # full track width
    n_centerline: int = 512         # discretisation of the centre line
    mask_res: int = 128             # high-res boolean track mask
    x_range: float = 6.0            # mask covers [-x_range, x_range]
    y_range: float = 3.5


@dataclass
class CarConfig:
    speed: float = 0.05             # constant forward speed (m/step)
    turn_rate: float = 0.10         # rad/step at action=+/-1


@dataclass
class RenderConfig:
    img_size: int = 16              # render to img_size x img_size
    pixel_m: float = 0.20           # m per pixel  (so view ~ 3.2 m)


def build_track(cfg: TrackConfig):
    """Precompute centre line, cumulative arclength, and a high-res mask."""
    t = np.linspace(0.0, 2.0 * np.pi, cfg.n_centerline, endpoint=False)
    cx = (cfg.ax + cfg.bx * np.sin(2.0 * t)) * np.cos(t)
    cy = (cfg.ay + cfg.by * np.sin(2.0 * t)) * np.sin(t)
    cl = np.stack([cx, cy], axis=-1)            # (N_cl, 2)

    # cumulative arclength along the closed loop, with the same indexing as cl
    seg = np.linalg.norm(np.diff(cl, axis=0, append=cl[:1]), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    perimeter = float(seg.sum())

    # high-res "is this world (x, y) on the track" mask
    res = cfg.mask_res
    xs = np.linspace(-cfg.x_range, cfg.x_range, res)
    ys = np.linspace(-cfg.y_range, cfg.y_range, res)
    XX, YY = np.meshgrid(xs, ys)                # (res, res)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=-1)  # (res^2, 2)

    min_d = np.full(grid.shape[0], np.inf, dtype=np.float64)
    chunk = 64
    for s in range(0, cl.shape[0], chunk):
        cc = cl[s:s + chunk]                    # (chunk, 2)
        dx = grid[:, 0:1] - cc[:, 0]
        dy = grid[:, 1:2] - cc[:, 1]
        d = np.sqrt(dx * dx + dy * dy)
        min_d = np.minimum(min_d, d.min(axis=1))
    mask = (min_d < cfg.width * 0.5).reshape(res, res).astype(np.float32)

    return {
        "cl": cl,
        "cum": cum,
        "perimeter": perimeter,
        "mask": mask,
        "xs": xs,
        "ys": ys,
        "cfg": cfg,
    }


def sample_track(track: dict, world_xy: np.ndarray) -> np.ndarray:
    """Look up track-mask values at the given world (x, y) coordinates."""
    cfg: TrackConfig = track["cfg"]
    res = cfg.mask_res
    px = ((world_xy[..., 0] + cfg.x_range) / (2.0 * cfg.x_range) * res).astype(np.int32)
    py = ((world_xy[..., 1] + cfg.y_range) / (2.0 * cfg.y_range) * res).astype(np.int32)
    in_x = (px >= 0) & (px < res)
    in_y = (py >= 0) & (py < res)
    in_box = in_x & in_y
    px = np.clip(px, 0, res - 1)
    py = np.clip(py, 0, res - 1)
    vals = track["mask"][py, px]
    return np.where(in_box, vals, 0.0).astype(np.float32)


def render_view(track: dict, car: np.ndarray, rcfg: RenderConfig) -> np.ndarray:
    """Render an img_size x img_size top-down view, car-centred, heading-up."""
    n = rcfg.img_size
    # local pixel coordinates in metres, with (0, 0) at the centre and +y up
    half = (n - 1) * 0.5
    ii = (np.arange(n) - half) * rcfg.pixel_m            # +x = right
    jj = -(np.arange(n) - half) * rcfg.pixel_m           # +y = forward
    LX, LY = np.meshgrid(ii, jj)                         # (n, n)
    # rotate by car heading: forward = (cos th, sin th); right = (sin th, -cos th)
    cx, cy, th = float(car[0]), float(car[1]), float(car[2])
    cos_t, sin_t = np.cos(th), np.sin(th)
    world_x = cx + LX * sin_t + LY * cos_t
    world_y = cy - LX * cos_t + LY * sin_t
    coords = np.stack([world_x, world_y], axis=-1).reshape(-1, 2)
    img = sample_track(track, coords).reshape(n, n)
    return img.astype(np.float32)


def car_progress(track: dict, car: np.ndarray) -> float:
    """Approximate forward arclength along the centre line, in metres."""
    cl = track["cl"]
    cum = track["cum"]
    d = cl - car[None, :2]
    idx = int(np.argmin(np.einsum("ij,ij->i", d, d)))
    return float(cum[idx])


def step_car(car: np.ndarray, action: float, ccfg: CarConfig) -> np.ndarray:
    """Single Euler step. action is clipped to [-1, 1]; speed is fixed."""
    a = float(np.clip(action, -1.0, 1.0))
    th = car[2] + a * ccfg.turn_rate
    x = car[0] + ccfg.speed * np.cos(th)
    y = car[1] + ccfg.speed * np.sin(th)
    return np.array([x, y, th], dtype=np.float64)


# --------------------------------------------------------------------------- #
# DCT-compressed controller                                                   #
# --------------------------------------------------------------------------- #


def build_idct_matrix(N: int, K: int) -> np.ndarray:
    """Orthonormal IDCT-II basis. Returns M of shape (N, K).

    With K low-frequency coefficients C (K x K), the corresponding NxN image is
    `M @ C @ M.T`. This is the orthonormal DCT-II inverse, so it is also the
    forward DCT-II basis with the same scaling.
    """
    n = np.arange(N)[:, None].astype(np.float64)
    k = np.arange(K)[None, :].astype(np.float64)
    M = np.cos(np.pi * (n + 0.5) * k / N)
    scale = np.where(k == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
    return (M * scale).astype(np.float64)


@dataclass
class NetConfig:
    img_size: int = 16              # input is img_size x img_size
    hidden: int = 16
    output: int = 1
    dct_k: int = 4                  # keep KxK low-frequency DCT coefs per hidden unit


def n_compressed(nc: NetConfig) -> int:
    return nc.dct_k * nc.dct_k * nc.hidden + nc.hidden + nc.hidden * nc.output + nc.output


def n_raw(nc: NetConfig) -> int:
    inp = nc.img_size * nc.img_size
    return inp * nc.hidden + nc.hidden + nc.hidden * nc.output + nc.output


def split_params(theta: np.ndarray, nc: NetConfig) -> dict:
    """Split a flat parameter vector into (DCT coefs, b1, W2, b2)."""
    K, H, O = nc.dct_k, nc.hidden, nc.output
    o = 0
    coefs = theta[o:o + K * K * H].reshape(H, K, K); o += K * K * H
    b1 = theta[o:o + H]; o += H
    W2 = theta[o:o + H * O].reshape(H, O); o += H * O
    b2 = theta[o:o + O]; o += O
    assert o == theta.size, f"param mismatch: {o} vs {theta.size}"
    return {"coefs": coefs, "b1": b1, "W2": W2, "b2": b2}


def decode_W1(coefs: np.ndarray, M: np.ndarray, N: int) -> np.ndarray:
    """coefs: (H, K, K). Returns W1 of shape (N*N, H) via IDCT per hidden unit."""
    # img_h = M @ coefs[h] @ M.T  -> (H, N, N), then reshape
    imgs = np.einsum("nk,hkj,mj->hnm", M, coefs, M)     # (H, N, N)
    return imgs.reshape(coefs.shape[0], N * N).T        # (N*N, H)


def materialize_policy(theta: np.ndarray, nc: NetConfig, M: np.ndarray) -> dict:
    """Decode flat parameter vector into dense (W1, b1, W2, b2). The W1 IDCT
    is the only expensive part, so we do it ONCE per individual rather than
    on every env step."""
    p = split_params(theta, nc)
    W1 = decode_W1(p["coefs"], M, nc.img_size)          # (256, H)
    return {"W1": W1, "b1": p["b1"], "W2": p["W2"], "b2": p["b2"]}


def policy_forward(policy: dict, x: np.ndarray) -> float:
    """Forward pass on a single 16x16 image. Returns a scalar action in [-1, 1]."""
    h = np.tanh(x.ravel() @ policy["W1"] + policy["b1"])
    y = np.tanh(h @ policy["W2"] + policy["b2"])
    return float(y[0])


# --------------------------------------------------------------------------- #
# Episode rollout                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class EnvConfig:
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarConfig = field(default_factory=CarConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    max_steps: int = 500
    init_t: float = 0.0             # parametric position on centre line at reset
    init_theta_offsets: tuple = (-0.20, 0.0, 0.20)
    """Heading offsets (rad) added to the centre-line tangent for the
    multi-trial fitness eval. With the default 3 offsets, a constant-action
    policy cannot solve all three trials, so the controller is forced to
    use its visual input."""


def reset_car(env: EnvConfig, track: dict, theta_offset: float = 0.0) -> np.ndarray:
    """Place the car on the centre line at parametric t=init_t, heading along
    the centre-line tangent plus the given offset (in radians)."""
    cl = track["cl"]
    n_cl = cl.shape[0]
    idx = int(round((env.init_t / (2.0 * np.pi)) * n_cl)) % n_cl
    nxt = (idx + 1) % n_cl
    tangent = cl[nxt] - cl[idx]
    tangent_th = float(np.arctan2(tangent[1], tangent[0]))
    return np.array([cl[idx, 0], cl[idx, 1], tangent_th + theta_offset],
                    dtype=np.float64)


def rollout(theta: np.ndarray, nc: NetConfig, M: np.ndarray, track: dict,
            env: EnvConfig, return_traj: bool = False, theta_offset: float = 0.0):
    """Run one episode, return dict with progress, steps, trajectory.

    theta_offset rotates the initial heading away from the centre-line
    tangent. With non-zero offsets the agent must use its visual input to
    correct, since constant-action drift will exit the track.
    """
    car = reset_car(env, track, theta_offset=theta_offset)
    start_progress = car_progress(track, car)
    perim = track["perimeter"]
    last_progress = start_progress
    forward = 0.0          # cumulative forward arclength (handles wraparound)

    policy = materialize_policy(theta, nc, M)

    if return_traj:
        traj_car = [car.copy()]
        traj_obs = []
        traj_act = []

    last_action = 0.0
    for step in range(env.max_steps):
        obs = render_view(track, car, env.render)
        action = policy_forward(policy, obs)
        last_action = action
        car = step_car(car, action, env.car)

        if return_traj:
            traj_obs.append(obs.copy())
            traj_act.append(action)
            traj_car.append(car.copy())

        # off-track? terminate
        on_track = float(sample_track(track, car[None, :2])[0])
        if on_track < 0.5:
            break

        # update forward arclength using monotone wraparound
        cur = car_progress(track, car)
        delta = cur - last_progress
        if delta > perim * 0.5:
            delta -= perim
        elif delta < -perim * 0.5:
            delta += perim
        forward += delta
        last_progress = cur

    out = {
        "forward": float(forward),
        "perim": perim,
        "steps": step + 1,
        "lap_frac": float(forward / perim),
        "last_action": last_action,
    }
    if return_traj:
        out["traj_car"] = np.array(traj_car)
        out["traj_obs"] = np.array(traj_obs)
        out["traj_act"] = np.array(traj_act)
    return out


def fitness_multitrial(theta: np.ndarray, nc: NetConfig, M: np.ndarray,
                       track: dict, env: EnvConfig) -> Tuple[float, int, list]:
    """Mean lap fraction across env.init_theta_offsets. Returns (mean_lap,
    max_steps, per_trial_lap_list)."""
    laps = []
    max_steps = 0
    for off in env.init_theta_offsets:
        r = rollout(theta, nc, M, track, env, theta_offset=off)
        laps.append(r["lap_frac"])
        if r["steps"] > max_steps:
            max_steps = r["steps"]
    return float(np.mean(laps)), max_steps, laps


# --------------------------------------------------------------------------- #
# OpenAI-style natural ES                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ESConfig:
    pop: int = 32                   # antithetic, so 2*pop fitness evaluations
    sigma: float = 0.10
    lr: float = 0.05
    weight_decay: float = 0.005
    max_gen: int = 200
    target_lap: float = 1.05        # solve threshold (lap fraction)
    patience: int = 30              # allow this many gens after first solve to keep improving


def rank_centered_weights(fitness: np.ndarray) -> np.ndarray:
    """Map fitness to ranks in [-0.5, 0.5] (highest rank gets 0.5)."""
    n = fitness.size
    order = np.argsort(fitness)         # low to high
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return (ranks / (n - 1)) - 0.5


def evolve(theta0: np.ndarray, nc: NetConfig, M: np.ndarray, track: dict,
           env: EnvConfig, escfg: ESConfig, rng: np.random.Generator,
           on_gen=None) -> dict:
    """OpenAI-ES with antithetic sampling on the flat parameter vector."""
    theta = theta0.copy()
    history = {"gen": [], "best_lap": [], "mean_lap": [], "best_steps": []}
    best_theta = theta.copy()
    best_lap = -np.inf
    solved_at = None

    for g in range(escfg.max_gen):
        eps = rng.standard_normal((escfg.pop, theta.size))
        # antithetic
        all_eps = np.concatenate([eps, -eps], axis=0)
        F = np.zeros(all_eps.shape[0])
        steps = np.zeros(all_eps.shape[0], dtype=np.int32)
        for i, e in enumerate(all_eps):
            mean_lap, max_steps, _ = fitness_multitrial(
                theta + escfg.sigma * e, nc, M, track, env)
            F[i] = mean_lap
            steps[i] = max_steps
        gen_best = int(np.argmax(F))
        if F[gen_best] > best_lap:
            best_lap = float(F[gen_best])
            best_theta = (theta + escfg.sigma * all_eps[gen_best]).copy()

        # rank-shaped natural-gradient update (Salimans et al., 2017)
        weights = rank_centered_weights(F)
        grad = (all_eps * weights[:, None]).sum(axis=0) / (all_eps.shape[0] * escfg.sigma)
        theta = theta * (1.0 - escfg.weight_decay) + escfg.lr * grad

        history["gen"].append(g)
        history["best_lap"].append(float(F.max()))
        history["mean_lap"].append(float(F.mean()))
        history["best_steps"].append(int(steps.max()))
        if on_gen is not None:
            on_gen(g, F, steps, theta, best_theta, best_lap)

        if best_lap >= escfg.target_lap and solved_at is None:
            solved_at = g
        if solved_at is not None and (g - solved_at) >= escfg.patience:
            break

    return {
        "theta_final": theta,
        "theta_best": best_theta,
        "best_lap": best_lap,
        "solved_at": solved_at,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Top-level CLI                                                                #
# --------------------------------------------------------------------------- #


def env_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def init_theta(nc: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """Small Gaussian init for the flat parameter vector."""
    n = n_compressed(nc)
    return rng.standard_normal(n) * 0.10


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--dct-k", type=int, default=4,
                    help="keep KxK low-frequency DCT coefficients per hidden unit")
    ap.add_argument("--pop", type=int, default=32,
                    help="ES population (antithetic, so 2*pop evals per generation)")
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-gen", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=500,
                    help="max env steps per rollout (default 500 -> covers >1 lap)")
    ap.add_argument("--target-lap", type=float, default=1.05)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--save-json", type=str, default=None)
    ap.add_argument("--save-npz", type=str, default=None,
                    help="save final theta + history for downstream viz scripts")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    nc = NetConfig(hidden=args.hidden, dct_k=args.dct_k)
    env = EnvConfig(max_steps=args.max_steps)
    track = build_track(env.track)
    M = build_idct_matrix(nc.img_size, nc.dct_k)
    escfg = ESConfig(pop=args.pop, sigma=args.sigma, lr=args.lr,
                     max_gen=args.max_gen, target_lap=args.target_lap,
                     patience=args.patience)
    theta0 = init_theta(nc, rng)

    t0 = time.time()
    if args.quiet:
        out = evolve(theta0, nc, M, track, env, escfg, rng)
    else:
        def log(g, F, steps, theta, best_theta, best_lap):
            print(f"gen {g:3d}  best_lap {F.max():.3f}  mean_lap {F.mean():.3f}  "
                  f"best_steps {steps.max():4d}  global_best {best_lap:.3f}")
        out = evolve(theta0, nc, M, track, env, escfg, rng, on_gen=log)
    wall = time.time() - t0

    final_mean, final_steps, final_per_trial = fitness_multitrial(
        out["theta_best"], nc, M, track, env)
    summary = {
        "seed": args.seed,
        "wall": wall,
        "solved_at": out["solved_at"],
        "best_lap": float(out["best_lap"]),
        "final_eval_lap_mean": final_mean,
        "final_eval_lap_per_trial": [float(x) for x in final_per_trial],
        "final_eval_steps_max": int(final_steps),
        "n_compressed": n_compressed(nc),
        "n_raw": n_raw(nc),
        "compression_ratio": n_raw(nc) / n_compressed(nc),
        "config": {
            "hidden": nc.hidden,
            "dct_k": nc.dct_k,
            "pop": escfg.pop,
            "sigma": escfg.sigma,
            "lr": escfg.lr,
            "max_gen": escfg.max_gen,
            "max_steps": env.max_steps,
            "target_lap": escfg.target_lap,
            "track": asdict(env.track),
            "car": asdict(env.car),
            "render": asdict(env.render),
        },
        "env": env_metadata(),
        "history": out["history"],
    }

    if not args.quiet:
        print(f"\nseed={args.seed}  wall={wall:.1f}s  solved_at={out['solved_at']}  "
              f"best_lap={out['best_lap']:.3f}  final_eval_lap_mean={final_mean:.3f}  "
              f"per_trial={['%.2f' % x for x in final_per_trial]}")
        print(f"compressed params: {n_compressed(nc)}   raw params: {n_raw(nc)}   "
              f"ratio: {n_raw(nc)/n_compressed(nc):.1f}x")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
    if args.save_npz:
        np.savez(args.save_npz, theta_best=out["theta_best"],
                 theta_final=out["theta_final"],
                 history_best=np.array(out["history"]["best_lap"]),
                 history_mean=np.array(out["history"]["mean_lap"]))

    return summary


if __name__ == "__main__":
    run(parse_args())
