"""
nbb-xor — Neural Bucket Brigade (NBB) on XOR.

Schmidhuber, "A local learning algorithm for dynamic feedforward and recurrent
networks", Connection Science 1(4):403-412, 1989. Also FKI-124-90 (TUM).

Sources for the rule (the original FKI report PDF is degraded — we
reconstructed the algorithm from the IDSIA HTML transcription of the
1989/1990 paper sections):
  https://people.idsia.ch/~juergen/bucketbrigade/node3.html  (rule)
  https://people.idsia.ch/~juergen/bucketbrigade/node5.html  (continuous form)
  https://people.idsia.ch/~juergen/bucketbrigade/node6.html  (XOR experiment)

Architecture (from the IDSIA HTML node6 transcription):
  3 input units  =  bias + x1 + x2          (clamped, always 0/1)
  3 hidden units                            (one competitive subset, WTA)
  2 output units = "XOR=0" and "XOR=1"      (one competitive subset, WTA)
  Dense input -> hidden, dense hidden -> output. No skip connections.

Activation rule:
  Inputs: clamped from the pattern. Bias = 1 always.
  Hidden / Output: at every tick t, the unit with the largest *positive*
  net input in its subset wins, x_winner(t) = 1, others = 0. Activations
  reset to 0 between patterns.

Net input uses previous-tick activations:
  net_j(t) = sum_i x_i(t-1) * w_ij(t-1)
           = sum_i c_ij(t)       where c_ij(t) := x_i(t-1) * w_ij(t-1).

Bucket-brigade weight update (discrete-time form, applied at every tick):

  delta w_ij(t) = - lam * c_ij(t) * a_j(t)                          [pay out when j fires]
                + (c_ij(t-1) / sum_h c_hj(t-1)) * sum_k lam*c_jk(t)*a_k(t)
                                                                    [credit predecessors]
                + Ext_ij(t)                                         [external reward]

  a_j(t)  = 1 if unit j fires at tick t, else 0.
  Ext_ij(t) = eta * c_ij(t) on connections feeding the *correct* output
              when that output is firing; 0 otherwise.

This is a literal transcription of the discrete rule from the IDSIA HTML
transcription. The continuous form on node5 is shown only as a theoretical
construct; the paper says "the only experiments conducted so far were based
on the discrete time version."

Conservation: each tick, the substance paid out by a firing unit
(sum_k lam*c_jk(t)) is redistributed to its predecessor connections in
proportion to how much each contributed. The external reward (Ext) is the
only source of fresh substance; the system is otherwise dissipative
(weights decay through lam if there is no reward).

Hyperparameters (from the IDSIA HTML node6 + our seed sweep):
  lam = 0.005, eta = 0.05, weights init U(0.999, 1.001), 6 ticks/pattern
  (eta tuned upward from the paper's 0.005; see Deviations in README.)
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_xor_patterns() -> np.ndarray:
    """4 XOR patterns, columns = (x1, x2, target_output_index).

    target_output_index 0 means output unit 0 (the "XOR=0" unit) should win.
    """
    return np.array([
        [0, 0, 0],   # 0 XOR 0 = 0
        [0, 1, 1],   # 0 XOR 1 = 1
        [1, 0, 1],   # 1 XOR 0 = 1
        [1, 1, 0],   # 1 XOR 1 = 0
    ], dtype=np.int32)


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

class NBB:
    """Neural Bucket Brigade with two WTA subsets (hidden, output)."""

    def __init__(self,
                 n_input: int = 3,
                 n_hidden: int = 3,
                 n_output: int = 2,
                 lam: float = 0.005,
                 eta: float = 0.005,
                 init_lo: float = 0.999,
                 init_hi: float = 1.001,
                 seed: int = 0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lam = lam
        self.eta = eta

        # Single rng for init. WTA tie-breaking is deterministic (argmax),
        # so eval results don't depend on rng state at the point of measurement.
        # The init asymmetry (init_hi - init_lo) is what gives WTA an initial
        # preference between hidden / output units.
        init_rng = np.random.default_rng(seed)

        self.W_ih = init_rng.uniform(init_lo, init_hi, (n_input, n_hidden))
        self.W_ho = init_rng.uniform(init_lo, init_hi, (n_hidden, n_output))

        self._reset_state()

    def _reset_state(self) -> None:
        """Clear all activation history. Called between patterns."""
        self.x_i = np.zeros(self.n_input)
        self.x_h = np.zeros(self.n_hidden)
        self.x_o = np.zeros(self.n_output)
        # c at current tick t and previous tick t-1
        self.c_ih = np.zeros((self.n_input, self.n_hidden))
        self.c_ho = np.zeros((self.n_hidden, self.n_output))
        self.c_ih_prev = np.zeros_like(self.c_ih)
        self.c_ho_prev = np.zeros_like(self.c_ho)
        # snapshots of the previous tick's activations (for c_ij(t) computation)
        self.x_i_prev = np.zeros(self.n_input)
        self.x_h_prev = np.zeros(self.n_hidden)
        self.x_o_prev = np.zeros(self.n_output)

    def _wta(self, net: np.ndarray) -> np.ndarray:
        """Winner-take-all on a subset.

        Largest positive net-input wins. If all <= 0, no unit fires.
        Ties are broken deterministically (lowest index). The init
        asymmetry on weights provides initial differentiation; the
        learning rule amplifies it.
        """
        x = np.zeros_like(net)
        if net.max() > 0:
            x[int(np.argmax(net))] = 1.0
        return x

    def _step(self, input_clamp: np.ndarray) -> None:
        """Advance one tick. Input clamped; hidden/output via WTA."""
        # Snapshot tick-t-1 state.
        self.x_i_prev[:] = self.x_i
        self.x_h_prev[:] = self.x_h
        self.x_o_prev[:] = self.x_o
        self.c_ih_prev[:] = self.c_ih
        self.c_ho_prev[:] = self.c_ho

        # c_ij(t) = x_i(t-1) * w_ij(t-1)  -- weights *before* this tick's update.
        self.c_ih = self.x_i_prev[:, None] * self.W_ih
        self.c_ho = self.x_h_prev[:, None] * self.W_ho

        # Net inputs from c_ij(t).
        net_h = self.c_ih.sum(axis=0)
        net_o = self.c_ho.sum(axis=0)

        # Apply activation rule.
        self.x_i = input_clamp.copy()
        self.x_h = self._wta(net_h)
        self.x_o = self._wta(net_o)

    def _bucket_brigade_update(self, target_out_idx: int) -> None:
        """Apply the bucket-brigade weight update for the current tick.

        Three terms:
          1) -lam * c_ij(t)              when j is firing
          2) (c_ij(t-1) / sum_h c_hj(t-1)) * sum_k lam*c_jk(t)*a_k(t)
                                         when j is firing
          3) eta * c_ij(t)               for connections feeding the correct
                                         output, when that output is firing.

        Term 2 has no contribution for hidden->output weights (outputs have
        no successors in this 2-layer network).
        """
        active_h = self.x_h > 0  # shape (n_hidden,)
        active_o = self.x_o > 0  # shape (n_output,)

        # ---- input -> hidden ----
        delta_ih = np.zeros_like(self.W_ih)
        # Term 1: connections into firing hidden units pay out.
        delta_ih -= self.lam * self.c_ih * active_h[None, :]
        # Term 2: redistribute payments j makes to its successors back to
        # j's predecessors, in proportion to c_ij(t-1).
        # j here is hidden, k is output.
        paid_out_by_h = (self.lam * self.c_ho * active_o[None, :]).sum(axis=1)
        # share_ih[i, h] = c_ih_prev[i, h] / sum_i' c_i'h_prev[h]
        denom_h = self.c_ih_prev.sum(axis=0)
        # Avoid division by zero: if no predecessor fired at t-1, no redistribution.
        safe = denom_h > 1e-12
        share_ih = np.zeros_like(self.W_ih)
        if safe.any():
            share_ih[:, safe] = self.c_ih_prev[:, safe] / denom_h[safe]
        delta_ih += share_ih * paid_out_by_h[None, :]

        # ---- hidden -> output ----
        delta_ho = np.zeros_like(self.W_ho)
        # Term 1: connections into firing output units pay out.
        delta_ho -= self.lam * self.c_ho * active_o[None, :]
        # Term 2: outputs have no successors -> no redistribution into hidden->output.
        # Term 3 (Ext): reward connections feeding the correct output, if it fires.
        if active_o[target_out_idx]:
            delta_ho[:, target_out_idx] += self.eta * self.c_ho[:, target_out_idx]

        self.W_ih += delta_ih
        self.W_ho += delta_ho

    def present(self,
                x1: int,
                x2: int,
                target_idx: int,
                n_ticks: int = 6,
                learn: bool = True) -> int:
        """Present one pattern for n_ticks; return the most recent output index.

        The output index is the argmax of x_o at the final tick that an output
        unit was firing. Returns -1 if no output unit ever fired.
        """
        self._reset_state()
        bias = 1.0
        clamp = np.array([bias, float(x1), float(x2)])

        last_out = -1
        for _ in range(n_ticks):
            self._step(clamp)
            if learn:
                self._bucket_brigade_update(target_idx)
            if self.x_o.sum() > 0:
                last_out = int(np.argmax(self.x_o))
        return last_out


# ----------------------------------------------------------------------
# Eval / training
# ----------------------------------------------------------------------

def evaluate(nbb: NBB, patterns: np.ndarray, n_ticks: int = 6) -> tuple[int, list[int]]:
    """Run all 4 patterns with learning frozen. Return (correct_count, per_pattern_outputs)."""
    correct = 0
    outs: list[int] = []
    for x1, x2, target in patterns:
        out = nbb.present(int(x1), int(x2), int(target), n_ticks=n_ticks, learn=False)
        outs.append(out)
        if out == int(target):
            correct += 1
    return correct, outs


def train(seed: int = 0,
          max_presentations: int = 5000,
          n_ticks: int = 6,
          lam: float = 0.005,
          eta: float = 0.005,
          n_hidden: int = 3,
          init_lo: float = 0.999,
          init_hi: float = 1.001,
          log_every: int = 4,
          verbose: bool = True,
          history: dict | None = None,
          snapshot_callback=None,
          snapshot_every: int = 50) -> tuple[NBB, int, int]:
    """Train NBB on XOR. Returns (network, presentations_used, final_correct).

    presentations_used is the number of single-pattern presentations consumed
    until the network first solves all 4 patterns under frozen-eval, or
    max_presentations if it never converges.
    """
    nbb = NBB(n_hidden=n_hidden, lam=lam, eta=eta,
              init_lo=init_lo, init_hi=init_hi, seed=seed)
    patterns = make_xor_patterns()
    pres_rng = np.random.default_rng(seed + 12345)

    presentations = 0
    converged_at = -1
    while presentations < max_presentations:
        order = pres_rng.permutation(4)
        for p_idx in order:
            x1, x2, target = patterns[p_idx]
            nbb.present(int(x1), int(x2), int(target), n_ticks=n_ticks, learn=True)
            presentations += 1

            if history is not None and presentations % log_every == 0:
                acc, _ = evaluate(nbb, patterns, n_ticks)
                history["presentations"].append(presentations)
                history["accuracy"].append(acc)
                history["W_ih_norm"].append(float(np.linalg.norm(nbb.W_ih)))
                history["W_ho_norm"].append(float(np.linalg.norm(nbb.W_ho)))
                history["total_substance"].append(
                    float(nbb.W_ih.sum() + nbb.W_ho.sum()))

            if snapshot_callback is not None and presentations % snapshot_every == 0:
                snapshot_callback(presentations, nbb, history)

            if presentations >= max_presentations:
                break

        # Check convergence at end of cycle.
        acc, _ = evaluate(nbb, patterns, n_ticks)
        if acc == 4 and converged_at < 0:
            converged_at = presentations
            if verbose:
                print(f"  converged at {presentations} presentations  (acc 4/4)")
            break

    final_acc, outs = evaluate(nbb, patterns, n_ticks)
    if verbose:
        print(f"  final accuracy: {final_acc}/4   outs: {outs}")
    return nbb, presentations, final_acc


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-presentations", type=int, default=5000)
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--ticks", type=int, default=6)
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="if >1, run a seed sweep starting from --seed and "
                        "report the average / success rate.")
    args = p.parse_args()

    if args.n_seeds == 1:
        t0 = time.time()
        nbb, presentations, acc = train(
            seed=args.seed,
            max_presentations=args.max_presentations,
            lam=args.lam, eta=args.eta,
            n_ticks=args.ticks, n_hidden=args.n_hidden,
            verbose=True,
        )
        elapsed = time.time() - t0
        print()
        print(f"Final accuracy:        {acc}/4 ({100*acc//4}%)")
        print(f"Pattern presentations: {presentations}")
        print(f"Wallclock:             {elapsed:.2f}s")
        print()
        print("Final W_ih (input -> hidden):")
        print(np.round(nbb.W_ih, 3))
        print()
        print("Final W_ho (hidden -> output):")
        print(np.round(nbb.W_ho, 3))
    else:
        t0 = time.time()
        results = []
        for s in range(args.seed, args.seed + args.n_seeds):
            _, pres, acc = train(
                seed=s,
                max_presentations=args.max_presentations,
                lam=args.lam, eta=args.eta,
                n_ticks=args.ticks, n_hidden=args.n_hidden,
                verbose=False,
            )
            results.append((s, pres, acc))
            tag = "OK " if acc == 4 else "BAD"
            print(f"  seed {s:3d}  {tag}  acc={acc}/4  presentations={pres}")
        elapsed = time.time() - t0
        n_solved = sum(1 for _, _, a in results if a == 4)
        if n_solved > 0:
            mean_pres = np.mean([p for _, p, a in results if a == 4])
        else:
            mean_pres = float("nan")
        print()
        print(f"Solved: {n_solved}/{args.n_seeds}")
        print(f"Mean presentations among solvers: {mean_pres:.0f}")
        print(f"Total wallclock: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
