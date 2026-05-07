"""
nbb-moving-light — Neural Bucket Brigade (NBB) on 1-D moving-light direction
discrimination.

Schmidhuber, "A local learning algorithm for dynamic feedforward and recurrent
networks", Connection Science 1(4):403-412, 1989. Also FKI-124-90 (TUM), and
"The neural bucket brigade" in Pfeifer et al., Connectionism in Perspective,
Elsevier, pp. 439-446 (1989).

Reconstructed from the IDSIA HTML transcription of the 1989/1990 paper:
  https://people.idsia.ch/~juergen/bucketbrigade/node3.html  (rule)
  https://people.idsia.ch/~juergen/bucketbrigade/node5.html  (continuous form)
  https://people.idsia.ch/~juergen/bucketbrigade/node6.html  (experiments,
                                                              including the
                                                              moving-light)

Architecture (verbatim from node6):
  "A one dimensional 'retina' consisting of 5 input units (plus one
   additional unit which was always turned on) was fully connected to a
   competitive subset of two output units. This subset of output units was
   completely connected to itself, in order to allow recurrency."

  6 input units = 5 retina cells + 1 bias (always on)
  2 output units forming one competitive WTA subset
  Direct  W_io : input -> output     (6 x 2)
  Recurrent W_oo : output -> output  (2 x 2)
  No hidden layer.

Task (verbatim from node6):
  "switch on the first output unit after an illumination point has wandered
   across the retina from the left to the right (within 5 time ticks), and
   to switch on the [other] output unit after the illumination point has
   wandered from the right to the left."

  At tick t the retina shows exactly one cell lit (other cells = 0):
    direction 0 (LR): cell t lit              -> target = output 0
    direction 1 (RL): cell (n_cells - 1 - t)  -> target = output 1
  The bias is on at every tick.

Activation rule (same as nbb-xor):
  Inputs clamped from the sequence (bias = 1, exactly one retina cell on).
  Outputs: at every tick t the unit with the largest *positive* net input
  in the WTA subset wins; x_winner(t) = 1, others = 0. Activations and
  recurrent state reset to 0 between sequence presentations.

  net_o(t) = sum_i x_i(t-1) * w_io(t-1)         (input contribution)
           + sum_k x_k(t-1) * w_ko(t-1)         (recurrent contribution)
           = sum_i c_io(t)  +  sum_k c_oo(t)
  where c_io(t) := x_i(t-1) * w_io(t-1)
        c_oo(t) := x_o(t-1) * w_oo(t-1).

Bucket-brigade weight update (discrete-time, applied at every tick):

  delta w_ij(t) = - lam * c_ij(t) * a_j(t)                   [pay out]
                + (c_ij(t-1) / D_j(t-1))                     [share]
                  * sum_k lam*c_jk(t)*a_k(t)                 [substance paid]
                + Ext_ij(t)                                  [external reward]

  D_j(t-1) = sum_h c_hj(t-1)  (sum over ALL predecessors of j, both
             input-side and recurrent-side, since outputs receive both).
  a_j(t) = 1 if unit j fires at tick t, else 0.
  Ext_ij(t) = eta * c_ij(t) on connections feeding the *correct* output
              when that output is firing; 0 otherwise.

Term 1 dissipates substance from connections feeding firing units.
Term 2 redistributes the substance the firing unit pays out (to its
       successors, here the recurrent loop) back to its predecessors,
       proportionally to their contribution at t-1. For the input->output
       weights, predecessors are inputs; for the output->output weights,
       predecessors are the other (and same) output. Because outputs feed
       into the same recurrent subset, both blocks get a share of the
       redistribution.
Term 3 is the only source of fresh substance and only fires for connections
       feeding the correct output when that output is on. Reward is local in
       both space and time.

Hyperparameters (paper / IDSIA transcription):
  lam = 0.005, eta = 0.005, weights init U(0.999, 1.001), 5 ticks/sequence.

Headline (seed 0, 5-cell retina): solves both directions under frozen-eval
in roughly a few hundred presentations on a laptop CPU; see README §Results
for the seed sweep.
"""

from __future__ import annotations
import argparse
import time

import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_sequence(direction: int, n_cells: int) -> np.ndarray:
    """Return the moving-light input sequence as an array of shape (n_cells, n_cells+1).

    Each row is the input clamp at one tick: index 0 is the bias (always 1),
    indices 1..n_cells encode the retina cells (exactly one is 1, others 0).
    direction 0 = light moves left -> right (cell t lit at tick t).
    direction 1 = light moves right -> left (cell n_cells-1-t lit at tick t).
    """
    n_input = n_cells + 1
    seq = np.zeros((n_cells, n_input), dtype=np.float64)
    seq[:, 0] = 1.0  # bias on at every tick
    for t in range(n_cells):
        cell = t if direction == 0 else (n_cells - 1 - t)
        seq[t, 1 + cell] = 1.0
    return seq


def make_all_sequences(n_cells: int) -> list[tuple[int, np.ndarray]]:
    """Return the two labelled sequences: (target_output_idx, input_sequence)."""
    return [(d, make_sequence(d, n_cells)) for d in (0, 1)]


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

class NBBMovingLight:
    """Neural Bucket Brigade for moving-light direction discrimination.

    Two-layer architecture:
      inputs (clamped) -> outputs (WTA, with output->output recurrence)
    """

    def __init__(self,
                 n_cells: int = 5,
                 lam: float = 0.005,
                 eta: float = 0.005,
                 init_lo: float = 0.999,
                 init_hi: float = 1.001,
                 seed: int = 0):
        self.n_cells = n_cells
        self.n_input = n_cells + 1   # 5 retina cells + bias
        self.n_output = 2
        self.lam = lam
        self.eta = eta

        # Single rng for init. WTA tie-breaking is deterministic argmax,
        # so eval results don't depend on rng state at the point of measurement.
        # The init asymmetry (init_hi - init_lo) is what gives WTA an initial
        # preference between output units before learning shapes them.
        init_rng = np.random.default_rng(seed)
        self.W_io = init_rng.uniform(init_lo, init_hi, (self.n_input, self.n_output))
        self.W_oo = init_rng.uniform(init_lo, init_hi, (self.n_output, self.n_output))

        self._reset_state()

    def _reset_state(self) -> None:
        """Clear all activations and recurrent state. Called between sequences."""
        self.x_i = np.zeros(self.n_input)
        self.x_o = np.zeros(self.n_output)
        self.x_i_prev = np.zeros(self.n_input)
        self.x_o_prev = np.zeros(self.n_output)

        self.c_io = np.zeros((self.n_input, self.n_output))
        self.c_oo = np.zeros((self.n_output, self.n_output))
        self.c_io_prev = np.zeros_like(self.c_io)
        self.c_oo_prev = np.zeros_like(self.c_oo)

    def _wta(self, net: np.ndarray) -> np.ndarray:
        """Winner-take-all: largest positive net input wins; ties -> lowest index."""
        x = np.zeros_like(net)
        if net.max() > 0:
            x[int(np.argmax(net))] = 1.0
        return x

    def _step(self, input_clamp: np.ndarray) -> None:
        """Advance one tick. Inputs clamped; outputs via WTA over net input."""
        # Snapshot tick t-1 state.
        self.x_i_prev[:] = self.x_i
        self.x_o_prev[:] = self.x_o
        self.c_io_prev[:] = self.c_io
        self.c_oo_prev[:] = self.c_oo

        # c_ij(t) = x_i(t-1) * w_ij(t-1)  -- weights *before* this tick's update.
        self.c_io = self.x_i_prev[:, None] * self.W_io
        self.c_oo = self.x_o_prev[:, None] * self.W_oo

        # Net input to each output: input + recurrent contributions.
        net_o = self.c_io.sum(axis=0) + self.c_oo.sum(axis=0)

        # Apply activation rule.
        self.x_i = input_clamp.copy()
        self.x_o = self._wta(net_o)

    def _bucket_brigade_update(self, target_out_idx: int) -> None:
        """Apply the bucket-brigade weight update for the current tick.

        Identical algebraic form to nbb-xor, but extended to cover the
        recurrent W_oo block. Outputs receive substance from BOTH input
        connections AND recurrent output connections, so the redistribution
        denominator D_o (sum over predecessors of o at t-1) sums over both
        blocks: D_o[j] = sum_i c_io_prev[i,j] + sum_k c_oo_prev[k,j].
        """
        active_o = self.x_o > 0  # shape (n_output,)

        # Substance each firing output pays out at this tick. For an output
        # j, the "successor" connections are the recurrent ones W_oo[j, :]
        # (j's outgoing edges in the recurrent loop). Inputs have no incoming
        # weight updates from "successors" (inputs aren't predecessors here;
        # their successor connections are W_io). So:
        #   paid_out_by_o[j] = sum_l lam * c_oo[j, l] * a_l(t)   (j's outgoing
        #                       through recurrent edges to firing outputs l)
        paid_out_by_o = (self.lam * self.c_oo * active_o[None, :]).sum(axis=1)

        # Predecessor totals at t-1 (for the redistribution share denominator).
        # Each output j's predecessors are inputs (rows of W_io) AND outputs
        # (rows of W_oo, the recurrent self-loop).
        denom_o = self.c_io_prev.sum(axis=0) + self.c_oo_prev.sum(axis=0)
        safe = denom_o > 1e-12

        # ---- input -> output (W_io) ----
        delta_io = np.zeros_like(self.W_io)
        # Term 1: connections into firing outputs pay out.
        delta_io -= self.lam * self.c_io * active_o[None, :]
        # Term 2: predecessor share of the redistribution. For W_io[i, j]:
        #   share = c_io_prev[i, j] / denom_o[j]
        share_io = np.zeros_like(self.W_io)
        if safe.any():
            share_io[:, safe] = self.c_io_prev[:, safe] / denom_o[safe]
        delta_io += share_io * paid_out_by_o[None, :]
        # Term 3: external reward on connections feeding the correct output
        # when that output is firing.
        if active_o[target_out_idx]:
            delta_io[:, target_out_idx] += self.eta * self.c_io[:, target_out_idx]

        # ---- output -> output (W_oo) ----
        delta_oo = np.zeros_like(self.W_oo)
        # Term 1: connections into firing outputs pay out.
        delta_oo -= self.lam * self.c_oo * active_o[None, :]
        # Term 2: predecessor share for recurrent edges.
        share_oo = np.zeros_like(self.W_oo)
        if safe.any():
            share_oo[:, safe] = self.c_oo_prev[:, safe] / denom_o[safe]
        delta_oo += share_oo * paid_out_by_o[None, :]
        # Term 3: external reward. The recurrent edges feeding the correct
        # output also receive Ext when that output fires; this is consistent
        # with the rule's "connections feeding the correct output" wording
        # (recurrent edges are also predecessors of the output).
        if active_o[target_out_idx]:
            delta_oo[:, target_out_idx] += self.eta * self.c_oo[:, target_out_idx]

        self.W_io += delta_io
        self.W_oo += delta_oo

    def present(self,
                seq: np.ndarray,
                target_idx: int,
                learn: bool = True,
                trace: bool = False) -> tuple[int, list[np.ndarray] | None]:
        """Present one sequence. Return (final_output_idx, optional trace).

        final_output_idx is the argmax of x_o at the last tick at which an
        output unit was firing; -1 if no output ever fires.
        If trace=True, also return the per-tick x_o vectors for animation.
        """
        self._reset_state()
        last_out = -1
        x_o_trace: list[np.ndarray] | None = [] if trace else None
        for t in range(len(seq)):
            self._step(seq[t])
            if learn:
                self._bucket_brigade_update(target_idx)
            if x_o_trace is not None:
                x_o_trace.append(self.x_o.copy())
            if self.x_o.sum() > 0:
                last_out = int(np.argmax(self.x_o))
        return last_out, x_o_trace


# ----------------------------------------------------------------------
# Eval / training
# ----------------------------------------------------------------------

def evaluate(nbb: NBBMovingLight) -> tuple[int, list[int]]:
    """Frozen-eval on both directions. Return (n_correct, per_direction_outputs)."""
    correct = 0
    outs: list[int] = []
    for direction, seq in make_all_sequences(nbb.n_cells):
        out, _ = nbb.present(seq, direction, learn=False)
        outs.append(out)
        if out == direction:
            correct += 1
    return correct, outs


def train(seed: int = 0,
          max_presentations: int = 5000,
          n_cells: int = 5,
          lam: float = 0.005,
          eta: float = 0.005,
          init_lo: float = 0.999,
          init_hi: float = 1.001,
          stable_window: int = 5,
          log_every: int = 4,
          verbose: bool = True,
          history: dict | None = None,
          snapshot_callback=None,
          snapshot_every: int = 50) -> tuple[NBBMovingLight, int, int]:
    """Train NBB on moving-light. Return (network, presentations_used, final_correct).

    Convergence criterion: the network must produce 2/2 correct under
    frozen-eval for `stable_window` consecutive evaluations. This matches
    the paper's "stable solution" tier (cf. node6).
    """
    nbb = NBBMovingLight(n_cells=n_cells, lam=lam, eta=eta,
                          init_lo=init_lo, init_hi=init_hi, seed=seed)
    seqs = make_all_sequences(n_cells)
    pres_rng = np.random.default_rng(seed + 12345)

    presentations = 0
    converged_at = -1
    consecutive = 0
    while presentations < max_presentations:
        order = pres_rng.permutation(2)
        for d_idx in order:
            direction, seq = seqs[d_idx]
            nbb.present(seq, direction, learn=True)
            presentations += 1

            if history is not None and presentations % log_every == 0:
                acc, _ = evaluate(nbb)
                history["presentations"].append(presentations)
                history["accuracy"].append(acc)
                history["W_io_norm"].append(float(np.linalg.norm(nbb.W_io)))
                history["W_oo_norm"].append(float(np.linalg.norm(nbb.W_oo)))
                history["total_substance"].append(
                    float(nbb.W_io.sum() + nbb.W_oo.sum()))

            if snapshot_callback is not None and presentations % snapshot_every == 0:
                snapshot_callback(presentations, nbb, history)

            if presentations >= max_presentations:
                break

        # Check convergence at end of cycle (one cycle = both directions).
        acc, _ = evaluate(nbb)
        if acc == 2:
            consecutive += 1
            if consecutive >= stable_window and converged_at < 0:
                converged_at = presentations
                if verbose:
                    print(f"  stable solution at {presentations} presentations  "
                          f"(2/2 for {stable_window} consecutive cycles)")
                break
        else:
            consecutive = 0

    final_acc, outs = evaluate(nbb)
    if verbose:
        print(f"  final accuracy: {final_acc}/2   outs: {outs}  "
              f"(out[0]=LR target, out[1]=RL target)")
    return nbb, presentations, final_acc


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-presentations", type=int, default=5000)
    p.add_argument("--n-cells", type=int, default=5,
                   help="Retina length. Paper used 5.")
    p.add_argument("--lam", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--stable-window", type=int, default=5,
                   help="Consecutive 2/2 evals required to declare convergence.")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="If >1, run a sweep starting from --seed.")
    args = p.parse_args()

    if args.n_seeds == 1:
        t0 = time.time()
        nbb, presentations, acc = train(
            seed=args.seed,
            max_presentations=args.max_presentations,
            n_cells=args.n_cells,
            lam=args.lam, eta=args.eta,
            stable_window=args.stable_window,
            verbose=True,
        )
        elapsed = time.time() - t0
        print()
        print(f"Final accuracy:        {acc}/2 ({100*acc//2}%)")
        print(f"Sequence presentations: {presentations}")
        print(f"Wallclock:             {elapsed:.2f}s")
        print()
        print("Final W_io (input -> output):")
        print(np.round(nbb.W_io, 4))
        print()
        print("Final W_oo (output -> output, recurrent):")
        print(np.round(nbb.W_oo, 4))
    else:
        t0 = time.time()
        results = []
        for s in range(args.seed, args.seed + args.n_seeds):
            _, pres, acc = train(
                seed=s,
                max_presentations=args.max_presentations,
                n_cells=args.n_cells,
                lam=args.lam, eta=args.eta,
                stable_window=args.stable_window,
                verbose=False,
            )
            results.append((s, pres, acc))
            tag = "OK " if acc == 2 else "BAD"
            print(f"  seed {s:3d}  {tag}  acc={acc}/2  presentations={pres}")
        elapsed = time.time() - t0
        n_solved = sum(1 for _, _, a in results if a == 2)
        if n_solved > 0:
            mean_pres = float(np.mean([p for _, p, a in results if a == 2]))
        else:
            mean_pres = float("nan")
        print()
        print(f"Solved: {n_solved}/{args.n_seeds}")
        print(f"Mean presentations among solvers: {mean_pres:.0f}")
        print(f"Total wallclock: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
