/-
  Theorem 1(b): Buffer floor at canonical FLOPs.

  Among schedules with no recomputation (m_i = q_i), CASAP achieves the
  minimum peak buffer S_*(G).

  STATUS: axiomatized. Full proof requires porting the CASAP
  exchange argument from a concurrent anonymous submission (which
  also has its own Lean port underway).
-/

import ParetoCorner.Definitions

namespace ParetoCorner

/-! ## CASAP optimality (axiomatized)

The CASAP schedule fires, at each step, the unique most-downstream-enabled
actor. Among canonical schedules (m_i = q_i for all i), CASAP achieves the
minimum peak buffer.

The exchange-argument proof is in the concurrent submission §III-D. The full Lean port
mirrors the structure of
the concurrent submission's Lean port).

For the present formalization we axiomatize the result. The corner Pareto-
optimality (Theorem 1(c)) and the linear bound (Theorem 1(d)) both depend
on this axiom; replacing the axiom with a proof closes the dependency.
-/

/-- **Theorem 1(b)** [AXIOM, port from the concurrent submission]: Among canonical schedules
    of a chain-with-skips non-expanding DNN-SDF graph G, the CASAP-minimum
    buffer `S_*(G)` is a lower bound on peak buffer.

    This axiom is replaced by the constructive CASAP exchange-argument
    proof in `BufferFloorProof.lean` (currently a stub matching the concurrent submission
    Theorem 1). -/
axiom bufferFloorAtCanonical {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (hCanonical : σ.isCanonical) :
    G.bufferCanonical ≤ σ.peakBuffer

/-- **Constructive corner**: there exists a canonical schedule σ_CASAP
    achieving the canonical-buffer minimum. -/
axiom casap_exists {n : Nat} (G : DnnSdfGraph n) :
    ∃ σ : Schedule G, σ.isCanonical ∧ σ.peakBuffer = G.bufferCanonical

end ParetoCorner
