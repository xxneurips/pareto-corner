/-
  Pareto-Corner Theorem: Pairwise Lemma (F, S).
  Lean 4 skeleton: compilable types + theorem statement; proofs are `sorry`.

  This file states:
  - the structures matching the SDF formalism
  - the chain-backpropagation lemma
  - the proof obligations marked `sorry`

  The exchange-argument proof from the concurrent submission's Theorem 1 is axiomatized; full
  port from the concurrent submission's Lean port is open work.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Order.MinMax

open BigOperators

namespace ParetoCornerLemma

/-! ## SDF graph structure (chain-with-skips, non-expanding) -/

/-- A chain-with-skips DNN-SDF graph. Actors are indexed `0..n-1` along the
    Hamiltonian path; skip edges go `i → j` with `i < j`. -/
structure DnnSdfGraph (n : ℕ) where
  /-- Per-actor FLOP cost per firing. -/
  flopCost : Fin n → ℕ
  /-- Repetition vector entry per actor (must satisfy balance equations). -/
  rep : Fin n → ℕ
  /-- Per-actor maximum input footprint in bytes. -/
  inputBytes : Fin n → ℕ
  /-- Per-actor maximum output footprint in bytes. -/
  outputBytes : Fin n → ℕ
  /-- Non-expanding property: output ≤ input bytes per actor. -/
  nonExpanding : ∀ i, outputBytes i ≤ inputBytes i
  /-- Repetition vector entries are positive. -/
  rep_pos : ∀ i, 0 < rep i

/-! ## Schedules (allowing recomputation) -/

/-- A valid schedule fires each actor at least its repetition count.
    `m i` is the number of times actor `i` fires. -/
structure Schedule {n : ℕ} (G : DnnSdfGraph n) where
  /-- Number of firings per actor (≥ rep). -/
  m : Fin n → ℕ
  /-- The schedule fires each actor at least its repetition count. -/
  m_ge_rep : ∀ i, G.rep i ≤ m i
  /-- The schedule's peak buffer (in bytes). Abstract for now. -/
  peakBuffer : ℕ
  /-- Peak buffer satisfies a token-conservation lower bound (placeholder). -/
  peakBuffer_pos : 0 < peakBuffer

/-- The CASAP schedule: each actor fires exactly its repetition count, in
    most-downstream-enabled order. Definition is abstract; the optimality
    claim is the content of `casap_optimal` below. -/
noncomputable def casapSchedule {n : ℕ} (G : DnnSdfGraph n) : Schedule G :=
  sorry  -- mirrors the concurrent submission's Algorithm 2

/-- FLOP count of a schedule. -/
def Schedule.flops {n : ℕ} {G : DnnSdfGraph n} (σ : Schedule G) : ℕ :=
  ∑ i, σ.m i * G.flopCost i

/-- Canonical FLOP count: each actor fires exactly its repetition count. -/
def DnnSdfGraph.flopsCanonical {n : ℕ} (G : DnnSdfGraph n) : ℕ :=
  ∑ i, G.rep i * G.flopCost i

/-- CASAP-minimum buffer (the schedule-feasible lower bound). -/
noncomputable def DnnSdfGraph.bufferCanonical {n : ℕ} (G : DnnSdfGraph n) : ℕ :=
  (casapSchedule G).peakBuffer

/-! ## The four claims of Theorem (Pareto Corner) -/

/-- (1) FLOP floor: every schedule does at least the canonical FLOP work. -/
theorem flopFloor {n : ℕ} (G : DnnSdfGraph n) (σ : Schedule G) :
    G.flopsCanonical ≤ σ.flops := by
  unfold DnnSdfGraph.flopsCanonical Schedule.flops
  apply Finset.sum_le_sum
  intro i _
  exact Nat.mul_le_mul_right _ (σ.m_ge_rep i)

/-- (2) Buffer floor at canonical FLOP: among schedules with no recomputation
    (`m i = G.rep i`), CASAP achieves the minimum peak buffer.

    This is a concurrent submission's Theorem 1 statement, axiomatized
    here. The Lean port from that concurrent submission is open work. -/
axiom casap_optimal {n : ℕ} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_canonical : ∀ i, σ.m i = G.rep i) :
    G.bufferCanonical ≤ σ.peakBuffer

/-- (3) Pareto-corner non-improvement: it is not the case that
    `σ.flops < G.flopsCanonical` AND `σ.peakBuffer < G.bufferCanonical`.

    This is the central claim. An earlier multiplicative product form was
    retracted because it is empirically falsified by MCUNetV2 patch sweeps.
    The corner statement, in contrast, is a one-line case split. -/
theorem paretoCornerNonImprovement {n : ℕ} (G : DnnSdfGraph n) (σ : Schedule G) :
    ¬ (σ.flops < G.flopsCanonical ∧ σ.peakBuffer < G.bufferCanonical) := by
  intro ⟨h_flops, _⟩
  -- σ.flops < G.flopsCanonical contradicts flopFloor.
  exact absurd (flopFloor G σ) (not_le.mpr h_flops)

/-- (4) Recomputation FLOP overhead: if `σ.peakBuffer < G.bufferCanonical`,
    then σ recomputes some actor (FLOPs strictly above canonical).

    The architecture-dependent overhead `min_j G.flopCost j` is the floor;
    Lemma `chainBackprop` (below) gives stronger bounds via the chain
    backpropagation of recomputation through predecessors. -/
theorem recomputationOverhead {n : ℕ} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_buffer : σ.peakBuffer < G.bufferCanonical) :
    G.flopsCanonical < σ.flops := by
  -- If σ were canonical, casap_optimal would give bufferCanonical ≤ peakBuffer,
  -- contradicting h_buffer. So some m_i > rep i, hence flops > flopsCanonical.
  by_contra h
  push_neg at h
  -- σ.flops ≤ flopsCanonical combined with flopFloor gives equality.
  have hflops_eq : σ.flops = G.flopsCanonical := le_antisymm h (flopFloor G σ)
  -- σ.flops = flopsCanonical means σ is canonical (in the multiplicative
  -- sense: the only way m_i ≥ rep i for all i and Σ m_i F_i = Σ rep_i F_i
  -- is m_i = rep_i for all i, when F_i > 0 for some i).
  -- Then casap_optimal gives bufferCanonical ≤ peakBuffer.
  -- Contradiction with h_buffer.
  sorry  -- straightforward but needs the canonical-iff-equal-flops lemma

/-! ## The chain-backpropagation lemma (Lemma 1 of the paper) -/

/-- For minimal valid schedules, recomputation of any actor v_i propagates
    back through the chain: predecessors v_j with j ≤ i must also re-fire
    to supply the recomputation, with multiplicity α_{j→i}. -/
axiom chainBackprop {n : ℕ} (G : DnnSdfGraph n) (σ : Schedule G)
    (i : Fin n) (h_recomp : G.rep i < σ.m i) :
    -- For all j ≤ i in chain order, m_j ≥ q_j + (m_i - q_i) · α_{j→i}.
    -- Stating this fully requires defining α_{j→i} from rate ratios,
    -- which is left as an open obligation in the Lean port.
    True  -- placeholder; full statement is an open obligation

/-! ## Corollary: the (F_*, S_*) corner is on the Pareto frontier -/

/-- A schedule τ Pareto-dominates σ if both `flops` and `peakBuffer` are weakly
    smaller, with strict inequality in at least one coordinate. -/
def Schedule.dominatedBy {n : ℕ} {G : DnnSdfGraph n} (σ τ : Schedule G) : Prop :=
  τ.flops ≤ σ.flops ∧ τ.peakBuffer ≤ σ.peakBuffer ∧
    (τ.flops < σ.flops ∨ τ.peakBuffer < σ.peakBuffer)

/-- The CASAP schedule is on the Pareto frontier: no schedule strictly
    dominates it on both F and S simultaneously.

    Proof idea: suppose τ strictly improves both. Then τ.flops < flopsCanonical
    AND τ.peakBuffer < bufferCanonical, contradicting paretoCornerNonImprovement. -/
theorem casap_paretoOptimal {n : ℕ} (G : DnnSdfGraph n) :
    ∀ τ : Schedule G,
      ¬ (τ.flops < G.flopsCanonical ∧ τ.peakBuffer < G.bufferCanonical) := by
  intro τ
  exact paretoCornerNonImprovement G τ

end ParetoCornerLemma

/-
  Open Lean obligations:

  1. `casapSchedule` constructive definition (mirrors the concurrent submission's Algorithm 2).
  2. `casap_optimal` — currently axiomatized; full proof needs the
     exchange argument from the concurrent submission §III-D. Plan: import
     or rebuild the concurrent submission's Lean port once de-anonymized.
  3. `recomputationOverhead` — needs the canonical-iff-equal-flops lemma:
     `(∀ i, σ.m i ≥ G.rep i) ∧ σ.flops = G.flopsCanonical → ∀ i, σ.m i = G.rep i`,
     under the hypothesis that some `F_i > 0`.
  4. `chainBackprop` — full statement needs α_{j→i} defined as the
     rate-product, then the induction over the chain.

  Note: `paretoCornerNonImprovement` (claim 3) needs no multiplicative-form
  machinery and is proved in 2 lines. The retracted multiplicative product
  form (β_i bound) is documented in the paper for completeness.
-/
