/-
  Definitions for the Pareto-Corner Theorem.

  This file encodes the SDF-graph formalism from §2 of the paper:
  DnnSdfGraph, Schedule, FLOPs, peakBuffer, gamma.

  Conventions:
  - Actors are indexed `Fin n` for explicit graph size.
  - All quantities are nonneg integers (Nat). Bytes are int8 baseline.
  - Schedules are *minimal valid* — no firing produces unconsumed tokens.
-/

import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Order.MinMax
import Mathlib.Data.Finset.Card

open BigOperators

namespace ParetoCorner

/-! ## DNN-SDF graph -/

/-- A chain-with-skips non-expanding DNN-SDF graph on `n` actors.

We encode actors `0, 1, ..., n-1` along a Hamiltonian chain. The skip
edges are not stored explicitly here because the chain-backpropagation
lemma only depends on the chain edge rate-product (skip edges give
parallel constraints that are weaker for our lower bounds; see paper §3).
-/
structure DnnSdfGraph (n : Nat) where
  /-- Per-actor FLOP cost per firing. -/
  flopCost : Fin n → Nat
  /-- Repetition vector entry per actor (unique smallest positive solution
      to the balance equations Γq = 0). -/
  rep : Fin n → Nat
  /-- All repetition counts are positive. -/
  rep_pos : ∀ i, 0 < rep i
  /-- Per-edge consume rate along the chain `(v_k, v_{k+1})` for `k < n-1`. -/
  cons : ∀ (k : Fin n), k.val + 1 < n → Nat
  /-- Per-edge produce rate along the chain. -/
  prod : ∀ (k : Fin n), k.val + 1 < n → Nat
  /-- All chain rates are positive. -/
  cons_pos : ∀ k h, 0 < cons k h
  prod_pos : ∀ k h, 0 < prod k h
  /-- Output edge byte size for actor v_i (its primary chain output). -/
  outputBytes : Fin n → Nat
  /-- Input edge byte size (cumulative over input edges). -/
  inputBytes : Fin n → Nat
  /-- Non-expanding property: output bytes ≤ input bytes per actor. -/
  nonExpanding : ∀ i, outputBytes i ≤ inputBytes i
  /-- Output bytes positive except possibly at the sink. -/
  outputBytes_pos : ∀ i : Fin n, i.val + 1 < n → 0 < outputBytes i
  /-- At least one actor has positive output bytes (used for non-emptiness
      of the `gamma` infimum). For chain-with-skips graphs this is the stem
      v_0 itself when n ≥ 2; we make the assumption explicit. -/
  someOutputPositive : ∃ i : Fin n, 0 < outputBytes i

/-! ## Canonical FLOP count and buffer -/

/-- The architecture's canonical FLOP count: each actor fires q_i times. -/
def DnnSdfGraph.flopsCanonical {n : Nat} (G : DnnSdfGraph n) : Nat :=
  ∑ i, G.rep i * G.flopCost i

/-! ## Schedule -/

/-- A *valid* schedule of `G`. Each actor `i` is fired `m i ≥ G.rep i`
    times. The schedule's peak buffer is given as a parameter (its
    derivation from the firing sequence is in `BufferFloor.lean`). -/
structure Schedule {n : Nat} (G : DnnSdfGraph n) where
  /-- Number of firings per actor (≥ rep). -/
  m : Fin n → Nat
  /-- Validity: every actor fires at least its repetition count. -/
  m_ge_rep : ∀ i, G.rep i ≤ m i
  /-- Peak SRAM in bytes during execution. -/
  peakBuffer : Nat

/-- FLOP count of a schedule. -/
def Schedule.flops {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) : Nat :=
  ∑ i, σ.m i * G.flopCost i

/-- The set of recomputed actors `R(σ) = {i : m_i > q_i}`. -/
def Schedule.recomputed {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) : Finset (Fin n) :=
  Finset.univ.filter (fun i => G.rep i < σ.m i)

/-- The recomputation count `ρ_i = m_i - q_i` for `i ∈ R(σ)`. -/
def Schedule.rho {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) (i : Fin n) : Nat :=
  σ.m i - G.rep i

/-- A schedule is *canonical* if it does not recompute any actor. -/
def Schedule.isCanonical {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) : Prop :=
  ∀ i, σ.m i = G.rep i

/-! ## Chain rate-product `α_{j → i}` -/

/-- Per-edge rate ratio `cons(k,k+1) / prod(k,k+1)` along the chain.
    Returns 1 when `k+1 = n` (no successor; vacuous edge). -/
noncomputable def DnnSdfGraph.edgeRatio {n : Nat} (G : DnnSdfGraph n) (k : Fin n) : ℚ :=
  if h : k.val + 1 < n then (G.cons k h : ℚ) / (G.prod k h : ℚ) else 1

/-- The rate-product along the chain segment `j → j+1 → ... → i` for `j ≤ i`.

For `j = i` the rate-product is 1 (empty product). For `j < i` it accumulates
`cons(k, k+1) / prod(k, k+1)` along the chain. We use rationals because
rate ratios may not be integers in general; the lemma statements then take
the ceiling for firing-count obligations. -/
noncomputable def DnnSdfGraph.alpha {n : Nat} (G : DnnSdfGraph n) (j i : Fin n) : ℚ :=
  if j.val ≤ i.val then
    -- product over k ∈ [j, i) of edgeRatio at k
    ∏ k ∈ Finset.univ.filter (fun k : Fin n => j.val ≤ k.val ∧ k.val < i.val),
      G.edgeRatio k
  else 0

/-- α at the diagonal is 1 (empty product). -/
lemma DnnSdfGraph.alpha_self {n : Nat} (G : DnnSdfGraph n) (i : Fin n) :
    G.alpha i i = 1 := by
  unfold DnnSdfGraph.alpha
  -- `i.val ≤ i.val` is true, so the `if` reduces to the product branch.
  rw [if_pos (le_refl _)]
  -- Show the filter is empty (then the empty product is 1).
  have hempty : (Finset.univ.filter
      (fun k : Fin n => i.val ≤ k.val ∧ k.val < i.val)) = ∅ := by
    apply Finset.filter_false_of_mem
    intro k _ hk
    -- hk : i.val ≤ k.val ∧ k.val < i.val — contradiction
    omega
  rw [hempty, Finset.prod_empty]

/-! ## Recompute cost γ(G) -/

/-- Chain-cost-up-to-i: `cc(i) = Σ_{j ≤ i} α_{j → i} F_j`.

Returns a rational because α_{j→i} is rational. -/
noncomputable def DnnSdfGraph.cc {n : Nat} (G : DnnSdfGraph n) (i : Fin n) : ℚ :=
  ∑ j : Fin n, if j.val ≤ i.val then G.alpha j i * (G.flopCost j : ℚ) else 0

/-- The non-empty Finset of actors with positive output bytes. Used as the
    domain of the `gamma` infimum. -/
noncomputable def DnnSdfGraph.positiveOutputActors {n : Nat} (G : DnnSdfGraph n) :
    Finset (Fin n) :=
  Finset.univ.filter (fun i : Fin n => 0 < G.outputBytes i)

/-- The set of actors with positive output bytes is non-empty
    (witness: `someOutputPositive`). -/
lemma DnnSdfGraph.positiveOutputActors_nonempty {n : Nat} (G : DnnSdfGraph n) :
    G.positiveOutputActors.Nonempty := by
  obtain ⟨i, hi⟩ := G.someOutputPositive
  refine ⟨i, ?_⟩
  rw [DnnSdfGraph.positiveOutputActors, Finset.mem_filter]
  exact ⟨Finset.mem_univ _, hi⟩

/-- The graph's recompute cost: minimum FLOPs-per-byte ratio over all
    actors with positive output. -/
noncomputable def DnnSdfGraph.gamma {n : Nat} (G : DnnSdfGraph n) : ℚ :=
  Finset.inf' G.positiveOutputActors G.positiveOutputActors_nonempty
    (fun i => G.cc i / (G.outputBytes i : ℚ))

/-! ## CASAP-minimum buffer

The CASAP buffer `S_*(G)` is the minimum peak buffer over canonical
schedules. Its constructive existence is provided by the CASAP algorithm
(`BufferFloor.lean`); for the present file we *axiomatize* the existence
of at least one canonical schedule, which is a standard SDF result.

This axiom is enumerated in the paper's "axioms list" (Appendix A of
the supplementary). It is a substantive but well-documented obligation:
existence of a valid canonical schedule for any chain-with-skips
non-expanding DNN-SDF graph follows from Lee–Messerschmitt 1987.
-/

/-- **AXIOM (existence of canonical schedule)**: every chain-with-skips
    non-expanding DNN-SDF graph admits at least one canonical schedule.
    This is a standard SDF result (Lee–Messerschmitt 1987); we make it
    an explicit axiom rather than provide a constructive existence
    proof, which would require defining the firing-sequence semantics. -/
axiom canonicalSchedule_exists {n : Nat} (G : DnnSdfGraph n) :
    ∃ σ : Schedule G, σ.isCanonical

/-- The CASAP buffer `S_*(G)` is the minimum peak buffer over canonical
    schedules. Defined via `Nat.find` over a non-empty set of achievable
    buffer values. We use classical decidability since the existential
    over Schedule structures is not constructively decidable. -/
noncomputable def DnnSdfGraph.bufferCanonical {n : Nat} (G : DnnSdfGraph n) : Nat :=
  letI : DecidablePred (fun S : Nat =>
      ∃ σ : Schedule G, σ.isCanonical ∧ σ.peakBuffer ≤ S) :=
    fun _ => Classical.propDecidable _
  Nat.find (P := fun S => ∃ σ : Schedule G, σ.isCanonical ∧ σ.peakBuffer ≤ S)
    (by
      -- Existence: take any canonical schedule σ₀ (axiom) and let S = σ₀.peakBuffer.
      obtain ⟨σ₀, hσ₀⟩ := canonicalSchedule_exists G
      exact ⟨σ₀.peakBuffer, σ₀, hσ₀, le_refl _⟩)

/-- A schedule σ is *minimal* if it fires each actor exactly its repetition
    count (i.e., is canonical). This is the formal version of "minimal
    valid schedule" used in the paper to rule out wasted recomputation
    of zero-FLOP actors. -/
def Schedule.isMinimal {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) : Prop :=
  σ.isCanonical

/-! ## Enumeration of axioms in this formalization

For paper auditability we list every `axiom` declaration in the
ParetoCorner namespace. Theorems 1(a) and 1(c) depend only on the
SDF axioms below; they contain no `sorry`.

| Axiom                          | File                | Statement                                                           |
|--------------------------------|---------------------|---------------------------------------------------------------------|
| `canonicalSchedule_exists`     | Definitions.lean    | Every chain-with-skips graph has a canonical schedule (Lee-M. 1987) |
| `tokenBalance`                 | ChainBackprop.lean  | `cons · m_{j+1} ≤ prod · m_j` along chain edges (SDF semantics)     |
| `rep_balance`                  | ChainBackprop.lean  | `cons · q_{j+1} = prod · q_j` (canonical balance equation)          |
| `bufferFloorAtCanonical`       | BufferFloor.lean    | Theorem 1(b): canonical schedules have peak buffer ≥ S_*(G)         |
| `casap_exists`                 | BufferFloor.lean    | A canonical schedule attains S_*(G)                                 |

`canonicalSchedule_exists`, `tokenBalance`, `rep_balance` are standard SDF
results derivable from the firing-sequence semantics (Lee-Messerschmitt 1987).
`bufferFloorAtCanonical` and `casap_exists` together encode Theorem 1(b),
whose self-contained proof appears in supplementary Appendix A; the Lean
port requires `Schedule` to carry a firing-order list (a multi-week
structural refactor).
-/

end ParetoCorner
