/-
  Theorem 1(a): FLOP floor.

  Every valid schedule does at least the canonical FLOP work.
  This is immediate from validity and is fully proven (no `sorry`).
-/

import ParetoCorner.Definitions

open BigOperators

namespace ParetoCorner

/-- **Theorem 1(a):** For every valid schedule σ of a DNN-SDF graph G,
    the schedule's FLOP count is at least the canonical FLOP count. -/
theorem flopFloor {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G) :
    G.flopsCanonical ≤ σ.flops := by
  unfold DnnSdfGraph.flopsCanonical Schedule.flops
  apply Finset.sum_le_sum
  intro i _
  exact Nat.mul_le_mul_right _ (σ.m_ge_rep i)

/-- **Corollary:** The FLOP floor is tight when the schedule is canonical. -/
theorem flopFloor_tight_iff_canonical {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G) :
    σ.flops = G.flopsCanonical ↔ ∀ i, σ.m i * G.flopCost i = G.rep i * G.flopCost i := by
  unfold DnnSdfGraph.flopsCanonical Schedule.flops
  constructor
  · intro h
    -- Equality of sums plus pointwise ≥ implies pointwise =
    have hpoint : ∀ i ∈ Finset.univ,
      G.rep i * G.flopCost i ≤ σ.m i * G.flopCost i := by
      intro i _
      exact Nat.mul_le_mul_right _ (σ.m_ge_rep i)
    intro i
    by_contra hne
    -- If some i has strict inequality, the sum is strictly larger
    have hi_strict : G.rep i * G.flopCost i < σ.m i * G.flopCost i := by
      have hi_ge : G.rep i * G.flopCost i ≤ σ.m i * G.flopCost i := hpoint i (Finset.mem_univ _)
      omega
    have : ∑ j, G.rep j * G.flopCost j < ∑ j, σ.m j * G.flopCost j :=
      Finset.sum_lt_sum_of_nonempty_of_lt_of_le
        (Finset.univ_nonempty_iff.mpr ⟨i⟩)
        (fun j _ => hpoint j (Finset.mem_univ _))
        ⟨i, Finset.mem_univ _, hi_strict⟩
    omega
  · intro h
    apply Finset.sum_congr rfl
    intros i _
    rw [h]

/-- **Useful corollary for Pareto-optimality:** If `σ.flops = G.flopsCanonical`
    and there exists at least one actor with positive FLOP cost, then σ is
    canonical (m i = q i) on every actor with F_i > 0. -/
theorem canonical_of_flops_eq {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h : σ.flops = G.flopsCanonical) :
    ∀ i, 0 < G.flopCost i → σ.m i = G.rep i := by
  intro i hFi
  have hpoint := (flopFloor_tight_iff_canonical G σ).mp h i
  -- σ.m i * G.flopCost i = G.rep i * G.flopCost i with G.flopCost i > 0 forces equality
  have hge : G.rep i ≤ σ.m i := σ.m_ge_rep i
  have : σ.m i * G.flopCost i = G.rep i * G.flopCost i := hpoint
  have := Nat.eq_of_mul_eq_mul_right hFi this
  omega

end ParetoCorner
