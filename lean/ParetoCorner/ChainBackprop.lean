/-
  Lemma 1: Chain-backpropagation of recomputation.

  For any minimal valid schedule σ with m_i > q_i, every predecessor v_j
  (j ≤ i along the chain) satisfies m_j ≥ q_j + ρ_i · α_{j → i}.

  STATUS: structured outline. Proof reduces to a finite induction over
  chain edges combined with the SDF token-balance axiom (`tokenBalance`)
  encoding that every consumed token was previously produced.
-/

import ParetoCorner.Definitions
import Mathlib.Data.Rat.Cast.Order

namespace ParetoCorner

open BigOperators

/-! ## SDF token-balance axiom

The `Schedule` structure carries firing counts but no firing-sequence
order. The fact that "every firing of v_{j+1} consumes tokens previously
produced by v_j" cannot be derived from firing counts alone — it requires
firing-sequence semantics. We axiomatize the resulting integer balance
constraint, which is the *consequence* of the SDF semantics that downstream
proofs need:

    ∀ schedule σ, edge (j, j+1):  cons(j,j+1) · m_{j+1}  ≤  prod(j,j+1) · m_j

This axiom, together with `rep_balance` (the canonical balance equation
on the repetition vector q), is sufficient for the chain-backpropagation
lemma.
-/

/-- **AXIOM (SDF token balance for valid schedules)**: for any valid
    schedule σ and any chain edge (j, j+1), the cumulative tokens consumed
    by v_{j+1} cannot exceed the cumulative tokens produced by v_j. This
    is the integer-form consequence of SDF firing semantics; it is the
    rate-corrected statement of "every firing of v_{j+1} is preceded by
    enough firings of v_j to supply its inputs". -/
axiom tokenBalance {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (j : Fin n) (hj : j.val + 1 < n) :
    G.cons j hj * σ.m ⟨j.val + 1, hj⟩ ≤ G.prod j hj * σ.m j

/-- **AXIOM (canonical balance)**: the repetition vector q satisfies the
    SDF balance equations on every chain edge. This is Lee–Messerschmitt's
    classical balance equation; we make it an axiom rather than derive it
    from the unspecified balance-equation solver. -/
axiom rep_balance {n : Nat} (G : DnnSdfGraph n)
    (j : Fin n) (hj : j.val + 1 < n) :
    G.cons j hj * G.rep ⟨j.val + 1, hj⟩ = G.prod j hj * G.rep j

/-- **Step lemma**: rate-product `α_{j → i}` satisfies the recursion
    `α_{j → i} = (cons(j,j+1) / prod(j,j+1)) · α_{j+1 → i}` for `j < i`. -/
lemma alpha_step {n : Nat} (G : DnnSdfGraph n) (j i : Fin n)
    (hj : j.val + 1 < n) (hji : j.val < i.val) :
    G.alpha j i = G.edgeRatio j * G.alpha ⟨j.val + 1, hj⟩ i := by
  -- α_{j → i} = ∏_{k : j ≤ k.val < i.val} edgeRatio k
  --          = edgeRatio j · ∏_{k : j+1 ≤ k.val < i.val} edgeRatio k
  --          = edgeRatio j · α_{j+1 → i}
  -- Mathlib: factor out k = j from the Finset.prod.
  -- Open obligation: complete via `Finset.prod_filter` and `Finset.insert` manipulation.
  -- The set {k : j ≤ k.val < i.val} = {j} ∪ {k : j+1 ≤ k.val < i.val}
  -- when j.val < i.val.
  sorry

/-- The chain-backpropagation lemma in its rational form.

For a valid schedule σ (every firing produces consumed tokens, encoded
by `tokenBalance`), recomputing actor i propagates back through the chain
with rate-product α_{j → i}. We state this for rational firing-counts;
integer-rounded versions follow by ceiling and are weaker.
-/
theorem chainBackprop {n : Nat} (G : DnnSdfGraph n)
    (σ : Schedule G) (i j : Fin n) (hji : j.val ≤ i.val)
    (h_recomp : G.rep i < σ.m i) :
    (G.rep j : ℚ) + (σ.rho i : ℚ) * G.alpha j i ≤ (σ.m j : ℚ) := by
  -- Strategy: strong induction on the chain distance d := i.val - j.val.
  -- We use `Nat.strongRecOn` / `Nat.le_induction` on the gap.
  --
  -- Base case (d = 0, j = i):
  --   α_{i → i} = 1 by `alpha_self`.
  --   Claim: q_i + ρ_i · 1 ≤ m_i, i.e., q_i + (m_i - q_i) ≤ m_i. ✓
  --
  -- Step case (d = (j'+1) - j', so j+1 ≤ i):
  --   By IH applied to j+1 ≤ i:
  --     q_{j+1} + ρ_i · α_{j+1 → i} ≤ m_{j+1}              (in ℚ)
  --   By `tokenBalance` at edge (j, j+1):
  --     cons · m_{j+1} ≤ prod · m_j
  --   So:
  --     m_j ≥ (cons / prod) · m_{j+1}
  --         ≥ (cons / prod) · (q_{j+1} + ρ_i · α_{j+1 → i})
  --         = (cons / prod) · q_{j+1} + ρ_i · (cons / prod) · α_{j+1 → i}
  --   By `rep_balance`: cons · q_{j+1} = prod · q_j, so
  --     (cons / prod) · q_{j+1} = q_j  (in ℚ; valid since prod > 0).
  --   By `alpha_step`:
  --     (cons / prod) · α_{j+1 → i} = edgeRatio j · α_{j+1 → i} = α_{j → i}.
  --   Substituting:
  --     m_j ≥ q_j + ρ_i · α_{j → i}.   ✓
  --
  -- Status: structured outline complete; remaining `sorry` is the
  -- Mathlib-level `Finset.prod` manipulation in `alpha_step` and the
  -- ℕ→ℚ inequality bookkeeping (≈ 2 days of focused Lean work).
  sorry

/-- **Useful corollary:** for any single recomputed actor i ∈ R(σ), the
    FLOP overhead is at least ρ_i · cc(i). -/
theorem flop_overhead_single_witness {n : Nat} (G : DnnSdfGraph n)
    (σ : Schedule G) (i : Fin n) (hi_rec : G.rep i < σ.m i) :
    (σ.rho i : ℚ) * G.cc i ≤ (σ.flops : ℚ) - (G.flopsCanonical : ℚ) := by
  -- Set up the algebraic chain:
  --   σ.flops - flopsCanonical
  --     = Σ_j (m_j F_j - q_j F_j)
  --     = Σ_j (m_j - q_j) F_j                                   [pulling out F_j]
  --     ≥ Σ_{j ≤ i} (m_j - q_j) F_j                              [drop j > i terms, all ≥ 0]
  --     ≥ Σ_{j ≤ i} ρ_i · α_{j → i} · F_j                        [by chainBackprop]
  --     = ρ_i · Σ_{j ≤ i} α_{j → i} · F_j                        [factor]
  --     = ρ_i · cc(i)                                            [definition of cc]
  --
  -- Each step is a Mathlib `Finset.sum_le_sum` / `Finset.sum_filter` /
  -- `Finset.mul_sum` rewrite. The proof is mechanical but needs care
  -- with the ℕ→ℚ casts.
  sorry

/-- **Why we use a single witness, not a sum over R(σ).**

  Lemma `chainBackprop` gives `m_j - q_j ≥ ρ_i · α_{j → i}` for each
  i ∈ R(σ) with i ≥ j. Different i's impose constraints on the *same*
  m_j, so taking the maximum (not the sum) over R(σ) is the correct
  combination:

      m_j - q_j ≥ max_{i ∈ R, i ≥ j} ρ_i · α_{j → i}

  Therefore the per-actor lower bound on F - F_* via Σ_{i ∈ R} ρ_i · cc(i)
  is **invalid** (it would treat the constraints as independent).
  Only the per-witness bound `F - F_* ≥ ρ_i · cc(i)` for any single
  i ∈ R is valid.

  Maximizing over i ∈ R gives the tightest single-witness bound, which
  is what `LinearBound.lean` uses. -/

end ParetoCorner
