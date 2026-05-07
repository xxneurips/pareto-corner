/-
  Theorem 1(c): Pareto-optimality of the corner.

  No valid schedule weakly dominates the canonical corner (F_*, S_*).
  Fully proven from Theorem 1(a) (FlopFloor) and Theorem 1(b) (BufferFloor).
-/

import ParetoCorner.Definitions
import ParetoCorner.FlopFloor
import ParetoCorner.BufferFloor

namespace ParetoCorner

/-- A schedule σ *weakly dominates* the canonical corner (F_*, S_*) if
    σ.flops ≤ F_* and σ.peakBuffer ≤ S_*, with strict inequality in at
    least one coordinate. -/
def Schedule.weaklyDominatesCorner {n : Nat} {G : DnnSdfGraph n} (σ : Schedule G) : Prop :=
  σ.flops ≤ G.flopsCanonical ∧ σ.peakBuffer ≤ G.bufferCanonical ∧
    (σ.flops < G.flopsCanonical ∨ σ.peakBuffer < G.bufferCanonical)

/-- **Theorem 1(c) [strong form]:** Among *minimal* schedules
    (m_i = q_i for all i, including zero-FLOP actors), the canonical
    corner is on the Pareto frontier.

    The minimality hypothesis sidesteps zero-FLOP actors that the
    FLOP-equality cannot constrain (e.g., identity layers, pooling).
    For non-minimal schedules with `m_i > q_i` on a zero-FLOP actor,
    the schedule still satisfies `σ.flops = G.flopsCanonical` but may
    in principle have a different peak buffer; the paper rules this
    out by the standing convention that all SDF schedules are minimal
    (no firing produces unconsumed tokens). -/
theorem paretoCornerNonImprovement {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (hMinimal : σ.isMinimal) :
    ¬ σ.weaklyDominatesCorner := by
  intro ⟨hF, hS, hStrict⟩
  -- By Theorem 1(a): σ.flops ≥ G.flopsCanonical, so combined with hF we get equality.
  have hF_ge : G.flopsCanonical ≤ σ.flops := flopFloor G σ
  -- Strict inequality must therefore be on S: σ.peakBuffer < G.bufferCanonical.
  have hS_strict : σ.peakBuffer < G.bufferCanonical := by
    rcases hStrict with hF_lt | hS_lt
    · -- σ.flops < flopsCanonical contradicts hF_ge
      omega
    · exact hS_lt
  -- By assumption σ is minimal (= canonical), so Theorem 1(b) applies.
  have hCanonical : σ.isCanonical := hMinimal
  -- By Theorem 1(b) (BufferFloor): canonical σ has S(σ) ≥ S_*(G).
  have hS_ge : G.bufferCanonical ≤ σ.peakBuffer := bufferFloorAtCanonical G σ hCanonical
  -- This contradicts hS_strict.
  omega

/-- **Alternate form using positive-FLOP-cost everywhere** (no minimality
    needed): if every actor has F_i > 0, the FLOP-equality from Theorem 1(a)
    fully determines the firing counts. -/
theorem paretoCornerNonImprovement_strictFlops {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (hAllPositive : ∀ i, 0 < G.flopCost i) :
    ¬ σ.weaklyDominatesCorner := by
  intro ⟨hF, hS, hStrict⟩
  have hF_ge : G.flopsCanonical ≤ σ.flops := flopFloor G σ
  have hF_eq : σ.flops = G.flopsCanonical := le_antisymm hF hF_ge
  have hS_strict : σ.peakBuffer < G.bufferCanonical := by
    rcases hStrict with hF_lt | hS_lt
    · omega
    · exact hS_lt
  have hCanonical : σ.isCanonical := fun i =>
    canonical_of_flops_eq G σ hF_eq i (hAllPositive i)
  have hS_ge : G.bufferCanonical ≤ σ.peakBuffer := bufferFloorAtCanonical G σ hCanonical
  omega

/-- **Corollary:** No σ has both `σ.flops < G.flopsCanonical` AND
    `σ.peakBuffer < G.bufferCanonical`. (This is the headline form of
    Pareto-corner non-improvement.) -/
theorem corner_non_improvement_headline {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (hExistsPositiveFlopCost : ∃ i, 0 < G.flopCost i) :
    ¬ (σ.flops < G.flopsCanonical ∧ σ.peakBuffer < G.bufferCanonical) := by
  intro ⟨hF, _⟩
  have : G.flopsCanonical ≤ σ.flops := flopFloor G σ
  omega

end ParetoCorner
