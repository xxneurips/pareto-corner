/-
  Theorem 2: Loose harmonic lower bound (uniform-density case).

  Under uniform FLOP density F_i = F_*/n, every graph-internal σ with
  S(σ) < S_*(G) satisfies F - F_* ≥ (F_*/2) · (S_* - S)/S.

  STATUS: skeleton; proof relies on a "k-pass" lemma that formalizes
  the buffer-constraint-implies-pass-count argument from the paper.
-/

import ParetoCorner.Definitions
import ParetoCorner.BufferFloor

namespace ParetoCorner

/-- A graph G has *uniform FLOP density* if every actor has the same
    per-firing FLOP cost F_*/n. -/
def DnnSdfGraph.hasUniformDensity {n : Nat} (G : DnnSdfGraph n) : Prop :=
  ∀ i j : Fin n, G.flopCost i = G.flopCost j

/-- **Key sub-lemma**: under uniform density and buffer S(σ) ≤ S_*/k for
    integer k ≥ 1, σ must process the chain in at least k "passes,"
    where pass i ∈ {1, ..., k} re-fires layers 1..⌈(i-1)·n/k⌉. -/
theorem k_pass_lemma {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_uniform : G.hasUniformDensity) (k : Nat) (hk : 1 ≤ k)
    (h_buffer : σ.peakBuffer * k ≤ G.bufferCanonical) :
    -- σ.flops contains at least the k-pass overhead
    (G.flopsCanonical * (k - 1) : ℤ) / 2 ≤ (σ.flops : ℤ) - (G.flopsCanonical : ℤ) := by
  -- Strategy:
  -- 1. Define a "pass" formally: maximal interval during which the
  --    live-token set on the chain prefix decreases.
  -- 2. Show: with peak buffer S, the schedule has at least k passes
  --    where k = ⌈S_* / S⌉.
  -- 3. Each pass i ∈ {2, ..., k} requires re-firing the upstream segment
  --    freed in pass i-1.
  -- 4. Under uniform density, pass i's overhead is F_* · (i-1)/k.
  -- 5. Sum: Σ_{i=1}^{k} F_*(i-1)/k = F_*(k-1)/2.
  --
  -- This is the most architecturally specific lemma in the formalization
  -- and is the bottleneck for proving Theorem 2 universally (without
  -- uniform density).
  sorry

/-- **Theorem 2:** Loose harmonic lower bound, uniform-density case. -/
theorem looseHarmonicBound {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_uniform : G.hasUniformDensity) (h_pos : 0 < σ.peakBuffer)
    (h_below : σ.peakBuffer ≤ G.bufferCanonical) :
    (G.flopsCanonical : ℚ) * (G.bufferCanonical - σ.peakBuffer : ℚ) / (2 * σ.peakBuffer : ℚ)
      ≤ (σ.flops - G.flopsCanonical : ℚ) := by
  -- Strategy: apply k_pass_lemma with k = ⌈S_*/S⌉ and rearrange.
  -- For S(σ) = S_*/k exactly: F - F_* ≥ F_* (k-1)/2 = F_* (S_*/S - 1)/2 = F_*(S_* - S)/(2S).
  -- For S(σ) ≤ S_*/k: substitute the inequality and the bound goes through.
  sorry

end ParetoCorner
