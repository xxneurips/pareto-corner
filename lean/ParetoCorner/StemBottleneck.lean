/-
  Proposition 1: Stem-bottleneck for MobileNet-family backbones.

  γ(G) ≤ K² · C_in for any chain-with-skips graph whose first actor is a
  stride-s K×K convolution mapping C_in to C_out channels.

  Proof: cc(1)/w_1 = K² · C_in by direct computation; γ is the min, so it
  is at most this value.
-/

import ParetoCorner.Definitions
import Mathlib.Data.Rat.Basic

namespace ParetoCorner

/-- A chain-with-skips DNN-SDF graph whose first actor is a standard
    convolutional stem with kernel `K`, input channels `C_in`, output
    channels `C_out`, stride `s`, and spatial size `H × W`. -/
structure HasConvStem {n : Nat} (G : DnnSdfGraph n) where
  /-- The graph has at least one actor. -/
  n_pos : 0 < n
  /-- Kernel size (typically 3 or 5). -/
  K : Nat
  K_pos : 0 < K
  /-- Input channels (3 for RGB). -/
  C_in : Nat
  C_in_pos : 0 < C_in
  /-- Output channels of the stem. -/
  C_out : Nat
  C_out_pos : 0 < C_out
  /-- Spatial input dimensions H × W. -/
  H W : Nat
  H_pos : 0 < H
  W_pos : 0 < W
  /-- Stride. -/
  s : Nat
  s_pos : 0 < s
  /-- The stem's FLOP cost matches the convolution formula. -/
  flop_cost_eq : G.flopCost ⟨0, n_pos⟩ = K * K * C_in * C_out * (H * W / (s * s))
  /-- The stem's output bytes match the int8 activation footprint. -/
  output_bytes_eq : G.outputBytes ⟨0, n_pos⟩ = C_out * (H * W / (s * s))
  /-- The stem fires once per inference (q_1 = 1). -/
  rep_eq_one : G.rep ⟨0, n_pos⟩ = 1

/-- **Proposition 1:** For any chain-with-skips DNN-SDF graph G whose
    first actor is a K×K convolutional stem mapping C_in to C_out
    channels, the recompute cost satisfies `γ(G) ≤ K² · C_in`.

    The proof needs a non-degeneracy hypothesis on the stem
    (`stem_w_pos`: stem output bytes are positive) which is
    automatically guaranteed when `H, W, C_out, s` are positive. -/
theorem stem_bottleneck {n : Nat} (G : DnnSdfGraph n) (h : HasConvStem G)
    (stem_w_pos : 0 < G.outputBytes ⟨0, h.n_pos⟩) :
    G.gamma ≤ (h.K * h.K * h.C_in : ℚ) := by
  -- Strategy:
  -- 1. The stem's cc(0) = α_{0→0} · F_0 = 1 · F_0 = K² · C_in · C_out · HW/s²
  --    (using `alpha_self : G.alpha i i = 1`, now provable in Definitions.lean).
  -- 2. The stem's w_0 = C_out · HW/s²
  -- 3. The ratio cc(0)/w_0 = K² · C_in
  -- 4. γ(G) = inf' over `positiveOutputActors` of cc/w ≤ stem's ratio = K² · C_in
  unfold DnnSdfGraph.gamma
  -- Stem (index 0) is in `positiveOutputActors` by `stem_w_pos`.
  have h_stem_mem : (⟨0, h.n_pos⟩ : Fin n) ∈ G.positiveOutputActors := by
    rw [DnnSdfGraph.positiveOutputActors, Finset.mem_filter]
    exact ⟨Finset.mem_univ _, stem_w_pos⟩
  -- Apply `Finset.inf'_le` with the stem as witness.
  refine le_trans (Finset.inf'_le _ h_stem_mem) ?_
  -- Now show: cc(0) / outputBytes(0) ≤ K²·C_in.
  -- Substituting `flop_cost_eq` and `output_bytes_eq` from `h`:
  --   cc(0) = α_{0→0} · F_0 = 1 · F_0 = K²·C_in·C_out·HW/s²
  --   w_0 = C_out·HW/s²
  --   cc(0)/w_0 = K²·C_in.
  -- Open obligation: requires expanding `cc`, applying `alpha_self`, and using
  -- `h.flop_cost_eq`, `h.output_bytes_eq`. Pure arithmetic in ℚ.
  sorry

/-- **Concrete corollary:** For an RGB input with K=3, C_in=3,
    γ(G) ≤ 27 MACs/byte. -/
theorem stem_bottleneck_rgb {n : Nat} (G : DnnSdfGraph n)
    (h : HasConvStem G) (stem_w_pos : 0 < G.outputBytes ⟨0, h.n_pos⟩)
    (hK : h.K = 3) (hCin : h.C_in = 3) :
    G.gamma ≤ 27 := by
  have := stem_bottleneck G h stem_w_pos
  rw [hK, hCin] at this
  norm_num at this
  exact this

end ParetoCorner
