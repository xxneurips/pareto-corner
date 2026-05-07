/-
  Theorem 1(d): Linear FLOP-SRAM trade-off.

  For graph-internal σ recomputing R(σ) ⊆ V:
    F(σ) - F_*(G) ≥ (γ(G) / |R(σ)|) · (S_*(G) - S(σ))^+

  STATUS: skeleton with proof structure; the key sub-lemma
  `buffer_reduction_bound` requires a careful peak-buffer-reduction
  argument that is the longest non-trivial Lean obligation in this
  formalization (~3 days of focused work).
-/

import ParetoCorner.Definitions
import ParetoCorner.ChainBackprop
import ParetoCorner.BufferFloor

namespace ParetoCorner

/-- **Key buffer-reduction lemma**: the canonical-CASAP-buffer minus the
    schedule's peak buffer is at most the cumulative re-supplied byte
    count via recomputation.

    Specifically: each recomputation event of actor v_i can "free" at most
    w_i bytes from the live-token set during the period the recomputed
    token is absent. Summing over R(σ) gives the bound.

    Open structural obligation: this lemma requires firing-sequence
    semantics (which token is live at which time), but `Schedule` only
    carries firing counts. A full Lean proof requires redefining
    `Schedule` to carry an ordered list of firings
    (`firings : List (Fin n)` with appropriate validity), then defining
    `liveBytes : Schedule G → Nat → Nat` and `peakBuffer` as the max
    over time. The same structural refactor is needed for Theorem 1(b)
    (CASAP exchange argument) and this `buffer_reduction_bound`.
-/
theorem buffer_reduction_bound {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G) :
    (G.bufferCanonical : ℤ) - (σ.peakBuffer : ℤ) ≤
      ∑ i in σ.recomputed, (σ.rho i : ℤ) * (G.outputBytes i : ℤ) := by
  -- Strategy (requires firing-sequence semantics, currently absent):
  -- 1. For each time t in σ, compute live-byte sum L(t) = Σ_e buf_e^σ(t) w(e).
  -- 2. peakBuffer = max_t L(t).
  -- 3. Show: at every time t, L(t) ≥ L_canonical(t) - Σ_{recomputed-and-currently-freed} ρ_i w_i.
  -- 4. Take min_t to get peakBuffer ≥ S_* - (max recomputation-freed-bytes-at-some-instant).
  -- 5. The maximum-freed-bytes-at-an-instant is at most Σ_R ρ_i w_i (very loose; could be tightened).
  --
  -- This is the longest non-trivial proof in the formalization. Estimated
  -- ~3 days of Lean work *after* the firing-sequence refactor of `Schedule`.
  sorry

/-- **Theorem 1(d):** Linear FLOP-SRAM trade-off for graph-internal schedules.

    The bound holds when the buffer reduction is non-negative
    (`σ.peakBuffer ≤ G.bufferCanonical`); otherwise the RHS is non-positive
    and the bound is trivial. We state the version under the assumption
    that the schedule lies below the corner, matching the paper's
    "graph-internal σ" hypothesis. -/
theorem linearBound {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_pos : (σ.recomputed.card : ℕ) > 0)
    (h_below : σ.peakBuffer ≤ G.bufferCanonical) :
    (G.gamma / (σ.recomputed.card : ℚ)) * ((G.bufferCanonical : ℤ) - (σ.peakBuffer : ℤ) : ℚ)
      ≤ ((σ.flops : ℤ) - (G.flopsCanonical : ℤ) : ℚ) := by
  -- Strategy (mechanical chain, modulo `chainBackprop` from Lemma 1
  -- and `buffer_reduction_bound`):
  --
  -- 1. From `flop_overhead_single_witness` for any i ∈ R(σ):
  --      σ.flops - flopsCanonical ≥ ρ_i · cc(i)                       (*)
  --
  -- 2. Pick i* = argmax_{i ∈ R(σ)} ρ_i · w_i. (R(σ) is non-empty by h_pos;
  --    the argmax exists in a finite Finset by `Finset.exists_max_image`.)
  --
  -- 3. By Definition `gamma`: γ ≤ cc(i*)/w_i*  (since i* ∈ R(σ) ⊆
  --    positiveOutputActors). So:
  --      ρ_i* · cc(i*) ≥ ρ_i* · w_i* · γ                              (**)
  --
  -- 4. By max ≥ avg over R(σ):
  --      ρ_i* · w_i* ≥ (Σ_{i ∈ R(σ)} ρ_i · w_i) / |R(σ)|              (***)
  --
  -- 5. By `buffer_reduction_bound`:
  --      Σ_{i ∈ R(σ)} ρ_i · w_i ≥ S_*(G) - σ.peakBuffer               (#)
  --
  -- 6. Chain (*), (**), (***), (#):
  --      σ.flops - flopsCanonical
  --        ≥ ρ_i* · cc(i*)              [(*)]
  --        ≥ γ · ρ_i* · w_i*            [(**)]
  --        ≥ γ · (S_* - S) / |R(σ)|     [(***), (#)]
  --      = (γ / |R(σ)|) · (S_* - S).    ✓
  --
  -- Status: chain (*) reduces to `flop_overhead_single_witness`
  -- (currently `sorry`); chain (#) is `buffer_reduction_bound` (currently
  -- `sorry`). Steps (**), (***) are pure arithmetic on rationals/integers
  -- with `Finset.exists_max_image` and `div_le_one`; estimate ~1 day to
  -- close mechanically once the two named lemmas are available.
  sorry

/-- **Universal corollary:** divides by `n` (chain length) for the bound
    that holds for every valid schedule, regardless of |R(σ)|.
    Requires `n > 0` and `σ.peakBuffer ≤ G.bufferCanonical`. -/
theorem linearBound_universal {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (hn : 0 < n) (h_below : σ.peakBuffer ≤ G.bufferCanonical) :
    (G.gamma / (n : ℚ)) * ((G.bufferCanonical : ℤ) - (σ.peakBuffer : ℤ) : ℚ)
      ≤ ((σ.flops : ℤ) - (G.flopsCanonical : ℤ) : ℚ) := by
  -- Case 1: R(σ) = ∅. Then σ is canonical, so σ.flops = flopsCanonical
  -- (FLOP floor is tight) and σ.peakBuffer ≥ S_*(G) (buffer floor at
  -- canonical FLOPs). The LHS is then non-positive (since γ ≥ 0 and
  -- S_* - σ.peakBuffer ≤ 0), so the bound holds trivially.
  --
  -- Case 2: R(σ) ≠ ∅. Apply linearBound, then use |R(σ)| ≤ n to weaken
  -- (γ / |R|) · (S_* - S) ≥ (γ / n) · (S_* - S) when (S_* - S) ≥ 0.
  sorry

/-- **Single-actor specialization:** when |R(σ)| = 1, the bound simplifies
    to the strong form γ(G) · (S_* - S). -/
theorem linearBound_single_actor {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_one : σ.recomputed.card = 1)
    (h_below : σ.peakBuffer ≤ G.bufferCanonical) :
    G.gamma * ((G.bufferCanonical : ℤ) - (σ.peakBuffer : ℤ) : ℚ)
      ≤ ((σ.flops : ℤ) - (G.flopsCanonical : ℤ) : ℚ) := by
  have := linearBound G σ (by rw [h_one]; norm_num) h_below
  rw [h_one] at this
  simpa using this

end ParetoCorner
