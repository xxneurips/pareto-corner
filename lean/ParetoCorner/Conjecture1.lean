/-
  Conjecture 1: Tight harmonic asymptotic Φ_G.

  STATUS: open. This file states the conjecture and three concrete proof
  attempts (each as a `theorem` with `sorry`), so subsequent work can
  attack any of them and close the gap.

  The empirical evidence on MobileNetV2 fits a harmonic regression
  F/F_* - 1 ≈ 0.021 · (S_*/S - 1) with R² = 0.999, motivating the
  conjecture. Proving it would close the central open question.

  ## Three candidate proof approaches

  CASAP-critical-cut argument: identify the actor v_{i*} whose output
  saturates the canonical CASAP peak buffer (so w_{i*} = S_*), then
  show that any schedule with peak buffer S < S_* must process
  v_{i*}'s output in chunks of size at most S, requiring
  ⌈w_{i*}/S⌉ - 1 = (S_* - S)/S recomputations of layers 1..i*. Total
  overhead: ((S_* - S)/S) · cc(i*). This gives c_h(G) = cc(i*)/F_*.
  For MobileNetV2: cc(i*) ≈ 0.5 · F_*, giving c_h ≈ 0.5, overshooting
  the empirical 0.021 by 25×. The bound is correct but loose.

  Halo-aware extension: define an extended graph G' that includes
  spatial-partition actors and re-derive γ(G') and the linear bound
  on G'. The extended-graph constants account for halo overhead and
  predict the smaller empirical c_h.

  Information-theoretic: encode the chain computation as a streaming
  algorithm with limited working memory and prove a Hong-Kung-style
  bound F · S ≥ Ω(N · h(N/S)) where h is some smoothing function;
  specialize h for chain-with-skips.

  The first approach is the closest to provable; the latter two are
  longer-horizon research directions.
-/

import ParetoCorner.Definitions
import ParetoCorner.ChainBackprop
import ParetoCorner.LinearBound

namespace ParetoCorner

/-- **Conjecture 1 (tight harmonic asymptotic).**

There exists an architecture-dependent constant `c_h(G) ∈ (0, 1)` such
that for every valid graph-internal schedule σ with S(σ) < S_*(G):

    F(σ) - F_*(G) ≥ c_h(G) · F_*(G) · (S_*(G) - S(σ)) / S(σ).

Empirical evidence on MobileNetV2 ImageNet (n = 4 measurements) gives
c_h(MobileNetV2) ≈ 0.021 with R² = 0.999. Architecture-specific
dependence of c_h on graph properties is open. -/
def Conjecture1 (G : DnnSdfGraph 53) (σ : Schedule G) (c_h : ℚ) : Prop :=
  0 < c_h ∧ c_h < 1 ∧
  σ.peakBuffer < G.bufferCanonical →
  c_h * (G.flopsCanonical : ℚ) * (G.bufferCanonical - σ.peakBuffer : ℚ) / (σ.peakBuffer : ℚ)
    ≤ (σ.flops - G.flopsCanonical : ℚ)

/-! ## CASAP-critical-cut approach

Define the CASAP-critical actor `i_critical(G)` as the actor whose
live-token contribution saturates the peak buffer under CASAP.
Provable bound: `c_h ≥ cc(i_critical) / F_*`, which for MobileNetV2
gives `c_h ≥ 0.5` (loose relative to the empirical 0.021, but a real
lower bound).
-/

/-- The CASAP-critical actor: the actor whose primary output edge byte
    size achieves the canonical peak buffer S_*(G). For chain-with-skips
    non-expanding graphs, this is well-defined for the canonical
    schedule. -/
noncomputable def DnnSdfGraph.criticalActor {n : Nat} (G : DnnSdfGraph n) : Fin n :=
  -- The actor v_i whose w_i saturates S_*(G).
  -- Formally: the i ∈ V minimizing |w_i - S_*(G)|, picking the leftmost in case of ties.
  sorry

/-- **Theorem (weak harmonic, CASAP-critical-cut):**

For chain-with-skips non-expanding G and any graph-internal σ with
S(σ) < S_*(G), the FLOP overhead is bounded below by

    F(σ) - F_*(G) ≥ (cc(i_critical) / F_*(G)) · F_*(G) · (S_*(G) - S(σ)) / S(σ).

Equivalently, c_h(G) ≥ cc(i_critical(G)) / F_*(G) in the harmonic
conjecture. -/
theorem strategyA_weak_harmonic {n : Nat} (G : DnnSdfGraph n) (σ : Schedule G)
    (h_below : σ.peakBuffer < G.bufferCanonical) (h_pos : 0 < σ.peakBuffer) :
    G.cc (G.criticalActor) * ((G.bufferCanonical : ℚ) - σ.peakBuffer) / (σ.peakBuffer : ℚ)
      ≤ ((σ.flops : ℚ) - G.flopsCanonical) := by
  -- Strategy A proof:
  -- 1. Let i* = G.criticalActor with w_{i*} = S_*(G).
  -- 2. σ has peak buffer S < S_* = w_{i*}, so cannot hold v_{i*}'s output as a single
  --    contiguous tensor. Must process it in chunks of size at most S.
  -- 3. Number of chunks: ⌈w_{i*}/S⌉ ≥ S_*/S.
  -- 4. Each chunk requires re-firing layers 1..i* once: cost cc(i*).
  -- 5. Total overhead: (⌈w_{i*}/S⌉ - 1) · cc(i*) ≥ ((S_*/S) - 1) · cc(i*) = ((S_* - S)/S) · cc(i*).
  -- 6. Therefore F(σ) - F_*(G) ≥ ((S_* - S)/S) · cc(i*).
  --
  -- The key lemma is that "processing v_{i*}'s output in chunks of S"
  -- formally requires recomputing layers 1..i*. This is provable from
  -- the SDF semantics: each chunk requires the full upstream chain to
  -- regenerate the boundary tokens of that chunk.
  sorry

/-! ## Halo-aware extension approach

Extends the argument to handle MCUNetV2-style patch execution.
Requires defining the extended graph G' with split/merge actors. The
extended γ(G') and CASAP critical cut differ from G's, which accounts
for the smaller empirical c_h.
-/

/-- The extended SDF graph G' obtained from G by inserting a
    spatial-partition actor before the CASAP-critical cut. -/
noncomputable def DnnSdfGraph.extend {n : Nat} (G : DnnSdfGraph n) (k : Nat) (hk : 1 ≤ k) :
    DnnSdfGraph (n + 2) :=
  -- Insert split actor before criticalActor with prod=k, cons=1
  -- Insert merge actor after criticalActor with prod=1, cons=k
  -- Update rep vector accordingly.
  sorry

/-- **Theorem (extended-graph harmonic):**

For the extended graph G' with patch factor k, the linear bound from
Theorem 1(d) gives a harmonic-form bound on the original graph G's
trade-off, with c_h(G') depending on the extended-graph γ. -/
theorem strategyB_extended_harmonic {n : Nat} (G : DnnSdfGraph n) (k : Nat) (hk : 1 ≤ k) :
    True := by  -- statement TBD
  trivial

/-! ## Information-theoretic Hong-Kung specialization

The hardest of the three approaches; approaches the empirical
c_h ≈ 0.021 if it works. Idea: chain-with-skips graphs have an
inherent "information-flow density" that can be quantified using
Shannon-style entropy of the activation distribution. The Hong-Kung
red-blue pebble bound then specializes via this density. -/

/-- Stub for the information-theoretic approach. -/
theorem strategyC_information_theoretic : True := trivial

end ParetoCorner
