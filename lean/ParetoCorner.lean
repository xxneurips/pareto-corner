-- Pareto-Corner Non-Improvement Theorem for Chain-with-Skips DNN Inference
-- Top-level import: bundles all theorems and lemmas of the formalization.

import ParetoCorner.Definitions
import ParetoCorner.ChainBackprop
import ParetoCorner.FlopFloor
import ParetoCorner.BufferFloor
import ParetoCorner.ParetoOptimality
import ParetoCorner.LinearBound
import ParetoCorner.LooseHarmonic
import ParetoCorner.StemBottleneck
import ParetoCorner.Conjecture1

/-!
# The Pareto-Corner Theorem (Lean 4 formalization)

This file is the top-level import for the formalization of the
Pareto-Corner Non-Improvement Theorem, structured to mirror the paper's
sections.

## Status

| Result                                       | Lean status |
|----------------------------------------------|-------------|
| Theorem 1(a) FLOP floor                      | [proven]   |
| Theorem 1(b) CASAP buffer optimality         | [axiom] (axiomatized pending the concurrent submission's Lean port) |
| Theorem 1(c) Pareto-corner non-improvement   | [proven]   |
| Theorem 1(d) Linear trade-off                | [reduced] to chain-backprop + buffer-reduction lemmas |
| Lemma 1 Chain-backpropagation                | [stated]; proof needs balance-equation manipulation |
| Theorem 2 Loose harmonic (uniform density)   | [reduced] to k-pass lemma |
| Proposition 1 Stem-bottleneck γ ≤ K²·C_in    | [proven]   |
| Conjecture 1 Tight harmonic asymptotic       | [open]     |

The proven results (Theorems 1(a), 1(c), Proposition 1) provide the
machine-checkable foundation. The axiomatized result (CASAP optimality)
is cited from the concurrent submission with its own Lean port
underway. The skeletoned results have all proof obligations explicit.

## Repository structure

- `Definitions.lean` — DNN-SDF graph, schedule, peak buffer, γ(G)
- `ChainBackprop.lean` — Lemma 1
- `FlopFloor.lean` — Theorem 1(a), full proof
- `BufferFloor.lean` — Theorem 1(b), CASAP optimality (axiomatized)
- `ParetoOptimality.lean` — Theorem 1(c), full proof
- `LinearBound.lean` — Theorem 1(d), proof skeleton
- `LooseHarmonic.lean` — Theorem 2, proof skeleton
- `StemBottleneck.lean` — Proposition 1, full proof
- `Conjecture1.lean` — open conjecture statement + strategy
-/
