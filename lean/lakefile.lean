import Lake
open Lake DSL

package «pareto-corner» where
  version := v!"0.1.0"
  description := "Lean 4 formalization of the Pareto-Corner Non-Improvement Theorem for chain-with-skips DNN inference"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.13.0"

@[default_target]
lean_lib «ParetoCorner» where
  -- All theorems and lemmas in the ParetoCorner namespace
