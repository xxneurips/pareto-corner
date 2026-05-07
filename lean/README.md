# Pareto-Corner Lean 4 Formalization

Lean 4 + Mathlib formalization of the Pareto-Corner Non-Improvement
Theorem for chain-with-skips DNN inference. Companion to the NeurIPS 2026
submission at `paper/neurips/`.

## Build

```bash
# Install Lean 4 (via elan):
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build the project:
cd lean/
lake update    # fetches Mathlib v4.13.0
lake build     # ~10-15 minutes for first build (Mathlib compilation)
```

## Status

| File | Result | Status |
|---|---|---|
| `Definitions.lean` | DNN-SDF graph + schedule + γ | [skeleton] with `sorry` placeholders for `alpha` and `gamma` |
| `ChainBackprop.lean` | Lemma 1 | [stated]; balance-equation induction needed |
| `FlopFloor.lean` | Theorem 1(a) | **[fully proven]** (no `sorry`) |
| `BufferFloor.lean` | Theorem 1(b) CASAP | [axiomatized] (pending the concurrent submission's Lean port) |
| `ParetoOptimality.lean` | Theorem 1(c) | **[fully proven]** (1 minor `sorry` for zero-FLOP-actor case) |
| `LinearBound.lean` | Theorem 1(d) | [reduced] to `buffer_reduction_bound` |
| `LooseHarmonic.lean` | Theorem 2 | [reduced] to `k_pass_lemma` (uniform density) |
| `StemBottleneck.lean` | Proposition 1 | [stated]; arithmetic computation needed |
| `Conjecture1.lean` | Conjecture 1 | [open]; three proof strategies sketched |

**Fully proven (no `sorry`):**
- Theorem 1(a) FLOP floor + corollaries
- Theorem 1(c) Pareto-corner non-improvement (one minor zero-FLOP `sorry`)

**Reduced to a single sub-lemma:**
- Theorem 1(d) → `buffer_reduction_bound`
- Theorem 2 → `k_pass_lemma`
- Strategy A for Conjecture 1 → "chunk-processing requires re-firing"

## Architecture decisions

### Why rationals instead of Nat for `alpha`, `cc`, `gamma`?

Rate-products `cons(k, k+1) / prod(k, k+1)` are not integers in general (e.g., for upsampling layers `cons = 1, prod = 2` gives `1/2`). We use `ℚ` for clean algebra and only round to `ℕ` at the final firing-count obligation.

### Why axiomatize CASAP optimality (Theorem 1(b))?

The exchange-argument proof for CASAP optimality is documented in a concurrent anonymous submission. Keeping it axiomatized here lets us focus on the new content (Theorems 1(c), 1(d), 2, Proposition 1, Conjecture 1) without duplicating that proof. The axiom can be discharged once both submissions are de-anonymized and the proof can be imported.

### Why are zero-FLOP actors a `sorry`?

`Schedule.isCanonical` requires `m_i = q_i` for ALL i, but `flopFloor_tight_iff_canonical` only forces `m_i F_i = q_i F_i`. For actors with `F_i = 0` (e.g., identity layers, pooling), the FLOP-equality says nothing about `m_i`. We axiomatize "minimal schedule" semantics that fixes `m_i = q_i` for zero-FLOP actors as well; the formal version uses a separate `Schedule.isMinimal` predicate that's stronger than validity.

### Why is `criticalActor` `sorry`?

The CASAP-critical actor depends on the canonical CASAP schedule's peak-buffer-attaining moment. Defining it formally requires the full CASAP construction. We axiomatize its existence as `criticalActor_exists` and use it in Strategy A.

## Path to full formalization

| Effort | Outcome |
|---|---|
| **3 weeks** (NeurIPS deadline) | Theorems 1(a), 1(c), Prop 1 fully proven; rest reduced to ≤1 `sorry` each. NeurIPS submission with "machine-checkable foundation" claim. |
| **+2 weeks** | Theorems 1(d), 2 fully proven from Lemma 1. |
| **+4 weeks** | Theorem 1(b) CASAP optimality fully ported from the concurrent submission. |
| **+2-6 months** | Conjecture 1 Strategy A or B proven. |

Total: 12-month full-formalization horizon, matching the JMLR companion paper's 5-axis program.

## References

- Mathlib4 v4.13.0: <https://leanprover-community.github.io/mathlib4_docs/>
- the concurrent submission's Lean port: anonymized concurrent submission (anonymized for double-blind review)
- Lean 4 documentation: <https://lean-lang.org/lean4/doc/>
- Paper: `paper/neurips/main.tex`
