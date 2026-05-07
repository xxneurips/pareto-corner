# γ = K²C_in: Pareto-Corner Non-Improvement Theorem — code and Lean formalization

Anonymous code archive for NeurIPS 2026 submission *"γ = K²C_in: A Pareto-Corner Non-Improvement Theorem for Chain-with-Skips DNN Inference"*.

This repository contains:

- `proofs/` — measurement scripts (PyTorch forward hooks, patch-execution sweeps) and raw CSV results that anchor the empirical claims in §4 of the paper.
- `lean/` — Lean 4 + Mathlib formalization of Theorems 1(a) and 1(c), with five enumerated axioms (A1–A5) declared in the body of the paper.

Reviewer: this repo is anonymized for double-blind review. No identifying metadata in commits, file headers, or paths.

## Quick reproduction

```bash
# 1. Python environment for measurement scripts
pip install -r requirements.txt

# 2. Lean formalization (requires elan + Lean 4)
cd lean && lake build
```

Running `lake build` in `lean/` reproduces machine-verification of Theorems 1(a) and 1(c) modulo axioms A1–A5 (see `lean/ParetoCorner/Definitions.lean`, `lean/ParetoCorner/BufferFloor.lean`, `lean/ParetoCorner/ChainBackprop.lean`).

## Paper-to-code mapping

| Paper artifact | Script | Output CSV |
|---|---|---|
| Table 1 (patch sweep, 7 mit-han-lab backbones × 10 patch factors) | `proofs/real_patch_executor.py` | `proofs/real_patch_inference.csv` |
| Table 2 (γ sweep, 22 ImageNet backbones, three stem geometries) | `proofs/measure_real_models.py`, `proofs/measure_real_models_extended.py`, `proofs/measure_real_models_kvariety.py` | `proofs/extended_arch_table.csv`, `proofs/mcunet_arch_table.csv`, `proofs/kvariety_arch_table.csv` |
| Table 3 (large-memory aarch64 sanity check, 6 backbones) | `proofs/arm_silicon_validation.py` | `proofs/arm_validation_results.csv` |
| Conjecture 1 anchor (10-architecture analytical halo sweep, c_h ∈ [0.0014, 0.0253]) | `proofs/measure_ch_multiarch.py` | `proofs/ch_multiarch_results.csv` |
| Fork-join failure case (ViT/DeiT γ ≠ K²C_in) | `proofs/measure_fork_join.py` | `proofs/fork_join_failure_cases.csv` |
| Universal-form bound check across 60 patch configs | `proofs/patch_sweep_simulator.py`, `proofs/patch_sweep_extended.py` | `proofs/sweep_120_results.csv` |

## Lean formalization status

| Theorem | Status |
|---|---|
| 1(a) FLOP floor | [closed] modulo axioms A1–A5 |
| 1(b) Buffer floor at canonical FLOPs | Skeleton; encoded as axioms A4 + A5 pending Lake closure of supplementary Appendix A exchange-argument proof |
| 1(c) Pareto-optimality | [closed] modulo axioms A1–A5 |
| 1(d) Linear trade-off with γ(G)/|R| factor | Skeleton |
| 2 Loose harmonic bound (uniform density) | Skeleton |
| 3 Weak harmonic bound (CASAP-critical chunking) | Skeleton |
| 4 Partition-method saturation | Stated; proof sketch in body |

The five enumerated axioms appear at:

- `lean/ParetoCorner/Definitions.lean:178` — A3 canonical-schedule existence
- `lean/ParetoCorner/BufferFloor.lean:38`  — A5 canonical buffer floor (= Theorem 1(b))
- `lean/ParetoCorner/BufferFloor.lean:44`  — A4 constructive corner existence
- `lean/ParetoCorner/ChainBackprop.lean:41` — A1 token balance
- `lean/ParetoCorner/ChainBackprop.lean:49` — A2 repetition balance

A1–A3 are standard SDF results (Lee–Messerschmitt 1987, Bhattacharyya–Murthy–Lee 1996). A4 and A5 together capture the content of Theorem 1(b); axiomatising them in Lean leaves the exchange-argument proof of 1(b) as the single load-bearing remaining obligation for full formalization.

## Hardware requirements

- Graph extraction (Tables 1–2): one consumer GPU (single forward pass per architecture). Wall clock ≤ 5 minutes for all 22 architectures.
- Patch sweep (Table 1): same GPU, total wall clock ≤ 2 hours for the 70-config sweep.
- ARM validation (Table 3): aarch64 platform with sufficient unified memory; the paper used a large-memory data-center class chip as a sanity check (≠ Cortex-M SRAM regime; see paper §4 for explicit framing).
- Lean: any machine with `elan` + Lean 4 toolchain (specified in `lean/lean-toolchain`); Mathlib build is the dominant cost (~30 minutes cold).

No training is performed; all measurements use stock pretrained checkpoints (`torchvision`, `timm`, `mit-han-lab/mcunet`). No proprietary data.

## Licenses for existing assets

PyTorch (BSD-3), `torchvision` (BSD-3), `timm` (Apache-2.0), MCUNet checkpoints (`mit-han-lab`, MIT), Lean 4 (Apache-2.0), Mathlib (Apache-2.0), `fvcore` (Apache-2.0).

## License for this repository

Apache-2.0. See `LICENSE`.
