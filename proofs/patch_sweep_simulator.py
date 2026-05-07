"""
SDF patch-sweep simulator for the Pareto-Corner submission.

Given a chain-with-skips non-expanding DNN architecture spec, computes:
  - F* (canonical FLOP count)
  - S* (CASAP-minimum buffer for layer-by-layer execution)
  - gamma(G) (FLOP-per-byte recompute cost from Definition 7)
  - For each patch factor k: predicted (F, S) under MCUNetV2-style halo
    execution

Output: a TSV table matching the format of the paper's
multi-architecture empirical-evidence sweep.

Usage:
  python patch_sweep_simulator.py --arch mobilenetv2
  python patch_sweep_simulator.py --arch mcunet
  python patch_sweep_simulator.py --arch proxylessnas
  python patch_sweep_simulator.py --arch mnasnet
  python patch_sweep_simulator.py --all

The "actual measurement" entries should be replaced by extracting raw MAC
counts from each architecture's released code (e.g., torchprofile or
fvcore.nn.FlopCountAnalysis on the actual PyTorch model).

Author: Anonymous (NeurIPS 2026 double-blind submission)
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


# -----------------------------------------------------------------------------
# Architecture spec: one entry per layer (single-rate, MobileNet-family)
# -----------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """One layer of a chain-with-skips DNN.

    in_h, in_w: input spatial dims
    in_c, out_c: input/output channel counts
    kernel: kernel size (3 or 5 typical for MobileNet family)
    stride: stride (1 or 2)
    receptive_field_growth: pixels added to receptive field by this layer
    layer_type: 'conv', 'dwconv', 'pwconv', 'pool', 'fc', 'block' (IRB)
    """

    name: str
    in_h: int
    in_w: int
    in_c: int
    out_c: int
    kernel: int = 3
    stride: int = 1
    layer_type: str = "conv"

    @property
    def out_h(self) -> int:
        return self.in_h // self.stride

    @property
    def out_w(self) -> int:
        return self.in_w // self.stride

    @property
    def macs(self) -> int:
        """Multiply-accumulate count for this layer (single firing)."""
        if self.layer_type == "dwconv":
            return self.kernel * self.kernel * self.in_c * self.out_h * self.out_w
        elif self.layer_type == "pwconv":
            return self.in_c * self.out_c * self.out_h * self.out_w
        elif self.layer_type == "conv":
            return self.kernel * self.kernel * self.in_c * self.out_c * self.out_h * self.out_w
        elif self.layer_type == "fc":
            return self.in_c * self.out_c
        else:
            return 0  # pooling, identity, etc.

    @property
    def output_bytes(self) -> int:
        """Output activation tensor footprint in bytes (int8)."""
        return self.out_h * self.out_w * self.out_c

    @property
    def receptive_field_growth(self) -> int:
        """Pixels added to upstream receptive field by this layer."""
        return (self.kernel - 1) * self.stride


# -----------------------------------------------------------------------------
# Architecture specifications (simplified — replace per-arch from spec)
# -----------------------------------------------------------------------------


def mobilenetv2_spec() -> List[LayerSpec]:
    """Simplified MobileNetV2 ImageNet (224x224 input).

    Fully accurate: extract layer-by-layer from torchvision.models.mobilenet_v2.
    This simplified spec is for illustration; real numbers should come from
    the released model.
    """
    layers: List[LayerSpec] = []
    # Stem
    layers.append(LayerSpec("conv1", 224, 224, 3, 32, 3, 2, "conv"))
    # Inverted residual blocks (simplified — collapse expand+dw+project per block)
    # Block format: (expand_ratio, out_channels, n_blocks, stride)
    blocks = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    h, in_c = 112, 32
    for expand, out_c, n, stride in blocks:
        for i in range(n):
            s = stride if i == 0 else 1
            mid_c = in_c * expand
            # Expand pointwise
            layers.append(LayerSpec(f"irb_{out_c}_{i}_pw1", h, h, in_c, mid_c, 1, 1, "pwconv"))
            # Depthwise
            layers.append(LayerSpec(f"irb_{out_c}_{i}_dw", h, h, mid_c, mid_c, 3, s, "dwconv"))
            h = h // s
            # Project pointwise
            layers.append(LayerSpec(f"irb_{out_c}_{i}_pw2", h, h, mid_c, out_c, 1, 1, "pwconv"))
            in_c = out_c
    # Final pointwise + pool + classifier
    layers.append(LayerSpec("conv_last", h, h, in_c, 1280, 1, 1, "pwconv"))
    layers.append(LayerSpec("classifier", 1, 1, 1280, 1000, 1, 1, "fc"))
    return layers


def mcunet_spec() -> List[LayerSpec]:
    """MCUNet (Lin et al. 2020) approximate spec — searched architecture.

    For accurate numbers, use the released MCUNet code at
    https://github.com/mit-han-lab/mcunet
    """
    layers: List[LayerSpec] = []
    # MCUNet has ~25M MACs and ~500KB ROM, peak SRAM ~150KB on ImageNet
    # Simplified backbone matching the published profile
    layers.append(LayerSpec("conv1", 144, 144, 3, 16, 3, 2, "conv"))
    blocks = [
        (3, 8, 1, 1, 5),
        (4, 16, 2, 2, 5),
        (3, 24, 3, 2, 5),
        (4, 40, 4, 2, 5),
        (3, 48, 3, 1, 7),
        (4, 96, 3, 2, 7),
        (6, 160, 1, 1, 7),
    ]
    h, in_c = 72, 16
    for expand, out_c, n, stride, k in blocks:
        for i in range(n):
            s = stride if i == 0 else 1
            mid_c = in_c * expand
            layers.append(LayerSpec(f"mb_{out_c}_{i}_pw1", h, h, in_c, mid_c, 1, 1, "pwconv"))
            layers.append(LayerSpec(f"mb_{out_c}_{i}_dw", h, h, mid_c, mid_c, k, s, "dwconv"))
            h = h // s
            layers.append(LayerSpec(f"mb_{out_c}_{i}_pw2", h, h, mid_c, out_c, 1, 1, "pwconv"))
            in_c = out_c
    layers.append(LayerSpec("conv_last", h, h, in_c, 640, 1, 1, "pwconv"))
    layers.append(LayerSpec("classifier", 1, 1, 640, 1000, 1, 1, "fc"))
    return layers


def proxylessnas_spec() -> List[LayerSpec]:
    """ProxylessNAS-Mobile (Cai et al. 2019) approximate spec."""
    layers: List[LayerSpec] = []
    layers.append(LayerSpec("conv1", 224, 224, 3, 32, 3, 2, "conv"))
    blocks = [
        (1, 16, 1, 1, 3),
        (3, 32, 4, 2, 5),
        (3, 40, 4, 2, 7),
        (6, 80, 4, 2, 5),
        (6, 96, 4, 1, 5),
        (6, 192, 4, 2, 7),
        (6, 320, 1, 1, 7),
    ]
    h, in_c = 112, 32
    for expand, out_c, n, stride, k in blocks:
        for i in range(n):
            s = stride if i == 0 else 1
            mid_c = in_c * expand
            layers.append(LayerSpec(f"px_{out_c}_{i}_pw1", h, h, in_c, mid_c, 1, 1, "pwconv"))
            layers.append(LayerSpec(f"px_{out_c}_{i}_dw", h, h, mid_c, mid_c, k, s, "dwconv"))
            h = h // s
            layers.append(LayerSpec(f"px_{out_c}_{i}_pw2", h, h, mid_c, out_c, 1, 1, "pwconv"))
            in_c = out_c
    layers.append(LayerSpec("conv_last", h, h, in_c, 1280, 1, 1, "pwconv"))
    layers.append(LayerSpec("classifier", 1, 1, 1280, 1000, 1, 1, "fc"))
    return layers


def mnasnet_spec() -> List[LayerSpec]:
    """MnasNet-A1 (Tan et al. 2018) approximate spec."""
    layers: List[LayerSpec] = []
    layers.append(LayerSpec("conv1", 224, 224, 3, 32, 3, 2, "conv"))
    blocks = [
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (3, 40, 3, 2, 5),
        (6, 80, 4, 2, 3),
        (6, 112, 2, 1, 3),
        (6, 160, 3, 2, 5),
        (6, 320, 1, 1, 3),
    ]
    h, in_c = 112, 32
    for expand, out_c, n, stride, k in blocks:
        for i in range(n):
            s = stride if i == 0 else 1
            mid_c = in_c * expand
            layers.append(LayerSpec(f"mn_{out_c}_{i}_pw1", h, h, in_c, mid_c, 1, 1, "pwconv"))
            layers.append(LayerSpec(f"mn_{out_c}_{i}_dw", h, h, mid_c, mid_c, k, s, "dwconv"))
            h = h // s
            layers.append(LayerSpec(f"mn_{out_c}_{i}_pw2", h, h, mid_c, out_c, 1, 1, "pwconv"))
            in_c = out_c
    layers.append(LayerSpec("conv_last", h, h, in_c, 1280, 1, 1, "pwconv"))
    layers.append(LayerSpec("classifier", 1, 1, 1280, 1000, 1, 1, "fc"))
    return layers


# -----------------------------------------------------------------------------
# SDF metrics
# -----------------------------------------------------------------------------


def compute_canonical_metrics(layers: List[LayerSpec]) -> Tuple[int, int, float]:
    """Compute (F*, S*, gamma) for layer-by-layer execution.

    For single-rate networks (q_i = 1 for all i), this is straightforward:
      - F* = sum of per-layer MACs
      - S* = max activation footprint at any layer boundary (live tensors at any moment)
      - gamma = min_i (cumulative MACs up to layer i) / (output bytes of layer i)
    """
    flops_canonical = sum(layer.macs for layer in layers)

    # CASAP for single-rate networks: only one activation tensor live at a time.
    # S* = max layer's output footprint.
    s_star = max(layer.output_bytes for layer in layers)

    # gamma(G) = min over actors of cc(i) / w_i
    cumulative_macs = 0
    min_ratio = float("inf")
    for layer in layers:
        cumulative_macs += layer.macs
        if layer.output_bytes > 0:
            ratio = cumulative_macs / layer.output_bytes
            if ratio < min_ratio:
                min_ratio = ratio
    gamma = min_ratio

    return flops_canonical, s_star, gamma


def compute_receptive_field(layers: List[LayerSpec]) -> int:
    """Compute the cumulative receptive-field growth at network output (pixels)."""
    rf = 1
    for layer in layers:
        if layer.layer_type in ("conv", "dwconv"):
            rf += (layer.kernel - 1) * layer.stride
    return rf


def predict_patch_metrics(
    layers: List[LayerSpec],
    flops_canonical: int,
    s_star: int,
    k: int,
) -> Tuple[float, float]:
    """Predict (F/F*, S/S*) for patch factor k under the SDF halo model.

    For 1D-equivalent patch factor k (matching MCUNetV2's reported "patch factor"
    which is the SRAM reduction factor):
      - S/S* ≈ 1/k (peak buffer is per-patch activation)
      - F/F* ≈ 1 + halo_overhead, where halo_overhead is determined by
        the receptive-field-induced extra pixels per patch boundary

    The halo overhead per patch boundary is roughly
      halo_overhead ≈ (receptive_field * k_lin / image_size)
    where k_lin = sqrt(k) for 2D patches or = k for 1D patches. We use the 1D
    approximation here, matching MCUNetV2's reported patch-factor convention.
    """
    if k == 1:
        return 1.0, 1.0

    # For single-rate chain-with-skips networks, peak buffer scales as 1/k.
    s_ratio = 1.0 / k

    # Halo overhead computation
    receptive_field = compute_receptive_field(layers)
    image_size = layers[0].in_h  # input spatial dim

    # 1D patch overhead: extra pixels per patch boundary
    # halo_overhead ≈ (receptive_field / patch_size) * (number_of_patch_boundaries)
    # = (rf / (image_size / k)) * (k - 1)  (for 1D, k-1 boundaries)
    # = rf * k * (k - 1) / image_size
    # But this is per-patch; total over k patches is rf * k * (k - 1) / image_size
    # Then F/F* ≈ 1 + total_halo / total_pixels ≈ 1 + rf * (k - 1) / image_size
    # Simplification: F/F* − 1 ≈ rf / image_size * (k − 1)
    rf_fraction = receptive_field / image_size
    f_ratio = 1.0 + rf_fraction * (k - 1)

    return f_ratio, s_ratio


def compute_linear_bound(
    flops_canonical: int, s_star: int, gamma: float, s_ratio: float
) -> float:
    """Linear lower bound from Theorem 1(d): (F − F*) / F* ≥ γ · (S* − S) / F*."""
    s_reduction_bytes = s_star * (1.0 - s_ratio)
    f_overhead_lower_bound = gamma * s_reduction_bytes
    return f_overhead_lower_bound / flops_canonical


def compute_loose_harmonic_bound(s_ratio: float) -> float:
    """Loose harmonic bound from the proof attempt: (F − F*) / F* ≥ (1/2) · (1/S − 1)."""
    if s_ratio >= 1.0:
        return 0.0
    return 0.5 * (1.0 / s_ratio - 1.0)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def architecture_table(name: str, layers: List[LayerSpec]) -> Dict:
    flops_canonical, s_star, gamma = compute_canonical_metrics(layers)
    receptive_field = compute_receptive_field(layers)
    rows = []
    for k in [1, 2, 4, 6, 10]:
        f_ratio, s_ratio = predict_patch_metrics(layers, flops_canonical, s_star, k)
        linear_bound = compute_linear_bound(flops_canonical, s_star, gamma, s_ratio)
        loose_harmonic = compute_loose_harmonic_bound(s_ratio)
        # Empirical-style harmonic regression form (c = 0.021 from MobileNetV2 fit)
        emp_harmonic = 0.021 * (1.0 / s_ratio - 1.0) if s_ratio < 1.0 else 0.0
        rows.append({
            "k": k,
            "F/F*": f_ratio,
            "F/F* - 1": f_ratio - 1.0,
            "S/S*": s_ratio,
            "linear_bound": linear_bound,
            "loose_harmonic_bound": loose_harmonic,
            "empirical_harmonic_021": emp_harmonic,
            "linear_OK": (f_ratio - 1.0) >= linear_bound,
            "loose_harmonic_OK": (f_ratio - 1.0) >= loose_harmonic,
        })
    return {
        "name": name,
        "F*": flops_canonical,
        "S*": s_star,
        "gamma": gamma,
        "receptive_field": receptive_field,
        "rows": rows,
    }


def print_arch_summary(result: Dict) -> None:
    print(f"\n=== {result['name']} ===")
    print(f"  F* = {result['F*']:,} MACs")
    print(f"  S* = {result['S*']:,} bytes ({result['S*']/1024:.1f} KB)")
    print(f"  gamma(G) = {result['gamma']:.2f} MACs/byte")
    print(f"  receptive field = {result['receptive_field']} pixels")
    print(f"\n  {'k':>3} | {'F/F*':>6} | {'S/S*':>6} | {'F/F*-1':>7} | "
          f"{'lin bound':>10} | {'lin OK?':>8} | {'lh bound':>9} | {'lh OK?':>8}")
    print(f"  {'-'*3} | {'-'*6} | {'-'*6} | {'-'*7} | {'-'*10} | {'-'*8} | "
          f"{'-'*9} | {'-'*8}")
    for row in result["rows"]:
        ok_lin = "yes" if row["linear_OK"] else "NO"
        ok_lh = "yes" if row["loose_harmonic_OK"] else "NO"
        print(
            f"  {row['k']:>3} | {row['F/F*']:>6.3f} | {row['S/S*']:>6.3f} | "
            f"{row['F/F* - 1']:>7.4f} | {row['linear_bound']:>10.4f} | {ok_lin:>8} | "
            f"{row['loose_harmonic_bound']:>9.3f} | {ok_lh:>8}"
        )


def print_paper_table(results: List[Dict]) -> None:
    """Print in LaTeX format for direct paste into the paper the empirical-evidence section."""
    print("\n\n% ===== LaTeX table for the empirical-evidence section =====")
    print("% Auto-generated by patch_sweep_simulator.py; replace empirical column with")
    print("% real measurements from each architecture's released codebase.")
    print()
    for result in results:
        for row in result["rows"]:
            if row["k"] == 1:
                continue  # skip k=1 baseline
            ok = "\\checkmark" if row["linear_OK"] else "$\\times$"
            print(
                f"{result['name']} & {row['k']} & {row['S/S*']:.2f} & "
                f"{row['F/F* - 1']:.3f} & {row['linear_bound']:.3f} & {ok} \\\\"
            )
        print("\\midrule")


def main():
    parser = argparse.ArgumentParser(
        description="Patch-sweep simulator for the Pareto-Corner submission"
    )
    parser.add_argument(
        "--arch",
        choices=["mobilenetv2", "mcunet", "proxylessnas", "mnasnet"],
        help="Single architecture to simulate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Simulate all 4 architectures",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table for the empirical-evidence section",
    )
    args = parser.parse_args()

    spec_fns = {
        "mobilenetv2": mobilenetv2_spec,
        "mcunet": mcunet_spec,
        "proxylessnas": proxylessnas_spec,
        "mnasnet": mnasnet_spec,
    }

    results = []
    if args.all:
        for name, fn in spec_fns.items():
            r = architecture_table(name, fn())
            print_arch_summary(r)
            results.append(r)
    elif args.arch:
        r = architecture_table(args.arch, spec_fns[args.arch]())
        print_arch_summary(r)
        results.append(r)
    else:
        parser.error("Specify --arch <name> or --all")

    if args.latex:
        print_paper_table(results)


if __name__ == "__main__":
    main()
