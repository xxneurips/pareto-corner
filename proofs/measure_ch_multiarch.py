"""
Conjecture 1 c_h multi-architecture sweep (Priority C).

Computes a per-layer receptive-field-aware patch-execution FLOP overhead
for each architecture, then extracts the empirical halo coefficient c_h
defined by the harmonic ansatz of Conjecture 1:

    F/F* - 1 ≈ c_h * (S* - S) / S

For each architecture:
  - Run a forward hook trace to extract (F_i, w_i, rf_growth_i, output_h, output_w)
    per layer i.
  - For each patch factor k ∈ {2, 4, 6, 10}:
      * patch input size = image_size / k pixels per side
      * For layer i: cumulative receptive-field-aware padded input
        ≈ patch_size + rf_cumulative_pre(i)
      * Per-layer halo scaling = (padded_size / patch_size)^2
      * Total F(k) = k^2 * sum_i F_i * scaling_i,
        capped where padded_size exceeds image_size (no halo benefit beyond)
      * S(k) ≈ S*/k (peak buffer scales by patch fraction)
      * c_h(k) = (F(k)/F* - 1) * S(k) / (S* - S(k))
  - Report c_h per (arch, k) plus the harmonic-fit slope for each arch.

This is a refined version of measure_real_models.predict_patch which used
only the global receptive field. Here we account for per-layer rf_pre,
which gives a less pessimistic and architecture-distinguishing c_h.

Reference c_h(MobileNetV2) ≈ 0.021 from MCUNetV2 Table 4 published numbers.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_real_models_extended import (  # noqa: E402
    load_mobilenet_v2_anchor,
    load_resnet50,
    load_efficientnet_lite0,
    load_regnety_002,
    load_shufflenet_v2,
    load_ghostnet,
)


class DetailedLayerProfile:
    """Per-layer (F_i, w_i, rf_growth_i, spatial dims)."""

    def __init__(self):
        self.records: List[Tuple[str, int, int, int, int, int]] = []

    def hook(self, name: str):
        def fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            if output.dim() < 4:
                # FC / pooled output. Treat as 1x1 spatial.
                out_h = out_w = 1
            else:
                out_h, out_w = output.shape[-2], output.shape[-1]
            output_bytes = output.numel() // output.shape[0]  # int8 baseline
            macs = 0
            rf_growth = 0
            if isinstance(module, nn.Conv2d):
                in_c = module.in_channels // module.groups
                out_c = module.out_channels
                k_h, k_w = module.kernel_size
                macs = k_h * k_w * in_c * out_c * out_h * out_w
                rf_growth = max(k_h - 1, k_w - 1) * max(module.stride)
            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
            if macs > 0 or output_bytes > 0:
                self.records.append((name, macs, output_bytes, rf_growth, out_h, out_w))

        return fn

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                module.register_forward_hook(self.hook(name))


def measure_detailed(loader_fn) -> dict:
    model, image_size, name = loader_fn()
    profile = DetailedLayerProfile()
    profile.attach(model)
    with torch.no_grad():
        x = torch.randn(1, 3, image_size, image_size)
        _ = model(x)
    F_star = sum(r[1] for r in profile.records)
    S_star = max(r[2] for r in profile.records) if profile.records else 0
    return {
        "name": name.split(" (")[0],
        "image_size": image_size,
        "F_star": F_star,
        "S_star": S_star,
        "records": profile.records,
    }


def patch_overhead(records, image_size: int, k: int) -> float:
    """Receptive-field-aware halo FLOP overhead summed over patch-applicable layers.

    For layer i with cumulative receptive field rf_pre, the per-patch effective
    input is (patch_size + rf_pre) per side, vs full-image (image_size). The
    per-patch FLOPs scale by ((patch_size + rf_pre) / patch_size)^2; total for
    k^2 patches is k^2 * F_i * ((patch_size + rf_pre)/patch_size)^2.

    Original (k=1) FLOPs are F_i. Halo overhead therefore:
        F_i_patch / F_i = k^2 * ((patch_size + rf_pre) / patch_size)^2 / k^2
                        = ((patch_size + rf_pre) / patch_size)^2  ... if patch_size > 0
    But total work over k^2 patches at patch_size each is the multiplier above
    times F_i, not divided by k^2 -- we are reproducing more pixels per patch.
    Halo only kicks in until padded_size >= image_size (then full-image execution).

    Returns total F(k) - F* (overhead) summed over patch-applicable layers.
    """
    if k <= 1:
        return 0.0
    patch_size = image_size / k
    rf_cum_pre = 0
    F_overhead = 0.0
    for (name, macs, w, rf_growth, oh, ow) in records:
        if macs <= 0:
            rf_cum_pre += rf_growth
            continue
        padded = patch_size + rf_cum_pre
        if padded >= image_size:
            # Full-image execution from here onward; no further halo overhead.
            rf_cum_pre += rf_growth
            continue
        # Per-patch overhead: (padded/patch)^2 vs (image/k=patch)^2
        scale = (padded / patch_size) ** 2  # halo factor per patch
        # Each patch processes scale * (patch_size^2) pixels worth of layer-i FLOPs.
        # Total over k^2 patches: k^2 * scale * F_i_per_patch_pixel * patch_size^2
        # Compared to full image: F_i = F_i_per_patch_pixel * image_size^2 = F_i_per_patch_pixel * (k*patch_size)^2
        # So total F_i_patch / F_i = scale.
        F_overhead += macs * (scale - 1.0)
        rf_cum_pre += rf_growth
    return F_overhead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ch_multiarch_results.csv")
    args = parser.parse_args()

    loaders = [
        load_mobilenet_v2_anchor,
        load_resnet50,
        load_efficientnet_lite0,
        load_regnety_002,
        load_shufflenet_v2,
        load_ghostnet,
    ]
    patch_factors = [2, 4, 6, 10]

    rows = []
    print(f"{'Arch':<22} {'k':>3} {'S/S*':>6} {'F/F*-1':>9} {'c_h':>8}")
    print("-" * 55)
    for loader in loaders:
        try:
            stats = measure_detailed(loader)
        except Exception as e:
            print(f"[skip {loader.__name__}: {e}]")
            continue
        per_arch_ch = []
        for k in patch_factors:
            ovh = patch_overhead(stats["records"], stats["image_size"], k)
            f_ratio = ovh / stats["F_star"]
            s_ratio = 1.0 / k
            # c_h formula: (F/F* - 1) = c_h * (S* - S)/S = c_h * (1/s_ratio - 1)
            denom = (1.0 / s_ratio) - 1.0
            c_h = f_ratio / denom if denom > 0 else float("nan")
            per_arch_ch.append(c_h)
            print(f"{stats['name']:<22} {k:>3} {s_ratio:>6.3f} {f_ratio:>9.4f} {c_h:>8.4f}")
            rows.append({
                "arch": stats["name"],
                "F_star_M": stats["F_star"] / 1e6,
                "S_star_KB": stats["S_star"] / 1024,
                "k": k,
                "S_over_Sstar": s_ratio,
                "F_over_Fstar_minus_1": f_ratio,
                "c_h": c_h,
            })
        # Average c_h over patch factors -> single per-arch number
        avg_ch = sum(per_arch_ch) / len(per_arch_ch)
        print(f"{stats['name']:<22} avg c_h = {avg_ch:.4f}")
        print()

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(rows[0].keys()) if rows else []
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    print(f"[wrote {args.csv}]")


if __name__ == "__main__":
    main()
