"""
Extended real-model measurement for the Pareto-Corner the empirical-evidence section.

Adds 5 non-MobileNet-family backbones to test the stem-bottleneck claim
across diverse stem geometries:
  - resnet50 (3->64, K=7)         expected gamma ~49
  - efficientnet_lite0 (3->32, K=3) expected gamma ~27 (different family, same stem geom)
  - regnety_002 (3->32, K=3)        expected gamma ~27 (different family)
  - shufflenet_v2_x1_0 (3->24, K=3) expected gamma ~9  (smaller stem)
  - ghostnet_100 (3->16, K=3)       expected gamma ~? (different stem)

Imports the LayerProfile / predict_patch utilities from measure_real_models.py
so we keep one source of truth.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List

# Allow running from the proofs/ dir directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_real_models import (  # noqa: E402
    LayerProfile,
    measure,
    predict_patch,
    print_arch,
    print_paper_table_real,
)


# -----------------------------------------------------------------------------
# Extended model loaders (non-MobileNet stems)
# -----------------------------------------------------------------------------


def load_resnet50():
    from torchvision.models import resnet50
    m = resnet50(weights=None)
    m.eval()
    return m, 224, "ResNet-50 (torchvision)"


def load_efficientnet_lite0():
    import timm
    m = timm.create_model("tf_efficientnet_lite0", pretrained=False)
    m.eval()
    return m, 224, "EfficientNet-Lite0 (timm)"


def load_regnety_002():
    from torchvision.models import regnet_y_400mf
    m = regnet_y_400mf(weights=None)
    m.eval()
    return m, 224, "RegNetY-400MF (torchvision)"


def load_shufflenet_v2():
    from torchvision.models import shufflenet_v2_x1_0
    m = shufflenet_v2_x1_0(weights=None)
    m.eval()
    return m, 224, "ShuffleNetV2-x1.0 (torchvision)"


def load_ghostnet():
    import timm
    m = timm.create_model("ghostnet_100", pretrained=False)
    m.eval()
    return m, 224, "GhostNet-100 (timm)"


def load_mobilenet_v2_anchor():
    from torchvision.models import mobilenet_v2
    m = mobilenet_v2(weights=None)
    m.eval()
    return m, 224, "MobileNetV2 (anchor)"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extended real-model patch-sweep measurement")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--csv", default="extended_arch_table.csv", help="CSV output path")
    args = parser.parse_args()

    # MobileNetV2 is included as an anchor so we can verify gamma=27 still hits
    # on this aarch64 + this PyTorch build.
    loaders = [
        load_mobilenet_v2_anchor,
        load_resnet50,
        load_efficientnet_lite0,
        load_regnety_002,
        load_shufflenet_v2,
        load_ghostnet,
    ]
    patch_factors = [1, 2, 4, 6, 10]

    all_stats = []
    for loader in loaders:
        try:
            stats = measure(loader)
            print_arch(stats, patch_factors)
            all_stats.append(stats)
        except Exception as e:
            print(f"[skipped {loader.__name__}: {type(e).__name__}: {e}]")

    # Write CSV for paper integration
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "F_star_M", "S_star_KB", "gamma", "rf", "image_size",
                    "k", "S_over_Sstar", "F_over_Fstar_minus_1", "linear_bound", "linear_OK"])
        for stats in all_stats:
            for k in patch_factors:
                pred = predict_patch(stats, stats["image_size"], k)
                w.writerow([
                    stats["name"].split(" (")[0],
                    f"{stats['F_star']/1e6:.2f}",
                    f"{stats['S_star']/1024:.2f}",
                    f"{stats['gamma']:.3f}",
                    stats["receptive_field"],
                    stats["image_size"],
                    k,
                    f"{pred['S/S*']:.4f}",
                    f"{pred['F/F*-1 (sim upper)']:.4f}",
                    f"{pred['linear_bound']:.4f}",
                    "1" if pred["linear_OK"] else "0",
                ])
    print(f"\n[wrote {args.csv}]")

    if args.latex:
        print_paper_table_real(all_stats, patch_factors)


if __name__ == "__main__":
    main()
