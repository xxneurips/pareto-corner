"""
Extended γ sweep across STEM K values (Priority A).

Tests Proposition 1's K^2 * C_in formula across 4 distinct stem K values:
  - VGG-16  (K=3, but very deep + plain chain) → predicts γ = 27
  - ResNet-18 (K=7) → predicts γ = 147
  - DenseNet-121 (K=7) → predicts γ = 147 (different family with same stem)
  - ConvNeXt-Tiny (K=4) → predicts γ = 16 * 3 = 48
  - AlexNet (K=11) → predicts γ = 121 * 3 = 363

Combined with the K=3 (γ=27) and K=7 ResNet-50 (γ=147) data we already
have, this gives 4 distinct γ values across stem geometries: 27, 48, 147, 363.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_real_models import measure, predict_patch, print_arch  # noqa: E402


def load_resnet18():
    from torchvision.models import resnet18
    m = resnet18(weights=None); m.eval()
    return m, 224, "ResNet-18 (torchvision)"


def load_vgg16():
    from torchvision.models import vgg16
    m = vgg16(weights=None); m.eval()
    return m, 224, "VGG-16 (torchvision)"


def load_densenet121():
    from torchvision.models import densenet121
    m = densenet121(weights=None); m.eval()
    return m, 224, "DenseNet-121 (torchvision)"


def load_convnext_tiny():
    from torchvision.models import convnext_tiny
    m = convnext_tiny(weights=None); m.eval()
    return m, 224, "ConvNeXt-Tiny (torchvision)"


def load_alexnet():
    from torchvision.models import alexnet
    m = alexnet(weights=None); m.eval()
    return m, 224, "AlexNet (torchvision)"


def load_efficientnet_b0():
    from torchvision.models import efficientnet_b0
    m = efficientnet_b0(weights=None); m.eval()
    return m, 224, "EfficientNet-B0 (torchvision; SE-block, expect failure)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="kvariety_arch_table.csv")
    args = parser.parse_args()

    loaders = [
        load_alexnet,           # K=11
        load_resnet18,          # K=7
        load_densenet121,       # K=7
        load_convnext_tiny,     # K=4
        load_vgg16,             # K=3, but very deep
        load_efficientnet_b0,   # SE-block: failure-mode demo
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


if __name__ == "__main__":
    main()
