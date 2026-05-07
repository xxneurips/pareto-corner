"""
Fork-join / non-chain failure-case measurement for the Pareto-Corner paper §5.

Tests whether Proposition 1's stem-bottleneck bound  gamma(G) <= K^2 * C_in
predicts gamma correctly OUTSIDE the chain-with-skips non-expanding hypothesis.

Architectures tested:
  - vit_tiny_patch16_224 (timm)        : multi-head attention = fork-join,
                                         stem patch-conv K=16, C_in=3, predicted
                                         K^2 C_in = 768 if Prop 1 still applied.
  - deit_tiny_patch16_224 (timm)       : same patch-embed stem, distillation token.
  - fcn_resnet50 (torchvision)         : encoder-decoder with bilinear upsampling
                                         (violates non-expansion).
  - deeplabv3_resnet50 (torchvision)   : ASPP parallel-branch fork-join structure.

For each architecture we run the same forward-hook gamma extraction used in
measure_real_models.py (cumulative MACs / per-layer output bytes, min over
actors). The measurement is mechanically defined for any DAG, but the chain-
with-skips hypothesis underlying Proposition 1 is violated, so we expect
gamma_measured != K^2 * C_in (i.e., the bound is non-predictive outside scope).

Writes  fork_join_failure_cases.csv  next to this file.

Author: Anonymous (NeurIPS 2026 double-blind submission)
"""

from __future__ import annotations
import csv
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_real_models import LayerProfile  # noqa: E402


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def load_vit_tiny() -> Tuple[nn.Module, int, str, int, int, str]:
    import timm
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    m.eval()
    # patch-embed conv: 16x16, in_channels=3
    return m, 224, "ViT-Tiny/16", 16, 3, "transformer (multi-head attn = fork-join)"


def load_deit_tiny() -> Tuple[nn.Module, int, str, int, int, str]:
    import timm
    m = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    m.eval()
    return m, 224, "DeiT-Tiny/16", 16, 3, "transformer + distillation token (fork-join)"


def load_fcn_resnet50() -> Tuple[nn.Module, int, str, int, int, str]:
    from torchvision.models.segmentation import fcn_resnet50
    m = fcn_resnet50(weights=None, weights_backbone=None)
    m.eval()
    return m, 224, "FCN-ResNet50", 7, 3, "encoder-decoder with bilinear upsampling (non-expansion violated)"


def load_deeplabv3() -> Tuple[nn.Module, int, str, int, int, str]:
    from torchvision.models.segmentation import deeplabv3_resnet50
    m = deeplabv3_resnet50(weights=None, weights_backbone=None)
    m.eval()
    return m, 224, "DeepLabv3-ResNet50", 7, 3, "ASPP parallel branches (fork-join) + upsampling"


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------


def measure_one(loader_fn):
    model, image_size, name, K, C_in, failure_mode = loader_fn()
    profile = LayerProfile()
    profile.attach(model)
    with torch.no_grad():
        x = torch.randn(1, 3, image_size, image_size)
        try:
            _ = model(x)
        except Exception as e:
            return {
                "name": name,
                "K": K,
                "C_in": C_in,
                "predicted": K * K * C_in,
                "measured": None,
                "F_star": None,
                "S_star": None,
                "n_layers": 0,
                "image_size": image_size,
                "failure_mode": f"forward-pass failure: {type(e).__name__}: {e}",
            }
    s = profile.summary()
    s["name"] = name
    s["image_size"] = image_size
    s["K"] = K
    s["C_in"] = C_in
    s["predicted"] = K * K * C_in
    s["measured"] = s.get("gamma", None)
    s["failure_mode"] = failure_mode
    return s


def main():
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fork_join_failure_cases.csv")
    loaders = [
        load_vit_tiny,
        load_deit_tiny,
        load_fcn_resnet50,
        load_deeplabv3,
    ]

    rows = []
    for loader in loaders:
        try:
            s = measure_one(loader)
        except Exception as e:
            print(f"[failed {loader.__name__}: {type(e).__name__}: {e}]")
            continue
        family = "transformer" if "ViT" in s["name"] or "DeiT" in s["name"] else "segmentation"
        measured = s["measured"]
        predicted = s["predicted"]
        if measured is None:
            agreement = "undefined (forward failed)"
        else:
            ratio = measured / predicted if predicted else float("inf")
            agreement = f"measured={measured:.2f} vs predicted={predicted} (ratio={ratio:.3f})"
        print(f"\n=== {s['name']} ===")
        print(f"  family            = {family}")
        print(f"  in-scope (Prop 1) = no  ({s['failure_mode']})")
        print(f"  K, C_in           = {s['K']}, {s['C_in']}")
        print(f"  predicted gamma   = K^2 * C_in = {predicted}")
        print(f"  measured gamma    = {measured}")
        print(f"  F*                = {s.get('F_star', 'N/A')}")
        print(f"  S*                = {s.get('S_star', 'N/A')}")
        print(f"  n_layers          = {s.get('n_layers', 0)}")
        print(f"  agreement         = {agreement}")

        rows.append({
            "arch": s["name"],
            "family": family,
            "in_scope_of_prop1": "no",
            "K": s["K"],
            "C_in": s["C_in"],
            "F_star_M": (s.get("F_star") or 0) / 1e6 if s.get("F_star") else "",
            "S_star_KB": (s.get("S_star") or 0) / 1024 if s.get("S_star") else "",
            "measured_gamma": (f"{measured:.4f}" if measured is not None else "undefined"),
            "predicted_K_squared_Cin": predicted,
            "agreement": agreement,
            "failure_mode": s["failure_mode"],
        })

    fieldnames = ["arch", "family", "in_scope_of_prop1", "K", "C_in",
                  "F_star_M", "S_star_KB",
                  "measured_gamma", "predicted_K_squared_Cin",
                  "agreement", "failure_mode"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[wrote {out_csv}]")


if __name__ == "__main__":
    main()
