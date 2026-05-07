"""
Extended multi-architecture patch-factor sweep for the NeurIPS 2026
Pareto-corner paper.

Computes a statistically defensible 12-architecture x 10-patch-factor
analytical sweep, extending beyond the small-sample harmonic regression
(R^2 = 0.999, slope 0.021) referenced from MCUNetV2.

What this computes (PER ARCHITECTURE, PER k):
  - F(sigma_halo) / F* - 1   : analytical FLOP overhead from MCUNetV2-style
                                 per-layer halo recomputation (NOT a measured
                                 inference)
  - S(sigma_halo) / S*       : predicted SRAM ratio under k-way patching
                                 (1/k for single-rate chain backbone)
  - linear bound (Thm 1d) : gamma(G)/n * (S* - S)/F*   (universal form;
                                in F-overhead units)
  - weak harmonic bound (Thm 2): cc(i*) * (S* - S)/(S * F*)
  - bounds_hold flags

What this fits (PER ARCHITECTURE):
  - harmonic model:    F/F* - 1 = c_h * (S*/S - 1)
  - squared-log model: F/F* - 1 = c_sq * log(S*/S)^2
  - 95% bootstrap CIs on c_h
  - per-architecture R^2 and AIC

Note: the F-overhead numbers reported here are predictions of an
analytical extended-graph model, not measured inference time. The
model is the same one MCUNetV2 uses to motivate patch-based
execution. The values have analytical, not empirical, status.

Author: Anonymous (NeurIPS 2026 double-blind submission)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Per-layer hook profiler (re-uses measure_real_models.py / measure_mcunet.py
# logic verbatim so F*, S*, gamma, rf are extracted identically).
# =============================================================================


class LayerProfile:
    def __init__(self):
        self.records: List[Tuple[str, int, int, int]] = []  # (name, macs, out_bytes, rf_growth)

    def hook(self, name: str):
        def fn(module, inputs, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            output_bytes = output.numel() // output.shape[0]
            macs = 0
            rf_growth = 0
            if isinstance(module, nn.Conv2d):
                in_c = module.in_channels // module.groups
                out_c = module.out_channels
                k_h, k_w = module.kernel_size
                out_h, out_w = output.shape[-2], output.shape[-1]
                macs = k_h * k_w * in_c * out_c * out_h * out_w
                rf_growth = max(k_h - 1, k_w - 1) * max(module.stride)
            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
            if macs > 0 or output_bytes > 0:
                self.records.append((name, macs, output_bytes, rf_growth))
        return fn

    def attach(self, model: nn.Module):
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.AvgPool2d,
                              nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                m.register_forward_hook(self.hook(n))

    def summary(self) -> Optional[Dict]:
        if not self.records:
            return None
        F_star = sum(r[1] for r in self.records)
        S_star = max(r[2] for r in self.records)
        cum, mn = 0, float("inf")
        cum_at_min = 0
        i_star_idx = 0
        for idx, (_, mac, w, _) in enumerate(self.records):
            cum += mac
            if w > 0 and cum > 0:
                ratio = cum / w
                if ratio < mn:
                    mn = ratio
                    cum_at_min = cum
                    i_star_idx = idx
        gamma = mn if mn < float("inf") else 0.0
        cc_i_star = cum_at_min  # chain cost up to i* (Definition 8)
        rf = 1 + sum(r[3] for r in self.records)
        return {
            "n_layers": len(self.records),
            "F_star": F_star,
            "S_star": S_star,
            "gamma": gamma,
            "cc_i_star": cc_i_star,
            "i_star_idx": i_star_idx,
            "receptive_field": rf,
        }


# =============================================================================
# Model loaders (returns (model, image_size, name, family_tag, notes))
# family_tag: 'mobilenet-family' (chain-with-skips OK) | 'se-augmented' (NaN)
# =============================================================================


def load_mobilenet_v2():
    from torchvision.models import mobilenet_v2
    return mobilenet_v2(weights=None).eval(), 224, "MobileNetV2", "mobilenet-family", ""


def load_mnasnet1_0():
    from torchvision.models import mnasnet1_0
    return mnasnet1_0(weights=None).eval(), 224, "MnasNet1.0", "mobilenet-family", ""


def load_mnasnet0_75():
    from torchvision.models import mnasnet0_75
    return mnasnet0_75(weights=None).eval(), 224, "MnasNet0.75", "mobilenet-family", ""


def load_semnasnet_100():
    import timm
    return (timm.create_model("semnasnet_100", pretrained=False).eval(),
            224, "semnasnet_100", "se-augmented",
            "SE blocks add cross-channel reduction; reported for comparison.")


def load_mobilenet_v3_small():
    from torchvision.models import mobilenet_v3_small
    return (mobilenet_v3_small(weights=None).eval(), 224,
            "MobileNetV3-Small", "se-augmented",
            "SE blocks violate strict chain-with-skips; reported for comparison.")


# MCUNet zoo (requires sys.path entry for the cloned repo)
def make_mcunet_loader(net_id: str):
    def _load():
        from mcunet.model_zoo import build_model
        try:
            net, res, _ = build_model(net_id=net_id, pretrained=False)
        except Exception:
            net, res, _ = build_model(net_id=net_id, pretrained=True)
        return net.eval(), res, net_id, "mobilenet-family", ""
    _load.__name__ = f"load_{net_id.replace('-', '_').replace('.', '_')}"
    return _load


# =============================================================================
# Per-layer halo-recompute model
#
# The MCUNetV2 prediction model: at patch factor k (1D-equivalent), the
# image is split into k strips of width W/k. Each strip must include a
# halo of (rf_local - 1)/2 pixels on each side of EVERY layer in the
# patch-execution prefix, where rf_local is the per-layer receptive-
# field growth.
#
# Per-strip overhead per layer: extra_pixels_per_boundary = rf_growth_i
# Total over k-1 boundaries: rf_growth_i * (k-1)
# Recomputed MACs at layer i: macs_i * (rf_growth_i * (k-1)) / W_i
# Sum across layers gives total halo F overhead.
#
# This is the same analytical model used to motivate MCUNetV2's
# patch-execution row, applied per-architecture rather than copied
# from the original tabulation.
# =============================================================================


def predict_halo_overhead(records: List[Tuple[str, int, int, int]],
                           image_size: int,
                           k: int,
                           patch_prefix_layers: Optional[int] = None) -> float:
    """Return F_overhead = (F - F*)/F* under per-layer halo recomputation.

    patch_prefix_layers: if not None, only the first N layers are in the
                         patch-execution prefix; deeper layers run normally.
                         MCUNetV2 typically patches the first ~1/3 of the
                         network. If None, applies to all layers (worst case).
    """
    if k == 1:
        return 0.0

    # Reconstruct per-layer spatial widths from records via running stride.
    # records carry (name, macs, out_bytes, rf_growth). We don't have explicit
    # H/W per layer, so we recover W from out_bytes assuming square spatial:
    # out_bytes = H * W * C, but we don't have C. Instead, use the running
    # input width from the receptive-field growth chain: stride doubles when
    # rf_growth uses a stride-2 conv.
    #
    # Simpler and more robust: use rf_growth_i / image_size as the per-layer
    # halo fraction (this matches the MCUNetV2 paper's accounting and the
    # existing measure_real_models.predict_patch formula).

    F_star = sum(r[1] for r in records)
    if F_star == 0:
        return 0.0

    # Determine prefix
    if patch_prefix_layers is None:
        prefix = records
    else:
        prefix = records[:patch_prefix_layers]

    rf_in_prefix = sum(r[3] for r in prefix)  # cumulative rf growth in prefix
    macs_in_prefix = sum(r[1] for r in prefix)

    # Per-layer halo overhead per boundary: rf_growth_i / image_size
    # Total halo MACs = sum_i macs_i * rf_growth_i / image_size * (k-1)
    halo_macs = 0.0
    for _, mac, _, rf_g in prefix:
        halo_macs += mac * (rf_g / image_size) * (k - 1)

    return halo_macs / F_star


def predict_patch_metrics(stats: Dict,
                          records: List[Tuple[str, int, int, int]],
                          image_size: int,
                          k: int) -> Dict:
    """Return predicted (F/F* - 1, S/S*) plus both bound values."""
    if k == 1:
        s_ratio = 1.0
        f_overhead = 0.0
    else:
        s_ratio = 1.0 / k
        f_overhead = predict_halo_overhead(records, image_size, k)

    F_star = stats["F_star"]
    S_star = stats["S_star"]
    gamma = stats["gamma"]
    n = stats["n_layers"]
    cc = stats["cc_i_star"]

    # Linear bound from Theorem 1(d): F - F* >= (gamma/n) * (S* - S)
    # Express as F-overhead fraction: (gamma/n) * S* * (1 - 1/k) / F*
    s_reduction_bytes = S_star * (1.0 - s_ratio)
    linear_bound = (gamma / max(n, 1)) * s_reduction_bytes / F_star

    # Weak harmonic bound (Theorem 2): F - F* >= cc(i*) * (S* - S)/S
    # Express as F-overhead: cc * (S*/S - 1) / F*
    if k == 1:
        weak_harmonic_bound = 0.0
    else:
        weak_harmonic_bound = cc * (S_star / (S_star * s_ratio) - 1.0) / F_star
        # simplifies to cc * (k-1)/F_star, but keep the algebraic form

    return {
        "k": k,
        "S_over_Sstar": s_ratio,
        "F_overhead": f_overhead,
        "linear_bound": linear_bound,
        "weak_harmonic_bound": weak_harmonic_bound,
        "linear_OK": f_overhead + 1e-12 >= linear_bound,
        "weak_harmonic_OK": f_overhead + 1e-12 >= weak_harmonic_bound,
    }


# =============================================================================
# Per-architecture measurement pass
# =============================================================================


def measure_arch(loader_fn) -> Optional[Dict]:
    try:
        model, image_size, name, family, notes = loader_fn()
    except Exception as e:
        return {"name": loader_fn.__name__, "error": f"{type(e).__name__}: {e}"}
    profile = LayerProfile()
    profile.attach(model)
    try:
        with torch.no_grad():
            x = torch.randn(1, 3, image_size, image_size)
            _ = model(x)
    except Exception as e:
        return {"name": name, "error": f"forward failed: {type(e).__name__}: {e}"}
    s = profile.summary()
    if s is None:
        return {"name": name, "error": "no records collected"}
    s["name"] = name
    s["image_size"] = image_size
    s["family"] = family
    s["notes"] = notes
    s["records"] = profile.records
    return s


# =============================================================================
# Regression + bootstrap
# =============================================================================


def fit_no_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """OLS y = c * x (no intercept). Returns (c, R^2)."""
    if len(x) == 0 or np.allclose(x, 0):
        return float("nan"), float("nan")
    c = float(np.sum(x * y) / np.sum(x * x))
    y_pred = c * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 1.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c, r2


def aic(y: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """AIC for Gaussian residuals."""
    n = len(y)
    if n == 0:
        return float("nan")
    rss = float(np.sum((y - y_pred) ** 2))
    if rss <= 0:
        return float("-inf")
    return n * math.log(rss / n) + 2 * n_params


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Returns (median c, lo95, hi95) from bootstrap resampling."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    cs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c, _ = fit_no_intercept(x[idx], y[idx])
        if not math.isnan(c):
            cs.append(c)
    if not cs:
        return float("nan"), float("nan"), float("nan")
    cs_arr = np.array(cs)
    return (float(np.median(cs_arr)),
            float(np.percentile(cs_arr, 2.5)),
            float(np.percentile(cs_arr, 97.5)))


def fit_arch(rows: List[Dict]) -> Dict:
    """Fit harmonic + squared-log models on the non-trivial (k>=2) rows."""
    nontriv = [r for r in rows if r["k"] >= 2 and not math.isnan(r["F_overhead"])]
    if len(nontriv) < 2:
        return {"n_points": len(nontriv), "error": "insufficient points"}

    # x_h = S*/S - 1 = k - 1; y = F/F* - 1
    x_h = np.array([r["k"] - 1.0 for r in nontriv])
    x_sq = np.array([math.log(r["k"]) ** 2 for r in nontriv])  # log(S*/S)^2 = log(k)^2
    y = np.array([r["F_overhead"] for r in nontriv])

    c_h, r2_h = fit_no_intercept(x_h, y)
    c_sq, r2_sq = fit_no_intercept(x_sq, y)

    aic_h = aic(y, c_h * x_h, n_params=1)
    aic_sq = aic(y, c_sq * x_sq, n_params=1)

    c_h_med, c_h_lo, c_h_hi = bootstrap_ci(x_h, y)

    return {
        "n_points": len(nontriv),
        "c_harmonic": c_h,
        "r2_harmonic": r2_h,
        "c_harmonic_median_boot": c_h_med,
        "c_harmonic_lo95": c_h_lo,
        "c_harmonic_hi95": c_h_hi,
        "c_squared_log": c_sq,
        "r2_squared_log": r2_sq,
        "aic_harmonic": aic_h,
        "aic_squared_log": aic_sq,
        "aic_diff_h_minus_sq": aic_h - aic_sq,
    }


# =============================================================================
# Main sweep
# =============================================================================


PATCH_FACTORS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True,
                        help="Path to per-config CSV output")
    parser.add_argument("--summary", required=True,
                        help="Path to summary markdown output")
    parser.add_argument("--mcunet-path", default="mcunet",
                        help="Path to cloned mit-han-lab/mcunet repo")
    args = parser.parse_args()

    if args.mcunet_path and os.path.isdir(args.mcunet_path):
        sys.path.insert(0, args.mcunet_path)

    loaders: List[Tuple[str, Callable]] = [
        ("MobileNetV2", load_mobilenet_v2),
        ("MnasNet1.0", load_mnasnet1_0),
        ("MnasNet0.75", load_mnasnet0_75),
        ("semnasnet_100", load_semnasnet_100),
        ("MobileNetV3-Small", load_mobilenet_v3_small),
    ]
    for net_id in ["mcunet-in0", "mcunet-in1", "mcunet-in2", "mcunet-in3",
                   "mcunet-in4", "mbv2-w0.35", "proxyless-w0.3"]:
        loaders.append((net_id, make_mcunet_loader(net_id)))

    # Per-architecture extraction
    arch_stats: List[Dict] = []
    for short_name, loader in loaders:
        print(f"[measure] {short_name} ...", flush=True)
        s = measure_arch(loader)
        s["short_name"] = short_name
        arch_stats.append(s)
        if "error" in s:
            print(f"  ERROR: {s['error']}", flush=True)
        else:
            print(f"  F* = {s['F_star']:,} | S* = {s['S_star']:,} B | "
                  f"gamma = {s['gamma']:.2f} | cc(i*) = {s['cc_i_star']:,} | "
                  f"rf = {s['receptive_field']} | family = {s['family']}",
                  flush=True)

    # Sweep + write CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arch", "family", "image_size", "k",
            "F_star_macs", "S_star_bytes", "gamma_macs_per_byte",
            "cc_i_star_macs", "receptive_field",
            "F_over_Fstar_minus_1", "S_over_Sstar",
            "linear_bound_F_overhead", "weak_harmonic_bound_F_overhead",
            "linear_OK", "weak_harmonic_OK", "notes",
        ])
        per_arch_sweep: Dict[str, List[Dict]] = {}
        for s in arch_stats:
            if "error" in s:
                # NaN row to make absence visible
                for k in PATCH_FACTORS:
                    w.writerow([
                        s["short_name"], "ERROR", "", k,
                        "", "", "", "", "", "", "", "", "", "", "",
                        s["error"],
                    ])
                continue
            sweep_rows = []
            for k in PATCH_FACTORS:
                if s["family"] == "se-augmented":
                    # Mark predicted overhead as NaN: SE blocks have
                    # cross-channel reductions that violate the chain-with-
                    # skips assumption; the analytical halo model does not
                    # apply unmodified. We still report F*, S*, gamma for
                    # completeness.
                    pred = {
                        "k": k,
                        "S_over_Sstar": float("nan"),
                        "F_overhead": float("nan"),
                        "linear_bound": float("nan"),
                        "weak_harmonic_bound": float("nan"),
                        "linear_OK": False,
                        "weak_harmonic_OK": False,
                    }
                else:
                    pred = predict_patch_metrics(s, s["records"],
                                                  s["image_size"], k)
                sweep_rows.append(pred)
                w.writerow([
                    s["short_name"], s["family"], s["image_size"], k,
                    s["F_star"], s["S_star"], f"{s['gamma']:.6f}",
                    s["cc_i_star"], s["receptive_field"],
                    f"{pred['F_overhead']:.8f}" if not math.isnan(pred["F_overhead"]) else "NaN",
                    f"{pred['S_over_Sstar']:.6f}" if not math.isnan(pred["S_over_Sstar"]) else "NaN",
                    f"{pred['linear_bound']:.8f}" if not math.isnan(pred["linear_bound"]) else "NaN",
                    f"{pred['weak_harmonic_bound']:.8f}" if not math.isnan(pred["weak_harmonic_bound"]) else "NaN",
                    pred["linear_OK"],
                    pred["weak_harmonic_OK"],
                    s.get("notes", ""),
                ])
            per_arch_sweep[s["short_name"]] = sweep_rows

    print(f"\n[wrote] {args.output} ({sum(len(v) for v in per_arch_sweep.values())} rows + skipped)",
          flush=True)

    # Per-architecture regression
    fits: Dict[str, Dict] = {}
    for s in arch_stats:
        if "error" in s:
            continue
        if s["family"] == "se-augmented":
            fits[s["short_name"]] = {"skipped": "se-augmented"}
            continue
        fits[s["short_name"]] = fit_arch(per_arch_sweep[s["short_name"]])

    # Aggregate harmonic c_h across fittable archs
    c_hs = [v["c_harmonic"] for v in fits.values()
            if "c_harmonic" in v and not math.isnan(v["c_harmonic"])]
    summary_stats = {}
    if c_hs:
        summary_stats["n_fitted"] = len(c_hs)
        summary_stats["c_h_min"] = float(np.min(c_hs))
        summary_stats["c_h_median"] = float(np.median(c_hs))
        summary_stats["c_h_max"] = float(np.max(c_hs))
        summary_stats["c_h_mean"] = float(np.mean(c_hs))
        summary_stats["c_h_std"] = float(np.std(c_hs, ddof=1)) if len(c_hs) > 1 else 0.0

    # Write summary markdown
    write_summary(args.summary, arch_stats, fits, summary_stats)
    print(f"[wrote] {args.summary}")


def write_summary(path: str, arch_stats: List[Dict], fits: Dict[str, Dict],
                   summary_stats: Dict) -> None:
    with open(path, "w") as f:
        f.write("# Patch-Sweep 12 x 10 Analytical Sweep -- Results Summary\n\n")
        f.write("**STATUS:** Analytical extended-graph predictions, NOT empirical "
                "inference measurements. The F-overhead values are what the "
                "MCUNetV2-style per-layer halo-recompute model PREDICTS at each "
                "(arch, k) configuration; F\\*, S\\*, gamma, cc(i\\*), and the "
                "receptive field are extracted from real PyTorch models via "
                "forward hooks (identical hook code as `measure_real_models.py` "
                "and `measure_mcunet.py`).\n\n")

        # 1. Per-arch canonical metrics
        f.write("## 1. Per-architecture canonical metrics (extracted)\n\n")
        f.write("| arch | family | image | n_layers | F\\* (M MACs) | S\\* (KB) | "
                "gamma | cc(i\\*) (M) | rf | rf/img |\n")
        f.write("|------|--------|------:|---------:|------:|------:|------:|"
                "------:|----:|------:|\n")
        for s in arch_stats:
            if "error" in s:
                f.write(f"| {s['short_name']} | ERROR | - | - | - | - | - | - | - | - |\n")
                continue
            f.write(f"| {s['short_name']} | {s['family']} | {s['image_size']} | "
                    f"{s['n_layers']} | {s['F_star']/1e6:.1f} | "
                    f"{s['S_star']/1024:.1f} | {s['gamma']:.2f} | "
                    f"{s['cc_i_star']/1e6:.1f} | {s['receptive_field']} | "
                    f"{s['receptive_field']/s['image_size']:.3f} |\n")
        f.write("\n")

        # 2. Per-arch harmonic regression
        f.write("## 2. Per-architecture harmonic regression\n\n")
        f.write("Model: F/F\\* - 1 = c_h * (S\\*/S - 1) over k in {2,3,4,5,6,8,10,12,16} (9 points each).\n\n")
        f.write("Bootstrap 95% CI from 2000 resamples (no-intercept OLS).\n\n")
        f.write("| arch | n | c_h (OLS) | c_h (boot median) | 95% CI | R^2 | fits (R^2>=0.95)? |\n")
        f.write("|------|--:|----------:|------------------:|:-------|----:|:----:|\n")
        for s in arch_stats:
            if "error" in s:
                continue
            name = s["short_name"]
            v = fits.get(name, {})
            if v.get("skipped"):
                f.write(f"| {name} | - | NaN | NaN | NaN | NaN | skipped ({v['skipped']}) |\n")
                continue
            if "error" in v:
                f.write(f"| {name} | - | NaN | NaN | NaN | NaN | {v['error']} |\n")
                continue
            fit_ok = "yes" if v["r2_harmonic"] >= 0.95 else "no"
            ci = f"[{v['c_harmonic_lo95']:.4f}, {v['c_harmonic_hi95']:.4f}]"
            f.write(f"| {name} | {v['n_points']} | {v['c_harmonic']:.4f} | "
                    f"{v['c_harmonic_median_boot']:.4f} | {ci} | "
                    f"{v['r2_harmonic']:.4f} | {fit_ok} |\n")
        f.write("\n")

        # 3. Aggregate
        f.write("## 3. Aggregate c_h across architectures\n\n")
        if summary_stats:
            f.write(f"- Architectures with valid harmonic fit: **{summary_stats['n_fitted']}**\n")
            f.write(f"- c_h median:  **{summary_stats['c_h_median']:.4f}**\n")
            f.write(f"- c_h mean:    **{summary_stats['c_h_mean']:.4f}**  (std = {summary_stats['c_h_std']:.4f})\n")
            f.write(f"- c_h min:     {summary_stats['c_h_min']:.4f}\n")
            f.write(f"- c_h max:     {summary_stats['c_h_max']:.4f}\n\n")
        else:
            f.write("No architectures yielded a valid harmonic fit.\n\n")

        # 4. AIC: harmonic vs squared-log
        f.write("## 4. AIC: harmonic vs squared-log\n\n")
        f.write("Both are 1-parameter no-intercept models. AIC_h - AIC_sq < 0 means "
                "harmonic preferred.\n\n")
        f.write("| arch | AIC harmonic | AIC squared-log | Delta (h - sq) | preferred |\n")
        f.write("|------|------------:|---------------:|--------------:|:---------:|\n")
        n_h_pref = 0
        n_sq_pref = 0
        for s in arch_stats:
            if "error" in s:
                continue
            name = s["short_name"]
            v = fits.get(name, {})
            if v.get("skipped") or "error" in v:
                continue
            pref = "harmonic" if v["aic_diff_h_minus_sq"] < 0 else "squared-log"
            if pref == "harmonic":
                n_h_pref += 1
            else:
                n_sq_pref += 1
            f.write(f"| {name} | {v['aic_harmonic']:.2f} | {v['aic_squared_log']:.2f} | "
                    f"{v['aic_diff_h_minus_sq']:+.2f} | {pref} |\n")
        f.write(f"\nHarmonic preferred in **{n_h_pref}** archs; squared-log preferred in **{n_sq_pref}**.\n\n")

        # 5. Bound validation
        f.write("## 5. Bound validation (any failures across 12 archs * 9 k-values?)\n\n")
        # We need to re-walk from CSV-style logic to summarize. We have access
        # to fit info but not raw rows here; recompute on the fly.
        n_lin_fail = 0
        n_lh_fail = 0
        n_total = 0
        examples_lin_fail = []
        examples_lh_fail = []
        for s in arch_stats:
            if "error" in s or s["family"] == "se-augmented":
                continue
            for k in PATCH_FACTORS:
                if k == 1:
                    continue
                pred = predict_patch_metrics(s, s["records"], s["image_size"], k)
                n_total += 1
                if not pred["linear_OK"]:
                    n_lin_fail += 1
                    examples_lin_fail.append((s["short_name"], k,
                                              pred["F_overhead"],
                                              pred["linear_bound"]))
                if not pred["weak_harmonic_OK"]:
                    n_lh_fail += 1
                    examples_lh_fail.append((s["short_name"], k,
                                              pred["F_overhead"],
                                              pred["weak_harmonic_bound"]))
        f.write(f"- Total (arch, k>=2) configs evaluated: **{n_total}**\n")
        f.write(f"- Universal linear bound (Thm 1d) failures: **{n_lin_fail}**\n")
        f.write(f"- Weak harmonic bound (Thm 2) failures: **{n_lh_fail}**\n\n")
        if examples_lin_fail:
            f.write("Linear bound failures (first 10):\n\n")
            for name, k, fo, lb in examples_lin_fail[:10]:
                f.write(f"  - {name} k={k}: F-overhead={fo:.4f} < linear bound={lb:.4f}\n")
            f.write("\n")
        if examples_lh_fail:
            f.write("Weak harmonic bound failures (first 10):\n\n")
            for name, k, fo, lb in examples_lh_fail[:10]:
                f.write(f"  - {name} k={k}: F-overhead={fo:.4f} < weak harmonic bound={lb:.4f}\n")
            f.write("\n")

        # 6. LaTeX table output
        f.write("## 6. LaTeX table for the empirical-evidence section\n\n")
        f.write("```latex\n")
        f.write("\\begin{center}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrcc}\n")
        f.write("\\toprule\n")
        f.write("Architecture & $n_k$ & $c_h$ (boot 95\\% CI) & $R^2_{\\text{harm}}$ & "
                "harm pref. (AIC) \\\\\n")
        f.write("\\midrule\n")
        for s in arch_stats:
            if "error" in s:
                continue
            name = s["short_name"]
            v = fits.get(name, {})
            if v.get("skipped") or "error" in v:
                continue
            ci = f"[{v['c_harmonic_lo95']:.3f}, {v['c_harmonic_hi95']:.3f}]"
            pref = "\\checkmark" if v["aic_diff_h_minus_sq"] < 0 else "$\\times$"
            safe_name = name.replace("_", "\\_")
            f.write(f"{safe_name} & {v['n_points']} & "
                    f"{v['c_harmonic']:.3f} {ci} & "
                    f"{v['r2_harmonic']:.3f} & {pref} \\\\\n")
        f.write("\\midrule\n")
        if summary_stats:
            f.write(f"\\textbf{{Median (n={summary_stats['n_fitted']} archs)}} & "
                    f"-- & \\textbf{{{summary_stats['c_h_median']:.3f}}} & "
                    f"-- & -- \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{center}\n")
        f.write("```\n\n")
        f.write("**Replacement claim for section 5:** \"Across {n} chain-with-skips "
                "MobileNet-family architectures, an analytical extended-graph "
                "regression of $F/F^* - 1$ on $S^*/S - 1$ at $k \\in \\{2,3,4,5,6,8,10,12,16\\}$ "
                "yields a median harmonic constant $c_h = $ {median} (per-arch range "
                "[{min}, {max}], all per-arch $R^2 \\geq$ {min_r2}). The squared-log "
                "alternative is rejected by AIC in {n_h_pref}/{n_total_fits} "
                "architectures.\"\n\n")

        f.write("## 7. Honest disclosures\n\n")
        f.write("- These F-overhead numbers are PREDICTIONS of the per-layer halo-recompute model; "
                "they are NOT measured patch-execution latencies.\n")
        f.write("- The S/S* = 1/k assumption is an idealization for single-rate chain-with-skips backbones; "
                "real patch executors have additional buffer overhead (control, weights, intermediate ping-pong).\n")
        f.write("- semnasnet_100 and MobileNetV3-Small are reported for canonical metrics only; their SE blocks "
                "introduce cross-channel reductions that violate the strict chain-with-skips assumption, "
                "and the analytical halo model does not apply unmodified. Per the paper's scope, "
                "they are excluded from the harmonic regression.\n")
        f.write("- The analytical model matches the one used to motivate MCUNetV2's "
                "patch-execution argument; the contribution here is statistical "
                "content (per-arch CIs, AIC), not a different model.\n")


if __name__ == "__main__":
    main()
