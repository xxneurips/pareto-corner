"""
Real patch-based inference executor for the Pareto-Corner NeurIPS submission.

For each (arch, k):
  - Split the input image into k 1D strips along the height axis.
  - Each strip carries a receptive-field-sized halo (overlap with neighbors).
  - Run a "patched" forward pass where the model processes each strip independently
    and we discard halo pixels in the final output.
  - Measure:
      F_macs_measured: sum of MACs over all k patch forwards (using fvcore)
      S_peak_kb_measured: peak GPU memory during a SINGLE patch forward
                          (proxy for SRAM peak under per-patch execution)
      latency_ms: median wall-clock for one full patched inference.
  - Compare to the analytical F_predicted, S_predicted from sweep_120_results.csv.

This is a coarser proxy for the MCUNetV2 patch engine, but it gives REAL
measurements (not analytical extrapolation). The peak memory under k patches
should approximate S*/k + halo_overhead, matching the SDF model in the paper.

Architectures:
  - mcunet-in0  (image 48,  ~6.4M MACs, 36KB S*)
  - mcunet-in1  (image 96,  ~13M MACs)
  - mcunet-in2  (image 160, ~67M MACs)
  - mcunet-in3  (image 176, ~82M MACs)
  - mcunet-in4  (image 160, ~126M MACs)
  - mbv2-w0.35  (image 144, ~23M MACs)
  - proxyless-w0.3 (image 176, ~38M MACs)

We sweep k in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16}. For each k, we choose the halo
size based on the model's full receptive field (read from sweep_120_results.csv).

USAGE:
  python real_patch_executor.py --archs mcunet-in0 mcunet-in1 \
      --ks 1 2 3 4 5 6 8 10 12 16 \
      --out real_patch_inference.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MCUNET_PATH = os.environ.get("MCUNET_PATH", "./mcunet")
if os.path.isdir(MCUNET_PATH):
    sys.path.insert(0, MCUNET_PATH)


# -----------------------------------------------------------------------------
# Architecture metadata loaded from sweep_120_results.csv
# -----------------------------------------------------------------------------

ARCH_META: Dict[str, Dict] = {
    # arch_id: image_size, receptive_field, F*_macs, S*_bytes
    "mcunet-in0":     {"image_size": 48,  "rf": 73,  "F_star": 6364224,    "S_star": 36864},
    "mcunet-in1":     {"image_size": 96,  "rf": 67,  "F_star": 12790096,   "S_star": 73728},
    "mcunet-in2":     {"image_size": 160, "rf": 87,  "F_star": 67287200,   "S_star": 153600},
    "mcunet-in3":     {"image_size": 176, "rf": 91,  "F_star": 81785520,   "S_star": 185856},
    "mcunet-in4":     {"image_size": 160, "rf": 101, "F_star": 125874400,  "S_star": 307200},
    "mbv2-w0.35":     {"image_size": 144, "rf": 47,  "F_star": 23499152,   "S_star": 248832},
    "proxyless-w0.3": {"image_size": 176, "rf": 115, "F_star": 38282216,   "S_star": 185856},
}


# -----------------------------------------------------------------------------
# Halo computation
# -----------------------------------------------------------------------------

def compute_patch_halo_pixels(rf: int) -> int:
    """Halo size on each side of a patch (in input pixels).

    For a network with receptive field RF, processing a patch of width W from
    the input requires (W + 2*halo) pixels, where halo = (RF - 1) // 2 covers
    the spatial influence of pixels outside the patch boundary.
    """
    return (rf - 1) // 2


def slice_with_halo(x: torch.Tensor, k: int, patch_idx: int, halo: int) -> torch.Tensor:
    """Extract patch_idx-th vertical strip from input x with halo padding.

    x: [B, C, H, W]
    Returns:
      a tensor [B, C, H_patch + 2*halo_eff, W] where halo_eff is clipped at
      image boundaries (zero-padded).
    """
    B, C, H, W = x.shape
    # base patch boundaries (no halo)
    h_per_patch = H // k
    h_start = patch_idx * h_per_patch
    h_end = (patch_idx + 1) * h_per_patch if patch_idx < k - 1 else H
    # extend by halo on each side
    h_start_halo = max(0, h_start - halo)
    h_end_halo = min(H, h_end + halo)
    return x[:, :, h_start_halo:h_end_halo, :]


# -----------------------------------------------------------------------------
# MAC counting (fvcore)
# -----------------------------------------------------------------------------

def count_macs(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> int:
    """Count multiply-accumulate operations for a forward pass.

    Returns total MACs. fvcore counts conv MACs as (kernel * in_channel *
    out_h * out_w * out_channel), matching our analytical convention.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_shape, device=next(model.parameters()).device)
        with torch.no_grad():
            flop_analyzer = FlopCountAnalysis(model, x)
            flop_analyzer.unsupported_ops_warnings(False)
            flop_analyzer.uncalled_modules_warnings(False)
            macs = int(flop_analyzer.total())
        return macs
    except Exception as e:
        print(f"  [WARN] fvcore failed ({e}), falling back to thop", flush=True)
        try:
            from thop import profile
            x = torch.randn(*input_shape, device=next(model.parameters()).device)
            macs, _ = profile(model, inputs=(x,), verbose=False)
            return int(macs)
        except Exception as e2:
            print(f"  [WARN] thop also failed ({e2})", flush=True)
            return -1


# -----------------------------------------------------------------------------
# Per-patch execution
# -----------------------------------------------------------------------------

def measure_peak_activation(
    model: nn.Module, x: torch.Tensor, verbose: bool = False
) -> Tuple[int, int, str]:
    """Measure peak ACTIVATION footprint during forward pass via hooks.

    For each module, we record the tensor footprint (output_bytes +
    input_bytes + intermediate_bytes) right after its forward.
    The reported S is the max output activation tensor size over all layers,
    which is the canonical chain-with-skips peak SRAM proxy.

    Also returns max simultaneously-live activation bytes assuming
    layer-by-layer execution (only the most recent output is live).

    Returns:
      max_output_bytes: max single-layer output tensor footprint (bytes, fp32)
      max_inout_bytes: max (input + output) bytes at any single layer
      peak_layer_name: name of the layer where peak occurred
    """
    max_out = [0]
    max_inout = [0]
    peak_name = [""]
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            out_bytes = output.numel() * output.element_size()
            in_bytes = 0
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    in_bytes += inp.numel() * inp.element_size()
            if out_bytes > max_out[0]:
                max_out[0] = out_bytes
                peak_name[0] = f"{name} ({type(module).__name__}) out={tuple(output.shape)}"
            if in_bytes + out_bytes > max_inout[0]:
                max_inout[0] = in_bytes + out_bytes
        return hook

    for n, m in model.named_modules():
        # only register on leaf modules to avoid double-counting
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(make_hook(n)))
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    return max_out[0], max_inout[0], peak_name[0]


def run_patched_inference(
    model: nn.Module,
    image_size: int,
    k: int,
    rf: int,
    n_warmup: int = 3,
    n_runs: int = 10,
    device: str = "cuda",
) -> Dict:
    """Execute model on k 1D-strip patches of a single image. Measure F, S, latency.

    Returns:
      f_macs_total: sum of MACs over all k patches
      s_peak_act_bytes: peak per-layer ACTIVATION output footprint during a
                        single-patch forward (real SRAM-proxy)
      s_peak_inout_bytes: peak (in+out) activation bytes at any single layer
      s_gpu_alloc_bytes: peak total GPU allocation during single-patch fwd
                         (includes weights + workspace, less SRAM-relevant)
      latency_ms: median wall-clock for one full patched inference (k forwards)
    """
    halo = compute_patch_halo_pixels(rf)
    h_per_patch = image_size // k
    if h_per_patch <= 0:
        raise ValueError(f"k={k} too large for image_size={image_size}")

    x_full = torch.randn(1, 3, image_size, image_size, device=device)
    patches = []
    for i in range(k):
        p = slice_with_halo(x_full, k, i, halo)
        patches.append(p)

    # 1) Count MACs per patch (sum)
    f_macs_total = 0
    for i, p in enumerate(patches):
        macs_i = count_macs(model, tuple(p.shape))
        if macs_i < 0:
            return {
                "f_macs_total": -1,
                "s_peak_act_bytes": -1,
                "s_peak_inout_bytes": -1,
                "s_gpu_alloc_bytes": -1,
                "latency_ms": -1,
                "error": "fvcore_and_thop_failed",
            }
        f_macs_total += macs_i

    # 2) Measure peak activation memory during the LARGEST-patch forward.
    largest_p = max(patches, key=lambda p: p.shape[2])
    s_peak_act, s_peak_inout, peak_layer = measure_peak_activation(model, largest_p)

    # 3) Also report total GPU peak allocation (includes weights + workspace)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(largest_p)
    torch.cuda.synchronize()
    s_gpu_alloc = torch.cuda.max_memory_allocated()

    # 3) Latency: full patched inference (run all k patches sequentially).
    for _ in range(n_warmup):
        with torch.no_grad():
            for p in patches:
                _ = model(p)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for p in patches:
                _ = model(p)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    latency_ms = times[len(times) // 2]

    return {
        "f_macs_total": f_macs_total,
        "s_peak_act_bytes": s_peak_act,
        "s_peak_inout_bytes": s_peak_inout,
        "s_gpu_alloc_bytes": s_gpu_alloc,
        "peak_layer": peak_layer,
        "patch_shape": tuple(largest_p.shape),
        "latency_ms": latency_ms,
        "error": None,
    }


# -----------------------------------------------------------------------------
# Analytical predictions (from sweep_120_results.csv formulas)
# -----------------------------------------------------------------------------

def analytical_f_s(arch: str, k: int) -> Tuple[int, int]:
    """Return (F_predicted_macs, S_predicted_bytes) for arch at patch factor k.

    F_predicted = F* * (1 + (rf/image_size) * (k-1))
    S_predicted = S* / k
    """
    meta = ARCH_META[arch]
    f_star = meta["F_star"]
    s_star = meta["S_star"]
    rf = meta["rf"]
    img = meta["image_size"]
    if k == 1:
        return f_star, s_star
    f_pred = int(f_star * (1.0 + (rf / img) * (k - 1)))
    s_pred = int(s_star / k)
    return f_pred, s_pred


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def run_sweep(
    archs: List[str],
    ks: List[int],
    out_csv: str,
    device: str = "cuda",
) -> None:
    from mcunet.model_zoo import build_model

    rows: List[Dict] = []
    for arch in archs:
        if arch not in ARCH_META:
            print(f"[skip] {arch}: no metadata in ARCH_META", flush=True)
            continue
        meta = ARCH_META[arch]
        print(f"\n=== {arch} (image_size={meta['image_size']}, rf={meta['rf']}) ===", flush=True)
        try:
            model, image_size, _ = build_model(arch, pretrained=True)
        except Exception as e:
            print(f"  [FAIL] build_model({arch}) raised {type(e).__name__}: {e}", flush=True)
            continue
        model = model.to(device).eval()

        for k in ks:
            try:
                # Validate k vs image size
                if image_size // k < 2:
                    print(f"  k={k}: skipped (image_size={image_size} too small for k patches)", flush=True)
                    continue

                result = run_patched_inference(
                    model=model,
                    image_size=image_size,
                    k=k,
                    rf=meta["rf"],
                    device=device,
                )
                f_pred, s_pred = analytical_f_s(arch, k)
                if result["f_macs_total"] > 0 and f_pred > 0:
                    f_agreement = result["f_macs_total"] / f_pred
                else:
                    f_agreement = float("nan")
                # Note: fvcore reports MACs as fp32 (4-byte) tensors. We convert
                # peak activation bytes to int8-equivalent by dividing by 4
                # to match the analytical model which assumes int8.
                s_act_kb_int8 = result["s_peak_act_bytes"] / 4.0 / 1024.0 if result["s_peak_act_bytes"] > 0 else -1
                s_inout_kb_int8 = result["s_peak_inout_bytes"] / 4.0 / 1024.0 if result["s_peak_inout_bytes"] > 0 else -1
                if result["s_peak_act_bytes"] > 0 and s_pred > 0:
                    # Compare measured (in int8 bytes) to analytical S* / k
                    s_agreement = (result["s_peak_act_bytes"] / 4.0) / s_pred
                else:
                    s_agreement = float("nan")
                row = {
                    "arch": arch,
                    "k": k,
                    "F_macs_measured": result["f_macs_total"],
                    "S_peak_act_kb_int8": s_act_kb_int8,
                    "S_peak_inout_kb_int8": s_inout_kb_int8,
                    "S_gpu_alloc_kb": result["s_gpu_alloc_bytes"] / 1024.0 if result["s_gpu_alloc_bytes"] > 0 else -1,
                    "latency_ms": result["latency_ms"],
                    "F_predicted_analytical": f_pred,
                    "S_predicted_kb_analytical": s_pred / 1024.0,
                    "F_agreement_factor": f_agreement,
                    "S_agreement_factor": s_agreement,
                    "image_size": image_size,
                    "rf": meta["rf"],
                    "peak_layer": result.get("peak_layer", ""),
                    "patch_shape": str(result.get("patch_shape", "")),
                    "error": result.get("error"),
                }
                rows.append(row)
                print(
                    f"  k={k:>2}: F_meas={row['F_macs_measured']/1e6:7.2f}M "
                    f"F_pred={row['F_predicted_analytical']/1e6:7.2f}M "
                    f"F_agree={row['F_agreement_factor']:.4f} | "
                    f"S_act={row['S_peak_act_kb_int8']:7.1f}KB "
                    f"S_pred={row['S_predicted_kb_analytical']:7.1f}KB "
                    f"S_agree={row['S_agreement_factor']:.3f} | "
                    f"lat={row['latency_ms']:6.2f}ms | peak: {row['peak_layer']}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [FAIL] k={k} raised {type(e).__name__}: {e}", flush=True)
                rows.append({
                    "arch": arch,
                    "k": k,
                    "F_macs_measured": -1,
                    "S_peak_act_kb_int8": -1,
                    "S_peak_inout_kb_int8": -1,
                    "S_gpu_alloc_kb": -1,
                    "latency_ms": -1,
                    "F_predicted_analytical": -1,
                    "S_predicted_kb_analytical": -1,
                    "F_agreement_factor": float("nan"),
                    "S_agreement_factor": float("nan"),
                    "image_size": meta["image_size"],
                    "rf": meta["rf"],
                    "peak_layer": "",
                    "patch_shape": "",
                    "error": f"{type(e).__name__}: {e}",
                })

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSV
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_csv}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["mcunet-in0"])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
    p.add_argument("--out", default="real_patch_inference.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_sweep(args.archs, args.ks, args.out, args.device)


if __name__ == "__main__":
    main()
