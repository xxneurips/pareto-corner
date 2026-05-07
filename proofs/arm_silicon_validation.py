"""
ARM-silicon validation for the Pareto-Corner Theorem (see paper for the large-memory aarch64 sanity check).

Runs the 6 architectures on the actual GB10 Grace Blackwell aarch64 platform
and measures REAL peak activation memory + inference latency, then compares
to the theoretical (F*, S*) the paper computes.

Anchors the activation-footprint model against real-hardware peak
  memory on a large-memory aarch64 platform (does not test the
  Cortex-M SRAM regime; see paper limitations).

Method:
  For each architecture:
    1. Load on CUDA (unified memory on GB10).
    2. Warmup 5 forwards.
    3. Reset CUDA peak memory stats.
    4. Run 20 forward passes; record per-iter peak memory + latency.
    5. Param memory = sum of parameter sizes.
    6. Peak activation memory = peak total memory - param memory.
    7. Compare measured peak activation to theoretical S* from forward hooks.

Output: arm_validation_results.csv

Usage:
  python arm_silicon_validation.py
"""

from __future__ import annotations
import csv
import gc
import os
import sys
import time
from typing import Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_real_models import LayerProfile  # noqa: E402
from measure_real_models_extended import (  # noqa: E402
    load_mobilenet_v2_anchor,
    load_resnet50,
    load_efficientnet_lite0,
    load_regnety_002,
    load_shufflenet_v2,
    load_ghostnet,
)


def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def theoretical_s_star_int8(loader_fn) -> Dict:
    """Compute S* (max layer activation footprint, int8 baseline) and F*."""
    model, image_size, name = loader_fn()
    profile = LayerProfile()
    profile.attach(model)
    with torch.no_grad():
        x = torch.randn(1, 3, image_size, image_size)
        _ = model(x)
    s = profile.summary()
    return {
        "name": name.split(" (")[0],
        "image_size": image_size,
        "F_star_M": s["F_star"] / 1e6,
        "S_star_int8_KB": s["S_star"] / 1024,
        "gamma": s["gamma"],
        "rf": s["receptive_field"],
    }


def measure_arm_silicon(loader_fn, n_warmup: int = 5, n_iter: int = 20) -> Dict:
    """Measure real peak activation memory + latency on GB10 aarch64."""
    model, image_size, name = loader_fn()
    short = name.split(" (")[0]
    device = torch.device("cuda")
    model = model.to(device).eval()
    p_bytes = param_bytes(model)

    x = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    pre_alloc = torch.cuda.memory_allocated()

    latencies = []
    with torch.no_grad():
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    peak_alloc = torch.cuda.max_memory_allocated()
    # Activation peak = peak total alloc during forward - param/buffer alloc
    # (input + activation tensors).
    # On unified memory GB10, all "cuda" allocs come from the 128 GB pool.
    activation_peak_bytes = peak_alloc - pre_alloc
    # In float32, baseline. Convert to int8-equivalent for paper's S* comparison.
    activation_peak_KB_fp32 = activation_peak_bytes / 1024
    activation_peak_KB_int8 = activation_peak_KB_fp32 / 4  # 4x smaller in int8

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]
    p10_ms = latencies[max(0, len(latencies) // 10)]
    p90_ms = latencies[min(len(latencies) - 1, (9 * len(latencies)) // 10)]

    # Cleanup
    del model
    del x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": short,
        "param_KB": p_bytes / 1024,
        "measured_peak_act_KB_fp32": activation_peak_KB_fp32,
        "measured_peak_act_KB_int8": activation_peak_KB_int8,
        "latency_p10_ms": p10_ms,
        "latency_median_ms": median_ms,
        "latency_p90_ms": p90_ms,
    }


def main():
    loaders = [
        load_mobilenet_v2_anchor,
        load_resnet50,
        load_efficientnet_lite0,
        load_regnety_002,
        load_shufflenet_v2,
        load_ghostnet,
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Torch:  {torch.__version__}")
    print(f"Arch:   {os.uname().machine}")
    print(f"CUDA:   {torch.version.cuda}")
    print()

    rows = []
    for loader in loaders:
        print(f"--- {loader.__name__} ---")
        # Theoretical (CPU)
        t = theoretical_s_star_int8(loader)
        print(f"  theoretical: F*={t['F_star_M']:.1f} M, S*_int8={t['S_star_int8_KB']:.1f} KB, gamma={t['gamma']:.1f}")
        # Empirical (GPU)
        try:
            e = measure_arm_silicon(loader)
            print(f"  measured:    peak_act_fp32={e['measured_peak_act_KB_fp32']:.1f} KB, "
                  f"peak_act_int8_eq={e['measured_peak_act_KB_int8']:.1f} KB, "
                  f"latency_p50={e['latency_median_ms']:.2f} ms")
            ratio_int8 = e["measured_peak_act_KB_int8"] / t["S_star_int8_KB"] if t["S_star_int8_KB"] > 0 else 0
            print(f"  ratio measured_int8_eq / theoretical_S*_int8 = {ratio_int8:.2f}")
            row = {**t, **e, "ratio_int8": ratio_int8}
            rows.append(row)
        except Exception as ex:
            print(f"  [measure failed: {type(ex).__name__}: {ex}]")
        print()

    # Write CSV
    out = "arm_validation_results.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r[k] for k in keys])
        print(f"[wrote {out}]")

    # LaTeX-ready summary
    print("\n% --- ARM silicon validation summary (LaTeX) ---")
    print("% Cols: Arch | F* (M) | S*_int8 (KB) | Measured peak fp32 (KB) | "
          "p50 latency (ms)")
    print()
    for r in rows:
        print(f"{r['name']} & {r['F_star_M']:.0f} & {r['S_star_int8_KB']:.0f} & "
              f"{r['measured_peak_act_KB_fp32']:.0f} & {r['latency_median_ms']:.2f} \\\\")


if __name__ == "__main__":
    main()
