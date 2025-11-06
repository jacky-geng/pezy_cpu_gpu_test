#!/usr/bin/env python3
"""Generate a PEZY-SC3 speedup summary table from merged runtime results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

OUTPUT_FIELDS = [
    "kernel_name",
    "dtype",
    "input_size",
    "pezy_runtime(ms)",
    "speedup_vs_cuda",
    "speedup_vs_opencl",
]

FLOAT_DTYPES = {"FP32", "FP64"}

KERNEL_NAME_MAP = {
    "CUDA": {
        "vecadd_basic": "Vector Add",
        "dot_global": "Vector Dot",
        "gemv_global": "GEMV",
        "matmul_global": "MatMul",
        "conv2d_global": "Conv2D",
        "layernorm_basic": "LayerNorm",
        "fft1d_global": "FFT",
        "montecarlo_basic": "MonteCarloPi",
        "stencil2d_3x3": "Stencil2D 3x3",
        "stencil2d_5x5": "Stencil2D 5x5",
        "stencil3d_global": "Stencil3D Global",
        "stencil3d_shared": "Stencil3D Shared",
    },
    "OpenCL": {
        "vecadd_basic": "Vector Add",
        "dot_global": "Vector Dot",
        "gemv_global": "GEMV",
        "matmul_global": "MatMul",
        "conv2d_global": "Conv2D",
        "layernorm_basic": "LayerNorm",
        "fft1d_global": "FFT",
        "montecarlo_basic": "MonteCarloPi",
        "stencil2d_3x3": "Stencil2D 3x3",
        "stencil2d_5x5": "Stencil2D 5x5",
        "stencil3d_global": "Stencil3D Global",
        "stencil3d_shared": "Stencil3D Shared",
    },
    "PEZY-SC3": {
        "Vector Add": "Vector Add",
        "Vector Dot": "Vector Dot",
        "GEMV": "GEMV",
        "MatMul": "MatMul",
        "Conv2D": "Conv2D",
        "LayerNorm": "LayerNorm",
        "FFT": "FFT",
        "MonteCarloPi": "MonteCarloPi",
        "Stencil2D 3x3": "Stencil2D 3x3",
        "Stencil2D 5x5": "Stencil2D 5x5",
        "Stencil3D Global": "Stencil3D Global",
        "Stencil3D Shared": "Stencil3D Shared",
    },
}


def canonical_kernel_name(device: str, name: str) -> str | None:
    mapping = KERNEL_NAME_MAP.get(device)
    if not mapping:
        return None
    return mapping.get(name)


def parse_runtime(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def aggregate_runtimes(input_path: Path) -> List[Dict[str, str]]:
    buckets: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_kernel = (row.get("kernel_name") or "").strip()
            dtype = (row.get("dtype") or "").strip()
            input_size = (row.get("input_size") or "").strip()
            device = (row.get("device_name") or "").strip()

            if not raw_kernel or not dtype or not input_size or not device:
                continue

            canonical_kernel = canonical_kernel_name(device, raw_kernel)
            if not canonical_kernel:
                continue

            if dtype not in FLOAT_DTYPES:
                continue

            runtime = parse_runtime(row.get("runtime"))
            if runtime is None:
                continue

            buckets[(canonical_kernel, dtype, input_size)][device].append(runtime)

    summary: List[Dict[str, str]] = []
    for (kernel, dtype, input_size), device_map in buckets.items():
        pezy_runs = device_map.get("PEZY-SC3")
        if not pezy_runs:
            continue

        pezy_runtime = sum(pezy_runs) / len(pezy_runs)
        if pezy_runtime <= 0:
            continue

        cuda_runs = device_map.get("CUDA")
        opencl_runs = device_map.get("OpenCL")

        cuda_runtime = sum(cuda_runs) / len(cuda_runs) if cuda_runs else None
        opencl_runtime = sum(opencl_runs) / len(opencl_runs) if opencl_runs else None

        speedup_cuda = cuda_runtime / pezy_runtime if cuda_runtime else None
        speedup_opencl = opencl_runtime / pezy_runtime if opencl_runtime else None

        summary.append(
            {
                "kernel_name": kernel,
                "dtype": dtype,
                "input_size": input_size,
                "pezy_runtime(ms)": f"{pezy_runtime:.6f}",
                "speedup_vs_cuda": f"{speedup_cuda:.6f}" if speedup_cuda is not None else "",
                "speedup_vs_opencl": f"{speedup_opencl:.6f}" if speedup_opencl is not None else "",
            }
        )

    summary.sort(key=lambda row: (row["kernel_name"], row["dtype"], row["input_size"]))
    return summary


def write_summary(rows: List[Dict[str, str]], output_path: Path) -> int:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize PEZY-SC3 performance against CUDA and OpenCL runtimes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_core_results.csv"),
        help="Path to the merged runtime CSV (default: combined_core_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pezy_speedup_summary.csv"),
        help="Destination path for the summary CSV (default: pezy_speedup_summary.csv)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file {args.input} not found.")

    rows = aggregate_runtimes(args.input)
    write_summary(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
