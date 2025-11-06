#!/usr/bin/env python3
"""Merge the CSV results into a single file with a shared schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


CORE_COLUMNS = ["kernel_name", "dtype", "input_size", "runtime", "correct", "device_name"]


def normalize_correct(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip().lower()
    if text in {"true", "pass", "1", "yes"}:
        return "PASS"
    if text in {"false", "fail", "0", "no"}:
        return "FAIL"
    return value.strip()


def load_cuda(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) < 6:
                continue
            yield {
                "kernel_name": row[0].strip(),
                "dtype": row[1].strip().upper(),
                "input_size": row[2].strip(),
                "runtime": row[3].strip(),
                "correct": normalize_correct(row[4]),
                "device_name": "CUDA",
            }


def load_opencl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            yield {
                "kernel_name": (row.get("kernel_name") or "").strip(),
                "dtype": (row.get("dtype") or "").strip().upper(),
                "input_size": (row.get("input_size") or "").strip(),
                "runtime": (row.get("runtime_ms") or "").strip(),
                "correct": normalize_correct(row.get("correct")),
                "device_name": "OpenCL",
            }


def load_pezy(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            yield {
                "kernel_name": (row.get("Benchmark") or "").strip(),
                "dtype": (row.get("Precision") or "").strip().upper(),
                "input_size": (row.get("Size") or "").strip(),
                "runtime": (row.get("Time(ms)") or "").strip(),
                "correct": normalize_correct(row.get("Result")),
                "device_name": "PEZY-SC3",
            }


LOADER_BY_KEYWORD = [
    ("cuda", load_cuda),
    ("opencl", load_opencl),
    ("pezy", load_pezy),
]


def select_loader(path: Path):
    name = path.stem.lower()
    for keyword, loader in LOADER_BY_KEYWORD:
        if keyword in name:
            return loader
    raise ValueError(f"No loader registered for file {path.name}")


def merge_results(input_dir: Path, output_path: Path) -> int:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        loader = select_loader(csv_path)
        rows.extend(loader(csv_path))

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CSV result files into a common format.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("csv"),
        help="Directory containing the CSV inputs (default: ./csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_core_results.csv"),
        help="Path for the merged CSV (default: ./combined_core_results.csv)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory {args.input_dir} does not exist or is not a directory.")

    total = merge_results(args.input_dir, args.output)
    print(f"Wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
