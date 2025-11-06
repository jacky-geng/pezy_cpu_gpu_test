# pezy_cpu_gpu_test

## Layout

- `csv/` – canonical location for generated benchmark CSV files (CUDA/OpenCL outputs and merged summaries).
- `scripts/` – helper utilities such as `merge_csv.py` and `summarize_speedups.py` for post-processing results.
- `cuda/`, `opencl/` – platform-specific benchmarks; each binary now emits its CSV directly into `../csv`.
