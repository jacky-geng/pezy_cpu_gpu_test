#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#define CUDA_CHECK(err) { if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " (" << __FILE__ << ":" << __LINE__ << ")\n"; exit(1);} }

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer(){ cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end() { cudaEventRecord(stop); cudaEventSynchronize(stop);
                  float ms; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// Return CUDA device name for CSV output
inline std::string cuda_device_name() {
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return std::string(prop.name);
}

// Compare two arrays with relative tolerance
template<typename T>
inline bool compare_results(const std::vector<T>& a, const std::vector<T>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = static_cast<double>(a[i]);
        double db = static_cast<double>(b[i]);
        if (std::fabs(da - db) > tol * (1.0 + std::fabs(db))) return false;
    }
    return true;
}

// Simple CSV row writer (same columns as OpenCL)
inline void write_csv_row_cuda(const std::string& path,
                               const std::string& kernel,
                               const std::string& dtype,
                               const std::string& input_size,
                               double runtime_ms,
                               bool correct,
                               const std::string& device_name,
                               size_t g0, size_t g1, size_t g2,
                               size_t l0, size_t l1, size_t l2,
                               double flops_est,
                               double bw_GBps) {
    FILE* f = fopen(path.c_str(), "a");
    if (!f) return;
    fprintf(f, "%s,%s,%s,%.6f,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%.6e,%.6f\n",
            kernel.c_str(), dtype.c_str(), input_size.c_str(), runtime_ms,
            correct ? "true" : "false", device_name.c_str(),
            g0, g1, g2, l0, l1, l2, flops_est, bw_GBps);
    fclose(f);
}
