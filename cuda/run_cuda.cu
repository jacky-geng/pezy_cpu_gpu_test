#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <string>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "utils_cuda.hpp"
#include "../common/benchmark_config.hpp"
#include "../common/math_utils.hpp"
#include "../common/sequence_configs.hpp"

// Kernel prototypes from kernels_all.cu
extern "C" __global__ void vecadd_f32(const float*, const float*, float*, int);
extern "C" __global__ void vecadd_f64(const double*, const double*, double*, int);
extern "C" __global__ void dot_f32(const float*, const float*, float*, int);
extern "C" __global__ void dot_f64(const double*, const double*, double*, int);
extern "C" __global__ void reduction_f32(const float*, float*, int);
extern "C" __global__ void reduction_f64(const double*, double*, int);
extern "C" __global__ void gemv_global_f32(const float*, const float*, float*, int, int);
extern "C" __global__ void gemv_global_f64(const double*, const double*, double*, int, int);
extern "C" __global__ void gemv_shared_f32(const float*, const float*, float*, int, int);
extern "C" __global__ void gemv_shared_f64(const double*, const double*, double*, int, int);
extern "C" __global__ void matmul_global_f32(const float*, const float*, float*, int, int, int);
extern "C" __global__ void matmul_global_f64(const double*, const double*, double*, int, int, int);
extern "C" __global__ void matmul_shared_f32(const float*, const float*, float*, int, int, int);
extern "C" __global__ void matmul_shared_f64(const double*, const double*, double*, int, int, int);
extern "C" void launch_scan_f32(const float*, float*, float*, int, dim3, dim3, cudaStream_t);
extern "C" void launch_scan_f64(const double*, double*, double*, int, dim3, dim3, cudaStream_t);
extern "C" void launch_scan_add_f32(float*, const float*, int, dim3, dim3, cudaStream_t);
extern "C" void launch_scan_add_f64(double*, const double*, int, dim3, dim3, cudaStream_t);
extern "C" __global__ void spmv_csr_f32(const int*, const int*, const float*, const float*, float*, int);
extern "C" __global__ void spmv_csr_f64(const int*, const int*, const double*, const double*, double*, int);
extern "C" __global__ void conv2d_global_f32(const float*, const float*, float*, int, int);
extern "C" __global__ void conv2d_global_f64(const double*, const double*, double*, int, int);
extern "C" __global__ void conv2d_shared_f32(const float*, const float*, float*, int, int);
extern "C" __global__ void conv2d_shared_f64(const double*, const double*, double*, int, int);
extern "C" __global__ void depthwiseconv_global_f32(const float*, const float*, float*, int, int, int);
extern "C" __global__ void depthwiseconv_global_f64(const double*, const double*, double*, int, int, int);
extern "C" __global__ void depthwiseconv_tiled_f32(const float*, const float*, float*, int, int, int);
extern "C" __global__ void depthwiseconv_tiled_f64(const double*, const double*, double*, int, int, int);
extern "C" __global__ void softmax_basic_f32(const float*, float*, int);
extern "C" __global__ void softmax_basic_f64(const double*, double*, int);
extern "C" __global__ void layernorm_basic_f32(const float*, float*, int, float);
extern "C" __global__ void layernorm_basic_f64(const double*, double*, int, double);
extern "C" __global__ void activation_relu_f32(const float*, float*, int);
extern "C" __global__ void activation_relu_f64(const double*, double*, int);
extern "C" __global__ void activation_gelu_f32(const float*, float*, int);
extern "C" __global__ void activation_gelu_f64(const double*, double*, int);
extern "C" __global__ void bfs_basic(const int*, const int*, const unsigned int*, unsigned int*, unsigned int*, int);
extern "C" __global__ void dfs_basic(const int*, const int*, const unsigned int*, unsigned int*, unsigned int*, int);
extern "C" __global__ void pagerank_basic_f32(const int*, const int*, const int*, const float*, float*, float, int);
extern "C" __global__ void pagerank_basic_f64(const int*, const int*, const int*, const double*, double*, double, int);
extern "C" __global__ void stencil2d_3x3_f32(const float*, float*, int, int);
extern "C" __global__ void stencil2d_3x3_f64(const double*, double*, int, int);
extern "C" __global__ void stencil2d_5x5_f32(const float*, float*, int, int);
extern "C" __global__ void stencil2d_5x5_f64(const double*, double*, int, int);
extern "C" __global__ void stencil3d_global_f32(const float*, float*, int, int, int);
extern "C" __global__ void stencil3d_global_f64(const double*, double*, int, int, int);
extern "C" __global__ void stencil3d_shared_f32(const float*, float*, int, int, int);
extern "C" __global__ void stencil3d_shared_f64(const double*, double*, int, int, int);
extern "C" __global__ void histogram_global(const unsigned int*, unsigned int*, int);
extern "C" __global__ void histogram_shared(const unsigned int*, unsigned int*, int);
extern "C" __global__ void sort_bitonic(uint32_t*, int, int, int);
extern "C" __global__ void montecarlo_basic_f32(const float*, unsigned int*, int);
extern "C" __global__ void montecarlo_basic_f64(const double*, unsigned int*, int);
extern "C" __global__ void fft1d_global_f32(float2*, int, int);
extern "C" __global__ void fft1d_global_f64(double2*, int, int);
extern "C" __global__ void fft1d_staged_f32(float2*, int, int);
extern "C" __global__ void fft1d_staged_f64(double2*, int, int);
extern "C" __global__ void smithwaterman_basic_kernel(const uint8_t*, const uint8_t*, int*, int, int, int, int, int);
extern "C" __global__ void smithwaterman_wavefront_kernel(const uint8_t*, const uint8_t*, int*, int, int, int, int, int);
extern "C" __global__ void wfa_editdistance_kernel(const uint8_t*, const uint8_t*, int*, int, int);

struct RunInfo {
    size_t g0 = 0, g1 = 1, g2 = 1;
    size_t l0 = 0, l1 = 1, l2 = 1;
    double flops_est = 0.0;
    double bytes_moved = 0.0;
    double bw_GBps = 0.0;
};

static std::mt19937_64& global_rng() {
    static std::mt19937_64 rng(12345);
    return rng;
}

template <typename T>
void fill_uniform(std::vector<T>& v, T lo, T hi) {
    std::uniform_real_distribution<double> dist(static_cast<double>(lo), static_cast<double>(hi));
    auto& rng = global_rng();
    for (auto& x : v) x = static_cast<T>(dist(rng));
}

struct CSRGraph {
    std::vector<int> rowptr;
    std::vector<int> colind;
};

inline CSRGraph build_random_graph(int V, int avg_deg) {
    CSRGraph g;
    g.rowptr.resize(V + 1);
    g.colind.clear();
    auto& rng = global_rng();
    std::uniform_int_distribution<int> dst(0, V - 1);
    int nnz = 0;
    for (int v = 0; v < V; ++v) {
        g.rowptr[v] = nnz;
        for (int d = 0; d < avg_deg; ++d) {
            g.colind.push_back(dst(rng));
            ++nnz;
        }
    }
    g.rowptr[V] = nnz;
    return g;
}

template <typename T>
T cpu_dot(const std::vector<T>& a, const std::vector<T>& b) {
    T acc = T(0);
    for (size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
    return acc;
}

static void fill_random_bases(std::vector<uint8_t>& data) {
    auto& rng = global_rng();
    std::uniform_int_distribution<int> dist(0, 3);
    for (auto& v : data)
        v = static_cast<uint8_t>(dist(rng));
}

static int cpu_smithwaterman_pair(const uint8_t* a,
                                  const uint8_t* b,
                                  int len,
                                  int match_score,
                                  int mismatch_score,
                                  int gap_score) {
    std::vector<int> prev(len + 1, 0), curr(len + 1, 0);
    int best = 0;
    for (int i = 1; i <= len; ++i) {
        curr[0] = 0;
        for (int j = 1; j <= len; ++j) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? match_score : mismatch_score);
            int up = prev[j] + gap_score;
            int left = curr[j - 1] + gap_score;
            int val = std::max({0, diag, up, left});
            curr[j] = val;
            best = std::max(best, val);
        }
        std::swap(prev, curr);
    }
    return best;
}

static int cpu_edit_distance_pair(const uint8_t* a,
                                  const uint8_t* b,
                                  int len) {
    std::vector<int> prev(len + 1);
    std::vector<int> curr(len + 1);
    for (int j = 0; j <= len; ++j) prev[j] = j;
    for (int i = 1; i <= len; ++i) {
        curr[0] = i;
        for (int j = 1; j <= len; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[len];
}

struct SmithWatermanDataset {
    std::vector<uint8_t> seqA;
    std::vector<uint8_t> seqB;
    std::vector<int> reference;
    size_t verify_pairs = 0;
};

static SmithWatermanDataset build_smithwaterman_dataset(size_t num_pairs,
                                                        int len,
                                                        int match_score,
                                                        int mismatch_score,
                                                        int gap_score) {
    SmithWatermanDataset data;
    size_t total_elems = num_pairs * static_cast<size_t>(len);
    data.seqA.resize(total_elems);
    data.seqB.resize(total_elems);
    fill_random_bases(data.seqA);
    fill_random_bases(data.seqB);
    data.reference.resize(num_pairs);
    data.verify_pairs = std::min<size_t>(num_pairs, 2048);
    for (size_t p = 0; p < data.verify_pairs; ++p) {
        const uint8_t* a = data.seqA.data() + p * len;
        const uint8_t* b = data.seqB.data() + p * len;
        data.reference[p] = cpu_smithwaterman_pair(a, b, len, match_score, mismatch_score, gap_score);
    }
    for (size_t p = data.verify_pairs; p < num_pairs; ++p)
        data.reference[p] = std::numeric_limits<int>::min();
    return data;
}

static std::string size_label_for(const std::string& kernel_name, size_t sz) {
    if (kernel_name == "smithwaterman_basic" || kernel_name == "smithwaterman_wavefront") {
        const auto& sizes = smithwaterman_problem_sizes();
        if (sz < sizes.size()) return sizes[sz].label;
    } else if (kernel_name == "wfa_editdistance") {
        const auto& sizes = wfa_problem_sizes();
        if (sz < sizes.size()) return sizes[sz].label;
    }
    return std::to_string(sz);
}

template <typename T>
double run_vecadd(size_t N, bool& correct, RunInfo& info) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    size_t bytes = sizeof(T) * N;

    std::vector<T> A(N), B(N), C(N), Ref(N);
    fill_uniform(A, T(-1), T(1));
    fill_uniform(B, T(-1), T(1));
    for (size_t i = 0; i < N; ++i) Ref[i] = A[i] + B[i];

    T *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T, float>)
        vecadd_f32<<<grid, block>>>(dA, dB, dC, static_cast<int>(N));
    else
        vecadd_f64<<<grid, block>>>(dA, dB, dC, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    correct = compare_results(C, Ref, std::is_same_v<T,float> ? 1e-5 : 1e-9);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(N);
    info.bytes_moved = 3.0 * bytes;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return ms;
}

template <typename T>
double run_dot(const std::string& variant, size_t N, bool& correct, RunInfo& info) {
    dim3 block(256);
    dim3 grid(std::min<size_t>((N + block.x - 1) / block.x, 4096));
    size_t bytes = sizeof(T) * N;

    std::vector<T> A(N), B(N);
    fill_uniform(A, T(-1), T(1));
    fill_uniform(B, T(-1), T(1));

    T *dA, *dB, *dP;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dP, grid.x * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        dot_f32<<<grid, block>>>(dA, dB, dP, static_cast<int>(N));
    else
        dot_f64<<<grid, block>>>(dA, dB, dP, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<T> partial(grid.x);
    CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x * sizeof(T), cudaMemcpyDeviceToHost));
    double gpu = std::accumulate(partial.begin(), partial.end(), 0.0);
    double ref = cpu_dot(A, B);
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-9;
    correct = std::fabs(gpu - ref) <= tol * (1.0 + std::fabs(ref));

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = 2.0 * static_cast<double>(N);
    info.bytes_moved = 2.0 * bytes;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dP));
    return ms;
}

template <typename T>
double run_reduction(size_t N, bool& correct, RunInfo& info) {
    dim3 block(256);
    dim3 grid(std::min<size_t>((N + block.x - 1) / block.x, 4096));
    size_t bytes = sizeof(T) * N;

    std::vector<T> x(N);
    fill_uniform(x, T(0), T(1));
    T ref = std::accumulate(x.begin(), x.end(), T(0));

    T *dx, *dP;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dP, grid.x * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));

    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        reduction_f32<<<grid, block>>>(dx, dP, static_cast<int>(N));
    else
        reduction_f64<<<grid, block>>>(dx, dP, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<T> partial(grid.x);
    CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x * sizeof(T), cudaMemcpyDeviceToHost));
    double gpu = std::accumulate(partial.begin(), partial.end(), 0.0);
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-9;
    correct = std::fabs(gpu - ref) <= tol * (1.0 + std::fabs(ref));

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(N);
    info.bytes_moved = bytes;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dP));
    return ms;
}

template <typename T>
double run_gemv(const std::string& variant, size_t M, bool& correct, RunInfo& info) {
    int K = static_cast<int>(M);
    size_t mat_elems = M * K;
    size_t vec_elems = K;
    std::vector<T> A(mat_elems), x(vec_elems), y(M), ref(M);
    fill_uniform(A, T(-1), T(1));
    fill_uniform(x, T(-1), T(1));
    for (size_t row = 0; row < M; ++row) {
        T acc = T(0);
        for (size_t col = 0; col < K; ++col)
            acc += A[row * K + col] * x[col];
        ref[row] = acc;
    }

    T *dA, *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(T) * mat_elems));
    CUDA_CHECK(cudaMalloc(&dx, sizeof(T) * vec_elems));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(T) * M));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), sizeof(T) * mat_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), sizeof(T) * vec_elems, cudaMemcpyHostToDevice));

    double ms = 0.0;
    if (variant == "gemv_global") {
        dim3 block(256);
        dim3 grid((M + block.x - 1) / block.x);
        CudaTimer timer; timer.begin();
        if constexpr (std::is_same_v<T,float>)
            gemv_global_f32<<<grid, block>>>(dA, dx, dy, static_cast<int>(M), static_cast<int>(K));
        else
            gemv_global_f64<<<grid, block>>>(dA, dx, dy, static_cast<int>(M), static_cast<int>(K));
        CUDA_CHECK(cudaDeviceSynchronize());
        ms = timer.end();
        info.g0 = grid.x * block.x; info.l0 = block.x;
    } else {
        dim3 block(256);
        dim3 grid(M);
        CudaTimer timer; timer.begin();
        if constexpr (std::is_same_v<T,float>)
            gemv_shared_f32<<<grid, block>>>(dA, dx, dy, static_cast<int>(M), static_cast<int>(K));
        else
            gemv_shared_f64<<<grid, block>>>(dA, dx, dy, static_cast<int>(M), static_cast<int>(K));
        CUDA_CHECK(cudaDeviceSynchronize());
        ms = timer.end();
        info.g0 = grid.x * block.x; info.l0 = block.x;
    }

    CUDA_CHECK(cudaMemcpy(y.data(), dy, sizeof(T) * M, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-8;
    correct = compare_results(y, ref, tol);

    info.flops_est = 2.0 * static_cast<double>(M) * static_cast<double>(K);
    info.bytes_moved = sizeof(T) * (mat_elems + vec_elems + M);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    return ms;
}

template <typename T>
double run_matmul(const std::string& variant, size_t Ndim, bool& correct, RunInfo& info) {
    int M = static_cast<int>(Ndim);
    int N = static_cast<int>(Ndim);
    int K = static_cast<int>(Ndim);
    size_t elemsA = static_cast<size_t>(M) * K;
    size_t elemsB = static_cast<size_t>(K) * N;
    size_t elemsC = static_cast<size_t>(M) * N;
    std::vector<T> A(elemsA), B(elemsB), C(elemsC, T(0)), Ref(elemsC, T(0));
    fill_uniform(A, T(-1), T(1));
    fill_uniform(B, T(-1), T(1));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            T acc = T(0);
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            Ref[i * N + j] = acc;
        }

    T *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(T) * elemsA));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(T) * elemsB));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(T) * elemsC));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), sizeof(T) * elemsA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), sizeof(T) * elemsB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    CudaTimer timer; timer.begin();
    if (variant == "matmul_global") {
        if constexpr (std::is_same_v<T,float>)
            matmul_global_f32<<<grid, block>>>(dA, dB, dC, M, N, K);
        else
            matmul_global_f64<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else {
        if constexpr (std::is_same_v<T,float>)
            matmul_shared_f32<<<grid, block>>>(dA, dB, dC, M, N, K);
        else
            matmul_shared_f64<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeof(T) * elemsC, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-2 : 1e-8;
    correct = compare_results(C, Ref, tol);

    info.g0 = grid.x * block.x; info.g1 = grid.y * block.y; info.l0 = block.x; info.l1 = block.y;
    info.flops_est = 2.0 * static_cast<double>(M) * N * K;
    info.bytes_moved = sizeof(T) * (elemsA + elemsB + elemsC);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return ms;
}

template <typename T>
double run_scan(size_t N, bool& correct, RunInfo& info) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    size_t bytes = sizeof(T) * N;

    std::vector<T> x(N), out(N), ref(N);
    fill_uniform(x, T(0), T(1));
    T acc = T(0);
    for (size_t i = 0; i < N; ++i) {
        acc += x[i];
        ref[i] = acc;
    }

    T *dx, *dy, *dBlock;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dBlock, grid.x * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));

    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        launch_scan_f32(dx, dy, dBlock, static_cast<int>(N), grid, block, 0);
    else
        launch_scan_f64(dx, dy, dBlock, static_cast<int>(N), grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> block_sums(grid.x, T(0));
    if (grid.x > 0)
        CUDA_CHECK(cudaMemcpy(block_sums.data(), dBlock, grid.x * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<T> offsets(grid.x, T(0));
    for (size_t i = 1; i < grid.x; ++i)
        offsets[i] = offsets[i - 1] + block_sums[i - 1];
    if (grid.x > 0)
        CUDA_CHECK(cudaMemcpy(dBlock, offsets.data(), grid.x * sizeof(T), cudaMemcpyHostToDevice));
    if constexpr (std::is_same_v<T,float>)
        launch_scan_add_f32(dy, dBlock, static_cast<int>(N), grid, block, 0);
    else
        launch_scan_add_f64(dy, dBlock, static_cast<int>(N), grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), dy, bytes, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-4 : 1e-9;
    correct = compare_results(out, ref, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(N);
    info.bytes_moved = 2.0 * bytes;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dBlock));
    return ms;
}

template <typename T>
double run_spmv(size_t N, bool& correct, RunInfo& info) {
    int rows = static_cast<int>(N);
    int cols = static_cast<int>(N);
    int nnz_per_row = 5;
    int nnz = rows * nnz_per_row;

    std::vector<int> rowptr(rows + 1);
    std::vector<int> colind(nnz);
    std::vector<T> vals(nnz);
    std::vector<T> x(cols), y(rows, T(0)), ref(rows, T(0));
    fill_uniform(x, T(-1), T(1));

    auto& rng = global_rng();
    std::uniform_int_distribution<int> dcol(0, cols - 1);
    std::uniform_real_distribution<double> dval(-1.0, 1.0);
    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        rowptr[r] = idx;
        for (int k = 0; k < nnz_per_row; ++k) {
            int c = dcol(rng);
            colind[idx] = c;
            vals[idx] = static_cast<T>(dval(rng));
            ref[r] += vals[idx] * x[c];
            ++idx;
        }
    }
    rowptr[rows] = idx;

    T *d_vals, *d_x, *d_y;
    int *d_rowptr, *d_colind;
    CUDA_CHECK(cudaMalloc(&d_vals, sizeof(T) * nnz));
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(T) * cols));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(T) * rows));
    CUDA_CHECK(cudaMalloc(&d_rowptr, sizeof(int) * (rows + 1)));
    CUDA_CHECK(cudaMalloc(&d_colind, sizeof(int) * nnz));
    CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), sizeof(T) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(T) * cols, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowptr, rowptr.data(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colind, colind.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        spmv_csr_f32<<<grid, block>>>(d_rowptr, d_colind, d_vals, d_x, d_y, rows);
    else
        spmv_csr_f64<<<grid, block>>>(d_rowptr, d_colind, d_vals, d_x, d_y, rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, sizeof(T) * rows, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-8;
    correct = compare_results(y, ref, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = 2.0 * static_cast<double>(nnz);
    info.bytes_moved = sizeof(int) * (rowptr.size() + colind.size()) + sizeof(T) * (nnz + cols + rows);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_vals)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_rowptr)); CUDA_CHECK(cudaFree(d_colind));
    return ms;
}

double run_bfs_or_dfs(const std::string& name, size_t Vsz, bool& correct, RunInfo& info) {
    int V = static_cast<int>(Vsz);
    CSRGraph g = build_random_graph(V, 4);
    std::vector<unsigned int> frontier(V, 0), next(V, 0), visited(V, 0);
    frontier[0] = 1; visited[0] = 1;

    int *d_rowptr, *d_colind;
    unsigned int *d_frontier, *d_next, *d_visited;
    CUDA_CHECK(cudaMalloc(&d_rowptr, sizeof(int) * g.rowptr.size()));
    CUDA_CHECK(cudaMalloc(&d_colind, sizeof(int) * g.colind.size()));
    CUDA_CHECK(cudaMalloc(&d_frontier, sizeof(unsigned int) * V));
    CUDA_CHECK(cudaMalloc(&d_next, sizeof(unsigned int) * V));
    CUDA_CHECK(cudaMalloc(&d_visited, sizeof(unsigned int) * V));
    CUDA_CHECK(cudaMemcpy(d_rowptr, g.rowptr.data(), sizeof(int) * g.rowptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colind, g.colind.data(), sizeof(int) * g.colind.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, frontier.data(), sizeof(unsigned int) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, next.data(), sizeof(unsigned int) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, visited.data(), sizeof(unsigned int) * V, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((V + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    if (name == "bfs_basic")
        bfs_basic<<<grid, block>>>(d_rowptr, d_colind, d_frontier, d_next, d_visited, V);
    else
        dfs_basic<<<grid, block>>>(d_rowptr, d_colind, d_frontier, d_next, d_visited, V);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(next.data(), d_next, sizeof(unsigned int) * V, cudaMemcpyDeviceToHost));
    size_t count = 0;
    for (auto v : next) if (v) ++count;
    correct = (count > 0);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(int) * (g.rowptr.size() + g.colind.size()) + sizeof(unsigned int) * (3ull * V);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_rowptr)); CUDA_CHECK(cudaFree(d_colind));
    CUDA_CHECK(cudaFree(d_frontier)); CUDA_CHECK(cudaFree(d_next)); CUDA_CHECK(cudaFree(d_visited));
    return ms;
}

template <typename T>
double run_pagerank(size_t Vsz, bool use_fp64, bool& correct, RunInfo& info) {
    int V = static_cast<int>(Vsz);
    CSRGraph g = build_random_graph(V, 4);
    std::vector<int> outdeg(V);
    for (int v = 0; v < V; ++v) outdeg[v] = g.rowptr[v + 1] - g.rowptr[v];
    std::vector<T> pr(V, T(1) / T(V));
    std::vector<T> pr_next(V, T((1.0 - 0.85) / V));
    std::vector<T> pr_cpu(V, T((1.0 - 0.85) / V));
    for (int v = 0; v < V; ++v) {
        int start = g.rowptr[v];
        int end = g.rowptr[v + 1];
        T contrib = T(0.85) * pr[v] / T(max(outdeg[v], 1));
        for (int idx = start; idx < end; ++idx)
            pr_cpu[g.colind[idx]] += contrib;
    }

    int *d_rowptr, *d_colind, *d_outdeg;
    T *d_pr, *d_pr_next;
    CUDA_CHECK(cudaMalloc(&d_rowptr, sizeof(int) * g.rowptr.size()));
    CUDA_CHECK(cudaMalloc(&d_colind, sizeof(int) * g.colind.size()));
    CUDA_CHECK(cudaMalloc(&d_outdeg, sizeof(int) * V));
    CUDA_CHECK(cudaMalloc(&d_pr, sizeof(T) * V));
    CUDA_CHECK(cudaMalloc(&d_pr_next, sizeof(T) * V));
    CUDA_CHECK(cudaMemcpy(d_rowptr, g.rowptr.data(), sizeof(int) * g.rowptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colind, g.colind.data(), sizeof(int) * g.colind.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outdeg, outdeg.data(), sizeof(int) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr, pr.data(), sizeof(T) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr_next, pr_next.data(), sizeof(T) * V, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((V + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        pagerank_basic_f32<<<grid, block>>>(d_rowptr, d_colind, d_outdeg, d_pr, d_pr_next, 0.85f, V);
    else
        pagerank_basic_f64<<<grid, block>>>(d_rowptr, d_colind, d_outdeg, d_pr, d_pr_next, 0.85, V);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(pr_next.data(), d_pr_next, sizeof(T) * V, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 3e-3 : 1e-8;
    correct = compare_results(pr_next, pr_cpu, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(int) * (g.rowptr.size() + g.colind.size() + outdeg.size()) + sizeof(T) * (2ull * V);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_rowptr)); CUDA_CHECK(cudaFree(d_colind)); CUDA_CHECK(cudaFree(d_outdeg));
    CUDA_CHECK(cudaFree(d_pr)); CUDA_CHECK(cudaFree(d_pr_next));
    return ms;
}

template <typename T>
double run_stencil2d(const std::string& name, size_t S, bool use_fp64, bool& correct, RunInfo& info) {
    int H = static_cast<int>(S);
    int W = static_cast<int>(S);
    std::vector<T> in(H * W), out(H * W), ref(H * W);
    fill_uniform(in, T(-1), T(1));
    int radius = (name == "stencil2d_5x5") ? 2 : 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            T acc = T(0); int cnt = 0;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                        acc += in[yy * W + xx]; cnt++;
                    }
                }
            ref[y * W + x] = acc / static_cast<T>(cnt);
        }
    T *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, sizeof(T) * in.size()));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(T) * out.size()));
    CUDA_CHECK(cudaMemcpy(din, in.data(), sizeof(T) * in.size(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    CudaTimer timer; timer.begin();
    if (name == "stencil2d_5x5") {
        if constexpr (std::is_same_v<T,float>) stencil2d_5x5_f32<<<grid, block>>>(din, dout, H, W);
        else stencil2d_5x5_f64<<<grid, block>>>(din, dout, H, W);
    } else {
        if constexpr (std::is_same_v<T,float>) stencil2d_3x3_f32<<<grid, block>>>(din, dout, H, W);
        else stencil2d_3x3_f64<<<grid, block>>>(din, dout, H, W);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), dout, sizeof(T) * out.size(), cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-7;
    correct = compare_results(out, ref, tol);

    info.g0 = grid.x * block.x; info.g1 = grid.y * block.y; info.l0 = block.x; info.l1 = block.y;
    info.bytes_moved = sizeof(T) * (in.size() + out.size());
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(din)); CUDA_CHECK(cudaFree(dout));
    return ms;
}

template <typename T>
double run_stencil3d(const std::string& name, size_t S, bool use_fp64, bool& correct, RunInfo& info) {
    int D = static_cast<int>(S);
    int H = static_cast<int>(S);
    int W = static_cast<int>(S);
    size_t total = static_cast<size_t>(D) * H * W;
    std::vector<T> in(total), out(total), ref(total);
    fill_uniform(in, T(-1), T(1));
    auto idx = [&](int z, int y, int x) { return (z * H + y) * W + x; };
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                T acc = T(0);
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int zz = z + dz, yy = y + dy, xx = x + dx;
                            if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W)
                                acc += in[idx(zz, yy, xx)];
                        }
                ref[idx(z, y, x)] = acc;
            }
    T *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, sizeof(T) * total));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(T) * total));
    CUDA_CHECK(cudaMemcpy(din, in.data(), sizeof(T) * total, cudaMemcpyHostToDevice));

    dim3 block(8, 8, 2);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (D + block.z - 1) / block.z);
    CudaTimer timer; timer.begin();
    if (name == "stencil3d_shared") {
        if constexpr (std::is_same_v<T,float>) stencil3d_shared_f32<<<grid, block>>>(din, dout, D, H, W);
        else stencil3d_shared_f64<<<grid, block>>>(din, dout, D, H, W);
    } else {
        if constexpr (std::is_same_v<T,float>) stencil3d_global_f32<<<grid, block>>>(din, dout, D, H, W);
        else stencil3d_global_f64<<<grid, block>>>(din, dout, D, H, W);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), dout, sizeof(T) * total, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-7;
    correct = compare_results(out, ref, tol);

    info.g0 = grid.x * block.x; info.g1 = grid.y * block.y; info.g2 = grid.z * block.z;
    info.l0 = block.x; info.l1 = block.y; info.l2 = block.z;
    info.bytes_moved = sizeof(T) * (2ull * total);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(din)); CUDA_CHECK(cudaFree(dout));
    return ms;
}

double run_histogram(const std::string& name, size_t N, bool& correct, RunInfo& info) {
    std::vector<unsigned int> data(N);
    auto& rng = global_rng();
    std::uniform_int_distribution<unsigned int> dist(0, 255);
    for (auto& v : data) v = dist(rng);
    std::vector<unsigned int> ref(256, 0), out(256, 0);
    for (auto v : data) ref[v & 255u]++;

    unsigned int *d_data, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(unsigned int) * N));
    CUDA_CHECK(cudaMalloc(&d_hist, sizeof(unsigned int) * 256));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(unsigned int) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(unsigned int) * 256));

    dim3 block(256);
    dim3 grid(std::min<size_t>((N + block.x - 1) / block.x, 4096));
    CudaTimer timer; timer.begin();
    if (name == "histogram_shared")
        histogram_shared<<<grid, block>>>(d_data, d_hist, static_cast<int>(N));
    else
        histogram_global<<<grid, block>>>(d_data, d_hist, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_hist, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost));
    correct = (out == ref);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(unsigned int) * (N + 256);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_data)); CUDA_CHECK(cudaFree(d_hist));
    return ms;
}

double run_sort_bitonic(size_t N, bool& correct, RunInfo& info) {
    size_t padN = 1;
    while (padN < N) padN <<= 1;
    std::vector<uint32_t> data(padN, std::numeric_limits<uint32_t>::max());
    auto& rng = global_rng();
    std::uniform_int_distribution<uint32_t> dist(0, 1u << 24);
    for (size_t i = 0; i < N; ++i) data[i] = dist(rng);
    std::vector<uint32_t> ref(data.begin(), data.begin() + N);
    std::sort(ref.begin(), ref.end());

    uint32_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(uint32_t) * padN));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(uint32_t) * padN, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((padN / 2 + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    for (int k = 2; k <= static_cast<int>(padN); k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            sort_bitonic<<<grid, block>>>(d_data, static_cast<int>(padN), j, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(data.data(), d_data, sizeof(uint32_t) * padN, cudaMemcpyDeviceToHost));
    data.resize(N);
    correct = std::equal(data.begin(), data.end(), ref.begin());

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(uint32_t) * padN;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_data));
    return ms;
}

template <typename T>
double run_montecarlo(size_t N, bool use_fp64, bool& correct, RunInfo& info) {
    std::vector<T> xy(2 * N);
    fill_uniform(xy, T(0), T(1));
    T *d_xy;
    unsigned int *d_partial;
    CUDA_CHECK(cudaMalloc(&d_xy, sizeof(T) * xy.size()));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    size_t partial_elems = grid.x * block.x;
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(unsigned int) * partial_elems));
    CUDA_CHECK(cudaMemcpy(d_xy, xy.data(), sizeof(T) * xy.size(), cudaMemcpyHostToDevice));

    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        montecarlo_basic_f32<<<grid, block>>>(d_xy, d_partial, static_cast<int>(N));
    else
        montecarlo_basic_f64<<<grid, block>>>(d_xy, d_partial, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<unsigned int> partial(partial_elems);
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, sizeof(unsigned int) * partial.size(), cudaMemcpyDeviceToHost));
    uint64_t inside = 0;
    for (auto v : partial) inside += v;
    double pi_est = 4.0 * static_cast<double>(inside) / static_cast<double>(N);
    correct = (pi_est > 2.8 && pi_est < 3.4);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(T) * xy.size() + sizeof(unsigned int) * partial.size();
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_xy)); CUDA_CHECK(cudaFree(d_partial));
    return ms;
}

double run_smithwaterman(size_t config_idx, bool& correct, RunInfo& info) {
    const auto& spec = smithwaterman_problem_sizes().at(config_idx);
    const size_t num_pairs = spec.num_pairs;
    const int seq_len = static_cast<int>(spec.sequence_length);
    const int match_score = 2;
    const int mismatch_score = -1;
    const int gap_score = -2;

    SmithWatermanDataset data = build_smithwaterman_dataset(num_pairs, seq_len, match_score, mismatch_score, gap_score);
    const size_t total_elems = data.seqA.size();
    uint8_t *dA, *dB;
    int *dOut;
    CUDA_CHECK(cudaMalloc(&dA, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dB, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dOut, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, data.seqA.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, data.seqB.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(num_pairs) + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    smithwaterman_basic_kernel<<<grid, block>>>(
        dA, dB, dOut,
        static_cast<int>(num_pairs),
        seq_len,
        match_score,
        mismatch_score,
        gap_score);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<int> gpu(num_pairs);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, num_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < data.verify_pairs; ++i) {
        if (gpu[i] != data.reference[i]) { ok = false; break; }
    }
    correct = ok;

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(num_pairs) * seq_len * seq_len * 6.0;
    info.bytes_moved = static_cast<double>(num_pairs) *
                       (2.0 * seq_len * sizeof(uint8_t) + sizeof(int));
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dOut));
    return ms;
}

double run_smithwaterman_wavefront(size_t config_idx, bool& correct, RunInfo& info) {
    const auto& spec = smithwaterman_problem_sizes().at(config_idx);
    const size_t num_pairs = spec.num_pairs;
    const int seq_len = static_cast<int>(spec.sequence_length);
    const int match_score = 2;
    const int mismatch_score = -1;
    const int gap_score = -2;

    SmithWatermanDataset data = build_smithwaterman_dataset(num_pairs, seq_len, match_score, mismatch_score, gap_score);
    const size_t total_elems = data.seqA.size();

    uint8_t *dA, *dB;
    int *dOut;
    CUDA_CHECK(cudaMalloc(&dA, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dB, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dOut, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, data.seqA.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, data.seqB.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));

    unsigned int block_size = 1;
    while (block_size < static_cast<unsigned int>(seq_len) && block_size < 1024u) block_size <<= 1;
    if (block_size > 1024u) block_size = 1024u;
    dim3 block(block_size);
    dim3 grid(static_cast<unsigned int>(num_pairs));
    size_t shared_bytes = sizeof(int) * (3 * (seq_len + 1) + block.x);

    CudaTimer timer; timer.begin();
    smithwaterman_wavefront_kernel<<<grid, block, shared_bytes>>>(
        dA, dB, dOut,
        static_cast<int>(num_pairs),
        seq_len,
        match_score,
        mismatch_score,
        gap_score);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<int> gpu(num_pairs);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, num_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < data.verify_pairs; ++i) {
        if (gpu[i] != data.reference[i]) { ok = false; break; }
    }
    correct = ok;

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(num_pairs) * seq_len * seq_len * 6.0;
    info.bytes_moved = static_cast<double>(num_pairs) *
                       (2.0 * seq_len * sizeof(uint8_t) + sizeof(int));
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dOut));
    return ms;
}

double run_wfa(size_t config_idx, bool& correct, RunInfo& info) {
    const auto& spec = wfa_problem_sizes().at(config_idx);
    const size_t num_pairs = spec.num_pairs;
    const int seq_len = static_cast<int>(spec.sequence_length);
    size_t total_elems = num_pairs * static_cast<size_t>(seq_len);
    std::vector<uint8_t> seqA(total_elems);
    std::vector<uint8_t> seqB(total_elems);
    fill_random_bases(seqA);
    fill_random_bases(seqB);

    std::vector<int> ref(num_pairs, std::numeric_limits<int>::min());
    size_t verify_pairs = std::min<size_t>(num_pairs, 2048);
    for (size_t p = 0; p < verify_pairs; ++p) {
        const uint8_t* a = seqA.data() + p * seq_len;
        const uint8_t* b = seqB.data() + p * seq_len;
        ref[p] = cpu_edit_distance_pair(a, b, seq_len);
    }

    uint8_t *dA, *dB;
    int *dOut;
    CUDA_CHECK(cudaMalloc(&dA, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dB, total_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dOut, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, seqA.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, seqB.data(), total_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(num_pairs) + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    wfa_editdistance_kernel<<<grid, block>>>(
        dA, dB, dOut,
        static_cast<int>(num_pairs),
        seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    std::vector<int> gpu(num_pairs);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, num_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < verify_pairs; ++i) {
        if (gpu[i] != ref[i]) { ok = false; break; }
    }
    correct = ok;

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(num_pairs) * seq_len * seq_len * 4.0;
    info.bytes_moved = static_cast<double>(num_pairs) *
                       (2.0 * seq_len * sizeof(uint8_t) + sizeof(int));
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dOut));
    return ms;
}

template <typename Complex>
std::vector<Complex> cpu_dft(const std::vector<Complex>& in) {
    size_t N = in.size();
    std::vector<Complex> out(N);
    for (size_t k = 0; k < N; ++k) {
        double re = 0, im = 0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            re += in[n].x * cos(angle) - in[n].y * sin(angle);
            im += in[n].x * sin(angle) + in[n].y * cos(angle);
        }
        out[k].x = re;
        out[k].y = im;
    }
    return out;
}

template <typename T, typename Complex, typename Kernel>
double run_fft_impl(const std::string& variant, size_t N, Kernel kernel,
                    bool& correct, RunInfo& info) {
    std::vector<Complex> data(N);
    for (size_t i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        data[i].x = cos(theta);
        data[i].y = sin(theta);
    }
    std::vector<Complex> ref = (N <= 4096) ? cpu_dft(data) : data;

    Complex* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(Complex) * N));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(Complex) * N, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N / 2 + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    for (size_t mh = 1; mh < N; mh <<= 1)
        kernel<<<grid, block>>>(d_data, static_cast<int>(N), static_cast<int>(mh));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(data.data(), d_data, sizeof(Complex) * N, cudaMemcpyDeviceToHost));
    bool ok = true;
    if (N <= 4096) {
        for (size_t i = 0; i < N; ++i) {
            double mag_gpu = std::hypot(data[i].x, data[i].y);
            double mag_ref = std::hypot(ref[i].x, ref[i].y);
            double tol = std::is_same_v<T,float> ? 1e-2 : 1e-6;
            if (fabs(mag_gpu - mag_ref) > tol) { ok = false; break; }
        }
    }
    correct = ok;

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.bytes_moved = sizeof(Complex) * (2ull * N);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_data));
    return ms;
}

double run_fft(const std::string& name, size_t N, bool use_fp64, bool& correct, RunInfo& info) {
    if (use_fp64) {
        if (name == "fft1d_global")
            return run_fft_impl<double, double2>(name, N, fft1d_global_f64, correct, info);
        return run_fft_impl<double, double2>(name, N, fft1d_staged_f64, correct, info);
    }
    if (name == "fft1d_global")
        return run_fft_impl<float, float2>(name, N, fft1d_global_f32, correct, info);
    return run_fft_impl<float, float2>(name, N, fft1d_staged_f32, correct, info);
}

template <typename T>
double run_conv2d(const std::string& variant, size_t S, bool& correct, RunInfo& info) {
    int H = static_cast<int>(S);
    int W = static_cast<int>(S);
    size_t in_elems = static_cast<size_t>(H) * W;
    size_t fil_elems = 9;
    size_t out_elems = static_cast<size_t>(H) * W;

    std::vector<T> img(in_elems), filt(fil_elems), out(out_elems, T(0)), ref(out_elems, T(0));
    fill_uniform(img, T(-1), T(1));
    fill_uniform(filt, T(-1), T(1));

    auto idx_in = [&](int y, int x) { return y * W + x; };
    auto idx_out = [&](int y, int x) { return y * W + x; };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            T acc = T(0);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int yy = y + dy;
                    int xx = x + dx;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                        acc += img[idx_in(yy, xx)] * filt[(dy + 1) * 3 + (dx + 1)];
                }
            ref[idx_out(y, x)] = acc;
        }

    T *d_img, *d_filt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_img, sizeof(T) * in_elems));
    CUDA_CHECK(cudaMalloc(&d_filt, sizeof(T) * fil_elems));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * out_elems));
    CUDA_CHECK(cudaMemcpy(d_img, img.data(), sizeof(T) * in_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filt, filt.data(), sizeof(T) * fil_elems, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    CudaTimer timer; timer.begin();
    if (variant == "conv2d_shared") {
        if constexpr (std::is_same_v<T,float>)
            conv2d_shared_f32<<<grid, block>>>(d_img, d_filt, d_out, H, W);
        else
            conv2d_shared_f64<<<grid, block>>>(d_img, d_filt, d_out, H, W);
    } else {
        if constexpr (std::is_same_v<T,float>)
            conv2d_global_f32<<<grid, block>>>(d_img, d_filt, d_out, H, W);
        else
            conv2d_global_f64<<<grid, block>>>(d_img, d_filt, d_out, H, W);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(T) * out_elems, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-2 : 1e-8;
    correct = compare_results(out, ref, tol);

    info.g0 = grid.x * block.x; info.g1 = grid.y * block.y;
    info.l0 = block.x; info.l1 = block.y;
    info.flops_est = static_cast<double>(H) * W * 9 * 2.0;
    info.bytes_moved = sizeof(T) * (in_elems + fil_elems + out_elems);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_filt)); CUDA_CHECK(cudaFree(d_out));
    return ms;
}

template <typename T>
double run_depthwise(const std::string& variant, size_t S, bool& correct, RunInfo& info) {
    int H = static_cast<int>(S);
    int W = static_cast<int>(S);
    int C = 32;
    size_t in_elems = static_cast<size_t>(H) * W * C;
    size_t fil_elems = 9 * C;
    size_t out_elems = in_elems;

    std::vector<T> img(in_elems), filt(fil_elems), out(out_elems, T(0)), ref(out_elems, T(0));
    fill_uniform(img, T(-1), T(1));
    fill_uniform(filt, T(-1), T(1));

    auto idx = [&](int y, int x, int c) { return (c * H + y) * W + x; };
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                T acc = T(0);
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                            acc += img[idx(yy, xx, c)] * filt[c * 9 + (dy + 1) * 3 + (dx + 1)];
                    }
                ref[idx(y, x, c)] = acc;
            }

    T *d_img, *d_filt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_img, sizeof(T) * in_elems));
    CUDA_CHECK(cudaMalloc(&d_filt, sizeof(T) * fil_elems));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * out_elems));
    CUDA_CHECK(cudaMemcpy(d_img, img.data(), sizeof(T) * in_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filt, filt.data(), sizeof(T) * fil_elems, cudaMemcpyHostToDevice));

    dim3 block, grid;
    if (variant == "depthwiseconv_tiled") {
        block = dim3(8, 8, 1);
        grid = dim3((W + block.x - 1) / block.x,
                    (H + block.y - 1) / block.y,
                    C);
    } else {
        block = dim3(16, 16, 1);
        grid = dim3((W + block.x - 1) / block.x,
                    (H + block.y - 1) / block.y,
                    C);
    }

    CudaTimer timer; timer.begin();
    if (variant == "depthwiseconv_tiled") {
        if constexpr (std::is_same_v<T,float>)
            depthwiseconv_tiled_f32<<<grid, block>>>(d_img, d_filt, d_out, C, H, W);
        else
            depthwiseconv_tiled_f64<<<grid, block>>>(d_img, d_filt, d_out, C, H, W);
    } else {
        if constexpr (std::is_same_v<T,float>)
            depthwiseconv_global_f32<<<grid, block>>>(d_img, d_filt, d_out, C, H, W);
        else
            depthwiseconv_global_f64<<<grid, block>>>(d_img, d_filt, d_out, C, H, W);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(T) * out_elems, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 3e-3 : 1e-8;
    correct = compare_results(out, ref, tol);

    info.g0 = grid.x * block.x; info.g1 = grid.y * block.y; info.g2 = grid.z * block.z;
    info.l0 = block.x; info.l1 = block.y; info.l2 = block.z;
    info.flops_est = static_cast<double>(H) * W * C * 9 * 2.0;
    info.bytes_moved = sizeof(T) * (in_elems + fil_elems + out_elems);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_filt)); CUDA_CHECK(cudaFree(d_out));
    return ms;
}

template <typename T>
double run_softmax(size_t N, bool& correct, RunInfo& info) {
    std::vector<T> x(N), y(N), ref(N);
    fill_uniform(x, T(-4), T(4));
    T m = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double e = std::exp(static_cast<double>(x[i] - m));
        ref[i] = static_cast<T>(e);
        sum += e;
    }
    for (size_t i = 0; i < N; ++i)
        ref[i] = static_cast<T>(static_cast<double>(ref[i]) / sum);

    T *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, sizeof(T) * N));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(T) * N));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), sizeof(T) * N, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(1);
    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        softmax_basic_f32<<<grid, block>>>(dx, dy, static_cast<int>(N));
    else
        softmax_basic_f64<<<grid, block>>>(dx, dy, static_cast<int>(N));
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(y.data(), dy, sizeof(T) * N, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-9;
    correct = compare_results(y, ref, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = 4.0 * static_cast<double>(N);
    info.bytes_moved = sizeof(T) * (2 * N);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    return ms;
}

template <typename T>
double run_layernorm(size_t N, bool& correct, RunInfo& info) {
    std::vector<T> x(N), y(N), ref(N);
    fill_uniform(x, T(-3), T(3));
    double mean = 0.0;
    for (auto v : x) mean += static_cast<double>(v);
    mean /= static_cast<double>(N);
    double var = 0.0;
    for (auto v : x) {
        double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= static_cast<double>(N);
    double eps = std::is_same_v<T,float> ? 1e-5 : 1e-9;
    double inv = 1.0 / std::sqrt(var + eps);
    for (size_t i = 0; i < N; ++i)
        ref[i] = static_cast<T>((static_cast<double>(x[i]) - mean) * inv);

    T *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, sizeof(T) * N));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(T) * N));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), sizeof(T) * N, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(1);
    CudaTimer timer; timer.begin();
    if constexpr (std::is_same_v<T,float>)
        layernorm_basic_f32<<<grid, block>>>(dx, dy, static_cast<int>(N), static_cast<float>(eps));
    else
        layernorm_basic_f64<<<grid, block>>>(dx, dy, static_cast<int>(N), eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(y.data(), dy, sizeof(T) * N, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 5e-3 : 1e-7;
    correct = compare_results(y, ref, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = 6.0 * static_cast<double>(N);
    info.bytes_moved = sizeof(T) * (2 * N);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    return ms;
}

template <typename T>
double run_activation(const std::string& name, size_t N, bool& correct, RunInfo& info) {
    std::vector<T> x(N), y(N), ref(N);
    fill_uniform(x, T(-4), T(4));
    if (name == "activation_relu") {
        for (size_t i = 0; i < N; ++i)
            ref[i] = x[i] > T(0) ? x[i] : T(0);
    } else {
        constexpr double INV_SQRT2 = 0.70710678118654752440;
        for (size_t i = 0; i < N; ++i) {
            double v = static_cast<double>(x[i]);
            ref[i] = static_cast<T>(0.5 * v * (1.0 + std::erf(v * INV_SQRT2)));
        }
    }

    T *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, sizeof(T) * N));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(T) * N));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), sizeof(T) * N, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    CudaTimer timer; timer.begin();
    if (name == "activation_relu") {
        if constexpr (std::is_same_v<T,float>)
            activation_relu_f32<<<grid, block>>>(dx, dy, static_cast<int>(N));
        else
            activation_relu_f64<<<grid, block>>>(dx, dy, static_cast<int>(N));
    } else {
        if constexpr (std::is_same_v<T,float>)
            activation_gelu_f32<<<grid, block>>>(dx, dy, static_cast<int>(N));
        else
            activation_gelu_f64<<<grid, block>>>(dx, dy, static_cast<int>(N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = timer.end();

    CUDA_CHECK(cudaMemcpy(y.data(), dy, sizeof(T) * N, cudaMemcpyDeviceToHost));
    double tol = std::is_same_v<T,float> ? 1e-3 : 1e-7;
    correct = compare_results(y, ref, tol);

    info.g0 = grid.x * block.x; info.l0 = block.x;
    info.flops_est = static_cast<double>(N);
    info.bytes_moved = sizeof(T) * (2 * N);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    return ms;
}

struct ResultRow {
    std::string kernel;
    std::string dtype;
    std::string size_label;
    double ms;
    bool correct;
    RunInfo info;
};

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    std::string device_name = cuda_device_name();
    const std::string csv_path = "../csv/results_cuda.csv";
    std::remove(csv_path.c_str());

    auto run_float = [&](const std::string& name, size_t sz, bool want64, ResultRow& row) {
        bool correct = false;
        RunInfo info;
        double ms = 0.0;

        if (name == "vecadd_basic")
            ms = want64 ? run_vecadd<double>(sz, correct, info)
                        : run_vecadd<float>(sz, correct, info);
        else if (name == "dot_global" || name == "dot_shared")
            ms = want64 ? run_dot<double>(name, sz, correct, info)
                        : run_dot<float>(name, sz, correct, info);
        else if (name == "reduction_global" || name == "reduction_shared")
            ms = want64 ? run_reduction<double>(sz, correct, info)
                        : run_reduction<float>(sz, correct, info);
        else if (name == "gemv_global" || name == "gemv_shared")
            ms = want64 ? run_gemv<double>(name, sz, correct, info)
                        : run_gemv<float>(name, sz, correct, info);
        else if (name == "matmul_global" || name == "matmul_shared")
            ms = want64 ? run_matmul<double>(name, sz, correct, info)
                        : run_matmul<float>(name, sz, correct, info);
        else if (name == "scan_shared")
            ms = want64 ? run_scan<double>(sz, correct, info)
                        : run_scan<float>(sz, correct, info);
        else if (name == "spmv_csr")
            ms = want64 ? run_spmv<double>(sz, correct, info)
                        : run_spmv<float>(sz, correct, info);
        else if (name == "conv2d_global" || name == "conv2d_shared")
            ms = want64 ? run_conv2d<double>(name, sz, correct, info)
                        : run_conv2d<float>(name, sz, correct, info);
        else if (name == "depthwiseconv_global" || name == "depthwiseconv_tiled")
            ms = want64 ? run_depthwise<double>(name, sz, correct, info)
                        : run_depthwise<float>(name, sz, correct, info);
        else if (name == "softmax_basic")
            ms = want64 ? run_softmax<double>(sz, correct, info)
                        : run_softmax<float>(sz, correct, info);
        else if (name == "layernorm_basic")
            ms = want64 ? run_layernorm<double>(sz, correct, info)
                        : run_layernorm<float>(sz, correct, info);
        else if (name == "activation_relu" || name == "activation_gelu")
            ms = want64 ? run_activation<double>(name, sz, correct, info)
                        : run_activation<float>(name, sz, correct, info);
        else if (name == "pagerank_basic")
            ms = want64 ? run_pagerank<double>(sz, true, correct, info)
                        : run_pagerank<float>(sz, false, correct, info);
        else if (name == "stencil2d_3x3" || name == "stencil2d_5x5")
            ms = want64 ? run_stencil2d<double>(name, sz, true, correct, info)
                        : run_stencil2d<float>(name, sz, false, correct, info);
        else if (name == "stencil3d_global" || name == "stencil3d_shared")
            ms = want64 ? run_stencil3d<double>(name, sz, true, correct, info)
                        : run_stencil3d<float>(name, sz, false, correct, info);
        else if (name == "montecarlo_basic")
            ms = want64 ? run_montecarlo<double>(sz, true, correct, info)
                        : run_montecarlo<float>(sz, false, correct, info);
        else if (name == "fft1d_global" || name == "fft1d_staged")
            ms = run_fft(name, sz, want64, correct, info);
        else {
            std::cout << "[CUDA] Warning: kernel " << name << " not implemented.\n";
            correct = false; info = {}; ms = 0.0;
        }

        info.bw_GBps = (info.bytes_moved > 0 && ms > 0)
                        ? info.bytes_moved / (ms * 1e6)
                        : 0.0;

        row.ms = ms;
        row.correct = correct;
        row.info = info;
    };

    auto run_integer = [&](const std::string& name, size_t sz, ResultRow& row) {
        bool correct = false;
        RunInfo info;
        double ms = 0.0;

        if (name == "bfs_basic" || name == "dfs_basic")
            ms = run_bfs_or_dfs(name, sz, correct, info);
        else if (name == "histogram_global" || name == "histogram_shared")
            ms = run_histogram(name, sz, correct, info);
        else if (name == "sort_bitonic")
            ms = run_sort_bitonic(sz, correct, info);
        else if (name == "smithwaterman_basic")
            ms = run_smithwaterman(sz, correct, info);
        else if (name == "smithwaterman_wavefront")
            ms = run_smithwaterman_wavefront(sz, correct, info);
        else if (name == "wfa_editdistance")
            ms = run_wfa(sz, correct, info);
        else {
            std::cout << "[CUDA] Warning: integer kernel " << name << " not implemented.\n";
            correct = false; info = {}; ms = 0.0;
        }

        info.bw_GBps = (info.bytes_moved > 0 && ms > 0)
                        ? info.bytes_moved / (ms * 1e6)
                        : 0.0;

        row.ms = ms;
        row.correct = correct;
        row.info = info;
    };

    for (const auto& bench : BENCHMARKS) {
        if (!bench.enabled) continue;
        for (size_t sz : bench.sizes) {
            if (bench.dtype_mode == DTypeMode::FLOATING) {
                ResultRow row32;
                row32.kernel = bench.name;
                row32.dtype = "FP32";
                row32.size_label = size_label_for(bench.name, sz);
                std::cout << "[CUDA] Running kernel=" << bench.name << " dtype=FP32 size=" << sz << std::endl;
                run_float(bench.name, sz, false, row32);
                write_csv_row_cuda(csv_path, row32.kernel, row32.dtype, row32.size_label,
                                   row32.ms, row32.correct, device_name,
                                   row32.info.g0, row32.info.g1, row32.info.g2,
                                   row32.info.l0, row32.info.l1, row32.info.l2,
                                   row32.info.flops_est, row32.info.bw_GBps);

                ResultRow row64;
                row64.kernel = bench.name;
                row64.dtype = "FP64";
                row64.size_label = size_label_for(bench.name, sz);
                std::cout << "[CUDA] Running kernel=" << bench.name << " dtype=FP64 size=" << sz << std::endl;
                run_float(bench.name, sz, true, row64);
                write_csv_row_cuda(csv_path, row64.kernel, row64.dtype, row64.size_label,
                                   row64.ms, row64.correct, device_name,
                                   row64.info.g0, row64.info.g1, row64.info.g2,
                                   row64.info.l0, row64.info.l1, row64.info.l2,
                                   row64.info.flops_est, row64.info.bw_GBps);
            } else {
                ResultRow rowi;
                rowi.kernel = bench.name;
                rowi.dtype = "INT32";
                rowi.size_label = size_label_for(bench.name, sz);
                std::cout << "[CUDA] Running kernel=" << bench.name << " dtype=INT32 size=" << sz << std::endl;
                run_integer(bench.name, sz, rowi);
                write_csv_row_cuda(csv_path, rowi.kernel, rowi.dtype, rowi.size_label,
                                   rowi.ms, rowi.correct, device_name,
                                   rowi.info.g0, rowi.info.g1, rowi.info.g2,
                                   rowi.info.l0, rowi.info.l1, rowi.info.l2,
                                   rowi.info.flops_est, rowi.info.bw_GBps);
            }
        }
    }

    std::cout << "[CUDA] Completed benchmarks. Results in " << csv_path << std::endl;
    return 0;
}
