#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <string>
#include <cmath>

#include "utils_cuda.hpp"
#include "../common/csv_writer.hpp"      // optional (we use local writer in utils_cuda.hpp)
#include "../common/baseline_check.hpp"  // for CPU references if needed

// Kernel prototypes (must match kernels_all.cu)
extern "C" __global__ void vecadd_f32(const float*, const float*, float*, int);
extern "C" __global__ void vecadd_f64(const double*, const double*, double*, int);
extern "C" __global__ void dot_f32(const float*, const float*, float*, int);
extern "C" __global__ void dot_f64(const double*, const double*, double*, int);
extern "C" __global__ void reduction_f32(const float*, float*, int);
extern "C" __global__ void reduction_f64(const double*, double*, int);

static const int REPEAT = 3;

template<typename T>
void fill_random(std::vector<T>& v, T lo, T hi) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> U((double)lo, (double)hi);
    for (auto& x : v) x = (T)U(rng);
}

int main() {
    std::string device_name = cuda_device_name();
    const std::string csv_path = "results_cuda.csv";

    // ---------------- VecAdd FP32/FP64 ----------------
    {
        int N = 1<<20; // 1M
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        // FP32
        {
            size_t bytes = N * sizeof(float);
            std::vector<float> A(N), B(N), C(N), Ref(N);
            fill_random(A, -1.f, 1.f); fill_random(B, -1.f, 1.f);
            for (int i = 0; i < N; ++i) Ref[i] = A[i] + B[i];

            float *dA, *dB, *dC;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dB, bytes));
            CUDA_CHECK(cudaMalloc(&dC, bytes));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                vecadd_f32<<<grid, block>>>(dA, dB, dC, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;
            CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
            bool correct = compare_results(C, Ref, 1e-5);

            // bandwidth: 3*N*sizeof(T)
            double bytes_moved = 3.0 * bytes;
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda(csv_path, "vecadd_basic", "FP32", N, ms, correct,
                               device_name, grid.x*block.x, 0, 0, block.x, 0, 0, (double)N, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
            std::cout << "[CUDA] vecadd_basic FP32 N=" << N << " ms=" << ms
                      << " correct=" << (correct?"true":"false") << "\n";
        }

        // FP64
        {
            size_t bytes = N * sizeof(double);
            std::vector<double> A(N), B(N), C(N), Ref(N);
            fill_random(A, -1.0, 1.0); fill_random(B, -1.0, 1.0);
            for (int i = 0; i < N; ++i) Ref[i] = A[i] + B[i];

            double *dA, *dB, *dC;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dB, bytes));
            CUDA_CHECK(cudaMalloc(&dC, bytes));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                vecadd_f64<<<grid, block>>>(dA, dB, dC, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;
            CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
            bool correct = compare_results(C, Ref, 1e-9);

            double bytes_moved = 3.0 * bytes;
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda(csv_path, "vecadd_basic", "FP64", N, ms, correct,
                               device_name, grid.x*block.x, 0, 0, block.x, 0, 0, (double)N, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
            std::cout << "[CUDA] vecadd_basic FP64 N=" << N << " ms=" << ms
                      << " correct=" << (correct?"true":"false") << "\n";
        }
    }

    // ---------------- Dot FP32/FP64 ----------------
    {
        int N = 1<<20;
        dim3 block(256);
        dim3 grid( std::min( (N + block.x - 1)/block.x, 4096 ) );

        // FP32
        {
            size_t bytes = N * sizeof(float);
            std::vector<float> A(N), B(N);
            fill_random(A, -1.f, 1.f); fill_random(B, -1.f, 1.f);
            float *dA, *dB, *dP;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dB, bytes));
            CUDA_CHECK(cudaMalloc(&dP, grid.x * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                dot_f32<<<grid, block>>>(dA, dB, dP, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;

            std::vector<float> partial(grid.x);
            CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x*sizeof(float), cudaMemcpyDeviceToHost));
            double gpu = 0; for (auto v : partial) gpu += v;
            double ref = 0; for (int i=0;i<N;++i) ref += (double)A[i]*B[i];
            bool correct = std::fabs(gpu - ref) <= 1e-3 * (1.0 + std::fabs(ref));

            double flops_est = 2.0 * (double)N;
            double bytes_moved = 2.0 * bytes;
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda("results_cuda.csv","dot_global","FP32",N,ms,correct,
                               device_name, grid.x*block.x,0,0, block.x,0,0, flops_est, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dP));
            std::cout << "[CUDA] dot FP32 N="<<N<<" ms="<<ms<<" correct="<<(correct?"true":"false")<<"\n";
        }

        // FP64
        {
            size_t bytes = N * sizeof(double);
            std::vector<double> A(N), B(N);
            fill_random(A, -1.0, 1.0); fill_random(B, -1.0, 1.0);
            double *dA, *dB, *dP;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dB, bytes));
            CUDA_CHECK(cudaMalloc(&dP, grid.x * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                dot_f64<<<grid, block>>>(dA, dB, dP, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;

            std::vector<double> partial(grid.x);
            CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x*sizeof(double), cudaMemcpyDeviceToHost));
            double gpu = 0; for (auto v : partial) gpu += v;
            double ref = 0; for (int i=0;i<N;++i) ref += A[i]*B[i];
            bool correct = std::fabs(gpu - ref) <= 1e-9 * (1.0 + std::fabs(ref));

            double flops_est = 2.0 * (double)N;
            double bytes_moved = 2.0 * bytes;
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda("results_cuda.csv","dot_global","FP64",N,ms,correct,
                               device_name, grid.x*block.x,0,0, block.x,0,0, flops_est, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dP));
            std::cout << "[CUDA] dot FP64 N="<<N<<" ms="<<ms<<" correct="<<(correct?"true":"false")<<"\n";
        }
    }

    // ---------------- Reduction FP32/FP64 ----------------
    {
        int N = 1<<20;
        dim3 block(256);
        dim3 grid( std::min( (N + block.x - 1)/block.x, 4096 ) );

        // FP32
        {
            size_t bytes = N * sizeof(float);
            std::vector<float> A(N);
            fill_random(A, 0.f, 1.f);
            float *dA, *dP;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dP, grid.x*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                reduction_f32<<<grid, block>>>(dA, dP, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;

            std::vector<float> partial(grid.x);
            CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x*sizeof(float), cudaMemcpyDeviceToHost));
            double gpu = 0; for (auto v : partial) gpu += v;
            double ref = 0; for (int i=0;i<N;++i) ref += (double)A[i];
            bool correct = std::fabs(gpu - ref) <= 1e-3 * (1.0 + std::fabs(ref));

            double flops_est = (double)N;
            double bytes_moved = bytes; // read-only reduction
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda("results_cuda.csv","reduction_shared","FP32",N,ms,correct,
                               device_name, grid.x*block.x,0,0, block.x,0,0, flops_est, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dP));
            std::cout << "[CUDA] reduction FP32 N="<<N<<" ms="<<ms<<" correct="<<(correct?"true":"false")<<"\n";
        }

        // FP64
        {
            size_t bytes = N * sizeof(double);
            std::vector<double> A(N);
            fill_random(A, 0.0, 1.0);
            double *dA, *dP;
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dP, grid.x*sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

            double ms = 0.0;
            for (int r=0; r<REPEAT; ++r) {
                CudaTimer t; t.begin();
                reduction_f64<<<grid, block>>>(dA, dP, N);
                CUDA_CHECK(cudaDeviceSynchronize());
                ms += t.end();
            }
            ms /= REPEAT;

            std::vector<double> partial(grid.x);
            CUDA_CHECK(cudaMemcpy(partial.data(), dP, grid.x*sizeof(double), cudaMemcpyDeviceToHost));
            double gpu = 0; for (auto v : partial) gpu += v;
            double ref = 0; for (int i=0;i<N;++i) ref += A[i];
            bool correct = std::fabs(gpu - ref) <= 1e-9 * (1.0 + std::fabs(ref));

            double flops_est = (double)N;
            double bytes_moved = bytes;
            double bw_GBps = bytes_moved / (ms * 1e6);
            write_csv_row_cuda("results_cuda.csv","reduction_shared","FP64",N,ms,correct,
                               device_name, grid.x*block.x,0,0, block.x,0,0, flops_est, bw_GBps);

            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dP));
            std::cout << "[CUDA] reduction FP64 N="<<N<<" ms="<<ms<<" correct="<<(correct?"true":"false")<<"\n";
        }
    }

    return 0;
}
