// =============================================================
// baseline_check.hpp (excerpt additions)
// CPU reference implementations for new kernels.
// =============================================================
#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <complex>


// VecAdd
template<typename T>
inline void vecadd_ref(const std::vector<T>& A,
                       const std::vector<T>& B,
                       std::vector<T>& C)
{
    C.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) C[i] = A[i] + B[i];
}

// MatMul (C[MxN] = A[MxK] * B[KxN])
template<typename T>
inline void matmul_ref(const std::vector<T>& A,
                       const std::vector<T>& B,
                       std::vector<T>&       C,
                       int M, int N, int K)
{
    C.assign((size_t)M*(size_t)N, (T)0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            T a = A[(size_t)i*K + k];
            for (int j = 0; j < N; ++j) {
                C[(size_t)i*N + j] += a * B[(size_t)k*N + j];
            }
        }
    }
}

// Reduction sum
template<typename T>
inline T reduction_ref(const std::vector<T>& x) {
    T acc = (T)0;
    for (auto v : x) acc += v;
    return acc;
}

// Conv2D 3x3 single channel with zero padding
template<typename T>
inline void conv2d3x3_ref(const std::vector<T>& input,
                          const std::vector<T>& k3x3, // length 9
                          std::vector<T>&       out,
                          int H, int W)
{
    out.assign((size_t)H*(size_t)W, (T)0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            T acc = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;
                    T v = 0;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        v = input[(size_t)iy*W + ix];
                    }
                    acc += v * k3x3[(ky+1)*3 + (kx+1)];
                }
            }
            out[(size_t)y*W + x] = acc;
        }
    }
}

// Stencil 5-point
template<typename T>
inline void stencil2d_ref(const std::vector<T>& in,
                          std::vector<T>&       out,
                          int H, int W)
{
    out.assign((size_t)H*(size_t)W, (T)0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            T c = in[(size_t)y*W + x];
            T n = (y > 0)     ? in[(size_t)(y-1)*W + x] : c;
            T s = (y+1 < H)   ? in[(size_t)(y+1)*W + x] : c;
            T w = (x > 0)     ? in[(size_t)y*W + (x-1)] : c;
            T e = (x+1 < W)   ? in[(size_t)y*W + (x+1)] : c;
            out[(size_t)y*W + x] = (T)0.2 * (c + n + s + w + e);
        }
    }
}

// Build a simple CSR matrix with ~nnz_per_row random nonzeros (no duplicates)
template<typename T>
inline void build_synthetic_csr(int M, int N, int nnz_per_row,
                                std::vector<int>& row_ptr,
                                std::vector<int>& col_idx,
                                std::vector<T>&   values)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> col_dist(0, N-1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    row_ptr.resize(M+1);
    col_idx.clear();
    values.clear();

    int nnz_acc = 0;
    for (int r = 0; r < M; ++r) {
        row_ptr[r] = nnz_acc;
        // naive unique selection (small nnz_per_row)
        std::vector<int> cols;
        cols.reserve(nnz_per_row);
        while ((int)cols.size() < nnz_per_row) {
            int c = col_dist(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            col_idx.push_back(c);
            values.push_back((T)val_dist(rng));
        }
        nnz_acc += nnz_per_row;
    }
    row_ptr[M] = nnz_acc;
}

// SpMV CSR reference
template<typename T>
inline void spmv_csr_ref(const std::vector<int>& row_ptr,
                         const std::vector<int>& col_idx,
                         const std::vector<T>&   values,
                         const std::vector<T>&   x,
                         std::vector<T>&         y)
{
    int M = (int) (row_ptr.size() - 1);
    y.assign(M, (T)0);
    for (int r = 0; r < M; ++r) {
        T acc = 0;
        for (int p = row_ptr[r]; p < row_ptr[r+1]; ++p) {
            acc += values[p] * x[col_idx[p]];
        }
        y[r] = acc;
    }
}


// --- Dot ---
template<typename T>
inline T dot_ref(const std::vector<T>& a, const std::vector<T>& b) {
    T acc = (T)0; for (size_t i=0;i<a.size();++i) acc += a[i]*b[i]; return acc;
}

// --- GEMV: y = A(MxN) * x(N) ---
template<typename T>
inline void gemv_ref(const std::vector<T>& A, const std::vector<T>& x,
                     std::vector<T>& y, int M, int N) {
    y.assign(M,(T)0);
    for (int i=0;i<M;++i) {
        T acc=(T)0;
        for (int j=0;j<N;++j) acc+=A[(size_t)i*N+j]*x[j];
        y[i]=acc;
    }
}

// --- Softmax along a row (single row version) ---
template<typename T>
inline void softmax_row_ref(const std::vector<T>& in, std::vector<T>& out) {
    out.resize(in.size());
    T mx = in[0]; for (auto v:in) mx = std::max(mx, v);
    T sum=(T)0; for (auto v:in) sum += std::exp(v-mx);
    for (size_t i=0;i<in.size();++i) out[i]=std::exp(in[i]-mx)/sum;
}

// --- LayerNorm on one row: y = (x - mean)/sqrt(var+eps) ---
template<typename T>
inline void layernorm_row_ref(const std::vector<T>& in, std::vector<T>& out, T eps=(T)1e-5) {
    out.resize(in.size());
    T mean=(T)0; for (auto v:in) mean+=v; mean/= (T)in.size();
    T var=(T)0; for (auto v:in) { T d=v-mean; var+=d*d; } var/= (T)in.size();
    T inv = (T)1/std::sqrt(var+eps);
    for (size_t i=0;i<in.size();++i) out[i]=(in[i]-mean)*inv;
}

// --- Activations ---
template<typename T>
inline void relu_ref(const std::vector<T>& in, std::vector<T>& out) {
    out.resize(in.size());
    for (size_t i=0;i<in.size();++i) out[i] = std::max<T>(in[i], (T)0);
}
template<typename T>
inline void gelu_ref(const std::vector<T>& in, std::vector<T>& out) {
    out.resize(in.size());
    // Approximate GELU
    for (size_t i=0;i<in.size();++i) {
        T x = in[i];
        out[i] = (T)0.5 * x * ( (T)1 + std::erf(x/(T)std::sqrt(2.0)) );
    }
}

// --- Scan (inclusive) ---
template<typename T>
inline void scan_inclusive_ref(const std::vector<T>& in, std::vector<T>& out) {
    out.resize(in.size()); T acc=(T)0;
    for (size_t i=0;i<in.size();++i){ acc+=in[i]; out[i]=acc; }
}

// --- Histogram (bins=256) ---
inline void histogram_ref_u32(const std::vector<unsigned>& data,
                              std::vector<unsigned>& hist, unsigned bins=256) {
    hist.assign(bins,0u);
    for (auto v: data) hist[v % bins] ++;
}

// --- Sort (CPU baseline use std::sort) ---
inline void sort_u32_ref(std::vector<unsigned>& data) {
    std::sort(data.begin(), data.end());
}

// --- MonteCarlo Pi ---
template<typename T>
inline T montecarlo_pi_ref(size_t N, uint32_t seed=123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0,1.0);
    size_t inside=0;
    for (size_t i=0;i<N;++i) {
        double x=U(rng), y=U(rng);
        if (x*x + y*y <= 1.0) inside++;
    }
    return (T)4.0 * (T)inside / (T)N;
}