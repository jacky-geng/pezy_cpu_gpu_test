#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include <cstdint>
#include <type_traits>
#include <float.h>

// ---------------- Vector Add (FP32/FP64) ----------------
extern "C" __global__
void vecadd_f32(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" __global__
void vecadd_f64(const double* A, const double* B, double* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ---------------- Dot Product (FP32/FP64) ----------------
// Each block reduces a partial sum into 'partial[blockIdx.x]'
extern "C" __global__
void dot_f32(const float* A, const float* B, float* partial, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l   = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += A[i] * B[i];
    s[l] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (l < stride) s[l] += s[l + stride];
        __syncthreads();
    }
    if (l == 0) partial[blockIdx.x] = s[0];
}

extern "C" __global__
void dot_f64(const double* A, const double* B, double* partial, int N) {
    __shared__ double s[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l   = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += A[i] * B[i];
    s[l] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (l < stride) s[l] += s[l + stride];
        __syncthreads();
    }
    if (l == 0) partial[blockIdx.x] = s[0];
}

// ---------------- Reduction (sum) (FP32/FP64) ----------------
extern "C" __global__
void reduction_f32(const float* A, float* partial, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l   = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += A[i];
    s[l] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (l < stride) s[l] += s[l + stride];
        __syncthreads();
    }
    if (l == 0) partial[blockIdx.x] = s[0];
}

extern "C" __global__
void reduction_f64(const double* A, double* partial, int N) {
    __shared__ double s[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l   = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += A[i];
    s[l] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (l < stride) s[l] += s[l + stride];
        __syncthreads();
    }
    if (l == 0) partial[blockIdx.x] = s[0];
}

// ---------------- GEMV (global rows) ----------------
extern "C" __global__
void gemv_global_f32(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * x[k];
    y[row] = acc;
}

extern "C" __global__
void gemv_global_f64(const double* A, const double* x, double* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    double acc = 0.0;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * x[k];
    y[row] = acc;
}

// ---------------- GEMV (shared reduction per row) ----------------
extern "C" __global__
void gemv_shared_f32(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        sum += A[row * N + col] * x[col];
    }
    __shared__ float s[256];
    s[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

extern "C" __global__
void gemv_shared_f64(const double* A, const double* x, double* y, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int col = tid; col < N; col += blockDim.x) {
        sum += A[row * N + col] * x[col];
    }
    __shared__ double s[256];
    s[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = s[0];
}

// ---------------- MATMUL (global) ----------------
extern "C" __global__
void matmul_global_f32(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

extern "C" __global__
void matmul_global_f64(const double* A, const double* B, double* C,
                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------- MATMUL (tiled shared) ----------------
extern "C" __global__
void matmul_shared_f32(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}

extern "C" __global__
void matmul_shared_f64(const double* __restrict__ A,
                       const double* __restrict__ B,
                       double* __restrict__ C,
                       int M, int N, int K) {
    const int TILE = 16;
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------------- Scan (inclusive) ----------------
template <typename T>
__global__ void scan_block_kernel(const T* __restrict__ in, T* __restrict__ out,
                                  T* __restrict__ block_sums, int N) {
    int base = blockIdx.x * blockDim.x;
    int count = min(blockDim.x, N - base);
    if (count <= 0) return;
    int tid = threadIdx.x;
    __shared__ T s[256];
    T v = (tid < count) ? in[base + tid] : T(0);
    s[tid] = v;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        T tmp = (tid >= offset) ? s[tid - offset] : T(0);
        __syncthreads();
        if (tid < count && tid >= offset) s[tid] += tmp;
        __syncthreads();
    }
    if (tid < count) out[base + tid] = s[tid];
    if (block_sums && tid == count - 1) block_sums[blockIdx.x] = s[tid];
}

template <typename T>
__global__ void scan_add_offsets_kernel(T* __restrict__ data,
                                        const T* __restrict__ offsets,
                                        int N) {
    int base = blockIdx.x * blockDim.x;
    if (blockIdx.x == 0) return;
    int tid = threadIdx.x;
    int idx = base + tid;
    if (idx < N) data[idx] += offsets[blockIdx.x];
}

extern "C" void launch_scan_f32(const float* in, float* out,
                                float* block_sums, int N,
                                dim3 grid, dim3 block, cudaStream_t stream) {
    scan_block_kernel<float><<<grid, block, 0, stream>>>(in, out, block_sums, N);
}

extern "C" void launch_scan_f64(const double* in, double* out,
                                double* block_sums, int N,
                                dim3 grid, dim3 block, cudaStream_t stream) {
    scan_block_kernel<double><<<grid, block, 0, stream>>>(in, out, block_sums, N);
}

extern "C" void launch_scan_add_f32(float* data, const float* offsets, int N,
                                    dim3 grid, dim3 block, cudaStream_t stream) {
    scan_add_offsets_kernel<float><<<grid, block, 0, stream>>>(data, offsets, N);
}

extern "C" void launch_scan_add_f64(double* data, const double* offsets, int N,
                                    dim3 grid, dim3 block, cudaStream_t stream) {
    scan_add_offsets_kernel<double><<<grid, block, 0, stream>>>(data, offsets, N);
}

// ---------------- SpMV (CSR) ----------------
extern "C" __global__
void spmv_csr_f32(const int* rowptr, const int* colind,
                  const float* vals, const float* x, float* y, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    int start = rowptr[row];
    int end = rowptr[row + 1];
    for (int idx = start; idx < end; ++idx)
        sum += vals[idx] * x[colind[idx]];
    y[row] = sum;
}

extern "C" __global__
void spmv_csr_f64(const int* rowptr, const int* colind,
                  const double* vals, const double* x, double* y, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    double sum = 0.0;
    int start = rowptr[row];
    int end = rowptr[row + 1];
    for (int idx = start; idx < end; ++idx)
        sum += vals[idx] * x[colind[idx]];
    y[row] = sum;
}

// ---------------- Conv2d / Depthwise ----------------
extern "C" __global__
void conv2d_global_f32(const float* img, const float* filt, float* out,
                       int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            float val = (yy >= 0 && yy < H && xx >= 0 && xx < W)
                            ? img[yy * W + xx] : 0.0f;
            acc += val * filt[(dy + 1) * 3 + (dx + 1)];
        }
    out[y * W + x] = acc;
}

extern "C" __global__
void conv2d_global_f64(const double* img, const double* filt, double* out,
                       int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    double acc = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            double val = (yy >= 0 && yy < H && xx >= 0 && xx < W)
                             ? img[yy * W + xx] : 0.0;
            acc += val * filt[(dy + 1) * 3 + (dx + 1)];
        }
    out[y * W + x] = acc;
}

extern "C" __global__
void conv2d_shared_f32(const float* img, const float* filt, float* out,
                       int H, int W) {
    const int TILE = 16;
    __shared__ float tile[TILE + 2][TILE + 2];
    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    auto fetch = [&](int yy, int xx) -> float {
        return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : 0.0f;
    };
    tile[ly][lx] = fetch(gy, gx);
    if (threadIdx.x == 0) {
        tile[ly][lx - 1] = fetch(gy, gx - 1);
        tile[ly][lx + TILE] = fetch(gy, gx + TILE);
    }
    if (threadIdx.y == 0) {
        tile[ly - 1][lx] = fetch(gy - 1, gx);
        tile[ly + TILE][lx] = fetch(gy + TILE, gx);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[ly - 1][lx - 1] = fetch(gy - 1, gx - 1);
        tile[ly - 1][lx + TILE] = fetch(gy - 1, gx + TILE);
        tile[ly + TILE][lx - 1] = fetch(gy + TILE, gx - 1);
        tile[ly + TILE][lx + TILE] = fetch(gy + TILE, gx + TILE);
    }
    __syncthreads();
    if (gx >= W || gy >= H) return;
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            acc += tile[ly + dy][lx + dx] * filt[(dy + 1) * 3 + (dx + 1)];
    out[gy * W + gx] = acc;
}

extern "C" __global__
void conv2d_shared_f64(const double* img, const double* filt, double* out,
                       int H, int W) {
    const int TILE = 16;
    __shared__ double tile[TILE + 2][TILE + 2];
    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    auto fetch = [&](int yy, int xx) -> double {
        return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : 0.0;
    };
    tile[ly][lx] = fetch(gy, gx);
    if (threadIdx.x == 0) {
        tile[ly][lx - 1] = fetch(gy, gx - 1);
        tile[ly][lx + TILE] = fetch(gy, gx + TILE);
    }
    if (threadIdx.y == 0) {
        tile[ly - 1][lx] = fetch(gy - 1, gx);
        tile[ly + TILE][lx] = fetch(gy + TILE, gx);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[ly - 1][lx - 1] = fetch(gy - 1, gx - 1);
        tile[ly - 1][lx + TILE] = fetch(gy - 1, gx + TILE);
        tile[ly + TILE][lx - 1] = fetch(gy + TILE, gx - 1);
        tile[ly + TILE][lx + TILE] = fetch(gy + TILE, gx + TILE);
    }
    __syncthreads();
    if (gx >= W || gy >= H) return;
    double acc = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            acc += tile[ly + dy][lx + dx] * filt[(dy + 1) * 3 + (dx + 1)];
    out[gy * W + gx] = acc;
}

// ---------------- Depthwise Conv ----------------
extern "C" __global__
void depthwiseconv_global_f32(const float* img, const float* filt, float* out,
                              int C, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (x >= W || y >= H || c >= C) return;
    auto idx = [&](int yy, int xx, int cc) { return (cc * H + yy) * W + xx; };
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            float val = (yy >= 0 && yy < H && xx >= 0 && xx < W)
                            ? img[idx(yy, xx, c)] : 0.0f;
            acc += val * filt[c * 9 + (dy + 1) * 3 + (dx + 1)];
        }
    out[idx(y, x, c)] = acc;
}

extern "C" __global__
void depthwiseconv_global_f64(const double* img, const double* filt, double* out,
                              int C, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (x >= W || y >= H || c >= C) return;
    auto idx = [&](int yy, int xx, int cc) { return (cc * H + yy) * W + xx; };
    double acc = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            double val = (yy >= 0 && yy < H && xx >= 0 && xx < W)
                             ? img[idx(yy, xx, c)] : 0.0;
            acc += val * filt[c * 9 + (dy + 1) * 3 + (dx + 1)];
        }
    out[idx(y, x, c)] = acc;
}

extern "C" __global__
void depthwiseconv_tiled_f32(const float* img, const float* filt, float* out,
                             int C, int H, int W) {
    const int TILE = 8;
    __shared__ float tile[TILE + 2][TILE + 2];
    int c = blockIdx.z;
    if (c >= C) return;
    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    auto idx = [&](int yy, int xx) { return (c * H + yy) * W + xx; };
    auto fetch = [&](int yy, int xx) -> float {
        return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[idx(yy, xx)] : 0.0f;
    };
    tile[ly][lx] = fetch(gy, gx);
    if (threadIdx.x == 0) {
        tile[ly][lx - 1] = fetch(gy, gx - 1);
        tile[ly][lx + TILE] = fetch(gy, gx + TILE);
    }
    if (threadIdx.y == 0) {
        tile[ly - 1][lx] = fetch(gy - 1, gx);
        tile[ly + TILE][lx] = fetch(gy + TILE, gx);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[ly - 1][lx - 1] = fetch(gy - 1, gx - 1);
        tile[ly - 1][lx + TILE] = fetch(gy - 1, gx + TILE);
        tile[ly + TILE][lx - 1] = fetch(gy + TILE, gx - 1);
        tile[ly + TILE][lx + TILE] = fetch(gy + TILE, gx + TILE);
    }
    __syncthreads();
    if (gx >= W || gy >= H) return;
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            acc += tile[ly + dy][lx + dx] * filt[c * 9 + (dy + 1) * 3 + (dx + 1)];
    out[idx(gy, gx)] = acc;
}

extern "C" __global__
void depthwiseconv_tiled_f64(const double* img, const double* filt, double* out,
                             int C, int H, int W) {
    const int TILE = 8;
    __shared__ double tile[TILE + 2][TILE + 2];
    int c = blockIdx.z;
    if (c >= C) return;
    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    auto idx = [&](int yy, int xx) { return (c * H + yy) * W + xx; };
    auto fetch = [&](int yy, int xx) -> double {
        return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[idx(yy, xx)] : 0.0;
    };
    tile[ly][lx] = fetch(gy, gx);
    if (threadIdx.x == 0) {
        tile[ly][lx - 1] = fetch(gy, gx - 1);
        tile[ly][lx + TILE] = fetch(gy, gx + TILE);
    }
    if (threadIdx.y == 0) {
        tile[ly - 1][lx] = fetch(gy - 1, gx);
        tile[ly + TILE][lx] = fetch(gy + TILE, gx);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[ly - 1][lx - 1] = fetch(gy - 1, gx - 1);
        tile[ly - 1][lx + TILE] = fetch(gy - 1, gx + TILE);
        tile[ly + TILE][lx - 1] = fetch(gy + TILE, gx - 1);
        tile[ly + TILE][lx + TILE] = fetch(gy + TILE, gx + TILE);
    }
    __syncthreads();
    if (gx >= W || gy >= H) return;
    double acc = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            acc += tile[ly + dy][lx + dx] * filt[c * 9 + (dy + 1) * 3 + (dx + 1)];
    out[idx(gy, gx)] = acc;
}


// ---------------- Graph / Irregular ----------------
extern "C" __global__
void bfs_basic(const int* rowptr, const int* colind,
               const unsigned int* frontier,
               unsigned int* next_frontier,
               unsigned int* visited,
               int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    if (!frontier[v]) return;
    int start = rowptr[v];
    int end = rowptr[v + 1];
    for (int idx = start; idx < end; ++idx) {
        int u = colind[idx];
        if (atomicCAS(&visited[u], 0u, 1u) == 0u)
            next_frontier[u] = 1u;
    }
}

extern "C" __global__
void dfs_basic(const int* rowptr, const int* colind,
               const unsigned int* frontier,
               unsigned int* next_frontier,
               unsigned int* visited,
               int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    if (!frontier[v]) return;
    int start = rowptr[v];
    int end = rowptr[v + 1];
    for (int idx = start; idx < end; ++idx) {
        int u = colind[idx];
        if (atomicCAS(&visited[u], 0u, 1u) == 0u)
            next_frontier[u] = 1u;
    }
}

template <typename T>
__device__ inline void atomic_add_relaxed(T* addr, T val) {
    atomicAdd(addr, val);
}

template <>
__device__ inline void atomic_add_relaxed<double>(double* addr, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    auto address_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        double updated = __longlong_as_double(assumed) + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(updated));
    } while (assumed != old);
#endif
}

template <typename T>
__device__ __forceinline__ void pagerank_basic_kernel(const int* rowptr, const int* colind,
                                      const int* outdeg,
                                      const T* pr, T* pr_next,
                                      T d, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    int start = rowptr[v];
    int end = rowptr[v + 1];
    int deg = max(outdeg[v], 1);
    T contrib = d * pr[v] / static_cast<T>(deg);
    for (int idx = start; idx < end; ++idx) {
        int u = colind[idx];
        atomic_add_relaxed(&pr_next[u], contrib);
    }
}

extern "C" __global__
void pagerank_basic_f32(const int* rowptr, const int* colind,
                        const int* outdeg,
                        const float* pr, float* pr_next,
                        float d, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    pagerank_basic_kernel<float>(rowptr, colind, outdeg, pr, pr_next, d, V);
}

extern "C" __global__
void pagerank_basic_f64(const int* rowptr, const int* colind,
                        const int* outdeg,
                        const double* pr, double* pr_next,
                        double d, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    pagerank_basic_kernel<double>(rowptr, colind, outdeg, pr, pr_next, d, V);
}

// ---------------- Softmax ----------------
extern "C" __global__
void softmax_basic_f32(const float* x, float* y, int N) {
    __shared__ float smax[256];
    __shared__ float ssum[256];
    int tid = threadIdx.x;
    float val = (tid < N) ? x[tid] : -FLT_MAX;
    smax[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float m = smax[0];
    float e = (tid < N) ? expf(val - m) : 0.0f;
    ssum[tid] = e;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float sum = ssum[0];
    if (tid < N)
        y[tid] = e / sum;
}

extern "C" __global__
void softmax_basic_f64(const double* x, double* y, int N) {
    __shared__ double smax[256];
    __shared__ double ssum[256];
    int tid = threadIdx.x;
    double val = (tid < N) ? x[tid] : -DBL_MAX;
    smax[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smax[tid] = fmax(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    double m = smax[0];
    double e = (tid < N) ? exp(val - m) : 0.0;
    ssum[tid] = e;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    double sum = ssum[0];
    if (tid < N)
        y[tid] = e / sum;
}

// ---------------- LayerNorm ----------------
extern "C" __global__
void layernorm_basic_f32(const float* x, float* y, int N, float eps) {
    __shared__ float s0[256];
    __shared__ float s1[256];
    int tid = threadIdx.x;
    float val = (tid < N) ? x[tid] : 0.0f;
    s0[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s0[tid] += s0[tid + stride];
        __syncthreads();
    }
    float mean = s0[0] / static_cast<float>(N);
    float diff = val - mean;
    s1[tid] = (tid < N) ? diff * diff : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s1[tid] += s1[tid + stride];
        __syncthreads();
    }
    float var = s1[0] / static_cast<float>(N);
    float inv = rsqrtf(var + eps);
    if (tid < N)
        y[tid] = (x[tid] - mean) * inv;
}

extern "C" __global__
void layernorm_basic_f64(const double* x, double* y, int N, double eps) {
    __shared__ double s0[256];
    __shared__ double s1[256];
    int tid = threadIdx.x;
    double val = (tid < N) ? x[tid] : 0.0;
    s0[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s0[tid] += s0[tid + stride];
        __syncthreads();
    }
    double mean = s0[0] / static_cast<double>(N);
    double diff = val - mean;
    s1[tid] = (tid < N) ? diff * diff : 0.0;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s1[tid] += s1[tid + stride];
        __syncthreads();
    }
    double var = s1[0] / static_cast<double>(N);
    double inv = 1.0 / sqrt(var + eps);
    if (tid < N)
        y[tid] = (x[tid] - mean) * inv;
}

// ---------------- Activations ----------------
extern "C" __global__
void activation_relu_f32(const float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
}

extern "C" __global__
void activation_relu_f64(const double* x, double* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = x[idx] > 0.0 ? x[idx] : 0.0;
}

extern "C" __global__
void activation_gelu_f32(const float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = x[idx];
        y[idx] = 0.5f * v * (1.0f + erff(v * 0.70710678f));
    }
}

extern "C" __global__
void activation_gelu_f64(const double* x, double* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double v = x[idx];
        y[idx] = 0.5 * v * (1.0 + erf(v * 0.7071067811865476));
    }
}

// ---------------- Stencil 2D/3D ----------------
extern "C" __global__
void stencil2d_3x3_f32(const float* in, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f; int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                acc += in[yy * W + xx]; cnt++;
            }
        }
    out[y * W + x] = acc / cnt;
}

extern "C" __global__
void stencil2d_3x3_f64(const double* in, double* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    double acc = 0.0; int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                acc += in[yy * W + xx]; cnt++;
            }
        }
    out[y * W + x] = acc / cnt;
}

extern "C" __global__
void stencil2d_5x5_f32(const float* in, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f; int cnt = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                acc += in[yy * W + xx]; cnt++;
            }
        }
    out[y * W + x] = acc / cnt;
}

extern "C" __global__
void stencil2d_5x5_f64(const double* in, double* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    double acc = 0.0; int cnt = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                acc += in[yy * W + xx]; cnt++;
            }
        }
    out[y * W + x] = acc / cnt;
}

extern "C" __global__
void stencil3d_global_f32(const float* in, float* out, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    auto idx = [&](int zz, int yy, int xx) { return (zz * H + yy) * W + xx; };
    float acc = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int zz = z + dz, yy = y + dy, xx = x + dx;
                if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W)
                    acc += in[idx(zz, yy, xx)];
            }
    out[idx(z, y, x)] = acc;
}

extern "C" __global__
void stencil3d_global_f64(const double* in, double* out, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    auto idx = [&](int zz, int yy, int xx) { return (zz * H + yy) * W + xx; };
    double acc = 0.0;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int zz = z + dz, yy = y + dy, xx = x + dx;
                if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W)
                    acc += in[idx(zz, yy, xx)];
            }
    out[idx(z, y, x)] = acc;
}

extern "C" __global__
void stencil3d_shared_f32(const float* in, float* out, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    auto idx = [&](int zz, int yy, int xx) { return (zz * H + yy) * W + xx; };
    float acc = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int zz = z + dz, yy = y + dy, xx = x + dx;
                if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W)
                    acc += in[idx(zz, yy, xx)];
            }
    out[idx(z, y, x)] = acc;
}

extern "C" __global__
void stencil3d_shared_f64(const double* in, double* out, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    auto idx = [&](int zz, int yy, int xx) { return (zz * H + yy) * W + xx; };
    double acc = 0.0;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int zz = z + dz, yy = y + dy, xx = x + dx;
                if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W)
                    acc += in[idx(zz, yy, xx)];
            }
    out[idx(z, y, x)] = acc;
}

// ---------------- Histogram ----------------
extern "C" __global__
void histogram_global(const unsigned int* data, unsigned int* hist, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        unsigned int bin = data[i] & 255u;
        atomicAdd(&hist[bin], 1u);
    }
}

extern "C" __global__
void histogram_shared(const unsigned int* data, unsigned int* hist, int N) {
    __shared__ unsigned int bins[256];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x) bins[i] = 0;
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < N; i += stride) {
        unsigned int bin = data[i] & 255u;
        atomicAdd(&bins[bin], 1u);
    }
    __syncthreads();
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int val = bins[i];
        if (val) atomicAdd(&hist[i], val);
    }
}

// ---------------- Bitonic Sort ----------------
extern "C" __global__
void sort_bitonic(uint32_t* data, int N, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N / 2) return;
    int i = idx * 2;
    int ixj = i ^ j;
    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        uint32_t vi = data[i];
        uint32_t vx = data[ixj];
        bool swap = ascending ? (vi > vx) : (vi < vx);
        if (swap) {
            data[i] = vx;
            data[ixj] = vi;
        }
    }
}

// ---------------- Monte Carlo ----------------
extern "C" __global__
void montecarlo_basic_f32(const float* xy, unsigned int* partial, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int count = 0;
    for (int i = tid; i < N; i += stride) {
        float x = xy[2 * i];
        float y = xy[2 * i + 1];
        if (x * x + y * y <= 1.0f) count++;
    }
    partial[tid] = count;
}

extern "C" __global__
void montecarlo_basic_f64(const double* xy, unsigned int* partial, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int count = 0;
    for (int i = tid; i < N; i += stride) {
        double x = xy[2 * i];
        double y = xy[2 * i + 1];
        if (x * x + y * y <= 1.0) count++;
    }
    partial[tid] = count;
}

// ---------------- FFT ----------------
template <typename T, typename Complex>
__device__ __forceinline__ void fft1d_stage(Complex* data, int N, int tid, int mh) {
    int k = tid / mh;
    int j = tid % mh;
    int m = mh << 1;
    int idx1 = k * m + j;
    int idx2 = idx1 + mh;
    double angle = -M_PI * j / mh;
    double c = cos(angle);
    double s = sin(angle);
    Complex a = data[idx1];
    Complex b = data[idx2];
    Complex t;
    t.x = c * b.x - s * b.y;
    t.y = c * b.y + s * b.x;
    data[idx1].x = a.x + t.x;
    data[idx1].y = a.y + t.y;
    data[idx2].x = a.x - t.x;
    data[idx2].y = a.y - t.y;
}

extern "C" __global__
void fft1d_global_f32(float2* data, int N, int mh) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 2) return;
    fft1d_stage<float,float2>(data, N, tid, mh);
}

extern "C" __global__
void fft1d_global_f64(double2* data, int N, int mh) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 2) return;
    fft1d_stage<double,double2>(data, N, tid, mh);
}

extern "C" __global__
void fft1d_staged_f32(float2* data, int N, int mh) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 2) return;
    fft1d_stage<float,float2>(data, N, tid, mh);
}

extern "C" __global__
void fft1d_staged_f64(double2* data, int N, int mh) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 2) return;
    fft1d_stage<double,double2>(data, N, tid, mh);
}

// ---------------- Sequence Alignment (Smith-Waterman + WFA) ----------------
extern "C" __global__
void smithwaterman_basic_kernel(const uint8_t* seqA,
                                const uint8_t* seqB,
                                int* scores,
                                int num_pairs,
                                int len,
                                int match_score,
                                int mismatch_score,
                                int gap_score) {
    const int SW_MAX_LEN = 256;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    if (len > SW_MAX_LEN) len = SW_MAX_LEN;
    const uint8_t* a = seqA + static_cast<size_t>(tid) * len;
    const uint8_t* b = seqB + static_cast<size_t>(tid) * len;
    int prev[SW_MAX_LEN + 1];
    int curr[SW_MAX_LEN + 1];
    for (int j = 0; j <= len; ++j) prev[j] = 0;
    int best = 0;
    for (int i = 1; i <= len; ++i) {
        curr[0] = 0;
        uint8_t ai = a[i - 1];
        for (int j = 1; j <= len; ++j) {
            int diag = prev[j - 1] + (ai == b[j - 1] ? match_score : mismatch_score);
            int up   = prev[j] + gap_score;
            int left = curr[j - 1] + gap_score;
            int val = diag;
            if (up > val) val = up;
            if (left > val) val = left;
            if (val < 0) val = 0;
            curr[j] = val;
            if (val > best) best = val;
        }
        for (int j = 0; j <= len; ++j)
            prev[j] = curr[j];
    }
    scores[tid] = best;
}

extern "C" __global__
void wfa_editdistance_kernel(const uint8_t* seqA,
                             const uint8_t* seqB,
                             int* distances,
                             int num_pairs,
                             int len) {
    const int WFA_MAX_LEN = 256;
    const int DIAG_COUNT = 2 * WFA_MAX_LEN + 1;
    const int NEG_INF = -1000000;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;
    if (len > WFA_MAX_LEN) len = WFA_MAX_LEN;

    const uint8_t* a = seqA + static_cast<size_t>(tid) * len;
    const uint8_t* b = seqB + static_cast<size_t>(tid) * len;

    int prev[DIAG_COUNT];
    int curr[DIAG_COUNT];
    for (int i = 0; i < DIAG_COUNT; ++i) {
        prev[i] = NEG_INF;
        curr[i] = NEG_INF;
    }

    const int center = WFA_MAX_LEN;
    int offset = 0;
    while (offset < len && a[offset] == b[offset]) ++offset;
    if (offset >= len) {
        distances[tid] = 0;
        return;
    }
    prev[center] = offset;

    const int max_dist = len * 2;
    for (int dist = 1; dist <= max_dist; ++dist) {
        int diag_min = -dist;
        int diag_max = dist;
        for (int diag = diag_min; diag <= diag_max; ++diag) {
            int idx = center + diag;
            int best = NEG_INF;
            if (idx - 1 >= 0) {
                int cand = prev[idx - 1] + 1; // insertion
                if (cand > best) best = cand;
            }
            if (idx >= 0 && idx < DIAG_COUNT) {
                int cand = prev[idx] + 1; // mismatch
                if (cand > best) best = cand;
            }
            if (idx + 1 < DIAG_COUNT) {
                int cand = prev[idx + 1]; // deletion
                if (cand > best) best = cand;
            }
            if (best < 0) best = 0;
            int i = best;
            int j = i - diag;
            while (i < len && j < len && j >= 0 && a[i] == b[j]) {
                ++i;
                ++j;
            }
            curr[idx] = i;
            if (i >= len && j >= len) {
                distances[tid] = dist;
                return;
            }
        }
        for (int diag = diag_min; diag <= diag_max; ++diag) {
            int idx = center + diag;
            prev[idx] = curr[idx];
            curr[idx] = NEG_INF;
        }
    }
    distances[tid] = len;
}

extern "C" __global__
void smithwaterman_wavefront_kernel(const uint8_t* seqA,
                                    const uint8_t* seqB,
                                    int* scores,
                                    int num_pairs,
                                    int len,
                                    int match_score,
                                    int mismatch_score,
                                    int gap_score) {
    const int SW_MAX_LEN = 256;
    int pair = blockIdx.x;
    if (pair >= num_pairs) return;

    int effective_len = len > SW_MAX_LEN ? SW_MAX_LEN : len;
    const uint8_t* a = seqA + static_cast<size_t>(pair) * len;
    const uint8_t* b = seqB + static_cast<size_t>(pair) * len;

    extern __shared__ int shared[];
    int* diag_prev2 = shared;
    int* diag_prev  = diag_prev2 + (effective_len + 1);
    int* diag_curr  = diag_prev  + (effective_len + 1);
    int* best_buf   = diag_curr  + (effective_len + 1);

    for (int idx = threadIdx.x; idx <= effective_len; idx += blockDim.x) {
        diag_prev2[idx] = 0;
        diag_prev[idx] = 0;
        diag_curr[idx] = 0;
    }
    if (threadIdx.x < blockDim.x) best_buf[threadIdx.x] = 0;
    __syncthreads();

    int best_local = 0;
    int max_diag = 2 * effective_len;
    for (int diag = 1; diag <= max_diag; ++diag) {
        __syncthreads();
        int i_start = max(1, diag - effective_len);
        int i_end = min(effective_len, diag);

        int i = i_start + threadIdx.x;
        if (i <= i_end) {
            int j = diag - i;
            if (j >= 1 && j <= effective_len) {
                int diag_val = diag_prev2[i - 1];
                int up_val = diag_prev[i - 1];
                int left_val = diag_prev[i];
                int score = diag_val + (a[i - 1] == b[j - 1] ? match_score : mismatch_score);
                score = max(score, up_val + gap_score);
                score = max(score, left_val + gap_score);
                if (score < 0) score = 0;
                diag_curr[i] = score;
                best_local = max(best_local, score);
            } else {
                diag_curr[i] = 0;
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx <= effective_len; idx += blockDim.x) {
            diag_prev2[idx] = diag_prev[idx];
            diag_prev[idx] = diag_curr[idx];
            diag_curr[idx] = 0;
        }
        __syncthreads();
    }

    best_buf[threadIdx.x] = best_local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            best_buf[threadIdx.x] = max(best_buf[threadIdx.x], best_buf[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        scores[pair] = best_buf[0];
}
