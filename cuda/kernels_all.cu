#include <cuda_runtime.h>

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
