// =============================================================
// kernels_all.cl
// Unified OpenCL kernels for benchmarking (FP32/FP64 switchable)
// All comments are in English.
// =============================================================

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef USE_FP64
typedef double fp_t;
#else
typedef float  fp_t;
#endif
// -----------------------------
// 1) VecAdd (memory-bound)
// -----------------------------
__kernel void vecadd_basic(__global const fp_t* A,
                           __global const fp_t* B,
                           __global fp_t* C,
                           const int N)
{
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] + B[i];
}

// -----------------------------
// 2) MatMul (compute-bound)
//    - matmul_global: naive, all global loads
//    - matmul_tiled:  local tiling (BLOCK = 16)
// -----------------------------
__kernel void matmul_global(__global const fp_t* A,
                            __global const fp_t* B,
                            __global fp_t*       C,
                            const int M, const int N, const int K)
{
    // C[M x N] = A[M x K] * B[K x N]
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        fp_t acc = (fp_t)0;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

__kernel void matmul_tiled(__global const fp_t* A,
                           __global const fp_t* B,
                           __global fp_t*       C,
                           const int M, const int N, const int K)
{
    // Blocked GEMM with local memory tiles (BLOCK=16).
    // Local size should be set to (16,16).
    const int BLOCK = 16;

    __local fp_t Asub[16][16];
    __local fp_t Bsub[16][16];

    int row  = get_global_id(0);
    int col  = get_global_id(1);
    int lrow = get_local_id(0);
    int lcol = get_local_id(1);

    fp_t acc = (fp_t)0;

    for (int t = 0; t < K; t += BLOCK) {
        // Load tiles with bounds checks
        Asub[lrow][lcol] = (row < M && (t + lcol) < K) ? A[row * K + (t + lcol)] : (fp_t)0;
        Bsub[lrow][lcol] = (col < N && (t + lrow) < K) ? B[(t + lrow) * N + col] : (fp_t)0;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial product
        for (int k = 0; k < BLOCK; ++k) {
            acc += Asub[lrow][k] * Bsub[k][lcol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// -----------------------------
// 3) Reduction (sum)
//    - reduction_shared: group-wide tree reduction to partials
//      (host will accumulate partials)
// -----------------------------
__kernel void reduction_shared(__global const fp_t* input,
                               __global fp_t*       partial,
                               const int N)
{
    // Assumes local size is a power of two (e.g., 256).
    __local fp_t lbuf[256];

    int lid   = get_local_id(0);
    int gsize = get_global_size(0);
    int gid   = get_global_id(0);

    // Stride over the input to accumulate thread-local sum
    fp_t val = 0;
    for (int i = gid; i < N; i += gsize) {
        val += input[i];
    }
    lbuf[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary tree reduction within a work-group
    for (int s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            lbuf[lid] += lbuf[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[get_group_id(0)] = lbuf[0];
    }
}

// -----------------------------
// 4) Conv2D (3x3)
//    - conv2d_global: naive global reads
//    - conv2d_tiled : local tile with 1-pixel halo
// -----------------------------
__kernel void conv2d_global(__global const fp_t* input,
                            __global const fp_t* kernel3x3, // length 9 in row-major
                            __global fp_t*       output,
                            const int H, const int W)
{
    // Single-channel 2D conv with zero padding.
    int y = get_global_id(0);
    int x = get_global_id(1);

    if (y < H && x < W) {
        fp_t acc = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int iy = y + ky;
                int ix = x + kx;
                fp_t v = 0;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    v = input[iy * W + ix];
                }
                acc += v * kernel3x3[(ky + 1) * 3 + (kx + 1)];
            }
        }
        output[y * W + x] = acc;
    }
}

__kernel void conv2d_tiled(__global const fp_t* input,
                           __global const fp_t* kernel3x3,
                           __global fp_t*       output,
                           const int H, const int W)
{
    // Tiled conv with 1-pixel halo; local size e.g., (16,16)
    const int TILE = 16;
    __local fp_t tile[TILE + 2][TILE + 2];

    int y = get_global_id(0);
    int x = get_global_id(1);
    int ly = get_local_id(0) + 1; // +1 for halo shift
    int lx = get_local_id(1) + 1;

    // Load center
    tile[ly][lx] = (y < H && x < W) ? input[y * W + x] : (fp_t)0;

    // Load halos (edges). Do it with conditional guards to avoid OOB
    if (get_local_id(0) == 0) {
        // top row
        int yy = y - 1;
        int xx = x;
        tile[ly - 1][lx] = (yy >= 0 && xx < W) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(0) == TILE - 1) {
        // bottom row
        int yy = y + 1;
        int xx = x;
        tile[ly + 1][lx] = (yy < H && xx < W) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(1) == 0) {
        // left col
        int yy = y;
        int xx = x - 1;
        tile[ly][lx - 1] = (yy < H && xx >= 0) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(1) == TILE - 1) {
        // right col
        int yy = y;
        int xx = x + 1;
        tile[ly][lx + 1] = (yy < H && xx < W) ? input[yy * W + xx] : (fp_t)0;
    }

    // Corners
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        int yy = y - 1, xx = x - 1;
        tile[ly - 1][lx - 1] = (yy >= 0 && xx >= 0) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(0) == 0 && get_local_id(1) == TILE - 1) {
        int yy = y - 1, xx = x + 1;
        tile[ly - 1][lx + 1] = (yy >= 0 && xx < W) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(0) == TILE - 1 && get_local_id(1) == 0) {
        int yy = y + 1, xx = x - 1;
        tile[ly + 1][lx - 1] = (yy < H && xx >= 0) ? input[yy * W + xx] : (fp_t)0;
    }
    if (get_local_id(0) == TILE - 1 && get_local_id(1) == TILE - 1) {
        int yy = y + 1, xx = x + 1;
        tile[ly + 1][lx + 1] = (yy < H && xx < W) ? input[yy * W + xx] : (fp_t)0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (y < H && x < W) {
        fp_t acc = 0;
        #pragma unroll
        for (int ky = -1; ky <= 1; ++ky) {
            #pragma unroll
            for (int kx = -1; kx <= 1; ++kx) {
                acc += tile[ly + ky][lx + kx] * kernel3x3[(ky + 1) * 3 + (kx + 1)];
            }
        }
        output[y * W + x] = acc;
    }
}

// -----------------------------
// 5) Stencil2D (5-point Jacobi)
// -----------------------------
__kernel void stencil2d_global(__global const fp_t* input,
                               __global fp_t*       output,
                               const int H, const int W)
{
    int y = get_global_id(0);
    int x = get_global_id(1);

    if (y < H && x < W) {
        fp_t c = input[y * W + x];
        fp_t n = (y > 0)     ? input[(y - 1) * W + x] : c;
        fp_t s = (y + 1 < H) ? input[(y + 1) * W + x] : c;
        fp_t w = (x > 0)     ? input[y * W + (x - 1)] : c;
        fp_t e = (x + 1 < W) ? input[y * W + (x + 1)] : c;
        output[y * W + x] = (fp_t)0.2 * (c + n + s + w + e);
    }
}

__kernel void stencil2d_shared(__global const fp_t* input,
                               __global fp_t*       output,
                               const int H, const int W)
{
    // Same tiling structure as conv2d_tiled
    const int TILE = 16;
    __local fp_t tile[TILE + 2][TILE + 2];

    int y = get_global_id(0);
    int x = get_global_id(1);
    int ly = get_local_id(0) + 1;
    int lx = get_local_id(1) + 1;

    // Center
    tile[ly][lx] = (y < H && x < W) ? input[y * W + x] : (fp_t)0;

    // Halos (copy from conv2d_tiled pattern)
    if (get_local_id(0) == 0)         tile[ly - 1][lx] = (y > 0 && x < W) ? input[(y - 1) * W + x] : (fp_t)0;
    if (get_local_id(0) == TILE - 1)  tile[ly + 1][lx] = (y + 1 < H && x < W) ? input[(y + 1) * W + x] : (fp_t)0;
    if (get_local_id(1) == 0)         tile[ly][lx - 1] = (y < H && x > 0) ? input[y * W + (x - 1)] : (fp_t)0;
    if (get_local_id(1) == TILE - 1)  tile[ly][lx + 1] = (y < H && x + 1 < W) ? input[y * W + (x + 1)] : (fp_t)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (y < H && x < W) {
        fp_t c = tile[ly][lx];
        fp_t n = tile[ly - 1][lx];
        fp_t s = tile[ly + 1][lx];
        fp_t w = tile[ly][lx - 1];
        fp_t e = tile[ly][lx + 1];
        output[y * W + x] = (fp_t)0.2 * (c + n + s + w + e);
    }
}

// -----------------------------
// 6) SpMV (CSR format)
//    input: row_ptr[M+1], col_idx[nnz], values[nnz], x[N]
//    output: y[M]
// -----------------------------
__kernel void spmv_global(__global const int*  row_ptr,
                          __global const int*  col_idx,
                          __global const fp_t* values,
                          __global const fp_t* x,
                          __global fp_t*       y,
                          const int M)
{
    int row = get_global_id(0);
    if (row < M) {
        int start = row_ptr[row];
        int end   = row_ptr[row + 1];
        fp_t acc = (fp_t)0;
        for (int p = start; p < end; ++p) {
            acc += values[p] * x[col_idx[p]];
        }
        y[row] = acc;
    }
}

// (Optional) You can add spmv_cached here later if you want a cached variant.


// =============================================================
// ADDITIONS: more kernels to reach the 23-test suite
// =============================================================



// -----------------------------
// Dot Product (global / shared)
// -----------------------------
__kernel void dot_global(__global const fp_t* A,
                         __global const fp_t* B,
                         __global fp_t*       partial,
                         const int N)
{
    // Each work-item does strided dot and writes to partial per-group via local reduction.
    __local fp_t lbuf[256];
    int lid   = get_local_id(0);
    int gsize = get_global_size(0);
    int gid   = get_global_id(0);

    fp_t acc = (fp_t)0;
    for (int i = gid; i < N; i += gsize) acc += A[i]*B[i];
    lbuf[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s=get_local_size(0)>>1; s>0; s>>=1) {
        if (lid < s) lbuf[lid] += lbuf[lid+s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid==0) partial[get_group_id(0)] = lbuf[0];
}

__kernel void dot_shared(__global const fp_t* A,
                         __global const fp_t* B,
                         __global fp_t*       partial,
                         const int N)
{
    // Same as dot_global: explicit shared reduction (alias kept for A/B testing)
    __local fp_t lbuf[256];
    int lid   = get_local_id(0);
    int gsize = get_global_size(0);
    int gid   = get_global_id(0);

    fp_t acc = (fp_t)0;
    for (int i = gid; i < N; i += gsize) acc += A[i]*B[i];
    lbuf[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s=get_local_size(0)>>1; s>0; s>>=1) {
        if (lid < s) lbuf[lid] += lbuf[lid+s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid==0) partial[get_group_id(0)] = lbuf[0];
}

// -----------------------------
// GEMV: y = A(MxN) * x(N)
// -----------------------------
__kernel void gemv_global(__global const fp_t* A,
                          __global const fp_t* x,
                          __global fp_t*       y,
                          const int M, const int N)
{
    int row = get_global_id(0);
    if (row < M) {
        fp_t acc=(fp_t)0;
        for (int j=0;j<N;++j) acc += A[row*N+j]*x[j];
        y[row]=acc;
    }
}

__kernel void gemv_shared(__global const fp_t* A,
                          __global const fp_t* x,
                          __global fp_t*       y,
                          const int M, const int N)
{
    // Tile x into local memory for reuse (BLOCK=256)
    const int BLOCK = 256;
    __local fp_t xbuf[BLOCK];

    int row = get_global_id(0);
    if (row >= M) return;

    fp_t acc = (fp_t)0;
    for (int t=0; t<N; t+=BLOCK) {
        int lane = get_local_id(0);
        if (lane < BLOCK) {
            int col = t + lane;
            xbuf[lane] = (col < N) ? x[col] : (fp_t)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0;k<BLOCK;++k) {
            int col = t + k;
            if (col < N) acc += A[row*N + col] * xbuf[k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    y[row]=acc;
}

// -----------------------------
// Inclusive Scan (Blelloch-like per-block), write block scans
// Host can do a second pass if needed.
// -----------------------------
__kernel void scan_shared(__global const fp_t* in,
                          __global fp_t*       out,
                          const int N)
{
    __local fp_t temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    fp_t val = (gid < N) ? in[gid] : (fp_t)0;
    temp[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // upsweep
    for (int offset=1; offset<get_local_size(0); offset<<=1) {
        fp_t t = 0;
        if (((lid+1) % (offset<<1)) == 0) {
            int ai = lid - offset + 1;
            int bi = lid + 1;
            t = temp[ai-1] + temp[bi-1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (((lid+1) % (offset<<1)) == 0) temp[lid] = t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // downsweep to inclusive
    for (int offset=get_local_size(0)>>1; offset>0; offset>>=1) {
        fp_t t = temp[lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (((lid+1) % (offset<<1)) == 0) {
            int ai = lid - offset + 1;
            int bi = lid + 1;
            fp_t left = temp[ai-1];
            fp_t right= t;
            temp[ai-1] = left;
            temp[bi-1] = left + right;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gid < N) out[gid] = temp[lid];
}

// -----------------------------
// Softmax on a row vector
// -----------------------------
__kernel void softmax_basic(__global const fp_t* x,
                            __global fp_t*       y,
                            const int N)
{
    // Tile the input in chunks so N can be > 1024
    const int TILE = 1024;
    __local fp_t buf[TILE];
    int lid   = get_local_id(0);
    int lsize = get_local_size(0);
    int gnum  = get_num_groups(0);
    int gid0  = get_group_id(0);

    for (int base = gid0 * TILE; base < N; base += TILE * gnum) {
        int idx = base + lid;

        // 1) load and find max
        fp_t val = (idx < N) ? x[idx] : (fp_t)(-1e30);
        buf[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = lsize >> 1; s > 0; s >>= 1) {
            if (lid < s) buf[lid] = fmax(buf[lid], buf[lid + s]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        fp_t mx = buf[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2) exp and sum (use standard exp for better FP64 reproducibility)
        fp_t ex = (idx < N) ? exp(val - mx) : (fp_t)0;
        buf[lid] = ex;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = lsize >> 1; s > 0; s >>= 1) {
            if (lid < s) buf[lid] += buf[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        fp_t denom = buf[0];

        // 3) write
        if (idx < N) y[idx] = ex / denom;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// -----------------------------
// Activations
// -----------------------------
__kernel void activation_relu(__global const fp_t* x,
                              __global fp_t*       y,
                              const int N)
{
    int i = get_global_id(0);
    if (i < N) y[i] = fmax((fp_t)0, x[i]);
}
__kernel void activation_gelu(__global const fp_t* x,
                              __global fp_t*       y,
                              const int N)
{
    int i = get_global_id(0);
    if (i < N) {
        fp_t v = x[i];
        y[i] = (fp_t)0.5 * v * ((fp_t)1 + erf(v / (fp_t)1.41421356237));
    }
}

// -----------------------------
// LayerNorm on one row
// -----------------------------
__kernel void layernorm_basic(__global const fp_t* x,
                              __global fp_t*       y,
                              const int N, const fp_t eps)
{
    // One work-group computes layernorm for the whole row via two-pass tiling
    const int TILE = 1024;
    __local fp_t lbuf[1024];

    int lid   = get_local_id(0);
    int lsize = get_local_size(0);

    // ---- Pass 1: compute global mean/var ----
    fp_t priv_sum = (fp_t)0;
    fp_t priv_sumsq = (fp_t)0;

    for (int base = 0; base < N; base += TILE) {
        int idx = base + lid;
        fp_t v = (idx < N) ? x[idx] : (fp_t)0;
        priv_sum   += v;
        priv_sumsq += v * v;
    }

    // reduce private sums to group totals
    lbuf[lid] = priv_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) lbuf[lid] += lbuf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    fp_t sum = lbuf[0];

    lbuf[lid] = priv_sumsq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) lbuf[lid] += lbuf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    fp_t sumsq = lbuf[0];

    fp_t mean = sum / (fp_t)N;
    fp_t var  = sumsq / (fp_t)N - mean * mean;
    fp_t inv_std = rsqrt(var + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Pass 2: normalize and write ----
    for (int base = 0; base < N; base += TILE) {
        int idx = base + lid;
        if (idx < N) {
            fp_t v = x[idx];
            y[idx] = (v - mean) * inv_std;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}



// -----------------------------
// Histogram (256 bins) global vs shared
// -----------------------------
__kernel void histogram_global(__global const uint* data,
                               __global uint*       hist,
                               const int N)
{
    int gid = get_global_id(0);
    int gsz = get_global_size(0);
    for (int i=gid; i<N; i+=gsz) {
        uint v = data[i] & 255u;
        atomic_inc(&hist[v]);
    }
}
__kernel void histogram_shared(__global const uint* data,
                               __global uint*       hist,
                               const int N)
{
    __local uint lhist[256];
    int lid = get_local_id(0);
    for (int i=lid;i<256;i+=get_local_size(0)) lhist[i]=0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    int gid = get_global_id(0);
    int gsz = get_global_size(0);
    for (int i=gid;i<N;i+=gsz) {
        uint v = data[i] & 255u;
        atomic_inc(&lhist[v]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=lid;i<256;i+=get_local_size(0)) {
        uint c = lhist[i];
        if (c) atomic_add(&hist[i], c);
    }
}

/* ============================
   Templates for remaining ones
   ============================
   - depthwiseconv_global/tiled: per-channel 3x3 (H*W per channel)
   - stencil3d_global/shared   : 7-point 3D
   - fft1d_global/staged       : radix-2 butterfly (per stage in local mem)
   - sort_bitonic              : standard bitonic network in local mem
   - spmv_global               : already implemented
   - bfs_basic / dfs_basic     : level-synchronous frontier (host iterates)
   - pagerank_basic            : power iteration over CSR
   - attentioncore_*           : simplified QK^T softmax V (single head)
*/



// ===================== MatMul (Tiled Shared Memory) =====================
__kernel void matmul_shared(__global const fp_t* A,
                            __global const fp_t* B,
                            __global fp_t*       C,
                            int M, int N, int K)
{
    const int TS = 16; // tile size
    __local fp_t Asub[TS][TS];
    __local fp_t Bsub[TS][TS];

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    fp_t acc = (fp_t)0;
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t*TS + lx;
        int b_row = t*TS + ly;

        Asub[ly][lx] = (gy < M && a_col < K) ? A[gy*K + a_col] : (fp_t)0;
        Bsub[ly][lx] = (b_row < K && gx < N) ? B[b_row*N + gx] : (fp_t)0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
            acc += Asub[ly][k] * Bsub[k][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gy < M && gx < N)
        C[gy*N + gx] = acc;
}
