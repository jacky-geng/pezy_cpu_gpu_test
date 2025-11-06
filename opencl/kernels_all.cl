// =============================================================
// kernels_all.cl  (global-baseline + shared/tiled optimized variants)
// -------------------------------------------------------------
// Policy:
//  - "*_global": strictly naive implementations reading/writing
//    from/to global memory only. No local memory, no tiling.
//  - "*_shared"/"*_tiled": common, conservative optimizations
//    with local memory caching + barriers for data reuse.
//  - Numeric type: REAL (float) by default; define -DENABLE_FP64
//    to switch to double. The same kernel names work for FP32/FP64.
// =============================================================

#ifdef ENABLE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
typedef double REAL;
typedef double2 REAL2;
#define REAL_EXP exp
#define REAL_RSQRT rsqrt
#define REAL_ERF erf
#else
typedef float REAL;
typedef float2 REAL2;
#define REAL_EXP exp
#define REAL_RSQRT rsqrt
#define REAL_ERF erf
#endif

// ---------------- Common helpers ----------------
inline REAL2 cmul(REAL2 a, REAL2 b) {
  return (REAL2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline REAL rmax(REAL a, REAL b) { return a > b ? a : b; }

inline void atomic_add_real(__global REAL *addr, REAL val) {
#ifdef ENABLE_FP64
  // double atomic add (64-bit CAS loop)
  __global ulong *uaddr = (__global ulong *)addr;
  union {
    ulong u;
    REAL d;
  } oldv, newv;
  oldv.u = *uaddr;
  do {
    newv.d = as_double(oldv.u) + val;
  } while (atomic_cmpxchg(uaddr, oldv.u, as_ulong(newv.d)) != oldv.u);
#else
  // float atomic add (32-bit CAS loop)
  __global uint *uaddr = (__global uint *)addr;
  union {
    uint u;
    float f;
  } oldv, newv;
  oldv.u = *uaddr;
  do {
    newv.f = as_float(oldv.u) + val;
  } while (atomic_cmpxchg(uaddr, oldv.u, as_uint(newv.f)) != oldv.u);
#endif
}

// =============================================================
// 1) Linear Algebra
// =============================================================

// ---- vecadd_basic (global-only) ----
__kernel void vecadd_basic(__global const REAL *A, __global const REAL *B,
                           __global REAL *C, int N) {
  int gid = get_global_id(0);
  if (gid < N)
    C[gid] = A[gid] + B[gid];
}

// ---- dot_global (global-only) ----
__kernel void dot_global(__global const REAL *A, __global const REAL *B,
                         __global REAL *partial, int N) {
  int gid = get_global_id(0);
  int gsz = get_global_size(0);
  REAL s = (REAL)0;
  for (int i = gid; i < N; i += gsz)
    s += A[i] * B[i];
  // Each work-item writes its partial; host reduces (or run another kernel)
  partial[gid] = s;
}

// ---- dot_shared (WG reduction in local memory) ----
__kernel void dot_shared(__global const REAL *A, __global const REAL *B,
                         __global REAL *partial, int N) {
  __local REAL buf[256];
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int gsz = get_global_size(0);

  REAL s = (REAL)0;
  for (int i = gid; i < N; i += gsz)
    s += A[i] * B[i];

  buf[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      buf[lid] += buf[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partial[get_group_id(0)] = buf[0];
}

// ---- gemv_global (global-only) ----
__kernel void gemv_global(__global const REAL *A, __global const REAL *x,
                          __global REAL *y, int M, int N) {
  int row = get_global_id(0);
  if (row >= M)
    return;
  REAL acc = (REAL)0;
  for (int k = 0; k < N; k++)
    acc += A[row * N + k] * x[k];
  y[row] = acc;
}

// ---- gemv_shared (tile x into local memory) ----
__kernel void gemv_shared(__global const REAL *A, __global const REAL *x,
                          __global REAL *y, int M, int N) {
  int row = get_global_id(0);
  if (row >= M)
    return;

  __local REAL xTile[256];
  REAL acc = (REAL)0;
  for (int t = 0; t < N; t += 256) {
    int k = t + get_local_id(0);
    if (k < N)
      xTile[get_local_id(0)] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);
    int kend = min(t + 256, N);
    for (int kk = t; kk < kend; ++kk) {
      acc += A[row * N + kk] * xTile[kk - t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  y[row] = acc;
}

// ---- matmul_global (global-only) ----
__kernel void matmul_global(__global const REAL *A, __global const REAL *B,
                            __global REAL *C, int M, int N, int K) {
  int j = get_global_id(0); // col
  int i = get_global_id(1); // row
  if (i >= M || j >= N)
    return;
  REAL acc = (REAL)0;
  for (int k = 0; k < K; k++)
    acc += A[i * K + k] * B[k * N + j];
  C[i * N + j] = acc;
}

// ---- matmul_shared (tiled, local-memory cache) ----
__kernel void matmul_shared(__global const REAL *A, __global const REAL *B,
                            __global REAL *C, int M, int N, int K) {
  // Square tile size; 16 is conservative for most devices
  const int TS = 16;
  __local REAL As[TS][TS];
  __local REAL Bs[TS][TS];

  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int j = get_group_id(0) * TS + tx; // col
  int i = get_group_id(1) * TS + ty; // row

  REAL acc = (REAL)0;
  for (int t = 0; t < K; t += TS) {
    // Load tiles (with bounds checks)
    int a_col = t + tx;
    int b_row = t + ty;
    As[ty][tx] = (i < M && a_col < K) ? A[i * K + a_col] : (REAL)0;
    Bs[ty][tx] = (b_row < K && j < N) ? B[b_row * N + j] : (REAL)0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute partial for this tile
    for (int k = 0; k < TS; k++)
      acc += As[ty][k] * Bs[k][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (i < M && j < N)
    C[i * N + j] = acc;
}

// ---- reduction_global (global-only partials) ----
__kernel void reduction_global(__global const REAL *x, __global REAL *partial,
                               int N) {
  int gid = get_global_id(0);
  int gsz = get_global_size(0);
  REAL s = (REAL)0;
  for (int i = gid; i < N; i += gsz)
    s += x[i];
  partial[gid] = s;
}

// ---- reduction_shared (WG reduction) ----
__kernel void reduction_shared(__global const REAL *x, __global REAL *partial,
                               int N) {
  __local REAL buf[256];
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int gsz = get_global_size(0);

  REAL s = (REAL)0;
  for (int i = gid; i < N; i += gsz)
    s += x[i];
  buf[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      buf[lid] += buf[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partial[get_group_id(0)] = buf[0];
}

// ---- scan_shared (per-block inclusive scan; block-local only) ----
__kernel void scan_shared(__global const REAL *in, __global REAL *out, int N) {
  __local REAL temp[256];
  int gid = get_global_id(0), lid = get_local_id(0);
  int gsz = get_global_size(0);

  for (int i = gid; i < N; i += gsz) {
    temp[lid] = in[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Hillis-Steele inclusive within the WG
    for (int s = 1; s < get_local_size(0); s <<= 1) {
      REAL t = temp[lid];
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid >= s)
        t += temp[lid - s];
      barrier(CLK_LOCAL_MEM_FENCE);
      temp[lid] = t;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    out[i] = temp[lid];
  }
}

// ---- spmv_csr (global-only) ----
__kernel void spmv_csr(__global const int *rowptr, __global const int *colind,
                       __global const REAL *values, __global const REAL *x,
                       __global REAL *y, int M) {
  int row = get_global_id(0);
  if (row >= M)
    return;
  int beg = rowptr[row], end = rowptr[row + 1];
  REAL acc = (REAL)0;
  for (int jj = beg; jj < end; ++jj) {
    acc += values[jj] * x[colind[jj]];
  }
  y[row] = acc;
}

// =============================================================
// 2) ML Core Ops
// =============================================================

// ---- conv2d_global (3x3 SAME, global-only) ----
__kernel void conv2d_global(__global const REAL *img, __global const REAL *k3,
                            __global REAL *out, int H, int W) {
  int x = get_global_id(0), y = get_global_id(1);
  if (x >= W || y >= H)
    return;
  REAL acc = (REAL)0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      int yy = y + dy, xx = x + dx;
      REAL v =
          (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : (REAL)0;
      acc += v * k3[(dy + 1) * 3 + (dx + 1)];
    }
  out[y * W + x] = acc;
}

// ---- conv2d_shared (3x3 SAME with local tile + halo) ----
__kernel void conv2d_shared(__global const REAL *img, __global const REAL *k3,
                            __global REAL *out, int H, int W) {
  const int TS = 16;           // compute tile
  const int R = 1;             // radius for 3x3
  const int LDSW = TS + 2 * R; // shared width/height
  __local REAL tile[LDSW][LDSW];

  int gx = get_group_id(0) * TS + get_local_id(0);
  int gy = get_group_id(1) * TS + get_local_id(1);
  int lx = get_local_id(0) + R;
  int ly = get_local_id(1) + R;

  // Load center
  REAL center = (gx < W && gy < H) ? img[gy * W + gx] : (REAL)0;
  tile[ly][lx] = center;

  // Load halo in X
  if (get_local_id(0) < R) {
    int gxl = gx - R;
    int gxr = gx + TS;
    tile[ly][lx - R] = (gxl >= 0 && gy < H) ? img[gy * W + gxl] : (REAL)0;
    tile[ly][lx + TS] = (gxr < W && gy < H) ? img[gy * W + gxr] : (REAL)0;
  }
  // Load halo in Y
  if (get_local_id(1) < R) {
    int gyt = gy - R;
    int gyb = gy + TS;
    tile[ly - R][lx] = (gx < W && gyt >= 0) ? img[gyt * W + gx] : (REAL)0;
    tile[ly + TS][lx] = (gx < W && gyb < H) ? img[gyb * W + gx] : (REAL)0;
  }
  // Load corners if within first R in both axes
  if (get_local_id(0) < R && get_local_id(1) < R) {
    int gxl = gx - R, gxr = gx + TS;
    int gyt = gy - R, gyb = gy + TS;
    tile[ly - R][lx - R] =
        (gxl >= 0 && gyt >= 0) ? img[gyt * W + gxl] : (REAL)0;
    tile[ly - R][lx + TS] =
        (gxr < W && gyt >= 0) ? img[gyt * W + gxr] : (REAL)0;
    tile[ly + TS][lx - R] =
        (gxl >= 0 && gyb < H) ? img[gyb * W + gxl] : (REAL)0;
    tile[ly + TS][lx + TS] =
        (gxr < W && gyb < H) ? img[gyb * W + gxr] : (REAL)0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (gx < W && gy < H) {
    REAL acc = (REAL)0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++)
#pragma unroll
      for (int dx = -1; dx <= 1; dx++) {
        acc += tile[ly + dy][lx + dx] * k3[(dy + 1) * 3 + (dx + 1)];
      }
    out[gy * W + gx] = acc;
  }
}

// ---- depthwiseconv_global (3x3, global-only) ----
__kernel void depthwiseconv_global(__global const REAL *img,
                                   __global const REAL *k3, __global REAL *out,
                                   int C, int H, int W) {
  int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
  if (x >= W || y >= H || c >= C)
    return;
  REAL acc = (REAL)0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      int yy = y + dy, xx = x + dx;
      REAL v = (yy >= 0 && yy < H && xx >= 0 && xx < W)
                   ? img[c * H * W + yy * W + xx]
                   : (REAL)0;
      acc += v * k3[c * 9 + (dy + 1) * 3 + (dx + 1)];
    }
  out[c * H * W + y * W + x] = acc;
}

// ---- depthwiseconv_tiled (local tile per channel) ----
__kernel void depthwiseconv_tiled(__global const REAL *img,
                                  __global const REAL *k3, __global REAL *out,
                                  int C, int H, int W) {
  const int TS = 8, R = 1;
  const int LDSW = TS + 2 * R;
  __local REAL tile[LDSW][LDSW];

  int c = get_global_id(2);
  int gx = get_group_id(0) * TS + get_local_id(0);
  int gy = get_group_id(1) * TS + get_local_id(1);
  int lx = get_local_id(0) + R;
  int ly = get_local_id(1) + R;

  if (c >= C)
    return;

#define IMG(c, y, x) img[(c) * H * W + (y) * W + (x)]

  // Center
  REAL center = (gx < W && gy < H) ? IMG(c, gy, gx) : (REAL)0;
  tile[ly][lx] = center;

  // Halo loads
  if (get_local_id(0) < R) {
    int gxl = gx - R, gxr = gx + TS;
    tile[ly][lx - R] = (gxl >= 0 && gy < H) ? IMG(c, gy, gxl) : (REAL)0;
    tile[ly][lx + TS] = (gxr < W && gy < H) ? IMG(c, gy, gxr) : (REAL)0;
  }
  if (get_local_id(1) < R) {
    int gyt = gy - R, gyb = gy + TS;
    tile[ly - R][lx] = (gx < W && gyt >= 0) ? IMG(c, gyt, gx) : (REAL)0;
    tile[ly + TS][lx] = (gx < W && gyb < H) ? IMG(c, gyb, gx) : (REAL)0;
  }
  if (get_local_id(0) < R && get_local_id(1) < R) {
    int gxl = gx - R, gxr = gx + TS;
    int gyt = gy - R, gyb = gy + TS;
    tile[ly - R][lx - R] = (gxl >= 0 && gyt >= 0) ? IMG(c, gyt, gxl) : (REAL)0;
    tile[ly - R][lx + TS] = (gxr < W && gyt >= 0) ? IMG(c, gyt, gxr) : (REAL)0;
    tile[ly + TS][lx - R] = (gxl >= 0 && gyb < H) ? IMG(c, gyb, gxl) : (REAL)0;
    tile[ly + TS][lx + TS] = (gxr < W && gyb < H) ? IMG(c, gyb, gxr) : (REAL)0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (gx < W && gy < H) {
    REAL acc = (REAL)0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++)
#pragma unroll
      for (int dx = -1; dx <= 1; dx++) {
        acc += tile[ly + dy][lx + dx] * k3[c * 9 + (dy + 1) * 3 + (dx + 1)];
      }
    out[c * H * W + gy * W + gx] = acc;
  }
#undef IMG
}

// ---- softmax_basic (single row; block-local) ----
__kernel void softmax_basic(__global const REAL *x, __global REAL *y, int N) {
  __local REAL buf[256];
  int gid = get_global_id(0), lid = get_local_id(0), lsize = get_local_size(0);
  REAL v = (gid < N) ? x[gid] : (REAL)(-1.0 / 0.0); // -inf
  buf[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      buf[lid] = rmax(buf[lid], buf[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  REAL m = buf[0];
  REAL e = (gid < N) ? REAL_EXP(v - m) : (REAL)0;
  buf[lid] = e;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      buf[lid] += buf[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  REAL sum = buf[0];
  if (gid < N)
    y[gid] = e / sum;
}

// ---- layernorm_basic (per-row; expect single work-group for N) ----
__kernel void layernorm_basic(__global const REAL *x, __global REAL *y, int N,
                              REAL eps) {
  __local REAL buf0[256], buf1[256];
  int gid = get_global_id(0), lid = get_local_id(0), lsize = get_local_size(0);
  REAL v = (gid < N) ? x[gid] : (REAL)0;
  buf0[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      buf0[lid] += buf0[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  REAL mean = buf0[0] / (REAL)N;

  REAL dv = v - mean;
  buf1[lid] = (gid < N) ? dv * dv : (REAL)0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      buf1[lid] += buf1[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  REAL var = buf1[0] / (REAL)N;
  REAL inv = REAL_RSQRT(var + eps);

  if (gid < N)
    y[gid] = (x[gid] - mean) * inv;
}

// ---- activation_relu (global-only) ----
__kernel void activation_relu(__global const REAL *x, __global REAL *y, int N) {
  int gid = get_global_id(0);
  if (gid < N)
    y[gid] = x[gid] > (REAL)0 ? x[gid] : (REAL)0;
}

// ---- activation_gelu (global-only, exact erf form) ----
__kernel void activation_gelu(__global const REAL *x, __global REAL *y, int N) {
  int gid = get_global_id(0);
  if (gid < N) {
    REAL v = x[gid];
    y[gid] = (REAL)0.5 * v * ((REAL)1 + REAL_ERF(v * (REAL)0.7071067811865475));
  }
}

// =============================================================
// 3) Graph / Irregular
// =============================================================

// ---- bfs_basic (one relax step) ----
__kernel void bfs_basic(__global const int *rowptr, __global const int *colind,
                        __global const uint *frontier,
                        __global uint *next_frontier, __global uint *visited,
                        int V) {
  int v = get_global_id(0);
  if (v >= V)
    return;
  if (!frontier[v])
    return;
  int beg = rowptr[v], end = rowptr[v + 1];
  for (int jj = beg; jj < end; ++jj) {
    int u = colind[jj];
    if (atomic_xchg(&visited[u], 1u) == 0u) {
      next_frontier[u] = 1u;
    }
  }
}

// ---- dfs_basic (reachability step; baseline equals bfs step) ----
__kernel void dfs_basic(__global const int *rowptr, __global const int *colind,
                        __global const uint *frontier,
                        __global uint *next_frontier, __global uint *visited,
                        int V) {
  bfs_basic(rowptr, colind, frontier, next_frontier, visited, V);
}

// ---- pagerank_basic (push-style; atomic add to pr_next) ----
__kernel void pagerank_basic(__global const int *rowptr,
                             __global const int *colind,
                             __global const int *outdeg,
                             __global const REAL *pr, __global REAL *pr_next,
                             REAL d, int V) {
  int v = get_global_id(0);
  if (v >= V)
    return;
  int deg = max(outdeg[v], 1);
  REAL contrib = d * pr[v] / (REAL)deg;
  int beg = rowptr[v], end = rowptr[v + 1];
  for (int jj = beg; jj < end; ++jj) {
    int u = colind[jj];
    atomic_add_real(&pr_next[u], contrib);
  }
}

// =============================================================
// 4) Numerical / Physics
// =============================================================

// ---- stencil2d_3x3 (global-only mean filter) ----
__kernel void stencil2d_3x3(__global const REAL *in, __global REAL *out, int H,
                            int W) {
  int x = get_global_id(0), y = get_global_id(1);
  if (x >= W || y >= H)
    return;
  REAL acc = (REAL)0;
  int cnt = 0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      int yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
        acc += in[yy * W + xx];
        cnt++;
      }
    }
  out[y * W + x] = acc / (REAL)cnt;
}

// ---- stencil2d_5x5 (global-only mean filter) ----
__kernel void stencil2d_5x5(__global const REAL *in, __global REAL *out, int H,
                            int W) {
  int x = get_global_id(0), y = get_global_id(1);
  if (x >= W || y >= H)
    return;
  REAL acc = (REAL)0;
  int cnt = 0;
  for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
      int yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
        acc += in[yy * W + xx];
        cnt++;
      }
    }
  out[y * W + x] = acc / (REAL)cnt;
}

// ---- stencil3d_global (7-point; global-only) ----
__kernel void stencil3d_global(__global const REAL *in, __global REAL *out,
                               int D, int H, int W) {
  int x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  if (x >= W || y >= H || z >= D)
    return;
  int idx = z * H * W + y * W + x;
  REAL acc = in[idx] * (REAL)0.5;
  if (x > 0)
    acc += (REAL)0.0833 * in[idx - 1];
  if (x + 1 < W)
    acc += (REAL)0.0833 * in[idx + 1];
  if (y > 0)
    acc += (REAL)0.0833 * in[idx - W];
  if (y + 1 < H)
    acc += (REAL)0.0833 * in[idx + W];
  if (z > 0)
    acc += (REAL)0.0833 * in[idx - H * W];
  if (z + 1 < D)
    acc += (REAL)0.0833 * in[idx + H * W];
  out[idx] = acc;
}

// ---- stencil3d_shared (cache a 2D tile of current slice) ----
__kernel void stencil3d_shared(__global const REAL *in, __global REAL *out,
                               int D, int H, int W) {
  const int TS = 8;
  __local REAL tile[(TS + 2)][(TS + 2)];
  int xg = get_group_id(0) * TS + get_local_id(0);
  int yg = get_group_id(1) * TS + get_local_id(1);
  int z = get_global_id(2);
  if (z >= D)
    return;

  int lx = get_local_id(0) + 1, ly = get_local_id(1) + 1;
  REAL center = (xg < W && yg < H) ? in[z * H * W + yg * W + xg] : (REAL)0;
  tile[ly][lx] = center;

  // halo X
  if (get_local_id(0) == 0) {
    tile[ly][0] =
        (xg > 0 && yg < H) ? in[z * H * W + yg * W + (xg - 1)] : (REAL)0;
  }
  if (get_local_id(0) == TS - 1) {
    tile[ly][TS + 1] =
        (xg + 1 < W && yg < H) ? in[z * H * W + yg * W + (xg + 1)] : (REAL)0;
  }
  // halo Y
  if (get_local_id(1) == 0) {
    tile[0][lx] =
        (yg > 0 && xg < W) ? in[z * H * W + (yg - 1) * W + xg] : (REAL)0;
  }
  if (get_local_id(1) == TS - 1) {
    tile[TS + 1][lx] =
        (yg + 1 < H && xg < W) ? in[z * H * W + (yg + 1) * W + xg] : (REAL)0;
  }
  // corners (simplified)
  if (get_local_id(0) == 0 && get_local_id(1) == 0) {
    tile[0][0] =
        (xg > 0 && yg > 0) ? in[z * H * W + (yg - 1) * W + (xg - 1)] : (REAL)0;
  }
  if (get_local_id(0) == TS - 1 && get_local_id(1) == 0) {
    tile[0][TS + 1] = (xg + 1 < W && yg > 0)
                          ? in[z * H * W + (yg - 1) * W + (xg + 1)]
                          : (REAL)0;
  }
  if (get_local_id(0) == 0 && get_local_id(1) == TS - 1) {
    tile[TS + 1][0] = (xg > 0 && yg + 1 < H)
                          ? in[z * H * W + (yg + 1) * W + (xg - 1)]
                          : (REAL)0;
  }
  if (get_local_id(0) == TS - 1 && get_local_id(1) == TS - 1) {
    tile[TS + 1][TS + 1] = (xg + 1 < W && yg + 1 < H)
                               ? in[z * H * W + (yg + 1) * W + (xg + 1)]
                               : (REAL)0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (xg < W && yg < H) {
    int idx = z * H * W + yg * W + xg;
    REAL acc = tile[ly][lx] * (REAL)0.5;
    acc += (REAL)0.0833 * tile[ly][lx - 1];
    acc += (REAL)0.0833 * tile[ly][lx + 1];
    acc += (REAL)0.0833 * tile[ly - 1][lx];
    acc += (REAL)0.0833 * tile[ly + 1][lx];
    // z neighbors from global (no z-tiling to keep LDS usage low)
    if (z > 0)
      acc += (REAL)0.0833 * in[idx - H * W];
    if (z + 1 < D)
      acc += (REAL)0.0833 * in[idx + H * W];
    out[idx] = acc;
  }
}

// ---- histogram_global (global-only atomics to shared hist) ----
__kernel void histogram_global(__global const uint *data, __global uint *hist,
                               int N) {
  int gid = get_global_id(0), gsz = get_global_size(0);
  for (int i = gid; i < N; i += gsz) {
    uint bin = data[i] & 255u;
    atomic_inc(&hist[bin]);
  }
}

// ---- histogram_shared (per-WG local bins -> global merge) ----
__kernel void histogram_shared(__global const uint *data, __global uint *hist,
                               int N) {
  __local uint bins[256];
  int lid = get_local_id(0);
  for (int i = lid; i < 256; i += get_local_size(0))
    bins[i] = 0u;
  barrier(CLK_LOCAL_MEM_FENCE);

  int gid = get_global_id(0), gsz = get_global_size(0);
  for (int i = gid; i < N; i += gsz) {
    uint bin = data[i] & 255u;
    atomic_inc(&bins[bin]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lid; i < 256; i += get_local_size(0)) {
    uint v = bins[i];
    if (v)
      atomic_add(&hist[i], v);
  }
}

// ---- sort_bitonic (one step kernel) ----
__kernel void sort_bitonic(__global uint *data, int N, int j, int k) {
  int i = get_global_id(0);
  int ix = i << 1;
  if (ix + 1 >= N)
    return;
  int ixj = ix ^ j;
  if (ixj > ix) {
    bool up = ((ix & k) == 0);
    uint a = data[ix];
    uint b = data[ixj];
    bool cond = (up ? (a > b) : (a < b));
    if (cond) {
      data[ix] = b;
      data[ixj] = a;
    }
  }
}

// ---- montecarlo_basic (pi estimation; global-only) ----
__kernel void montecarlo_basic(__global const REAL *xy, __global uint *partial,
                               int N) {
  int gid = get_global_id(0);
  int gsz = get_global_size(0);
  uint c = 0;
  for (int i = gid; i < N; i += gsz) {
    REAL x = xy[2 * i], y = xy[2 * i + 1];
    REAL d = x * x + y * y;
    if (d <= (REAL)1)
      c++;
  }
  partial[gid] = c;
}

// ---- fft1d_global / fft1d_staged (per-stage butterfly) ----
__kernel void fft1d_global(__global REAL *data, int N,
                           int mh) // mh = m/2 for current stage
{
  int i = get_global_id(0);
  int k = i / mh;
  int j = i % mh;
  int m = mh << 1;
  int idx1 = k * m + j;
  int idx2 = idx1 + mh;
  if (2 * idx2 + 1 >= 2 * N)
    return;

  REAL theta = (REAL)(-3.141592653589793) * (REAL)j / (REAL)mh;
  REAL2 w = (REAL2)(cos(theta), sin(theta));

  REAL2 a = (REAL2)(data[2 * idx1], data[2 * idx1 + 1]);
  REAL2 b = (REAL2)(data[2 * idx2], data[2 * idx2 + 1]);
  REAL2 t = cmul(w, b);
  REAL2 u = (REAL2)(a.x + t.x, a.y + t.y);
  REAL2 v = (REAL2)(a.x - t.x, a.y - t.y);
  data[2 * idx1] = u.x;
  data[2 * idx1 + 1] = u.y;
  data[2 * idx2] = v.x;
  data[2 * idx2 + 1] = v.y;
}
__kernel void fft1d_staged(__global REAL *data, int N, int mh) {
  fft1d_global(data, N, mh);
}

// ---- smithwaterman_basic (per-thread pair alignment) ----
#define SW_MAX_LEN 256

__kernel void smithwaterman_basic(__global const uchar *seqA,
                                  __global const uchar *seqB,
                                  __global int *scores,
                                  int num_pairs,
                                  int len,
                                  int match_score,
                                  int mismatch_score,
                                  int gap_score) {
  int tid = get_global_id(0);
  if (tid >= num_pairs)
    return;
  if (len > SW_MAX_LEN)
    len = SW_MAX_LEN;

  __global const uchar *a = seqA + (size_t)tid * len;
  __global const uchar *b = seqB + (size_t)tid * len;
  int prev[SW_MAX_LEN + 1];
  int curr[SW_MAX_LEN + 1];
  for (int j = 0; j <= len; ++j)
    prev[j] = 0;
  int best = 0;
  for (int i = 1; i <= len; ++i) {
    curr[0] = 0;
    uchar ai = a[i - 1];
    for (int j = 1; j <= len; ++j) {
      int diag = prev[j - 1] + (ai == b[j - 1] ? match_score : mismatch_score);
      int up = prev[j] + gap_score;
      int left = curr[j - 1] + gap_score;
      int val = diag;
      if (up > val)
        val = up;
      if (left > val)
        val = left;
      if (val < 0)
        val = 0;
      curr[j] = val;
      if (val > best)
        best = val;
    }
    for (int j = 0; j <= len; ++j)
      prev[j] = curr[j];
  }
  scores[tid] = best;
}

__kernel void smithwaterman_wavefront(__global const uchar *seqA,
                                      __global const uchar *seqB,
                                      __global int *scores,
                                      int num_pairs,
                                      int len,
                                      int match_score,
                                      int mismatch_score,
                                      int gap_score) {
  int pair = get_group_id(0);
  if (pair >= num_pairs)
    return;

  int lsize = get_local_size(0);
  int lid = get_local_id(0);

  const int stride = len;
  if (len > SW_MAX_LEN)
    len = SW_MAX_LEN;

  __global const uchar *a = seqA + (size_t)pair * stride;
  __global const uchar *b = seqB + (size_t)pair * stride;

  __local int diag_prev2[SW_MAX_LEN + 1];
  __local int diag_prev[SW_MAX_LEN + 1];
  __local int diag_curr[SW_MAX_LEN + 1];
  __local int best_buf[SW_MAX_LEN];

  for (int idx = lid; idx <= len; idx += lsize) {
    diag_prev2[idx] = 0;
    diag_prev[idx] = 0;
    diag_curr[idx] = 0;
  }
  if (lid < SW_MAX_LEN)
    best_buf[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  int best_local = 0;
  int max_diag = 2 * len;
  for (int diag = 1; diag <= max_diag; ++diag) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int i_start = max(1, diag - len);
    int i_end = min(len, diag);

    int i = i_start + lid;
    if (i <= i_end) {
      int j = diag - i;
      int score = 0;
      if (j >= 1 && j <= len) {
        int diag_val = diag_prev2[i - 1];
        int up_val = diag_prev[i - 1];
        int left_val = diag_prev[i];
        int match = (a[i - 1] == b[j - 1]) ? match_score : mismatch_score;
        score = diag_val + match;
        score = max(score, up_val + gap_score);
        score = max(score, left_val + gap_score);
        if (score < 0)
          score = 0;
      }
      diag_curr[i] = score;
      if (score > best_local)
        best_local = score;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int idx = lid; idx <= len; idx += lsize) {
      diag_prev2[idx] = diag_prev[idx];
      diag_prev[idx] = diag_curr[idx];
      diag_curr[idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  best_buf[lid] = best_local;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
    if (lid < stride)
      best_buf[lid] = max(best_buf[lid], best_buf[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    scores[pair] = best_buf[0];
}

// ---- wfa_editdistance (per-thread wavefront edit distance) ----
#define WFA_MAX_LEN 256
#define WFA_DIAG_COUNT (2 * WFA_MAX_LEN + 1)

__kernel void wfa_editdistance(__global const uchar *seqA,
                               __global const uchar *seqB,
                               __global int *distances,
                               int num_pairs,
                               int len) {
  const int NEG_INF = -1000000;
  int tid = get_global_id(0);
  if (tid >= num_pairs)
    return;
  if (len > WFA_MAX_LEN)
    len = WFA_MAX_LEN;

  __global const uchar *a = seqA + (size_t)tid * len;
  __global const uchar *b = seqB + (size_t)tid * len;

  int prev[WFA_DIAG_COUNT];
  int curr[WFA_DIAG_COUNT];
  for (int i = 0; i < WFA_DIAG_COUNT; ++i) {
    prev[i] = NEG_INF;
    curr[i] = NEG_INF;
  }

  const int center = WFA_MAX_LEN;
  int offset = 0;
  while (offset < len && a[offset] == b[offset])
    ++offset;
  if (offset >= len) {
    distances[tid] = 0;
    return;
  }
  prev[center] = offset;

  int max_dist = len * 2;
  for (int dist = 1; dist <= max_dist; ++dist) {
    int diag_min = -dist;
    int diag_max = dist;
    for (int diag = diag_min; diag <= diag_max; ++diag) {
      int idx = center + diag;
      int best = NEG_INF;
      if (idx - 1 >= 0) {
        int cand = prev[idx - 1] + 1;
        if (cand > best)
          best = cand;
      }
      if (idx >= 0 && idx < WFA_DIAG_COUNT) {
        int cand = prev[idx] + 1;
        if (cand > best)
          best = cand;
      }
      if (idx + 1 < WFA_DIAG_COUNT) {
        int cand = prev[idx + 1];
        if (cand > best)
          best = cand;
      }
      if (best < 0)
        best = 0;
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
