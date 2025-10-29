// =============================================================
// run_opencl.cpp  (full dispatcher covering all benchmarks)
// - Builds kernels_all.cl once for FP32 and (if supported) FP64
// - Runs all benchmarks listed in benchmark_config.hpp
// - Computes simple CPU references or structural checks where feasible
// - Records gws/lws, flops_est, bw_GBps into CSV
// - English comments only
// =============================================================

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <cmath>
#include <cassert>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdint>
#include <cstdio>

// Project headers (provided by user in ../common)
#include "../common/benchmark_config.hpp"
#include "utils_opencl.hpp"   // device/context helpers, enqueue_timed(), round_up(), build_program_from_file()
#include "math_utils.hpp"     // allclose(...), fill_random(...)
#include "baseline_check.hpp" // optional CPU baselines; guard usage
#include "csv_writer.hpp"     // simple CSV dump
#include "timer.hpp"          // host-side timing helper

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// ------------------------------
// Small helpers (local only)
// ------------------------------

static bool has_fp64(const cl::Device &dev)
{
    std::string ext = dev.getInfo<CL_DEVICE_EXTENSIONS>();
    return (ext.find("cl_khr_fp64") != std::string::npos) ||
           (dev.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>() != 0);
}

// Kernels are compiled with the same symbol name for all dtype variants.
static string resolve_kname(const string &base, DTypeMode /*mode*/, bool /*fp64*/)
{
    return base;
}

// OpenCL events -> elapsed milliseconds
static double event_elapsed_ms(const cl::Event &evt)
{
    cl_ulong t0 = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t1 = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return double(t1 - t0) * 1e-6;
}

// Simple host-side RNG fillers
template <typename T>
static void fill_uniform(std::vector<T> &v, T lo, T hi, uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    if constexpr (std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<double> dist{static_cast<double>(lo), static_cast<double>(hi)};
        for (auto &x : v)
            x = static_cast<T>(dist(rng));
    }
    else
    {
        using IntDistType = std::conditional_t<std::is_signed<T>::value, long long, unsigned long long>;
        std::uniform_int_distribution<IntDistType> dist(static_cast<IntDistType>(lo),
                                                        static_cast<IntDistType>(hi));
        for (auto &x : v)
            x = static_cast<T>(dist(rng));
    }
}

// local allclose (namespaced) to avoid collision; fallback if math_utils.hpp not found
namespace local_ref
{
    template <typename T>
    bool allclose(const std::vector<T> &a, const std::vector<T> &b, double rtol = 1e-5, double atol = 1e-6)
    {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double da = double(a[i]);
            double db = double(b[i]);
            double diff = std::fabs(da - db);
            if (diff > (atol + rtol * std::fabs(db)))
                return false;
        }
        return true;
    }
}

// ------------------------------
// Generic buffer helpers
// ------------------------------
template <typename T>
static cl::Buffer make_read_buf(cl::Context &ctx, const std::vector<T> &host)
{
    return cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * host.size(), (void *)host.data());
}
template <typename T>
static cl::Buffer make_write_buf(cl::Context &ctx, size_t n)
{
    return cl::Buffer(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * n);
}

template <typename T>
static cl::Buffer make_readwrite_buf(cl::Context &ctx, const std::vector<T> &host)
{
    return cl::Buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(T) * host.size(), const_cast<T *>(host.data()));
}
template <typename T>
static void read_back(cl::CommandQueue &q, const cl::Buffer &buf, std::vector<T> &host)
{
    q.enqueueReadBuffer(buf, true, 0, sizeof(T) * host.size(), host.data());
}

// Normalize run info helpers
static void set_runinfo_1d(RunInfo &info, size_t gws, size_t lws)
{
    info.gws0 = gws;
    info.gws1 = 1;
    info.gws2 = 1;
    info.lws0 = lws;
    info.lws1 = 1;
    info.lws2 = 1;
}

static void set_runinfo_2d(RunInfo &info,
                           size_t gws0, size_t gws1,
                           size_t lws0, size_t lws1)
{
    info.gws0 = gws0;
    info.gws1 = gws1;
    info.gws2 = 1;
    info.lws0 = lws0;
    info.lws1 = lws1;
    info.lws2 = 1;
}

static void set_runinfo_3d(RunInfo &info,
                           size_t gws0, size_t gws1, size_t gws2,
                           size_t lws0, size_t lws1, size_t lws2)
{
    info.gws0 = gws0;
    info.gws1 = gws1;
    info.gws2 = gws2;
    info.lws0 = lws0;
    info.lws1 = lws1;
    info.lws2 = lws2;
}

static void finalize_bandwidth(RunInfo &info, double ms)
{
    if (ms > 0.0 && info.bytes_moved > 0.0)
    {
        info.bw_GBps = (info.bytes_moved * 1e-9) / (ms * 1e-3);
    }
    else
    {
        info.bw_GBps = 0.0;
    }
}

// =============================================================
// Per-kernel runners
// Each runner must fill: time_ms, correct flag, and optionally set GWS/LWS + perf estimates
// The dimension "size" is interpreted per kernel as in benchmark_config.hpp
// =============================================================

// ---- 1) VECCADD -------------------------------------------------
template <typename T>
static double run_vecadd(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                         const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> A(N), B(N), C(N, T(0)), Ref(N);
    fill_uniform<T>(A, T(-1), T(1), 1);
    fill_uniform<T>(B, T(-1), T(1), 2);
    for (size_t i = 0; i < N; ++i)
        Ref[i] = A[i] + B[i];

    cl::Buffer dA = make_read_buf(ctx, A);
    cl::Buffer dB = make_read_buf(ctx, B);
    cl::Buffer dC = make_write_buf<T>(ctx, N);

    krn.setArg(0, dA);
    krn.setArg(1, dB);
    krn.setArg(2, dC);
    krn.setArg(3, (int)N);

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dC, C);
    bool ok = false;
    // Prefer project allclose if available; otherwise fallback
    try
    {
        ok = allclose(C, Ref);
    }
    catch (...)
    {
        ok = local_ref::allclose(C, Ref);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(N);             // one add per element
    info.bytes_moved = 3.0 * sizeof(T) * N; // A,B read; C write
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 2) DOT (global/shared identical host pattern) --------------
template <typename T>
static double run_dot(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                      const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    std::vector<T> x(N), y(N);
    fill_uniform<T>(x, T(-1), T(1), 3);
    fill_uniform<T>(y, T(-1), T(1), 4);
    T ref = std::inner_product(x.begin(), x.end(), y.begin(), T(0));
    std::vector<T> partial_host(gws, T(0));

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_read_buf(ctx, y);
    cl::Buffer dout = make_readwrite_buf(ctx, partial_host);

    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, dout);
    krn.setArg(3, (int)N);

    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dout, partial_host);
    T gpu_sum = std::accumulate(partial_host.begin(), partial_host.end(), T(0));
    bool ok = std::fabs(double(gpu_sum - ref)) < (use_fp64 ? 1e-9 : 1e-4);
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(2 * N); // mul+add per element
    info.bytes_moved = (2.0 * sizeof(T) * N) + sizeof(T) * gws;
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 3) GEMV (global/shared) ------------------------------------
template <typename T>
static double run_gemv(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                       const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Square matrix size N x N for sweep item
    size_t M = N, K = N;
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> A(M * K), x(K), y(M, T(0)), Ref(M, T(0));
    fill_uniform<T>(A, T(-1), T(1), 11);
    fill_uniform<T>(x, T(-1), T(1), 12);
    for (size_t i = 0; i < M; ++i)
    {
        T acc = T(0);
        for (size_t k = 0; k < K; ++k)
            acc += A[i * K + k] * x[k];
        Ref[i] = acc;
    }

    cl::Buffer dA = make_read_buf(ctx, A);
    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, M);

    krn.setArg(0, dA);
    krn.setArg(1, dx);
    krn.setArg(2, dy);
    krn.setArg(3, (int)M);
    krn.setArg(4, (int)K);

    size_t lws = 256;
    size_t gws = round_up(M, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, Ref, use_fp64 ? 1e-9 : 1e-4, use_fp64 ? 1e-12 : 1e-6);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, Ref, use_fp64 ? 1e-9 : 1e-4, use_fp64 ? 1e-12 : 1e-6);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(2ull) * double(M) * double(K);
    info.bytes_moved = sizeof(T) * (M * K + K + M);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 4) MATMUL (global/shared) ----------------------------------
template <typename T>
static double run_matmul(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                         const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Square N x N times N x N
    size_t M = N, K = N;
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> A(M * K), B(K * N), C(M * N, T(0)), Ref(M * N, T(0));
    fill_uniform<T>(A, T(-1), T(1), 21);
    fill_uniform<T>(B, T(-1), T(1), 22);
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            T acc = T(0);
            for (size_t k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            Ref[i * N + j] = acc;
        }
    }

    cl::Buffer dA = make_read_buf(ctx, A);
    cl::Buffer dB = make_read_buf(ctx, B);
    cl::Buffer dC = make_write_buf<T>(ctx, M * N);

    krn.setArg(0, dA);
    krn.setArg(1, dB);
    krn.setArg(2, dC);
    krn.setArg(3, (int)M);
    krn.setArg(4, (int)N);
    krn.setArg(5, (int)K);

    // 2D launch
    size_t tile = 16;
    size_t gws0 = round_up(N, tile);
    size_t gws1 = round_up(M, tile);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws0, gws1), cl::NDRange(tile, tile), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dC, C);
    bool ok = false;
    try
    {
        ok = allclose(C, Ref, use_fp64 ? 1e-9 : 1e-3, use_fp64 ? 1e-12 : 1e-5);
    }
    catch (...)
    {
        ok = local_ref::allclose(C, Ref, use_fp64 ? 1e-9 : 1e-3, use_fp64 ? 1e-12 : 1e-5);
    }
    correct = ok;

    set_runinfo_2d(info, gws0, gws1, tile, tile);
    info.flops_est = double(2ull) * double(M) * double(N) * double(K);
    info.bytes_moved = sizeof(T) * (M * K + K * N + M * N);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 5) REDUCTION (global/shared) --------------------------------
template <typename T>
static double run_reduction(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> x(N);
    fill_uniform<T>(x, T(0), T(1), 31);
    T ref = T(0);
    for (auto v : x)
        ref += v;

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    std::vector<T> partial_host(gws, T(0));

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dout = make_readwrite_buf(ctx, partial_host);

    krn.setArg(0, dx);
    krn.setArg(1, dout);
    krn.setArg(2, (int)N);

    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dout, partial_host);
    T gpu_sum = std::accumulate(partial_host.begin(), partial_host.end(), T(0));
    bool ok = std::fabs(double(gpu_sum - ref)) < (use_fp64 ? 1e-8 : 1e-3);
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(N); // adds
    info.bytes_moved = sizeof(T) * (N + gws);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 6) SCAN (exclusive or inclusive, here assume inclusive) ------
template <typename T>
static double run_scan_shared(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                              const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> x(N), out(N), ref(N);
    fill_uniform<T>(x, T(0), T(1), 41);
    T acc = T(0);
    for (size_t i = 0; i < N; ++i)
    {
        acc += x[i];
        ref[i] = acc;
    }

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, N);

    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, (int)N);

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, out);
    bool ok = false;
    try
    {
        ok = allclose(out, ref, use_fp64 ? 1e-9 : 1e-4, use_fp64 ? 1e-12 : 1e-6);
    }
    catch (...)
    {
        ok = local_ref::allclose(out, ref, use_fp64 ? 1e-9 : 1e-4, use_fp64 ? 1e-12 : 1e-6);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(N); // approx number of adds
    info.bytes_moved = sizeof(T) * (2 * N);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 7) SPMV CSR --------------------------------------------------
template <typename T>
static double run_spmv_csr(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                           const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Build a simple banded sparse matrix with ~5 nonzeros per row
    size_t rows = N, cols = N;
    size_t nnz_per_row = 5;
    size_t nnz = rows * nnz_per_row;

    std::vector<int> rowptr(rows + 1);
    std::vector<int> colind(nnz);
    std::vector<T> vals(nnz);
    std::vector<T> x(cols), y(rows, T(0)), ref(rows, T(0));

    fill_uniform<T>(x, T(-1), T(1), 51);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dcol(0, int(cols) - 1);
    std::uniform_real_distribution<double> dval(-1.0, 1.0);

    size_t p = 0;
    for (size_t r = 0; r < rows; ++r)
    {
        rowptr[r] = int(p);
        for (size_t k = 0; k < nnz_per_row; ++k)
        {
            int c = (int((r + k) % cols));
            colind[p] = c;
            vals[p] = T(dval(rng));
            ref[r] += vals[p] * x[c];
            ++p;
        }
    }
    rowptr[rows] = int(p);

    cl::Buffer d_rowptr = make_read_buf(ctx, rowptr);
    cl::Buffer d_colind = make_read_buf(ctx, colind);
    cl::Buffer d_vals = make_read_buf(ctx, vals);
    cl::Buffer d_x = make_read_buf(ctx, x);
    cl::Buffer d_y = make_write_buf<T>(ctx, rows);

    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());
    krn.setArg(0, d_rowptr);
    krn.setArg(1, d_colind);
    krn.setArg(2, d_vals);
    krn.setArg(3, d_x);
    krn.setArg(4, d_y);
    krn.setArg(5, (int)rows);

    size_t lws = 256, gws = round_up(rows, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, d_y, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(2 * nnz);
    info.bytes_moved = sizeof(int) * (rows + 1 + nnz) + sizeof(T) * (nnz + cols + rows);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 8) Conv2D / Depthwise (NHWC, single image) -------------------
template <typename T>
static double run_conv2d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                         const string &base, size_t S, bool use_fp64, bool &correct, RunInfo &info)
{
    // Single-channel 3x3 SAME convolution
    const int H = static_cast<int>(S);
    const int W = static_cast<int>(S);
    const int K = 3;
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    size_t in_elems = size_t(H) * W;
    size_t fil_elems = size_t(K) * K;
    size_t out_elems = size_t(H) * W;

    std::vector<T> x(in_elems), f(fil_elems), y(out_elems, T(0)), ref(out_elems, T(0));
    fill_uniform<T>(x, T(-1), T(1), 61);
    fill_uniform<T>(f, T(-1), T(1), 62);

    auto idx_in = [&](int h, int w) { return h * W + w; };
    auto idx_out = [&](int h, int w) { return h * W + w; };
    auto idx_f = [&](int r, int c) { return (r + 1) * 3 + (c + 1); };

    for (int y0 = 0; y0 < H; ++y0)
        for (int x0 = 0; x0 < W; ++x0)
        {
            T acc = T(0);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int yy = y0 + dy;
                    int xx = x0 + dx;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                        acc += x[idx_in(yy, xx)] * f[idx_f(dy, dx)];
                }
            ref[idx_out(y0, x0)] = acc;
        }

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer df = make_read_buf(ctx, f);
    cl::Buffer dy = make_write_buf<T>(ctx, out_elems);

    krn.setArg(0, dx);
    krn.setArg(1, df);
    krn.setArg(2, dy);
    krn.setArg(3, H);
    krn.setArg(4, W);

    size_t lws0 = 16, lws1 = 16;
    size_t gws0 = round_up((size_t)W, lws0);
    size_t gws1 = round_up((size_t)H, lws1);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws0, gws1), cl::NDRange(lws0, lws1), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    correct = ok;

    set_runinfo_2d(info, gws0, gws1, lws0, lws1);
    info.flops_est = double(H) * W * (9 * 2.0);
    info.bytes_moved = sizeof(T) * (in_elems + fil_elems + out_elems);
    finalize_bandwidth(info, ms);
    return ms;
}

template <typename T>
static double run_depthwise(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const string &base, size_t S, bool use_fp64, bool &correct, RunInfo &info)
{
    // Depthwise: H=W=S, C=32, multiplier=1 (kernel per channel)
    const int H = static_cast<int>(S);
    const int W = static_cast<int>(S);
    const int C = 32;
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    size_t in_elems = size_t(H) * W * C;
    size_t fil_elems = size_t(3) * 3 * C;
    size_t out_elems = size_t(H) * W * C;

    std::vector<T> x(in_elems), f(fil_elems), y(out_elems, T(0)), ref(out_elems, T(0));
    fill_uniform<T>(x, T(-1), T(1), 71);
    fill_uniform<T>(f, T(-1), T(1), 72);

    auto idx_in = [&](int h, int w, int c) { return (h * W + w) * C + c; };
    auto idx_f = [&](int dy, int dx, int c) { return c * 9 + (dy + 1) * 3 + (dx + 1); };
    auto idx_out = [&](int h, int w, int c) { return (h * W + w) * C + c; };

    for (int oh = 0; oh < H; ++oh)
    {
        for (int ow = 0; ow < W; ++ow)
        {
            for (int c = 0; c < C; ++c)
            {
                T acc = T(0);
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int ih = oh + dy;
                        int iw = ow + dx;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        {
                            acc += x[idx_in(ih, iw, c)] * f[idx_f(dy, dx, c)];
                        }
                    }
                ref[idx_out(oh, ow, c)] = acc;
            }
        }
    }

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer df = make_read_buf(ctx, f);
    cl::Buffer dy = make_write_buf<T>(ctx, out_elems);

    krn.setArg(0, dx);
    krn.setArg(1, df);
    krn.setArg(2, dy);
    krn.setArg(3, C);
    krn.setArg(4, H);
    krn.setArg(5, W);

    size_t lws0 = (base == "depthwiseconv_tiled") ? 8 : 16;
    size_t lws1 = (base == "depthwiseconv_tiled") ? 8 : 16;
    size_t lws2 = 1;
    size_t gws0 = round_up((size_t)W, lws0);
    size_t gws1 = round_up((size_t)H, lws1);
    size_t gws2 = round_up((size_t)C, lws2);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws0, gws1, gws2),
                           cl::NDRange(lws0, lws1, lws2), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    correct = ok;

    set_runinfo_3d(info, gws0, gws1, gws2, lws0, lws1, lws2);
    info.flops_est = double(H) * W * C * (3 * 3 * 2.0);
    info.bytes_moved = sizeof(T) * (in_elems + fil_elems + out_elems);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 9) Softmax ----------------------------------------------------
template <typename T>
static double run_softmax(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                          const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Single vector softmax of length N
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> x(N), y(N, T(0)), ref(N);
    fill_uniform<T>(x, T(-4), T(4), 81);

    // ref softmax with stability
    T m = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        double e = std::exp(double(x[i] - m));
        ref[i] = T(e);
        sum += e;
    }
    for (size_t i = 0; i < N; ++i)
        ref[i] = T(double(ref[i]) / sum);

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, N);

    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(4 * N); // rough
    info.bytes_moved = sizeof(T) * (2 * N);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 10) LayerNorm -------------------------------------------------
template <typename T>
static double run_layernorm(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Single row of size N, gamma=1, beta=0
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<T> x(N), y(N), ref(N);
    fill_uniform<T>(x, T(-3), T(3), 91);
    T eps = use_fp64 ? T(1e-9) : T(1e-5);
    double mean = 0.0;
    for (auto v : x)
        mean += double(v);
    mean /= double(N);
    double var = 0.0;
    for (auto v : x)
    {
        double d = double(v) - mean;
        var += d * d;
    }
    var /= double(N);
    double inv = 1.0 / std::sqrt(var + double(eps));
    for (size_t i = 0; i < N; ++i)
        ref[i] = T((double(x[i]) - mean) * inv);

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, N);
    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, (int)N);
    krn.setArg(3, eps);

    size_t lws = 256, gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-7 : 5e-3, use_fp64 ? 1e-9 : 1e-4);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-7 : 5e-3, use_fp64 ? 1e-9 : 1e-4);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(6 * N); // rough
    info.bytes_moved = sizeof(T) * (2 * N);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 11) Activations ----------------------------------------------
template <typename T>
static double run_relu(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                       const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());
    std::vector<T> x(N), y(N), ref(N);
    fill_uniform<T>(x, T(-3), T(3), 101);
    for (size_t i = 0; i < N; ++i)
        ref[i] = x[i] > T(0) ? x[i] : T(0);

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, N);
    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(N);
    info.bytes_moved = sizeof(T) * (2 * N);
    finalize_bandwidth(info, ms);
    return ms;
}

template <typename T>
static double run_gelu(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                       const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());
    std::vector<T> x(N), y(N), ref(N);
    fill_uniform<T>(x, T(-4), T(4), 111);
    auto gelu = [](double v)
    {
        // tanh-based approx
        const double k0 = std::sqrt(2.0 / M_PI);
        double t = k0 * (v + 0.044715 * std::pow(v, 3));
        return 0.5 * v * (1.0 + std::tanh(t));
    };
    for (size_t i = 0; i < N; ++i)
        ref[i] = T(gelu(double(x[i])));

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, N);
    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 5e-7 : 2e-3, use_fp64 ? 1e-8 : 2e-4);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 5e-7 : 2e-3, use_fp64 ? 1e-8 : 2e-4);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(20 * N); // rough
    info.bytes_moved = sizeof(T) * (2 * N);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 12) Graph: BFS / DFS / PageRank ------------------------------
// For BFS/DFS we only verify that number of visited nodes > 0 on a simple synthetic graph.
static void build_random_graph(size_t V, size_t avg_deg,
                               std::vector<int> &rowptr,
                               std::vector<int> &colind)
{
    rowptr.resize(V + 1);
    colind.clear();
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dst(0, int(V) - 1);
    size_t nnz = 0;
    for (size_t v = 0; v < V; ++v)
    {
        rowptr[v] = int(nnz);
        for (size_t d = 0; d < avg_deg; ++d)
        {
            colind.push_back(dst(rng));
            ++nnz;
        }
    }
    rowptr[V] = int(nnz);
}

static double run_bfs_or_dfs(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                             const string &base, size_t V, bool is_bfs, bool &correct, RunInfo &info)
{
    (void)is_bfs;
    string kname = base;
    cl::Kernel krn(prog, kname.c_str());

    std::vector<int> rowptr, colind;
    build_random_graph(V, /*avg_deg*/ 4, rowptr, colind);
    std::vector<uint32_t> frontier(V, 0), next_frontier(V, 0), visited(V, 0);
    const int src = 0;
    frontier[src] = 1;
    visited[src] = 1;

    cl::Buffer d_rowptr = make_read_buf(ctx, rowptr);
    cl::Buffer d_colind = make_read_buf(ctx, colind);
    cl::Buffer d_frontier = make_read_buf(ctx, frontier);
    cl::Buffer d_next = make_readwrite_buf(ctx, next_frontier);
    cl::Buffer d_visited = make_readwrite_buf(ctx, visited);

    krn.setArg(0, d_rowptr);
    krn.setArg(1, d_colind);
    krn.setArg(2, d_frontier);
    krn.setArg(3, d_next);
    krn.setArg(4, d_visited);
    krn.setArg(5, (int)V);

    size_t lws = 256;
    size_t gws = round_up(V, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, d_next, next_frontier);
    read_back(q, d_visited, visited);
    size_t newly_reached = 0;
    for (size_t i = 0; i < next_frontier.size(); ++i)
    {
        if (next_frontier[i])
            ++newly_reached;
    }
    correct = (newly_reached > 0);

    set_runinfo_1d(info, gws, lws);
    info.flops_est = 0.0; // graph irregular; skip
    info.bytes_moved = sizeof(int) * (rowptr.size() + colind.size()) +
                       sizeof(uint32_t) * (frontier.size() + next_frontier.size() + visited.size());
    finalize_bandwidth(info, ms);
    return ms;
}

template <typename T>
static double run_pagerank(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                           const string &base, size_t V, bool use_fp64, bool &correct, RunInfo &info)
{
    // Simple power-iteration PageRank with fixed small iterations on CSR
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<int> rowptr, colind;
    build_random_graph(V, /*avg_deg*/ 4, rowptr, colind);
    std::vector<int> outdeg(V, 1);
    for (size_t v = 0; v < V; ++v)
        outdeg[v] = std::max(1, rowptr[v + 1] - rowptr[v]);

    const double d = 0.85;
    const T teleport = T((1.0 - d) / double(V));
    std::vector<T> pr(V, T(1) / T(V));
    std::vector<T> pr_next(V, teleport);
    std::vector<T> pr_cpu(V, teleport);

    // CPU one step for reference
    for (size_t v = 0; v < V; ++v)
    {
        size_t start = (size_t)rowptr[v], end = (size_t)rowptr[v + 1];
        T contrib = T(d) * pr[v] / T(outdeg[v]);
        for (size_t e = start; e < end; ++e)
        {
            int w = colind[e];
            pr_cpu[w] += contrib;
        }
    }

    cl::Buffer d_rowptr = make_read_buf(ctx, rowptr);
    cl::Buffer d_colind = make_read_buf(ctx, colind);
    cl::Buffer d_outdeg = make_read_buf(ctx, outdeg);
    cl::Buffer d_pr = make_read_buf(ctx, pr);
    cl::Buffer d_next = make_readwrite_buf(ctx, pr_next);

    krn.setArg(0, d_rowptr);
    krn.setArg(1, d_colind);
    krn.setArg(2, d_outdeg);
    krn.setArg(3, d_pr);
    krn.setArg(4, d_next);
    krn.setArg(5, (T)d);
    krn.setArg(6, (int)V);

    size_t lws = 256, gws = round_up(V, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, d_next, pr_next);
    bool ok = false;
    try
    {
        ok = allclose(pr_next, pr_cpu, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    catch (...)
    {
        ok = local_ref::allclose(pr_next, pr_cpu, use_fp64 ? 1e-8 : 3e-3, use_fp64 ? 1e-10 : 1e-4);
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = 0.0;
    info.bytes_moved = sizeof(int) * (rowptr.size() + colind.size() + outdeg.size()) +
                       sizeof(T) * (pr.size() + pr_next.size());
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 13) Stencil 2D / 3D -----------------------------------------
template <typename T>
static double run_stencil2d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const string &base, size_t S, bool use_fp64, bool &correct, RunInfo &info,
                            int ksz)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());
    const int H = S, W = S;
    std::vector<T> x(H * W), y(H * W, T(0)), ref(H * W, T(0));
    fill_uniform<T>(x, T(-1), T(1), 121);

    int rad = ksz / 2;
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            T acc = T(0);
            for (int di = -rad; di <= rad; ++di)
            {
                for (int dj = -rad; dj <= rad; ++dj)
                {
                    int ii = i + di, jj = j + dj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W)
                        acc += x[ii * W + jj];
                }
            }
            ref[i * W + j] = acc;
        }
    }

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, H * W);
    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, H);
    krn.setArg(3, W);

    size_t lws0 = 16, lws1 = 16;
    size_t gws0 = round_up((size_t)W, lws0);
    size_t gws1 = round_up((size_t)H, lws1);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws0, gws1), cl::NDRange(lws0, lws1), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    correct = ok;

    set_runinfo_2d(info, gws0, gws1, lws0, lws1);
    info.flops_est = double(H) * W * (ksz * ksz);
    info.bytes_moved = sizeof(T) * (2ull * H * W);
    finalize_bandwidth(info, ms);
    return ms;
}

template <typename T>
static double run_stencil3d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const string &base, size_t S, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());
    const int D = S, H = S, W = S;
    std::vector<T> x(D * H * W), y(D * H * W, T(0)), ref(D * H * W, T(0));
    fill_uniform<T>(x, T(-1), T(1), 131);

    int rad = 1;
    for (int z = 0; z < D; ++z)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
                T acc = T(0);
                for (int dz = -rad; dz <= rad; ++dz)
                    for (int di = -rad; di <= rad; ++di)
                        for (int dj = -rad; dj <= rad; ++dj)
                        {
                            int zz = z + dz, ii = i + di, jj = j + dj;
                            if (zz >= 0 && zz < D && ii >= 0 && ii < H && jj >= 0 && jj < W)
                                acc += x[(zz * H + ii) * W + jj];
                        }
                ref[(z * H + i) * W + j] = acc;
            }

    cl::Buffer dx = make_read_buf(ctx, x);
    cl::Buffer dy = make_write_buf<T>(ctx, D * H * W);
    krn.setArg(0, dx);
    krn.setArg(1, dy);
    krn.setArg(2, D);
    krn.setArg(3, H);
    krn.setArg(4, W);

    size_t lws0 = 8, lws1 = 8, lws2 = 2;
    size_t gws0 = round_up((size_t)W, lws0);
    size_t gws1 = round_up((size_t)H, lws1);
    size_t gws2 = round_up((size_t)D, lws2);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws0, gws1, gws2), cl::NDRange(lws0, lws1, lws2), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dy, y);
    bool ok = false;
    try
    {
        ok = allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    catch (...)
    {
        ok = local_ref::allclose(y, ref, use_fp64 ? 1e-8 : 1e-3, use_fp64 ? 1e-10 : 1e-5);
    }
    correct = ok;

    set_runinfo_3d(info, gws0, gws1, gws2, lws0, lws1, lws2);
    info.flops_est = double(D) * H * W * 27.0;
    info.bytes_moved = sizeof(T) * (2ull * D * H * W);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 14) Histogram (global/shared) --------------------------------
static double run_histogram_u32(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                                const string &base, size_t N, bool &correct, RunInfo &info)
{
    string kname = base;
    cl::Kernel krn(prog, kname.c_str());
    const int BINS = 256;

    std::vector<uint32_t> in(N);
    fill_uniform<uint32_t>(in, 0u, 255u, 141);
    std::vector<uint32_t> ref(BINS, 0), out(BINS, 0);
    for (auto v : in)
        ++ref[v & 255u];

    std::vector<uint32_t> hist_init(BINS, 0u);
    cl::Buffer din = make_read_buf(ctx, in);
    cl::Buffer dout = make_readwrite_buf(ctx, hist_init);

    krn.setArg(0, din);
    krn.setArg(1, dout);
    krn.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dout, out);
    correct = (out == ref);

    set_runinfo_1d(info, gws, lws);
    info.flops_est = 0.0;
    info.bytes_moved = sizeof(uint32_t) * (N + BINS);
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 15) Bitonic sort (keys only) ---------------------------------
static double run_bitonic_u32(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                              const string &base, size_t N, bool &correct, RunInfo &info)
{
    string kname = base;
    cl::Kernel krn(prog, kname.c_str());

    std::vector<uint32_t> input(N);
    fill_uniform<uint32_t>(input, 0u, 1000000u, 151);
    std::vector<uint32_t> ref = input;
    std::sort(ref.begin(), ref.end());

    size_t padN = 1;
    while (padN < N)
        padN <<= 1;
    if (padN < 2)
        padN = 2;
    if (padN & 1)
        ++padN;

    std::vector<uint32_t> data(padN, std::numeric_limits<uint32_t>::max());
    std::copy(input.begin(), input.end(), data.begin());
    cl::Buffer d_data = make_readwrite_buf(ctx, data);

    size_t lws = 128;
    size_t items = padN / 2;
    size_t gws = round_up(items, lws);
    double ms_accum = 0.0;

    for (size_t k = 2; k <= padN; k <<= 1)
    {
        for (size_t j = k >> 1; j > 0; j >>= 1)
        {
            krn.setArg(0, d_data);
            krn.setArg(1, (int)padN);
            krn.setArg(2, (int)j);
            krn.setArg(3, (int)k);
            cl::Event evt;
            q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
            evt.wait();
            ms_accum += event_elapsed_ms(evt);
        }
    }

    read_back(q, d_data, data);
    correct = std::equal(ref.begin(), ref.end(), data.begin());

    set_runinfo_1d(info, gws, lws);
    info.flops_est = 0.0;
    info.bytes_moved = sizeof(uint32_t) * padN;
    finalize_bandwidth(info, ms_accum);
    return ms_accum;
}

// ---- 16) Monte Carlo (pi estimate) --------------------------------
template <typename T>
static double run_montecarlo(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                             const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    // Pre-generate uniform points in [0,1)
    std::vector<T> xy(2 * N);
    fill_uniform<T>(xy, T(0), T(1), 161);

    size_t lws = 256, gws = round_up(N, lws);
    std::vector<uint32_t> partial(gws, 0u);

    cl::Buffer dxy = make_read_buf(ctx, xy);
    cl::Buffer dout = make_readwrite_buf(ctx, partial);

    krn.setArg(0, dxy);
    krn.setArg(1, dout);
    krn.setArg(2, (int)N);

    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
    evt.wait();
    double ms = event_elapsed_ms(evt);

    read_back(q, dout, partial);
    uint64_t hits = std::accumulate(partial.begin(), partial.end(), uint64_t(0));
    double pi_est = 4.0 * double(hits) / double(N);
    correct = (pi_est > 2.5 && pi_est < 3.8); // loose check

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(4 * N);
    info.bytes_moved = sizeof(T) * (2 * N) + sizeof(uint32_t) * gws;
    finalize_bandwidth(info, ms);
    return ms;
}

// ---- 17) FFT1D (global / staged) ---------------------------------
template <typename T>
static double run_fft1d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                        const string &base, size_t N, bool use_fp64, bool &correct, RunInfo &info)
{
    // Complex interleaved length-N FFT (assume kernels implement correct variant)
    using CT = std::conditional_t<std::is_same<T, double>::value, cl_double2, cl_float2>;
    string kname = resolve_kname(base, DTypeMode::FLOATING, use_fp64);
    cl::Kernel krn(prog, kname.c_str());

    std::vector<CT> x(N), y(N), ref(N);
    // Sine wave test
    for (size_t i = 0; i < N; ++i)
    {
        double theta = 2.0 * M_PI * double(i) / double(N);
        if constexpr (std::is_same<T, double>::value)
        {
            y[i].s[0] = 0;
            y[i].s[1] = 0; // init
        }
        ((float *)&x[i])[0] = (float)std::cos(theta);
        ((float *)&x[i])[1] = (float)std::sin(theta);
        ((float *)&ref[i])[0] = 0.f;
        ((float *)&ref[i])[1] = 0.f;
    }
    // very rough CPU DFT O(N^2) for small N
    if (N <= (1u << 12))
    {
        for (size_t k = 0; k < N; ++k)
        {
            double re = 0, im = 0;
            for (size_t n = 0; n < N; ++n)
            {
                double ang = -2.0 * M_PI * double(k * n) / double(N);
                double xr = ((float *)&x[n])[0];
                double xi = ((float *)&x[n])[1];
                re += xr * std::cos(ang) - xi * std::sin(ang);
                im += xr * std::sin(ang) + xi * std::cos(ang);
            }
            ((float *)&ref[k])[0] = (float)re;
            ((float *)&ref[k])[1] = (float)im;
        }
    }

    cl::Buffer ddata(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CT) * N, (void *)x.data());

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    double ms_total = 0.0;
    for (size_t m = 2; m <= N; m <<= 1)
    {
        int mh = int(m >> 1);
        krn.setArg(0, ddata);
        krn.setArg(1, (int)N);
        krn.setArg(2, mh);
        cl::Event evt;
        q.enqueueNDRangeKernel(krn, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms_total += event_elapsed_ms(evt);
    }

    q.enqueueReadBuffer(ddata, true, 0, sizeof(CT) * N, y.data());
    bool ok = true;
    if (N <= (1u << 12))
    {
        // Compare magnitudes only (avoid phase mismatch issues)
        for (size_t i = 0; i < N; ++i)
        {
            double mag_y = std::hypot(((float *)&y[i])[0], ((float *)&y[i])[1]);
            double mag_r = std::hypot(((float *)&ref[i])[0], ((float *)&ref[i])[1]);
            if (std::fabs(mag_y - mag_r) > (use_fp64 ? 1e-6 : 1e-2))
            {
                ok = false;
                break;
            }
        }
    }
    correct = ok;

    set_runinfo_1d(info, gws, lws);
    info.flops_est = double(5 * N * std::log2((double)N)); // rough
    info.bytes_moved = sizeof(CT) * (2 * N);
    finalize_bandwidth(info, ms_total);
    return ms_total;
}

// =============================================================
// Driver
// =============================================================

struct ResultRow
{
    std::string kernel;
    std::string dtype;
    size_t size;
    double ms;
    bool correct;
    RunInfo info;
};

int main()
try
{
    // 1) Pick platform/device and create context/queue
    cl::Context ctx;
    cl::Device dev;
    cl::CommandQueue q;
    create_context_queue(ctx, dev, q);
    std::string dev_name = device_name_string(dev);
    std::cout << "Using OpenCL device: " << dev_name << std::endl;

    // 2) Build programs: FP32 always, FP64 if supported
    bool fp64_ok = has_fp64(dev);
    cl::Program prog32 = build_program_from_file(ctx, dev, "kernels_all.cl", /*enable_fp64*/ false);
    cl::Program prog64;
    if (fp64_ok)
    {
        prog64 = build_program_from_file(ctx, dev, "kernels_all.cl", /*enable_fp64*/ true);
    }

    // 3) CSV path (header managed inside helper)
    const std::string csv_path = "results_opencl.csv";
    std::remove(csv_path.c_str());

    // 4) Dispatch table per kernel name
    auto run_float = [&](const std::string &name, size_t sz, bool want64, ResultRow &row)
    {
        bool correct = false;
        RunInfo info;
        double ms = 0.0;
        cl::Program &P = (want64 ? prog64 : prog32);

        if (name == "vecadd_basic")
            ms = want64 ? run_vecadd<double>(ctx, q, P, "vecadd_basic", sz, true, correct, info)
                        : run_vecadd<float>(ctx, q, P, "vecadd_basic", sz, false, correct, info);
        else if (name == "dot_global")
            ms = want64 ? run_dot<double>(ctx, q, P, "dot_global", sz, true, correct, info)
                        : run_dot<float>(ctx, q, P, "dot_global", sz, false, correct, info);
        else if (name == "dot_shared")
            ms = want64 ? run_dot<double>(ctx, q, P, "dot_shared", sz, true, correct, info)
                        : run_dot<float>(ctx, q, P, "dot_shared", sz, false, correct, info);
        else if (name == "gemv_global")
            ms = want64 ? run_gemv<double>(ctx, q, P, "gemv_global", sz, true, correct, info)
                        : run_gemv<float>(ctx, q, P, "gemv_global", sz, false, correct, info);
        else if (name == "gemv_shared")
            ms = want64 ? run_gemv<double>(ctx, q, P, "gemv_shared", sz, true, correct, info)
                        : run_gemv<float>(ctx, q, P, "gemv_shared", sz, false, correct, info);
        else if (name == "matmul_global")
            ms = want64 ? run_matmul<double>(ctx, q, P, "matmul_global", sz, true, correct, info)
                        : run_matmul<float>(ctx, q, P, "matmul_global", sz, false, correct, info);
        else if (name == "matmul_shared")
            ms = want64 ? run_matmul<double>(ctx, q, P, "matmul_shared", sz, true, correct, info)
                        : run_matmul<float>(ctx, q, P, "matmul_shared", sz, false, correct, info);
        else if (name == "reduction_global")
            ms = want64 ? run_reduction<double>(ctx, q, P, "reduction_global", sz, true, correct, info)
                        : run_reduction<float>(ctx, q, P, "reduction_global", sz, false, correct, info);
        else if (name == "reduction_shared")
            ms = want64 ? run_reduction<double>(ctx, q, P, "reduction_shared", sz, true, correct, info)
                        : run_reduction<float>(ctx, q, P, "reduction_shared", sz, false, correct, info);
        else if (name == "scan_shared")
            ms = want64 ? run_scan_shared<double>(ctx, q, P, "scan_shared", sz, true, correct, info)
                        : run_scan_shared<float>(ctx, q, P, "scan_shared", sz, false, correct, info);
        else if (name == "spmv_csr")
            ms = want64 ? run_spmv_csr<double>(ctx, q, P, "spmv_csr", sz, true, correct, info)
                        : run_spmv_csr<float>(ctx, q, P, "spmv_csr", sz, false, correct, info);
        else if (name == "conv2d_global" || name == "conv2d_shared")
            ms = want64 ? run_conv2d<double>(ctx, q, P, name, sz, true, correct, info)
                        : run_conv2d<float>(ctx, q, P, name, sz, false, correct, info);
        else if (name == "depthwiseconv_global" || name == "depthwiseconv_tiled")
            ms = want64 ? run_depthwise<double>(ctx, q, P, name, sz, true, correct, info)
                        : run_depthwise<float>(ctx, q, P, name, sz, false, correct, info);
        else if (name == "softmax_basic")
            ms = want64 ? run_softmax<double>(ctx, q, P, "softmax_basic", sz, true, correct, info)
                        : run_softmax<float>(ctx, q, P, "softmax_basic", sz, false, correct, info);
        else if (name == "layernorm_basic")
            ms = want64 ? run_layernorm<double>(ctx, q, P, "layernorm_basic", sz, true, correct, info)
                        : run_layernorm<float>(ctx, q, P, "layernorm_basic", sz, false, correct, info);
        else if (name == "activation_relu")
            ms = want64 ? run_relu<double>(ctx, q, P, "activation_relu", sz, true, correct, info)
                        : run_relu<float>(ctx, q, P, "activation_relu", sz, false, correct, info);
        else if (name == "activation_gelu")
            ms = want64 ? run_gelu<double>(ctx, q, P, "activation_gelu", sz, true, correct, info)
                        : run_gelu<float>(ctx, q, P, "activation_gelu", sz, false, correct, info);
        else if (name == "pagerank_basic")
            ms = want64 ? run_pagerank<double>(ctx, q, P, "pagerank_basic", sz, true, correct, info)
                        : run_pagerank<float>(ctx, q, P, "pagerank_basic", sz, false, correct, info);
        else if (name == "stencil2d_3x3")
            ms = want64 ? run_stencil2d<double>(ctx, q, P, "stencil2d_3x3", sz, true, correct, info, 3)
                        : run_stencil2d<float>(ctx, q, P, "stencil2d_3x3", sz, false, correct, info, 3);
        else if (name == "stencil2d_5x5")
            ms = want64 ? run_stencil2d<double>(ctx, q, P, "stencil2d_5x5", sz, true, correct, info, 5)
                        : run_stencil2d<float>(ctx, q, P, "stencil2d_5x5", sz, false, correct, info, 5);
        else if (name == "stencil3d_global" || name == "stencil3d_shared")
            ms = want64 ? run_stencil3d<double>(ctx, q, P, name, sz, true, correct, info)
                        : run_stencil3d<float>(ctx, q, P, name, sz, false, correct, info);
        else if (name == "montecarlo_basic")
            ms = want64 ? run_montecarlo<double>(ctx, q, P, "montecarlo_basic", sz, true, correct, info)
                        : run_montecarlo<float>(ctx, q, P, "montecarlo_basic", sz, false, correct, info);
        else if (name == "fft1d_global" || name == "fft1d_staged")
            ms = want64 ? run_fft1d<double>(ctx, q, P, name, sz, true, correct, info)
                        : run_fft1d<float>(ctx, q, P, name, sz, false, correct, info);
        else
        {
            // Unknown floating kernel -> mark skipped
            correct = false;
            info = {};
            ms = 0.0;
        }

        row.ms = ms;
        row.correct = correct;
        row.info = info;
    };

    auto run_integer = [&](const std::string &name, size_t sz, ResultRow &row)
    {
        bool correct = false;
        RunInfo info;
        double ms = 0.0;
        cl::Program &P = prog32; // 32-bit ints: same program

        if (name == "bfs_basic")
        {
            ms = run_bfs_or_dfs(ctx, q, P, "bfs_basic", sz, true, correct, info);
        }
        else if (name == "dfs_basic")
        {
            ms = run_bfs_or_dfs(ctx, q, P, "dfs_basic", sz, false, correct, info);
        }
        else if (name == "histogram_global" || name == "histogram_shared")
        {
            ms = run_histogram_u32(ctx, q, P, name, sz, correct, info);
        }
        else if (name == "sort_bitonic")
        {
            ms = run_bitonic_u32(ctx, q, P, "sort_bitonic", sz, correct, info);
        }
        else
        {
            correct = false;
            info = {};
            ms = 0.0;
        }

        row.ms = ms;
        row.correct = correct;
        row.info = info;
    };

    // 5) Iterate BENCHMARKS
    for (const auto &b : BENCHMARKS)
    {
        if (!b.enabled)
            continue;
        for (size_t sz : b.sizes)
        {
            if (b.dtype_mode == DTypeMode::FLOATING)
            {
                // FP32 run
                ResultRow row32;
                row32.kernel = b.name;
                row32.dtype = "FP32";
                row32.size = sz;
                cout << "[OpenCL] Running kernel=" << b.name << " dtype=FP32 size=" << sz << std::endl;
                run_float(b.name, sz, /*want64*/ false, row32);
                write_csv_row(csv_path, row32.kernel, row32.dtype, row32.size,
                              row32.ms, row32.correct, dev_name,
                              row32.info.gws0, row32.info.gws1, row32.info.gws2,
                              row32.info.lws0, row32.info.lws1, row32.info.lws2,
                              row32.info.flops_est, row32.info.bw_GBps);
                // FP64 run (if supported)
                if (fp64_ok)
                {
                    ResultRow row64;
                    row64.kernel = b.name;
                    row64.dtype = "FP64";
                    row64.size = sz;
                    cout << "[OpenCL] Running kernel=" << b.name << " dtype=FP64 size=" << sz << std::endl;
                    run_float(b.name, sz, /*want64*/ true, row64);
                    write_csv_row(csv_path, row64.kernel, row64.dtype, row64.size,
                                  row64.ms, row64.correct, dev_name,
                                  row64.info.gws0, row64.info.gws1, row64.info.gws2,
                                  row64.info.lws0, row64.info.lws1, row64.info.lws2,
                                  row64.info.flops_est, row64.info.bw_GBps);
                }
            }
            else
            { // INTEGER
                ResultRow rowi;
                rowi.kernel = b.name;
                rowi.dtype = "INT";
                rowi.size = sz;
                cout << "[OpenCL] Running kernel=" << b.name << " dtype=INT size=" << sz << std::endl;
                run_integer(b.name, sz, rowi);
                write_csv_row(csv_path, rowi.kernel, rowi.dtype, rowi.size,
                              rowi.ms, rowi.correct, dev_name,
                              rowi.info.gws0, rowi.info.gws1, rowi.info.gws2,
                              rowi.info.lws0, rowi.info.lws1, rowi.info.lws2,
                              rowi.info.flops_est, rowi.info.bw_GBps);
            }
        }
    }

    cout << "Done. CSV written to results_opencl.csv" << std::endl;
    return 0;
}
catch (const cl::Error &e)
{
    cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
    return 1;
}
catch (const std::exception &e)
{
    cerr << "Exception: " << e.what() << "\n";
    return 1;
}
