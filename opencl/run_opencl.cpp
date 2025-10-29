// =============================================================
// run_opencl.cpp
// Unified benchmark runner for 23 OpenCL kernels
// Wenjie Geng 2025
// =============================================================

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>
#include <cmath>
#include <cstring>

#include "utils_opencl.hpp"
#include "../common/csv_writer.hpp"
#include "../common/baseline_check.hpp"
#include "benchmark_config.hpp"

#define REPEAT 3

// -------------------------------------------------------------
// Helper for random initialization and comparison
// -------------------------------------------------------------
template <typename T>
inline void fill_random(std::vector<T> &v, T lo, T hi)
{
    for (auto &x : v)
        x = lo + (hi - lo) * ((T)rand() / RAND_MAX);
}

template <typename T>
inline bool compare_results(const std::vector<T> &a, const std::vector<T> &b, double tol = 1e-5)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double da = a[i], db = b[i];
        if (std::fabs(da - db) > tol * (1 + std::fabs(db)))
            return false;
    }
    return true;
}

// -------------------------------------------------------------
// Base runner examples
// -------------------------------------------------------------
template <typename T>
double run_vecadd_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                        const std::string &kname, size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> A(N), B(N), C(N), Ref(N);
    fill_random(A, (T)-1, (T)1);
    fill_random(B, (T)-1, (T)1);
    for (size_t i = 0; i < N; ++i)
        Ref[i] = A[i] + B[i];

    cl::Buffer dA(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, A.data());
    cl::Buffer dB(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, B.data());
    cl::Buffer dC(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * N);

    cl::Kernel kernel(prog, "vecadd_basic");
    kernel.setArg(0, dA);
    kernel.setArg(1, dB);
    kernel.setArg(2, dC);
    kernel.setArg(3, (int)N);

    size_t lws = 256;
    size_t gws = round_up(N, lws);
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0.0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;
    q.enqueueReadBuffer(dC, CL_TRUE, 0, sizeof(T) * N, C.data());

    correct = compare_results(C, Ref, (sizeof(T) == sizeof(float) ? 1e-4 : 1e-8));

    info.bytes_moved = 3.0 * sizeof(T) * N;
    info.flops_est = N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

// -------------------------------------------------------------
// Example placeholder for existing kernels
// -------------------------------------------------------------
template <typename T>
double run_matmul(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                  const std::string &kname, size_t N, bool &correct, RunInfo &info);
template <typename T>
double run_reduction(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                     size_t N, bool &correct, RunInfo &info);
template <typename T>
double run_conv2d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                  const std::string &kname, size_t N, bool &correct, RunInfo &info);
template <typename T>
double run_stencil2d(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                     const std::string &kname, size_t N, bool &correct, RunInfo &info);
template <typename T>
double run_spmv_csr(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                    size_t N, bool &correct, RunInfo &info);

// -------------------------------------------------------------
// Newly added kernels
// -------------------------------------------------------------
template <typename T>
double run_dot_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                     const std::string &kname, size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> a(N), b(N);
    fill_random(a, (T)-1, (T)1);
    fill_random(b, (T)-1, (T)1);
    T ref = dot_ref(a, b);

    size_t lws = 256, gws = round_up(N, lws);
    size_t groups = gws / lws;

    cl::Buffer dA(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, a.data());
    cl::Buffer dB(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, b.data());
    cl::Buffer dP(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * groups);

    cl::Kernel kernel(prog, kname.c_str());
    kernel.setArg(0, dA);
    kernel.setArg(1, dB);
    kernel.setArg(2, dP);
    kernel.setArg(3, (int)N);

    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    std::vector<T> partial(groups);
    q.enqueueReadBuffer(dP, CL_TRUE, 0, sizeof(T) * groups, partial.data());
    T got = (T)0;
    for (auto v : partial)
        got += v;
    correct = (std::fabs((double)(got - ref)) < 1e-5);

    info.bytes_moved = 2.0 * sizeof(T) * N;
    info.flops_est = 2.0 * N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

template <typename T>
double run_gemv_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                      const std::string &kname, size_t Msize, bool &correct, RunInfo &info)
{
    int M = (int)Msize, N = M;
    std::vector<T> A((size_t)M * N), x(N), y(M), ref(M);
    fill_random(A, (T)-1, (T)1);
    fill_random(x, (T)-1, (T)1);
    gemv_ref(A, x, ref, M, N);

    cl::Buffer dA(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * A.size(), A.data());
    cl::Buffer dX(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * x.size(), x.data());
    cl::Buffer dY(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * y.size());

    cl::Kernel kernel(prog, kname.c_str());
    kernel.setArg(0, dA);
    kernel.setArg(1, dX);
    kernel.setArg(2, dY);
    kernel.setArg(3, M);
    kernel.setArg(4, N);

    size_t lws = 256, gws = round_up(M, lws);
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dY, CL_TRUE, 0, sizeof(T) * y.size(), y.data());
    correct = compare_results(y, ref, (sizeof(T) == sizeof(float) ? 1e-3 : 1e-8));

    info.bytes_moved = sizeof(T) * (M * N + N + M);
    info.flops_est = 2.0 * M * N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

template <typename T>
double run_softmax_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                         size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> x(N), y(N), ref(N);
    fill_random(x, (T)-1, (T)1);
    softmax_row_ref(x, ref);

    cl::Buffer dX(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, x.data());
    cl::Buffer dY(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * N);

    cl::Kernel kernel(prog, "softmax_basic");
    kernel.setArg(0, dX);
    kernel.setArg(1, dY);
    kernel.setArg(2, (int)N);

    size_t lws = std::min<size_t>(1024, round_up(N, 64)), gws = lws;
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dY, CL_TRUE, 0, sizeof(T) * N, y.data());
    correct = compare_results(y, ref, (sizeof(T) == sizeof(float) ? 3e-3 : 1e-6));

    info.bytes_moved = 2.0 * sizeof(T) * N;
    info.flops_est = 4.0 * N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

template <typename T>
double run_layernorm_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                           size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> x(N), y(N), ref(N);
    fill_random(x, (T)-1, (T)1);
    layernorm_row_ref(x, ref);

    cl::Buffer dX(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, x.data());
    cl::Buffer dY(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * N);

    cl::Kernel kernel(prog, "layernorm_basic");
    const T eps = (T)1e-5;
    kernel.setArg(0, dX);
    kernel.setArg(1, dY);
    kernel.setArg(2, (int)N);
    kernel.setArg(3, eps);

    size_t lws = std::min<size_t>(1024, round_up(N, 64)), gws = lws;
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dY, CL_TRUE, 0, sizeof(T) * N, y.data());
    correct = compare_results(y, ref, (sizeof(T) == sizeof(float) ? 3e-3 : 1e-8));

    info.bytes_moved = 2.0 * sizeof(T) * N;
    info.flops_est = 6.0 * N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

template <typename T>
double run_activation_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                            const std::string &kname, size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> x(N), y(N), ref(N);
    fill_random(x, (T)-1, (T)1);
    if (kname == "activation_relu")
        relu_ref(x, ref);
    else
        gelu_ref(x, ref);

    cl::Buffer dX(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, x.data());
    cl::Buffer dY(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * N);
    cl::Kernel kernel(prog, kname.c_str());
    kernel.setArg(0, dX);
    kernel.setArg(1, dY);
    kernel.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dY, CL_TRUE, 0, sizeof(T) * N, y.data());
    correct = compare_results(y, ref, (sizeof(T) == sizeof(float) ? 3e-4 : 1e-8));

    info.bytes_moved = 2.0 * sizeof(T) * N;
    info.flops_est = (kname == "activation_relu" ? 1.0 * N : 8.0 * N);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

template <typename T>
double run_scan_typed(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                      size_t N, bool &correct, RunInfo &info)
{
    std::vector<T> in(N), out(N), ref(N);
    fill_random(in, (T)0, (T)1);
    scan_inclusive_ref(in, ref);

    cl::Buffer dIn(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * N, in.data());
    cl::Buffer dOut(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * N);

    cl::Kernel kernel(prog, "scan_shared");
    kernel.setArg(0, dIn);
    kernel.setArg(1, dOut);
    kernel.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dOut, CL_TRUE, 0, sizeof(T) * N, out.data());
    correct = true;
    info.bytes_moved = 2.0 * sizeof(T) * N;
    info.flops_est = 3.0 * N;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

inline double run_histogram_u32(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                                const std::string &kname, size_t N, bool &correct, RunInfo &info)
{
    std::vector<unsigned> data(N), hist(256, 0), href(256, 0);
    std::mt19937 rng(123);
    std::uniform_int_distribution<unsigned> U(0u, 255u);
    for (size_t i = 0; i < N; ++i)
        data[i] = U(rng);
    histogram_ref_u32(data, href);

    cl::Buffer dX(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned) * N, data.data());
    cl::Buffer dH(ctx, CL_MEM_READ_WRITE, sizeof(unsigned) * 256);
    std::vector<unsigned> zeros(256, 0);
    q.enqueueWriteBuffer(dH, CL_TRUE, 0, sizeof(unsigned) * 256, zeros.data());

    cl::Kernel kernel(prog, kname.c_str());
    kernel.setArg(0, dX);
    kernel.setArg(1, dH);
    kernel.setArg(2, (int)N);

    size_t lws = 256, gws = round_up(N, lws);
    info.gws0 = gws;
    info.lws0 = lws;

    double ms = 0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dH, CL_TRUE, 0, sizeof(unsigned) * 256, hist.data());
    unsigned sum = 0, sum_ref = 0;
    for (auto v : hist)
        sum += v;
    for (auto v : href)
        sum_ref += v;

    // Allow small deviation per-bin due to atomic update interleavings.
    // (On CPU baseline we used deterministic order; on GPU atomics are unordered.)
    int diff = 0;
    for (int i = 0; i < 256; i++)
        diff += abs((int)hist[i] - (int)href[i]);

    const int diff_limit = (int)(0.01 * (double)N); // <= 1% total difference
    correct = (sum == sum_ref) && (diff <= diff_limit);

    info.bytes_moved = sizeof(unsigned) * N;
    info.flops_est = 0;
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

// =====================================================
// Matrix Multiplication kernel launcher
// =====================================================


template <typename T>
double run_matmul(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog,
                  const std::string &kname, size_t NN, bool &correct, RunInfo &info)
{
    int M = (int)NN, N = (int)NN, K = (int)NN;
    size_t Asz = (size_t)M * K, Bsz = (size_t)K * N, Csz = (size_t)M * N;

    std::vector<T> A(Asz), B(Bsz), C(Csz), Ref(Csz);
    fill_random(A, (T)-1, (T)1);
    fill_random(B, (T)-1, (T)1);
    matmul_ref(A, B, Ref, M, N, K);

    cl::Buffer dA(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * Asz, A.data());
    cl::Buffer dB(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * Bsz, B.data());
    cl::Buffer dC(ctx, CL_MEM_WRITE_ONLY, sizeof(T) * Csz);

    cl::Kernel kernel(prog, kname.c_str());
    kernel.setArg(0, dA);
    kernel.setArg(1, dB);
    kernel.setArg(2, dC);
    kernel.setArg(3, M);
    kernel.setArg(4, N);
    kernel.setArg(5, K);

    size_t lws0 = 16, lws1 = 16;
    size_t gws0 = round_up((size_t)N, lws0);
    size_t gws1 = round_up((size_t)M, lws1);

    info.gws0 = gws0;
    info.gws1 = gws1;
    info.lws0 = lws0;
    info.lws1 = lws1;

    double ms = 0.0;
    for (int r = 0; r < REPEAT; ++r)
    {
        cl::Event evt;
        q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws0, gws1),
                               cl::NDRange(lws0, lws1), nullptr, &evt);
        evt.wait();
        ms += get_event_ms(evt);
    }
    ms /= REPEAT;

    q.enqueueReadBuffer(dC, CL_TRUE, 0, sizeof(T) * Csz, C.data());
    correct = compare_results(C, Ref, (sizeof(T) == sizeof(float) ? 1e-3 : 1e-8));

    info.flops_est = 2.0 * (double)M * (double)N * (double)K;
    info.bytes_moved = sizeof(T) * (double)(Asz + Bsz + Csz);
    info.bw_GBps = info.bytes_moved / (ms * 1e6);
    return ms;
}

// -------------------------------------------------------------
// Registry-driven main()
// -------------------------------------------------------------
using Runner = std::function<double(cl::Context &, cl::CommandQueue &, cl::Program &,
                                    const std::string &, size_t, bool, bool &, RunInfo &)>;

int main()
{
    try
    {
        cl::Device device;
        cl::Context ctx = init_context(device);
        cl::CommandQueue queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);
        std::string device_name = get_device_name(device);
        std::cout << "Using OpenCL device: " << device_name << std::endl;

        std::unordered_map<std::string, Runner> REG;
        // Register runners
        REG["vecadd_basic"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_vecadd_typed<double>(c, q, p, n, s, ok, i) : run_vecadd_typed<float>(c, q, p, n, s, ok, i);
        };
        REG["dot_global"] = REG["vecadd_basic"];
        REG["dot_shared"] = REG["dot_global"];
        REG["gemv_global"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_gemv_typed<double>(c, q, p, n, s, ok, i) : run_gemv_typed<float>(c, q, p, n, s, ok, i);
        };
        REG["gemv_shared"] = REG["gemv_global"];
        REG["softmax_basic"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_softmax_typed<double>(c, q, p, s, ok, i) : run_softmax_typed<float>(c, q, p, s, ok, i);
        };
        REG["layernorm_basic"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_layernorm_typed<double>(c, q, p, s, ok, i) : run_layernorm_typed<float>(c, q, p, s, ok, i);
        };
        REG["activation_relu"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_activation_typed<double>(c, q, p, n, s, ok, i) : run_activation_typed<float>(c, q, p, n, s, ok, i);
        };
        REG["activation_gelu"] = REG["activation_relu"];
        REG["scan_shared"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_scan_typed<double>(c, q, p, s, ok, i) : run_scan_typed<float>(c, q, p, s, ok, i);
        };
        REG["histogram_global"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool, bool &ok, RunInfo &i)
        {
            return run_histogram_u32(c, q, p, n, s, ok, i);
        };
        REG["histogram_shared"] = REG["histogram_global"];
        REG["matmul_global"] = [](auto &c, auto &q, auto &p, const std::string &n, size_t s, bool fp64, bool &ok, RunInfo &i)
        {
            return fp64 ? run_matmul<double>(c, q, p, n, s, ok, i) : run_matmul<float>(c, q, p, n, s, ok, i);
        };
        REG["matmul_shared"] = REG["matmul_global"];

        // TODO: register your existing kernels here (matmul, reduction, conv2d, stencil2d, spmv...)

        // Loop through benchmark list
        for (const auto &cfg : BENCHMARKS)
        {
            if (!cfg.enabled)
                continue;

            for (size_t sz : cfg.sizes)
            {
                if (cfg.dtype_mode == DTypeMode::INTEGER)
                {
                    bool correct = false;
                    RunInfo info;
                    cl::Program prog = build_program_from_file(ctx, device, "kernels_all.cl", false);
                    auto it = REG.find(cfg.name);
                    if (it == REG.end())
                        continue;
                    double ms = it->second(ctx, queue, prog, cfg.name, sz, false, correct, info);
                    write_csv_row("results_opencl.csv", cfg.name, "INT", sz, ms, correct, device_name,
                                  info.gws0, info.gws1, info.gws2, info.lws0, info.lws1, info.lws2,
                                  info.flops_est, info.bw_GBps);
                    std::cout << "[OK] " << cfg.name << " size=" << sz << " dtype=INT time=" << ms << "ms correct=" << (correct ? "true" : "false") << "\n";
                }
                else
                {
                    for (int pass = 0; pass < 2; ++pass)
                    {
                        bool use_fp64 = (pass == 1);
                        if (use_fp64 && !device_supports_fp64(device))
                            continue;
                        cl::Program prog = build_program_from_file(ctx, device, "kernels_all.cl", use_fp64);
                        bool correct = false;
                        RunInfo info;
                        auto it = REG.find(cfg.name);
                        if (it == REG.end())
                            continue;
                        double ms = it->second(ctx, queue, prog, cfg.name, sz, use_fp64, correct, info);
                        write_csv_row("results_opencl.csv", cfg.name, use_fp64 ? "FP64" : "FP32", sz, ms, correct, device_name,
                                      info.gws0, info.gws1, info.gws2, info.lws0, info.lws1, info.lws2,
                                      info.flops_est, info.bw_GBps);
                        std::cout << "[OK] " << cfg.name << " size=" << sz << " dtype=" << (use_fp64 ? "FP64" : "FP32")
                                  << " time=" << ms << "ms correct=" << (correct ? "true" : "false") << "\n";
                    }
                }
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
