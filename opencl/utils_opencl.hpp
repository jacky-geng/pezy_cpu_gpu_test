// =============================================================
// utils_opencl.hpp
// Common OpenCL utilities for context creation, timing, helpers.
// Compatible with run_opencl.cpp
// =============================================================
#pragma once

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif

#include <CL/opencl.hpp>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

// =============================================================
// Basic info structure for benchmark results
// =============================================================
struct RunInfo
{
    size_t gws0 = 0, gws1 = 1, gws2 = 1; // global work sizes
    size_t lws0 = 0, lws1 = 1, lws2 = 1; // local work sizes
    double flops_est = 0.0;              // estimated floating ops
    double bytes_moved = 0.0;            // estimated memory bytes moved
    double bw_GBps = 0.0;                // derived bandwidth (GB/s)
};

// =============================================================
// Device query helpers
// =============================================================
inline std::string device_name_string(const cl::Device &dev)
{
    std::string s;
    dev.getInfo(CL_DEVICE_NAME, &s);
    return s;
}

// =============================================================
// Rounding helpers
// =============================================================
inline size_t round_up(size_t x, size_t base)
{
    return (x + base - 1) / base * base;
}

inline size_t floor_pow2(size_t x, size_t cap)
{
    size_t p = 1;
    while ((p << 1) <= x && (p << 1) <= cap)
        p <<= 1;
    return p;
}

// =============================================================
// Context and queue creation
// =============================================================
inline void create_context_queue(cl::Context &ctx, cl::Device &dev, cl::CommandQueue &q)
{
    std::vector<cl::Platform> plats;
    cl::Platform::get(&plats);
    if (plats.empty())
        throw std::runtime_error("No OpenCL platforms found.");

    for (auto &p : plats)
    {
        std::vector<cl::Device> devs;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devs);
        if (!devs.empty())
        {
            // Prefer GPU, fallback to first device
            for (auto &d : devs)
            {
                cl_device_type type = d.getInfo<CL_DEVICE_TYPE>();
                if (type == CL_DEVICE_TYPE_GPU)
                {
                    dev = d;
                    ctx = cl::Context({d});
                    q = cl::CommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE);
                    return;
                }
            }
            dev = devs[0];
            ctx = cl::Context({dev});
            q = cl::CommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
            return;
        }
    }
    throw std::runtime_error("No OpenCL devices available.");
}

// =============================================================
// Timed kernel execution helper
// =============================================================
inline double enqueue_timed(cl::CommandQueue &q, cl::Kernel &krn,
                            cl::NDRange gws, cl::NDRange lws,
                            const std::vector<cl::Event> *wait = nullptr)
{
    cl::Event evt;
    q.enqueueNDRangeKernel(krn, cl::NullRange, gws, lws, wait, &evt);
    q.finish();
    cl_ulong t0 = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t1 = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (t1 - t0) * 1e-6; // milliseconds
}

// =============================================================
// Program builder with optional FP64 support
// =============================================================
inline cl::Program build_program_from_file(cl::Context &ctx, const cl::Device &dev,
                                           const std::string &path, bool enable_fp64)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open kernel file: " + path);
    }

    std::string src((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources;
    sources.push_back({src.c_str(), src.size()});
    cl::Program prog(ctx, sources);

    std::string version = dev.getInfo<CL_DEVICE_VERSION>();
    int major = 1, minor = 2;
    if (std::sscanf(version.c_str(), "OpenCL %d.%d", &major, &minor) != 2)
    {
        major = 1;
        minor = 2;
    }

    std::string opts = (major > 2 || (major == 2 && minor >= 0)) ? "-cl-std=CL2.0"
                                                                  : "-cl-std=CL1.2";
    if (enable_fp64)
        opts += " -DENABLE_FP64=1";

    try
    {
        prog.build({dev}, opts.c_str());
    }
    catch (const cl::Error &e)
    {
        std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        std::cerr << "[Build] Error code: " << e.err() << "\n[Build Log]\n"
                  << log << std::endl;
        throw;
    }
    return prog;
}
