// =============================================================
// utils_opencl.hpp
// OpenCL helpers: context/device init, program build, timing.
// =============================================================
#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS 0
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

struct RunInfo {
    // NDRange
    size_t gws0=0,gws1=0,gws2=0;
    size_t lws0=0,lws1=0,lws2=0;
    // Estimates (filled by host)
    double flops_est = 0.0;
    double bytes_moved = 0.0;
    double bw_GBps = 0.0; // computed as bytes_moved/(1e6*ms)
};

inline std::string get_device_name(const cl::Device& dev) {
    return dev.getInfo<CL_DEVICE_NAME>();
}
inline bool device_supports_fp64(const cl::Device& dev) {
    std::string ext = dev.getInfo<CL_DEVICE_EXTENSIONS>();
    return (ext.find("cl_khr_fp64") != std::string::npos);
}
inline cl::Context init_context(cl::Device& out_device) {
    std::vector<cl::Platform> plats; cl::Platform::get(&plats);
    if (plats.empty()) throw std::runtime_error("No OpenCL platforms found.");
    for (auto& p : plats) { std::vector<cl::Device> devs; p.getDevices(CL_DEVICE_TYPE_GPU, &devs); if (!devs.empty()) { out_device = devs[0]; return cl::Context(out_device); } }
    for (auto& p : plats) { std::vector<cl::Device> devs; p.getDevices(CL_DEVICE_TYPE_CPU, &devs); if (!devs.empty()) { out_device = devs[0]; return cl::Context(out_device); } }
    throw std::runtime_error("No suitable OpenCL device found.");
}
inline std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path); if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream oss; oss << ifs.rdbuf(); return oss.str();
}
inline cl::Program build_program_from_file(const cl::Context& ctx, const cl::Device& dev,
                                           const std::string& path, bool use_fp64)
{
    std::string src = read_text_file(path);
    cl::Program::Sources sources{ {src.c_str(), src.size()} };
    cl::Program prog(ctx, sources);
    std::string opts; if (use_fp64) opts += " -DUSE_FP64=1";
    cl_int err = prog.build({dev}, opts.c_str());
    if (err != CL_SUCCESS) {
        std::cerr << "[Build] Error code: " << err << "\n";
        std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        std::cerr << "[Build Log]\n" << log << "\n";
        throw std::runtime_error("OpenCL program build failed.");
    }
    return prog;
}
inline double get_event_ms(const cl::Event& evt) {
    cl_ulong s = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong e = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (double)(e - s) * 1e-6;
}
inline size_t round_up(size_t x, size_t base) {
    return ((x + base - 1) / base) * base;
}
