// =============================================================
// cuda/benchmark_config.hpp
// Linear Algebra benchmark definitions for CUDA path.
// Mirrors the OpenCL layout but starts with a reduced set.
// =============================================================
#pragma once
#include <vector>
#include <string>

enum class DTypeMode
{
    FLOATING,
    INTEGER
};

struct BenchConfig
{
    std::string name;           // kernel identifier
    std::vector<size_t> sizes;  // problem sizes (meaning depends on kernel)
    DTypeMode dtype_mode;       // FLOATING or INTEGER
    bool enabled;               // set false to skip a particular workload
};

// Focus on linear algebra kernels first; other categories can be added later.
static const std::vector<BenchConfig> BENCHMARKS = {
    {"vecadd_basic",     {1 << 16, 1 << 18, 1 << 20}, DTypeMode::FLOATING, true},
    {"dot_global",       {1 << 16, 1 << 18, 1 << 20}, DTypeMode::FLOATING, true},
    {"dot_shared",       {1 << 16, 1 << 18, 1 << 20}, DTypeMode::FLOATING, true},
    {"gemv_global",      {256, 512, 1024},            DTypeMode::FLOATING, true},
    {"gemv_shared",      {256, 512, 1024},            DTypeMode::FLOATING, true},
    {"matmul_global",    {64, 128, 256},              DTypeMode::FLOATING, true},
    {"matmul_shared",    {64, 128, 256},              DTypeMode::FLOATING, true},
    {"reduction_global", {1 << 18, 1 << 20},          DTypeMode::FLOATING, true},
    {"reduction_shared", {1 << 18, 1 << 20},          DTypeMode::FLOATING, true},
    {"scan_shared",      {1 << 16, 1 << 18},          DTypeMode::FLOATING, true},
    {"spmv_csr",         {1024, 4096},                DTypeMode::FLOATING, true},
    // 2. ML Core Ops
    {"conv2d_global",    {16, 32, 64},                DTypeMode::FLOATING, true},
    {"conv2d_shared",    {16, 32, 64},                DTypeMode::FLOATING, true},
    {"depthwiseconv_global", {16, 32, 64},            DTypeMode::FLOATING, true},
    {"depthwiseconv_tiled",  {16, 32, 64},            DTypeMode::FLOATING, true},
    {"softmax_basic",    {64, 128, 256},              DTypeMode::FLOATING, true},
    {"layernorm_basic",  {64, 128, 256},              DTypeMode::FLOATING, true},
    {"activation_relu",  {1 << 12, 1 << 14},          DTypeMode::FLOATING, true},
    {"activation_gelu",  {1 << 12, 1 << 14},          DTypeMode::FLOATING, true},
    // 3. Graph / Irregular
    {"bfs_basic",        {256, 512},                  DTypeMode::INTEGER, true},
    {"dfs_basic",        {256, 512},                  DTypeMode::INTEGER, true},
    {"pagerank_basic",   {128, 256},                  DTypeMode::FLOATING, true},
    // 4. Numerical / Physics
    {"stencil2d_3x3",    {32, 64},                    DTypeMode::FLOATING, true},
    {"stencil2d_5x5",    {32, 64},                    DTypeMode::FLOATING, true},
    {"stencil3d_global", {16, 32},                    DTypeMode::FLOATING, true},
    {"stencil3d_shared", {16, 32},                    DTypeMode::FLOATING, true},
    {"histogram_global", {1 << 10, 1 << 12},          DTypeMode::INTEGER, true},
    {"histogram_shared", {1 << 10, 1 << 12},          DTypeMode::INTEGER, true},
    {"sort_bitonic",     {1 << 8, 1 << 10},           DTypeMode::INTEGER, true},
    {"montecarlo_basic", {1 << 18, 1 << 20},          DTypeMode::FLOATING, true},
    {"fft1d_global",     {1 << 10, 1 << 12},          DTypeMode::FLOATING, true},
    {"fft1d_staged",     {1 << 10, 1 << 12},          DTypeMode::FLOATING, true}
    // Integer or advanced categories can be appended in future iterations.
};
