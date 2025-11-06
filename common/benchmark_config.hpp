#pragma once
#include <string>
#include <vector>

#include "sequence_configs.hpp"

// Sequence sweep indices used below:
//   0 -> pairs1024_len128
//   1 -> pairs2048_len256
//   2 -> pairs4096_len256
//   3 -> pairs8192_len256
//   4 -> pairs16384_len256
//   5 -> pairs32768_len256

enum class DTypeMode
{
    FLOATING,
    INTEGER
};

struct BenchConfig
{
    std::string name;
    std::vector<size_t> sizes;
    DTypeMode dtype_mode;
    bool enabled;
};

static const std::vector<BenchConfig> BENCHMARKS = {
    // 1. Linear Algebra
    {"vecadd_basic",     {1 << 6, 1 << 8, 1 << 10, 1 << 12}, DTypeMode::FLOATING, true},
    {"dot_global",       {1 << 8, 1 << 10, 1 << 12},         DTypeMode::FLOATING, true},
    {"dot_shared",       {1 << 8, 1 << 10, 1 << 12},         DTypeMode::FLOATING, true},
    {"gemv_global",      {16, 32, 64, 128},                  DTypeMode::FLOATING, true},
    {"gemv_shared",      {16, 32, 64, 128},                  DTypeMode::FLOATING, false},
    {"matmul_global",    {16, 32, 64, 128},                  DTypeMode::FLOATING, true},
    {"matmul_shared",    {16, 32, 64, 128},                  DTypeMode::FLOATING, true},
    {"reduction_global", {1 << 8, 1 << 10, 1 << 12},         DTypeMode::FLOATING, true},
    {"reduction_shared", {1 << 8, 1 << 10, 1 << 12},         DTypeMode::FLOATING, true},
    {"scan_shared",      {1 << 8, 1 << 10, 1 << 12},         DTypeMode::FLOATING, true},
    {"spmv_csr",         {128, 256, 512},                    DTypeMode::FLOATING, true},

    // 2. ML Core Ops
    {"conv2d_global",        {16, 32, 64},   DTypeMode::FLOATING, true},
    {"conv2d_shared",        {16, 32, 64},   DTypeMode::FLOATING, true},
    {"depthwiseconv_global", {16, 32, 64},   DTypeMode::FLOATING, true},
    {"depthwiseconv_tiled",  {16, 32, 64},   DTypeMode::FLOATING, true},
    {"softmax_basic",        {64, 128, 256}, DTypeMode::FLOATING, true},
    {"layernorm_basic",      {64, 128, 256}, DTypeMode::FLOATING, true},
    {"activation_relu",      {1 << 10, 1 << 12}, DTypeMode::FLOATING, true},
    {"activation_gelu",      {1 << 10, 1 << 12}, DTypeMode::FLOATING, true},

    // 3. Graph / Irregular
    {"bfs_basic",      {256, 512},    DTypeMode::INTEGER,  true},
    {"dfs_basic",      {256, 512},    DTypeMode::INTEGER,  true},
    {"pagerank_basic", {128, 256},    DTypeMode::FLOATING, true},

    // 4. Numerical / Physics
    {"stencil2d_3x3",    {32, 64},                  DTypeMode::FLOATING, true},
    {"stencil2d_5x5",    {32, 64},                  DTypeMode::FLOATING, true},
    {"stencil3d_global", {16, 32},                  DTypeMode::FLOATING, true},
    {"stencil3d_shared", {16, 32},                  DTypeMode::FLOATING, true},
    {"histogram_global", {1 << 10, 1 << 12},        DTypeMode::INTEGER,  true},
    {"histogram_shared", {1 << 10, 1 << 12},        DTypeMode::INTEGER,  true},
    {"sort_bitonic",     {1 << 8, 1 << 10},         DTypeMode::INTEGER,  true},
    {"montecarlo_basic", {1 << 10, 1 << 12},        DTypeMode::FLOATING, true},
    {"fft1d_global",     {1 << 8, 1 << 10, 1 << 12}, DTypeMode::FLOATING, true},
    {"fft1d_staged",     {1 << 8, 1 << 10, 1 << 12}, DTypeMode::FLOATING, true},

    // 5. Sequence Alignment
    {"smithwaterman_basic",     {0, 1, 2, 3, 4, 5}, DTypeMode::INTEGER, true},
    {"smithwaterman_wavefront", {0, 1, 2, 3, 4, 5}, DTypeMode::INTEGER, true},
    {"wfa_editdistance",        {0, 1, 2, 3, 4, 5}, DTypeMode::INTEGER, true}
};
