// =============================================================
// benchmark_config.hpp
// Defines which kernels to run, input sweep and dtype mode.
// CSV will include gws,lws,flops_est,bw_GBps.
// =============================================================
#pragma once
#include <vector>
#include <string>

enum class DTypeMode { FLOATING, INTEGER };

struct BenchConfig {
    std::string   name;          // kernel name (must match kernels_all.cl)
    std::vector<size_t> sizes;   // interpretation depends on kernel
    DTypeMode     dtype_mode;    // FLOATING (FP32/FP64) or INTEGER (INT)
    bool          enabled;       // whether to run
};

// NOTE: For INTEGER dtype kernels, host will only run "INT" path (no FP32/FP64 sweep).
// For FLOATING dtype kernels, host will sweep FP32 and FP64 (if device supports fp64).

static const std::vector<BenchConfig> BENCHMARKS = {
    // ---- Linear Algebra ----
    {"vecadd_basic",       {1ull<<14, 1ull<<16, 1ull<<18, 1ull<<20, 1ull<<22}, DTypeMode::FLOATING, true},
    {"dot_global",         {1ull<<16, 1ull<<18, 1ull<<20},                     DTypeMode::FLOATING, true},
    {"dot_shared",         {1ull<<16, 1ull<<18, 1ull<<20},                     DTypeMode::FLOATING, true},
    {"matmul_global",      {64, 128, 256, 512, 1024},                          DTypeMode::FLOATING, true},
    {"matmul_tiled",       {64, 128, 256, 512, 1024},                          DTypeMode::FLOATING, true},
    {"gemv_global",        {256, 512, 1024, 2048, 4096},                       DTypeMode::FLOATING, true},
    {"gemv_shared",        {256, 512, 1024, 2048, 4096},                       DTypeMode::FLOATING, true},
    {"reduction_shared",   {1ull<<16, 1ull<<18, 1ull<<20, 1ull<<22},           DTypeMode::FLOATING, true},
    {"scan_shared",        {1ull<<16, 1ull<<18, 1ull<<20},                     DTypeMode::FLOATING, true},

    // ---- ML Core Ops ----
    {"conv2d_global",      {64, 128, 256, 512},                                DTypeMode::FLOATING, true},
    {"conv2d_tiled",       {64, 128, 256, 512},                                DTypeMode::FLOATING, true},
    {"depthwiseconv_global",{64, 128, 256},                                    DTypeMode::FLOATING, false},
    {"depthwiseconv_tiled", {64, 128, 256},                                    DTypeMode::FLOATING, false},
    {"softmax_basic",      {256, 512, 1024, 2048},                             DTypeMode::FLOATING, true},
    {"layernorm_basic",    {256, 512, 1024, 2048},                             DTypeMode::FLOATING, true},
    {"activation_relu",    {1ull<<18, 1ull<<20, 1ull<<22},                     DTypeMode::FLOATING, true},
    {"activation_gelu",    {1ull<<18, 1ull<<20, 1ull<<22},                     DTypeMode::FLOATING, true},
    {"matmul_biasact_fused",{256, 512, 1024},                                  DTypeMode::FLOATING, false},
    {"attentioncore_basic",{256, 512},                                         DTypeMode::FLOATING, false},
    {"attentioncore_tiled",{256, 512},                                         DTypeMode::FLOATING, false},

    // ---- Graph / Irregular ----
    {"spmv_global",        {2048, 4096, 8192},                                 DTypeMode::FLOATING, true},
    {"bfs_basic",          {10000, 30000},                                     DTypeMode::INTEGER,  false},
    {"dfs_basic",          {10000, 30000},                                     DTypeMode::INTEGER,  false},
    {"pagerank_basic",     {4096, 8192},                                       DTypeMode::FLOATING, false},

    // ---- Numerical / Physics ----
    {"stencil2d_global",   {64, 128, 256, 512},                                DTypeMode::FLOATING, true},
    {"stencil2d_shared",   {64, 128, 256, 512},                                DTypeMode::FLOATING, true},
    {"stencil3d_global",   {64, 96, 128},                                      DTypeMode::FLOATING, false},
    {"stencil3d_shared",   {64, 96, 128},                                      DTypeMode::FLOATING, false},
    {"fft1d_global",       {1<<12, 1<<14, 1<<16},                              DTypeMode::FLOATING, false},
    {"fft1d_staged",       {1<<12, 1<<14, 1<<16},                              DTypeMode::FLOATING, false},
    {"histogram_global",   {1ull<<20, 1ull<<22},                               DTypeMode::INTEGER,  true},
    {"histogram_shared",   {1ull<<20, 1ull<<22},                               DTypeMode::INTEGER,  true},
    {"sort_bitonic",       {1<<16, 1<<18},                                     DTypeMode::INTEGER,  true},
    {"montecarlo_basic",   {1ull<<20, 1ull<<22},                               DTypeMode::FLOATING, true}
};
