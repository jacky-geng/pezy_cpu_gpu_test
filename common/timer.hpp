// =============================================================
// timer.hpp
// Simple CPU wall-clock timer (not used for kernel timing).
// =============================================================
#pragma once
#include <chrono>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void tic() { t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
