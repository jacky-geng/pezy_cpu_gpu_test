// =============================================================
// math_utils.hpp
// Simple numeric utilities: random fill, comparison, etc.
// =============================================================
#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// =============================================================
// Fill vector with random values between [lo, hi]
// =============================================================
template <typename T>
void fill_random(std::vector<T>& v, T lo, T hi) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist((double)lo, (double)hi);
    for (auto& x : v) x = static_cast<T>(dist(rng));
}

// =============================================================
// Compare two arrays within tolerance
// Equivalent to numpy.allclose()
// =============================================================
template <typename T>
bool allclose(const std::vector<T>& a, const std::vector<T>& b,
              double rtol = 1e-5, double atol = 1e-8) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs((double)a[i] - (double)b[i]);
        double tol = atol + rtol * std::abs((double)b[i]);
        if (diff > tol) return false;
    }
    return true;
}
