// =============================================================
// math_utils.hpp
// Random init and result comparison helpers.
// =============================================================
#pragma once
#include <vector>
#include <random>
#include <cmath>

template<typename T>
inline void fill_random(std::vector<T>& v, T low, T high, uint32_t seed = 123)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist((double)low, (double)high);
    for (auto& x : v) x = (T)dist(rng);
}

template<typename T>
inline bool compare_results(const std::vector<T>& a,
                            const std::vector<T>& b,
                            double eps)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = (double)a[i], db = (double)b[i];
        double diff = std::fabs(da - db);
        double denom = std::max(1.0, std::max(std::fabs(da), std::fabs(db)));
        if (diff / denom > eps) return false;
    }
    return true;
}
