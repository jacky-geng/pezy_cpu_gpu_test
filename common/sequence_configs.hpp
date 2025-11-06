#pragma once

#include <cstddef>
#include <vector>
#include <string>

struct SequenceProblemSize
{
    std::size_t num_pairs;
    std::size_t sequence_length;
    const char* label;
};

inline const std::vector<SequenceProblemSize>& smithwaterman_problem_sizes()
{
    static const std::vector<SequenceProblemSize> sizes = {
        {1024, 128,   "pairs1024_len128"},
        {2048, 256,   "pairs2048_len256"},
        {4096, 256,   "pairs4096_len256"},
        {8192, 256,   "pairs8192_len256"},
        {16384, 256,  "pairs16384_len256"},
        {32768, 256,  "pairs32768_len256"},
    };
    return sizes;
}

inline const std::vector<SequenceProblemSize>& wfa_problem_sizes()
{
    static const std::vector<SequenceProblemSize> sizes = {
        {1024, 128,   "pairs1024_len128"},
        {2048, 256,   "pairs2048_len256"},
        {4096, 256,   "pairs4096_len256"},
        {8192, 256,   "pairs8192_len256"},
        {16384, 256,  "pairs16384_len256"},
        {32768, 256,  "pairs32768_len256"},
    };
    return sizes;
}
