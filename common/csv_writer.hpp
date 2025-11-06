// =============================================================
// csv_writer.hpp
// Append benchmark results to a CSV file with extra fields.
// =============================================================
#pragma once
#include <fstream>
#include <string>
#include <sys/stat.h>

inline bool file_exists(const std::string& path) {
    struct stat st; return (stat(path.c_str(), &st) == 0);
}

inline void write_csv_row(const std::string& path,
                          const std::string& kernel,
                          const std::string& dtype,
                          const std::string& input_size,
                          double runtime_ms,
                          bool correct,
                          const std::string& device,
                          size_t gws0, size_t gws1, size_t gws2,
                          size_t lws0, size_t lws1, size_t lws2,
                          double flops_est, double bw_GBps)
{
    bool need_header = !file_exists(path);
    std::ofstream ofs(path, std::ios::app);
    if (need_header) {
        ofs << "kernel_name,dtype,input_size,runtime_ms,correct,device_name,"
               "gws0,gws1,gws2,lws0,lws1,lws2,flops_est,bw_GBps\n";
    }
    ofs << kernel << "," << dtype << "," << input_size << ","
        << runtime_ms << "," << (correct ? "true" : "false") << ","
        << "\"" << device << "\"" << ","
        << gws0 << "," << gws1 << "," << gws2 << ","
        << lws0 << "," << lws1 << "," << lws2 << ","
        << flops_est << "," << bw_GBps << "\n";
}
