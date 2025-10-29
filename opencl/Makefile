# =============================================================
# Makefile for OpenCL benchmarks (Linux default)
# Requires: OpenCL headers + libOpenCL
# On Ubuntu: sudo apt-get install ocl-icd-opencl-dev
# =============================================================

CXX      := g++
CXXFLAGS := -O2 -std=c++17 -I. -I../common
LDFLAGS  := -lOpenCL

SRC := run_opencl.cpp
BIN := run_opencl

all: $(BIN)

$(BIN): $(SRC) utils_opencl.hpp benchmark_config.hpp ../common/csv_writer.hpp ../common/math_utils.hpp ../common/baseline_check.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN)
