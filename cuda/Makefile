NVCC = nvcc
CFLAGS = -O2 -std=c++17
TARGET = run_cuda

$(TARGET): run_cuda.cu kernels_all.cu utils_cuda.hpp
	$(NVCC) $(CFLAGS) -o $(TARGET) run_cuda.cu kernels_all.cu


clean:
	rm -f $(TARGET)
