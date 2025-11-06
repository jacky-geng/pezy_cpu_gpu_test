NVCC   := nvcc
CFLAGS := -O2 -std=c++17
TARGET := run_cuda

.PHONY: all run clean

all: $(TARGET)

$(TARGET): run_cuda.cu kernels_all.cu utils_cuda.hpp
	$(NVCC) $(CFLAGS) -o $(TARGET) run_cuda.cu kernels_all.cu

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
