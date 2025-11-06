CUDA_DIR   := cuda
OPENCL_DIR := opencl

.PHONY: all cuda opencl build-cuda build-opencl clean

all: cuda opencl

build-cuda:
	$(MAKE) -C $(CUDA_DIR)

build-opencl:
	$(MAKE) -C $(OPENCL_DIR)

cuda:
	$(MAKE) -C $(CUDA_DIR) run

opencl:
	$(MAKE) -C $(OPENCL_DIR) run

clean:
	$(MAKE) -C $(CUDA_DIR) clean
	$(MAKE) -C $(OPENCL_DIR) clean
