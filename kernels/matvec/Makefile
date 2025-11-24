NVCC = nvcc
CFLAGS = -std=c++17 -allow-unsupported-compiler -lcublas

INCLUDE_DIR = include
KERNELS_DIR = kernels
BUILD_DIR = build

KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCES = src.cu
CUDA_SOURCES = $(KERNEL_SOURCES) $(ROOT_SOURCES)

CUDA_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%.obj,$(KERNEL_SOURCES)) $(BUILD_DIR)/src.obj

OUTPUT = matvec.exe

all: $(OUTPUT)

$(OUTPUT): $(CUDA_OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.obj: $(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/src.obj: src.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

clean:
	rm $(BUILD_DIR)/*.obj *.exe *.exp *.lib

.PHONY: all clean

