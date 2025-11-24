NVCC = nvcc
CFLAGS = -std=c++17 -allow-unsupported-compiler

INCLUDE_DIR = include
KERNELS_DIR = kernels
BUILD_DIR = build

SOURCE = main.cu
OUTPUT = layernorm.exe

all: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $<

clean:
	rm -rf $(BUILD_DIR) *.exe *.exp *.lib

.PHONY: all clean

