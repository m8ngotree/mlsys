NVCC = nvcc
CFLAGS = -std=c++11 -allow-unsupported-compiler

OUTPUT = pack_unpack_naive_0.exe

all: $(OUTPUT)

pack_unpack_naive_0.exe: pack_unpack_naive_0.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.exe *.exp *.lib

.PHONY: all clean

