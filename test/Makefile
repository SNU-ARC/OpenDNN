include ../common.mk

ifeq ($(TARGET), cuda)
BIN=unittest_gpu.exe
else ifeq ($(TARGET), ocl)
BIN=unittest_ocl.exe
else
BIN=unittest_cpu.exe
endif

all: $(BIN)

unittest_gpu.exe: unittest.cu
	@echo "TARGET: GPU-CUDA"
	$(NVCC) $< -L../lib -lopendnn -I../include -o $@

unittest_ocl.exe: unittest.cpp
	@echo "TARGET: OpenCL"
	$(GXX) $< -L../lib -lopendnn -I../include -o $@

unittest_cpu.exe: unittest.cpp
	@echo "TARGET: Sequential CPU"
	$(GXX) $< -L../lib -lopendnn -I../include -o $@

clean:
	rm -rf *.exe *.o
