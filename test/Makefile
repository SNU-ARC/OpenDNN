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
	@echo "TEST TARGET: cuda"
	$(MUTE) $(NVCC) $< -L../lib -lopendnn -I../include -o $@ -DUSE_CUDA

unittest_ocl.exe: unittest.cpp
	@echo "TEST TARGET: ocl"
	$(MUTE) $(GXX) $< -L../lib -lopendnn -I../include -o $@

unittest_cpu.exe: unittest.cpp
	@echo "TEST TARGET: cpu"
	$(MUTE) $(GXX) $< -L../lib -lopendnn -I../include -o $@

clean:
	@rm -rf *.exe *.o
