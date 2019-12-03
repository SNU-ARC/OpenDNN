include ../common.mk
  
# Generating shared & static library for CUDA
ifeq ($(USE_CUDA), 1)
all:
	@echo "CUDA is activated"
	$(NVCC) unittest.cu -L../lib -lopendnn -I../include -o unittest_gpu.exe
else
all:
	@echo "CUDA is not activated"
	$(GXX)  unittest.cpp -L../lib -lopendnn -I../include -o unittest_cpu.exe
endif

clean:
	rm -rf *.exe *.o
