include ../common.mk
  
# Generating shared & static library for CUDA
ifeq ($(IF_CUDA), 1)
all:
	$(NVCC) unittest.cpp -lopendnn -I../include -o unittest.exe
else
all:
	$(GXX)  unittest.cpp -lopendnn -I../include -o unittest.exe
endif

clean:
	rm -rf *.exe *.o
