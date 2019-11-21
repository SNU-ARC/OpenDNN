include ../common.mk
  
# Generating shared & static library for CUDA
ifeq ($(IF_CUDA), 1)
all:
	echo $(IF_CUDA)
	$(NVCC) unittest.cpp -L../src -lopendnn -I../include -o unittest.exe
else
all:
	$(GXX)  unittest.cpp -L../src -lopendnn -I../include -o unittest.exe
endif

clean:
	rm -rf *.exe *.o
