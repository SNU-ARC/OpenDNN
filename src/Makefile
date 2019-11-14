include ../common.mk

CXX_PRJ:=..

CFLAGS+=-I$(CXX_PRJ) -I.
# To use cuBLAS engine, use this flag
# LDFLAGS:=-L$(CUDA_DIR)/lib64 -lcublas -D__cuBLAS_ENGINE__

all: libopendnn.so libopendnn.a
	@echo "libnumber with opendnn is built sucessfully !"

# Generating shared library
libopendnn.so: opendnn.cu  opendnn_kernel.cuh
	$(SILENCE) $(NVCC) -shared $(filter-out $(wildcard *.cuh $(CXX_PRJ)/*.cuh), $^) -o $@ -Xcompiler -fPIC $(CFLAGS) $(LDFLAGS) -Dshared_lib

# Generating static library
opendnn.o: opendnn.cu  opendnn_kernel.cuh
	$(SILENCE) $(NVCC) -dc $< -o $@ $(CFLAGS) $(LDFLAGS) -Dstatic_lib

libopendnn.a: opendnn.o
	$(SILENCE) ar r libopendnn.a opendnn.o 2> /dev/null

clean:
	rm -f *.o *.so *.a
