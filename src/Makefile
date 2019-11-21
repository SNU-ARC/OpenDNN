include ../common.mk

CXX_PRJ:=..

CFLAGS+=-I$(CXX_PRJ) -I.
# To use cuBLAS engine, use this flag
# LDFLAGS:=-L$(CUDA_DIR)/lib64 -lcublas -D__cuBLAS_ENGINE__

all: libopendnn.so libopendnn.a
	@echo "libnumber with opendnn is built sucessfully !"
	@echo "use this command to set the path"
	@echo "export PATH:=/path/to/include"
	@echo "export LD_LIBRARY_PATH:=/path/to/lib(.so and .a)"

# Generating shared & static library for CUDA
ifeq ($(IF_CUDA), 1)
libopendnn.so: opendnn.cu  opendnn_kernel.cuh
	$(SILENCE) $(NVCC) -shared $(filter-out $(wildcard *.cuh $(CXX_PRJ)/*.cuh), $^) -o $@ -Xcompiler -fPIC -O3 $(LDFLAGS) -Dshared_lib
opendnn.o: opendnn.cu  opendnn_kernel.cuh
	$(SILENCE) $(NVCC) -dc $< -o $@ $(CFLAGS) $(LDFLAGS) -Dstatic_lib

# Generating shared & static library for CPP
else
libopendnn.so: opendnn.cpp
	$(SILENCE) $(GXX) -shared $(wildcard *.cpp $(CXX_PRJ)/*.cpp) -o $@ -fPIC -O3 $(LDFLAGS) -Dshared-lib
opendnn.o: opendnn.cpp
	$(SILENCE) $(GXX) -c opendnn.cpp -o $@ -O3 $(LDFLAGS) -Dstatic_lib
endif

libopendnn.a: opendnn.o
	$(SILENCE) ar r libopendnn.a opendnn.o 2> /dev/null

clean:
	rm -f *.o *.so *.a
