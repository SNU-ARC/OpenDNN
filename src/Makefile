include ../common.mk

# To use cuBLAS engine, use this flag
# LDFLAGS:=-L$(CUDA_DIR)/lib64 -lcublas -D__cuBLAS_ENGINE__

CFLAGS+=-O3 -I../include -I.

all: libopendnn.so libopendnn.a
	@echo "openDNN is built sucessfully !"
	@echo "use this command to set the path"
	@echo "export PATH:=/path/to/include"
	@echo "export LD_LIBRARY_PATH:=/path/to/lib(.so and .a)"

# Generating shared & static library for CUDA
ifeq ($(USE_CUDA), 1)
CFLAGS+=$(CUDA_ARCH)
libopendnn.so: opendnn.cu  opendnn_kernel.cuh
	$(MUTE) $(NVCC) -shared $< -o $@ -Xcompiler -fPIC $(CFLAGS) $(LDFLAGS) -DUSE_CUDA
opendnn.o: opendnn.cu  opendnn_kernel.cuh
	$(MUTE) $(NVCC) -dc $< -o $@ $(CFLAGS) $(LDFLAGS) -DUSE_CUDA

# Generating shared & static library for CPP
else
libopendnn.so: opendnn.cpp
	$(MUTE) $(GXX) -shared $< -o $@ -fPIC $(CFLAGS) $(LDFLAGS)
opendnn.o: opendnn.cpp
	$(MUTE) $(GXX) -c opendnn.cpp -o $@ $(CFLAGS) $(LDFLAGS)
endif

libopendnn.a: opendnn.o
	$(MUTE) ar r libopendnn.a opendnn.o 2> /dev/null

clean:
	$(MUTE) rm -f *.o *.so *.a
