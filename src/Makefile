include ../common.mk

# To use cuBLAS engine, use this flag
# LDFLAGS:=-L$(CUDA_DIR)/lib64 -lcublas -D__cuBLAS_ENGINE__

CFLAGS+=-O3 -I../include -I.

all: libopendnn.so
	@echo "openDNN is built sucessfully !"
	@echo "use this command to set the path"
	@echo "export PATH:=/path/to/include"
	@echo "export LD_LIBRARY_PATH:=/path/to/lib(.so)"

# Generating shared library for CUDA
ifeq ($(TARGET), cuda)
CFLAGS+=$(CUDA_ARCH)
libopendnn.so: opendnn.cu  opendnn_kernel.cuh
	$(MUTE) $(NVCC) -shared $< -o $@ -Xcompiler -fPIC $(CFLAGS) $(LDFLAGS) -DUSE_CUDA
	@cp $@ ../lib

# Generating shared library for OpenCL
else ifeq ($(TARGET), ocl)
LDFLAGS+=-lOpenCL
CFLAGS+=-std=c++11
libopendnn.so: opendnn_cl.cpp
	$(MUTE) $(GXX) -shared $< -o $@ -fPIC $(CFLAGS) $(LDFLAGS)
	@cp $@ opendnn_kernel.cl ../lib

# Generating shared library for CPP
else

libopendnn.so: opendnn.cpp
	$(MUTE) $(GXX) -shared $< -o $@ -fPIC $(CFLAGS) $(LDFLAGS)
	@cp $@ ../lib

endif

clean:
	$(MUTE) rm -f *.o *.so *.a
