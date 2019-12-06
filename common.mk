NVCC:=nvcc
GXX:=g++
CUDA_ARCH:= -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_52,code=sm_52
            # -gencode arch=compute_70,code=sm_70

CUDA_PATH:=/usr/local/cuda

# Mute all verbose messages
MUTE=@
MAKE:=@$(MAKE) --no-print-directory
MAKEVERBOSE:=make

GREEN=\033[1;32m
RED=\033[1;31m
END=\033[0m

# Target option
# CPU: cpu (opendnn.cpp)
# GPU-CUDA: cuda (opendnn.cu)
# OpenCL: ocl (opendnn_cl.c)
TARGET:=cuda
