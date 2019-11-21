NVCC:=nvcc
GXX:=g++
CUDA_ARCH:= -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_52,code=sm_52
            # -gencode arch=compute_70,code=sm_70
CFLAGS:=-O3 $(CUDA_ARCH)

CUDA_PATH:=/usr/local/cuda

# Silence all verbose messages
SILENCE=@
MAKE:=@$(MAKE) --no-print-directory
MAKEVERBOSE:=make

GREEN=\033[1;32m
RED=\033[1;31m
END=\033[0m

# Option for CPP and CU (Default: CPU (0))
IF_CUDA:=0
