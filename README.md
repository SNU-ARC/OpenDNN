# OpenDNN

OpenDNN is an open-source, cuDNN-like deep learning primitive library to support various framework and hardware architectures such as FPGA and acceleartor.
OpenDNN is implemented as CUDA and OpenCL and ported on popular DNN frameworks (Caffe, Tensorflow).

# Requirements (GPU)
## Nvidia Driver
If you use Nvidia GPU and want to use it to accelerate deep neural networks, you should download Nvidia Driver to communicate with GPU as general purpose. You can download it on the Nvidia homepage and check the installation by the command:
```nvidia-smi```
## CUDA
CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia. You can install it using:
```sudo apt-get install cuda-10-0```
or download it on Nvidia CUDA Driver homepage.
## CuDNN
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. You can install it using the command:
```sudo apt-get install libcudnn7-dev```
or download it on Nvidia CuDNN install homepage.

# Structure
```sh
├─OpenDNN
│  │  README.md
│  │  common.mk
│  ├─include
│  │     opendnn.h
│  └─src
│     │  opendnn.cpp
│     │  opendnn.cu
│     │  opendnn_kernel.cuh
│     │  Makefile
```

# Install
1. You select the CPU/GPU version library by changing `IF_CUDA` parameter to 0/1 in `common.mk`.
2. When you enter the command
```make```
in the `src` folder, then shared & static library will be built.
3. You should add the `PATH` and `LD_LIBRARY_PATH` on the current library, or copy it on the default library and binary path.

Here is the [http://s-space.snu.ac.kr/bitstream/10371/150799/1/000000154337.pdf](Paper) Link.
