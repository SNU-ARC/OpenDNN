# OpenDNN

OpenDNN is an open-source, cuDNN-like deep learning primitive library to support various framework and hardware architectures such as CPU, GPU, FPGA, and so on.
OpenDNN is implemented using CUDA and OpenCL and ported on popular DNN frameworks (Caffe, Tensorflow).
![OpenDNN Structure](/static/opendnn.png)

# Requirements for GPU
## Nvidia Driver
If you use an Nvidia GPU to run OpendNN, you should download Nvidia drivers first. You can download it on the Nvidia homepage and check the installation by the following command:
```nvidia-smi```
## CUDA
CUDA (Compute Unified Device Architecture) is a GPU-based parallel computing platform and application programming interface (API) created by Nvidia. You can install it using:
```sudo apt-get install cuda-10-0```
or download it from the Nvidia CUDA Driver homepage.
## CuDNN
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. You can install it using the command:
```sudo apt-get install libcudnn7-dev```
or download it from Nvidia CuDNN installation homepage.

# Directory Structure
```sh
├─OpenDNN
│  │  README.md
│  │  common.mk
│  ├─examples
│  │     download_and_patch_caffe.sh
│  │     Makefile.config
│  │     opendnn-to-caffe.patch
│  ├─test
│  │     unittest.cpp
│  │     unittest.cu
│  │     Makefile
│  ├─lib
│  ├─include
│  │     opendnn.h
│  └─src
│     │  opendnn.cpp
│     │  opendnn.cu
│     │  opendnn_kernel.cuh
│     │  Makefile

```

# Installation
1. Select the CPU or GPU version of OpenDNN by changing the `USE_CUDA` flag to 0 (CPU)/1 (GPU) in `common.mk`. (*Note*: An FPGA version is not currently open due to a licensing issue.)
2. Enter the command
```make```
at the root of OpenDNN folder, and shared & static library will be built targeting the device you chose.
3. You should add the `PATH` and `LD_LIBRARY_PATH` to the current library directory ($opendnn/lib), or copy it to the default library and binary paths. (e.g. /usr/lib)

# Hello World!
In `./test` folder, a small unit test for convolution is provided. You can build it by `make` and run it after setting `PATH` and `LD_LIBRARY_PATH` correctly.

# Caffe
1. Install opendnn into ```/usr/lib```. **Please check USE_CUDA option in ```common.mk``` is 1.** And the following instruction will copy opendnn library files into ```/usr/lib/``` which means from now on if you apply some changes on source codes and build them, you should update the file ```/usr/lib/libopendnn.so```.
```
make install
```
2. Download Caffe and apply patch.
```
cd examples
./download_and_patch_caffe.sh
```
3. You can now check the difference between original source codes and our patch.
```
cd caffe
git diff
```
4. Enable the cuDNN usage option in Caffe configuration file. (Makefile.config.example) We only tested our example with CUDA 9.0 and cuDNN 7.1.4. **Please make sure that ```USE_CUDNN``` must be set to 1 in Makefile.config.example**.

5. Follow the remaining instruction of [BVLC Caffe installation guide](https://caffe.berkeleyvision.org/install_apt.html)

6. Now you have built the openDNN port of Caffe. We recommend lenet with MNIST dataset for test. Please follow intructions of [```caffe/examples/mnist/README.md```](https://github.com/BVLC/caffe/blob/master/examples/mnist/readme.md).

# Reference
API descriptions and other information is available in the following thesis.

[Daeyeon Kim, "OpenDNN: An Open-source, cuDNN-like Deep Learning Primitive Library," M.S. Thesis, Department of Computer Science and Engineering, Seoul National University, February 2019.](http://s-space.snu.ac.kr/bitstream/10371/150799/1/000000154337.pdf)

