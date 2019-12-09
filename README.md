# OpenDNN

OpenDNN is an open-source, cuDNN-like deep learning primitive library to support various framework and hardware architectures such as CPU, GPU, FPGA, and so on.
OpenDNN is implemented using CUDA and OpenCL and ported on popular DNN frameworks (Caffe and Tensorflow).
![OpenDNN Structure](/static/opendnn.png)

# Requirements
## Nvidia Driver
If you use an Nvidia GPU to run OpendNN, you should download Nvidia drivers first. You can download it on the Nvidia homepage and check the installation by the following command:
```nvidia-smi```
## CUDA
CUDA (Compute Unified Device Architecture) is a GPU-based parallel computing platform and application programming interface (API) created by Nvidia. You can install it using:
```sudo apt-get install cuda-10-0```
or download it from the Nvidia CUDA Driver homepage.
## cuDNN
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. You can install it using the command:
```sudo apt-get install libcudnn7-dev```
or download it from Nvidia CuDNN installation homepage. (We used v7.1.4 for tests) OpenDNN does not have dependency to cuDNN but, [Caffe example](#framework-integration-examples) has.
## OpenCL
We used OpenCL with C++ bindings (c++11 standards). Although OpenCL itself has no dependency to a specific OpenCL-supported device, we tested it on Nvidia GPU (Titan Xp)

# Directory Structure
```sh
├─OpenDNN
│  │  README.md
│  │  common.mk        # A common source that speficies a compile target (cpu, cuda, ocl)
│  ├─examples
│  │     download_and_patch_caffe.sh
│  │     Makefile.config
│  │     opendnn-to-caffe.patch
│  │     opendnn-to-caffe-ocl.patch
│  ├─test
│  │     unittest.cpp  # Unit tests for CPU and OpenCL
│  │     unittest.cu   # CUDA
│  │     Makefile
│  ├─lib
│  ├─include
│  │     opendnn.h
│  └─src
│     │  opendnn.cpp
│     │  opendnn_cl.cpp
│     │  opendnn_kernel.cl
│     │  opendnn.cu
│     │  opendnn_kernel.cuh
│     │  Makefile

```

# Installation
1. Select the CPU / GPU(CUDA) / OpenCL version of OpenDNN by changing the `TARGET` option to cpu / cuda / ocl, respectively, in `common.mk`. (*Note*: An FPGA version is not currently open due to a licensing issue.)
2. Enter the command
```make```
at the root of OpenDNN folder, and shared library will be built targeting the device you chose.
3. You should add the `LD_LIBRARY_PATH` to the current library directory ($opendnn/lib), or copy it to the system library and header path. (e.g. /usr/lib, /usr/include, etc.)

# Hello World!
In `test` folder, a small unit test for convolution is provided. You can build it by `make` and run it after setting `LD_LIBRARY_PATH` correctly.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$opendnn_root/lib
cd test
./unittest_xxx.exe
```

# Framework integration examples
We provide Caffe and TensorFlow (Experimental) integration with OpenDNN. For brevity, we replace cuDNN convolution forward layer into OpenDNN which is compatible with all three supported cases. (`cpu / cuda / ocl`)

## Caffe
1. Install OpenDNN into ```/usr/lib```. **Please check the TARGET option in ```common.mk```.** And the following instruction will copy OpenDNN library files into ```/usr/lib/``` which means from now on if you apply some changes on source codes and build them, you should also update the file ```/usr/lib/libopendnn.so``` after build. To do that, just type this.
```
make install
```
2. Download Caffe and apply patch. You should specify corresponding `$target` which is consistent with TARGET option in `common.mk`. (You can replace `$target` into `cpu / cuda / ocl`)
```
cd examples
./download_and_patch_caffe.sh $target
```
3. You can now check the difference between original Caffe codes and our patch. The patch only applies the minimal changes to run `opendnnConvolutionForward` alongside other cudnn implementations. For example, backward convolution operations are still performed with cuDNN.
```
cd caffe
git diff
```

4. Enable the cuDNN usage option in Caffe configuration file. (Makefile.config.example) We only tested our example with CUDA 10.0 and cuDNN 7.1.4. **Please make sure that ```USE_CUDNN``` must be set to 1 in Makefile.config.example**.

5. Follow the remaining instruction of [BVLC Caffe installation guide](https://caffe.berkeleyvision.org/install_apt.html)

6. Now you have built the OpenDNN port of Caffe. We recommend lenet with MNIST dataset for test. Please follow intructions of [```caffe/examples/mnist/README.md```](https://github.com/BVLC/caffe/blob/master/examples/mnist/readme.md).

7. You can check opendnn linking with following commands, or you can insert simple debug codes to the library internal (`opendnn.x`)
```
ldd ./examples/caffe/build/tools/caffe.bin | grep opendnn
```

## Difference between OpenDNN-CUDA and OpenDNN-OpenCL version
We have some inevitable decisions which is not compatible each other between CUDA and OpenCL.
- CUDA data communication btw. the device and the host is explicitly managed by users. (e.g. cudaMemcpy) This is the exactly same way of cuDNN.
- OpenCL data communication is implicitly managed by each API internal. (e.g. cl::CreateBuffer) The input & output float array should reside at the host-side memory, not the device. (This incurs redundant memory transactions, but is practical to be compatible with cuDNN)
- You can check the difference when applying OpenDNN API by ...
```
diff ./examples/opendnn-to-caffe-ocl.patch ./examples/opendnn-to-caffe.patch
```

## TensorFlow 1.4.1 (Experimental)
We provide an experimental OpenDNN patch for TensorFlow 1.4.1. Building TensorFlow from a source code is a long way and the version is outdated already. Thus, we just log a patch and hope to be helpful for someone. Several original cuDNN implementation with optimization (e.g. Autotuning of cuDNN algorithmic selection) is just discarded for portability. You can refer it as a baseline and try the similar way of Caffe for up-to-date TensorFlow. (Note that you should turn on cuDNN for this patch.)

0. Install OpenDNN. (As you did with Caffe. Keep this file (`/usr/lib/libopendnn.so`) to be up-to-date, when you change source codes of OpenDNN.)
```
make install
```

1. Clone v1.4.1 and apply patch
```
cd examples
git clone https://github.com/tesorflow/tensorflow
cd tensorflow
git checkout v1.4.1
cp -rf ../tensorflow-1.4.1-unofficial-patch .
cd tensorflow-1.4.1-unofficial-patch
./patch.sh
cd ../
git diff
```

2. Please follow the instruction of [TensorFlow 1.x - build from source](https://www.tensorflow.org/install/source)

# Main code contributors
Daeyeon Kim, [Young H. Oh](https://snu-arc.github.io/people/oyh/index.html)

# Reference
API descriptions and other information is available in the following thesis.

[Daeyeon Kim, "OpenDNN: An Open-source, cuDNN-like Deep Learning Primitive Library," M.S. Thesis, Department of Computer Science and Engineering, Seoul National University, February 2019.](http://s-space.snu.ac.kr/bitstream/10371/150799/1/000000154337.pdf)
