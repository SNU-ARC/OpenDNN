#!/bin/bash

git clone https://github.com/BVLC/caffe
cd caffe
git checkout 04ab089d 
cp ../../include/opendnn.h include/caffe

target=$1
if [ $target == cuda ]; then
  echo "Patch CUDA version of Caffe"
  patch -p1 < ../opendnn-to-caffe.patch
elif [ $target == ocl ]; then
  echo "Patch OpenCL version of Caffe"
  patch -p1 < ../opendnn-to-caffe-ocl.patch
else
  echo "Patch CPU version of Caffe"
  patch -p1 < ../opendnn-to-caffe-ocl.patch
fi
