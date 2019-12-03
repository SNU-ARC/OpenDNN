#!/bin/bash

git clone https://github.com/BVLC/caffe
cd caffe
git checkout 04ab089d 
cp ../../include/opendnn.h include/caffe
patch -p1 < ../opendnn-to-caffe.patch
