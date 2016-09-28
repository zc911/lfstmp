#!/bin/bash
cuda_version=$( nvcc --version | grep release | awk -F , '{print $2}' | awk '{print $2}' )
echo "Current CUDA version:" $cuda_version
cp build/lib/* ../lib/dgcaffe/Linux-x86_64/$cuda_version
cp -r include ../include/dgcaffe


