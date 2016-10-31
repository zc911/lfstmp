# DGFace 

### Introduction
This repo is the SDK warpper for face detection, alignment, tracking and feature extraction algorithms.

### Team

- jiajiachen@deepglint.com
- xiaodongsun@deepglint.com
- yafengdeng@deepglint.com
- zhenchen@deepglint.com
- zhenzuo@deepglint.com
- ziyongfeng@deepglint.com

### Dependency

- dgcaffe
- cudart
- cudnn
- opencv
- protobuf
TBD



### How to build

1. Install Cmake
2. Install git-lfs for large file support, refer to doc/git-lfs.md
3. Install conan for libraries management, refer to doc/conan.md
4. $ mkdir build && cd build && conan install .. && cmake ..
5. $ make

