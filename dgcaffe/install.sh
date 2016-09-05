#!/bin/bash
cp build/lib/* ../apps/lib/dgcaffe/Linux-x86_64/
cp build/lib/* ../engine/lib/dgcaffe/Linux-x86_64/
cp build/lib/* ../apps/bin/Debug/libs/
cp build/lib/* ../apps/bin/Release/libs/
cp -r include/caffe ../engine/include/dgcaffe/

