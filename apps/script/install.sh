#!/bin/bash

if [ "$source" == "" ]; then source="http://192.168.2.21:80"; fi

# change apt source
wget -O sources.list "$source/apt/sources.list" && mv sources.list /etc/apt/sources.list

wget -O install_cuda.sh "$source/cuda/install.sh" && chmod +x install_cuda.sh
wget -O install_libs.sh "$source/libs/install.sh" && chmod +x install_libs.sh

wget -O dog_tool "$source/libs/dog_tool" && chmod +x dog_tool
wget -O pull.sh "$source/matrix/pull.sh" && chmod +x pull.sh
wget -O run.sh "$source/matrix/run.sh" && chmod +x run.sh

echo "install cuda libs ..."
./install_cuda.sh
mv deepvideo_env.sh env.sh # the environment script

echo "install dependency libs ..."
./install_libs.sh

echo "install matrix ..."
./pull.sh "matrix_apps" "0.1.1"

echo "remove temp data "
rm -rf install_cuda.sh install_libs.sh

echo "please enable functions by using dog_tool"
#echo "1 2 3 4 63" | ./dog_tool