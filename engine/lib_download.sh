#!/usr/bin/env bash

if [ ! -d lib ]; then
    mkdir "lib"
fi

if [ "`ls -A lib`" = "" ]; then

    wget -O ./lib/engine_libs.tar http://192.168.2.119/matrix/libs/engine_libs.tar
    tar -xvf ./lib/engine_libs.tar -C ./lib
    rm -rf ./lib/engine_libs.tar
else
    echo "Directory lib are ready."
fi