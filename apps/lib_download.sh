#!/usr/bin/env bash

FILE_SERVER="http://192.168.2.119/matrix/libs"
LIBS_TAR="apps_libs.tar"

if [ ! -d lib ]; then
    mkdir "lib"
fi

if [ "`ls -A lib`" = "" ]; then
    wget -O ./lib/$LIBS_TAR $FILE_SERVER/$LIBS_TAR
    tar -xvf ./lib/$LIBS_TAR -C ./lib
    rm -rf ./lib/$LIBS_TAR
    echo "Dependency library OK."
else
    echo "Dependency library already OK."
fi