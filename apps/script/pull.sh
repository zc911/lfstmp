#!/usr/bin/env bash
if [ -z "$1" ]; then
        echo "Please input module name!"
        exit
fi

if [ -z "$2" ]; then
	echo "Please input version number!"
	exit 
fi

module=$1
version=$2
platform=$(uname -s)-$(uname -p)
matrix_name=${module}_${version}
wget -O tmp_version "http://192.168.2.119/matrix/$platform/version"
version_num=$(cat tmp_version)
rm tmp_version
if [ "$2" == "latest" ]; then
	matrix_name=${module}_${version_num}
fi
wget -O ${matrix_name}.tar "http://192.168.2.119/matrix/$platform/$matrix_name.tar"

mkdir $matrix_name
mv $matrix_name.tar $matrix_name
cd $matrix_name
tar -xvf $matrix_name.tar
rm $matrix_name.tar
cd ..
rm latest
ln -s $matrix_name latest


