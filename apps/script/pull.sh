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
server="192.168.2.119"

platform=$(uname -s)-$(uname -p)
matrix_name=${module}_${version}
wget -O tmp_version "http://"$server"/matrix/$platform/version"
version_num=$(cat tmp_version)
rm tmp_version

if [ "$2" == "latest" ]; then
	matrix_name=${module}_${version_num}
fi

wget -O $matrix_name".tar" "http://"$server"/matrix/$platform/release/"$matrix_name".tar"
wget -O $module"_libs_"$version_num".tar" "http://"$server"/matrix/$platform/release/libs/"$module"_libs_"$version_num".tar"
wget -O $module"_data_"$version_num".tar" "http://"$server"/matrix/$platform/release/data/"$module"_data_"$version_num".tar"

rm -rf $matrix_name && mkdir $matrix_name

tar -xvf ${matrix_name}.tar -C ${matrix_name}
tar -xvf $module"_libs_"$version_num".tar" -C ${matrix_name}
tar -xvf $module"_data_"$version_num".tar" -C ${matrix_name}

rm -rf ${matrix_name}.tar $module"_libs_"$version_num".tar" $module"_data_"$version_num".tar"
rm latest
ln -s $matrix_name latest

