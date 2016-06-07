#!/usr/bin/env bash


if [ -z "$1" ]; then
	echo "Please input version number!"
	exit
fi

version=$1
name="matrix_apps"
platform=$(uname -s)-$(uname -p)
rm ../bin/Release/*.md
cp ../README.md ../bin/Release/
cp ../release/release_note_$version.md ../bin/Release/
tar -cvf $name"_"$version.tar ../bin/Release/*
rm version && touch version
echo $version > version
scp version dell@192.168.2.119:~/release/home/matrix/$platform/
scp $name"_"$version.tar  dell@192.168.2.119:~/release/home/matrix/$platform/
rm version
rm $name"_"$1.tar

echo "release " $version "done!"

