#!/usr/bin/env bash


if [ -z "$1" ]; then
	echo "Please input version number!"
	exit
fi

version=$1
name="matrix_apps"
platform=$(uname -s)-$(uname -p)
rm ../bin/Debug/*.md
cp ../README.md ../bin/Debug/
cp ../release/release_note_$version.md ../bin/Debug/
tar -cvf $name"_"$version.tar ../bin/Debug/*
#rm version && touch version
#echo $version > version
#scp version dell@192.168.2.119:~/release/home/matrix/$platform/
scp $name"_"$version.tar  dell@192.168.2.119:~/release/home/matrix/$platform/internal
rm version
rm $name"_"$1.tar

echo "release " $version "done!"

