#!/usr/bin/env bash

if [ -z "$1" ]; then
	echo "Please input version number!"
	exit
fi


server="192.168.2.119"
version=$1
name="matrix_apps"

# check remote version
platform=$(uname -s)-$(uname -p)
wget -O remote_version.tmp http://$server/matrix/Linux-x86_64/version
remote_version=$(cat remote_version.tmp)
rm remote_version.tmp

if [ "$remote_version" == "$version" ]; then
    echo "Version" $version "already exists, replaced? y/n"
    read replace
    if [[ "$replace" != "y" ]]; then
        echo "Release cancelled!"
        exit
    fi
fi


rm ../bin/Release/*.md
cp ../README.md ../bin/Release/
cp ../release/release_note_$version.md ../bin/Release/

tar -cvf $name"_libs_"$version.tar ../bin/Release/libs
tar -cvf $name"_data_"$version.tar ../bin/Release/data
tar -cvf $name"_"$version.tar --exclude ../bin/Release/libs --exclude ../bin/Release/data ../bin/Release/*


rm version.tmp && touch version.tmp
echo $version > version.tmp
chmod 400 ./release_id_rsa
scp -i ./release_id_rsa version.tmp dell@$server:~/release/home/matrix/$platform/version
scp -i ./release_id_rsa $name"_"$version.tar  dell@$server:~/release/home/matrix/$platform/release/
scp -i ./release_id_rsa $name"_libs_"$version.tar dell@$server:~/release/home/matrix/$platform/release/libs/
scp -i ./release_id_rsa $name"_data_"$version.tar dell@$server:~/release/home/matrix/$platform/release/data/
rm version.tmp $name"_"$1.tar $name"_libs_"$version.tar $name"_data_"$version.tar

#git commit -a -m $version
#git tag $version
#git push origin $version
#git checkout release && git merge master && git push
echo "release " $version "done!"

