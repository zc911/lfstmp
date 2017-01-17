#!/bin/bash

makesure_dir_exist(){
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}


if [ -z "$1" ]; then
    echo "Usage: proto_gen.sh cpp|python|golang"
    exit
fi

if [ "$1" == "cpp" ]; then
    grpc_x_plugin=`which grpc_cpp_plugin`
    target="--cpp_out"
    target_dir="../src/cpp"
    makesure_dir_exist $target_dir
    protoc -I . $target=$target_dir *.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} common.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} witness.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} ranker.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} system.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} skynet.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} matrix.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} spring.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} localcommon.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} dataservice.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} deepdatasingle.proto
elif [ "$1" == "python" ]; then
    grpc_x_plugin=`which grpc_python_plugin`
    target="--python_out"
    target_dir="../src/python"
    makesure_dir_exist $target_dir
    python -m grpc.tools.protoc -I . --python_out=$target_dir --grpc_python_out=$target_dir *.proto
elif [ "$1" == "golang" ]; then
    target="--go_out"
    target_dir="../src/golang"
    makesure_dir_exist $target_dir
    protoc -I . $target=$target_dir common.proto
    protoc -I . $target=$target_dir localcommon.proto
    protoc ranker.proto $target=plugins=grpc:$target_dir
    protoc witness.proto $target=plugins=grpc:$target_dir
    protoc system.proto $target=plugins=grpc:$target_dir
else
    echo "Usage: proto_gen.sh [cpp|python|golang]"
    exit
fi

echo "proto src file generated finished"




#generate grpc codes



