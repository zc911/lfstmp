#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: proto_gen.sh cpp|python"
    exit
fi

if [ "$1" == "cpp" ]; then
    grpc_x_plugin=`which grpc_cpp_plugin`
    target="--cpp_out"
    target_dir="../src/cpp"
    protoc -I . $target=$target_dir *.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} common.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} witness.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} ranker.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} system.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} skynet.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} matrix.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} spring.proto
    protoc -I . --grpc_out=$target_dir --plugin=protoc-gen-grpc=${grpc_x_plugin} localcommon.proto
elif [ "$1" == "python" ]; then
    grpc_x_plugin=`which grpc_python_plugin`
    target="--python_out"
    target_dir="../src/python"
    python -m grpc.tools.protoc -I . --python_out=$target_dir --grpc_python_out=$target_dir *.proto
else
    echo "Usage: proto_gen.sh [cpp|python]"
    exit
fi

echo "Finish"




#generate grpc codes



