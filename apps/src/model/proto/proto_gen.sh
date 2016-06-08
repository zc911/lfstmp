#!/bin/bash

protoc -I . --cpp_out=.. *.proto

#generate grpc codes
grpc_cpp_plugin=`which grpc_cpp_plugin`

protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} witness.proto
protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} ranker.proto
protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} system.proto
protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} skynet.proto
protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} matrix.proto
protoc -I . --grpc_out=.. --plugin=protoc-gen-grpc=${grpc_cpp_plugin} spring.proto

