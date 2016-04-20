
protoc -I . --grpc_out=../model/ --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin simservice.proto
protoc -I . --cpp_out=../model/ *.proto
