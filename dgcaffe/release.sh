system=$(uname -s)-$(uname -p)
cp build/lib/* ../engine/lib/caffe/$system/
cp -r include/* ../engine/include/caffe/
cp build/src/caffe/proto/caffe.pb.h ../engine/include/caffe/caffe/proto/

