LD_LIBRARY_PATH=/usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/lib/:/home/ubuntu/zdb/new_caffe/gflags/build/lib/:$LD_LIBRARY_PATH
PYTHONPATH=$(pwd)/python:$(pwd)/lib:$PYTHONPATH
#GLOG_logtostderr=1 ./build/examples/cpp_classification/main.bin < test.list
GLOG_logtostderr=1 ./build/examples/cpp_classification/main.bin < test_video.list
