PYTHONPATH=/home/ubuntu/zdb/new_caffe/lib:$PYTHONPATH
#LD_LIBRARY_PATH=/usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/lib/:/home/ubuntu/zdb/new_caffe/gflags/build/lib/:/usr/lib/arm-linux-gnueabihf/:$LD_LIBRARY_PATH
#./build/tools/caffe time --model models/small/googlenet_test/deploy.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model new_model.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/car_not_car/deploy.prototxt --iterations 100
#./build/tools/caffe time --model models/small/car_not_car/train.prototxt --iterations 100
#.build_release/tools/caffe time --model models/small/googlenet_deploy_half.prototxt --iterations 10 --gpu 0
#.build_release/tools/caffe time --model models/small/googlenet_deploy.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_deploy_half_stride8.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_deploy_stride8.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_deploy.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/rpn_16layer_small.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_test/half.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_test/quanter.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/alexnet_deploy_half_cpu.prototxt --iterations 5 
#./build/tools/caffe time --model ../models/GoogleNet_inception5/faster_rcnn_end2end/test.prototxt 
#./build/tools/caffe time --model models/small/googlenet_deploy_multi_scale.prototxt --iterations 10 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_deploy_multi_scale_python.prototxt --iterations 2 --gpu 0
#.build_release/tools/caffe time --model models/small/googlenet_test/quanter.prototxt --iterations 5 --gpu 0
#.build_release/tools/caffe time --model models/small/alexnet_deploy_half.prototxt --iterations 5  --gpu 0
.build_release/tools/caffe time --model ../models/GoogleNet_inception5/faster_rcnn_end2end/test.prototxt  --iterations 5 --gpu 0
