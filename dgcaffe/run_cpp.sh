
#find test_data/ -name "*.jpg" > test.input
#find test_data/ -name "*.png" >> test.input


#CAFFE path
CAFFE=.build_release/examples/cpp_classification/with_cls_to_jiajia.bin
#detection_model_conf_file
#argv1=./detection/faster_rcnn/detector/test.prototxt
#argv1=./faster_rcnn/detector_old/test.prototxt
argv1=./models/frcnn_fast/test_15.prototxt
#detection_model_trained_file
#argv2=./faster_rcnn/detector_old/googlenet_faster_rcnn_iter_48000.caffemodel
argv2=./models/frcnn_fast/frcnn_train_iter_250000.caffemodel
#input_image_list
#argv3=./test_video.list
argv3=./test_video_resize.list
#classfication_conf_file
#argv4=./detection/faster_rcnn/detector/deploy.prototxt

argv4=./faster_rcnn/detector_old/deploy.prototxt
#classfication_trained_file
#argv5=./detection/faster_rcnn/detector/car_not_car_train_iter_30000.caffemodel
argv5=./faster_rcnn/detector_old/car_not_car_train_iter_30000.caffemodel
#save_result_file
argv6=./detection_result

echo "$CAFFE $argv1 $argv2 $argv3 $argv4 $argv5 $argv6 > log_detection"
LD_LIBRARY_PATH=/usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/lib/:/home/ubuntu/zdb/new_caffe/gflags/build/lib/:$LD_LIBRARY_PATH
$CAFFE $argv1 $argv2 $argv3 $argv4 $argv5 $argv6 > log_detection 

#rm -rf  test_data_result/*
#python pred_full_8.py 0


#./build/tools/caffe time --model models/small/googlenet_test/deploy.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_test/half.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/googlenet_test/quanter.prototxt --iterations 5 --gpu 0
#./build/tools/caffe time --model models/small/alexnet_deploy_half_cpu.prototxt --iterations 5 
