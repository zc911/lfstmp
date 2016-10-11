#CAFFE path
#CAFFE=.build_release/examples/cpp_classification/with_cls_to_jiajia.bin
CAFFE=./build/examples/cpp_classification/main.bin
#detection_model_conf_file
#argv1=./models/frcnn_fast/test_15.prototxt
#argv1=./models/VGGNet/deepv_car_person/SSD_300x300/deploy.prototxt
#argv1=./models/SmallVGGNet/deepv_car_person/SSD_500x300/deploy.prototxt
argv1=./models/SmallVGGNet/deepv_car_person/SSD_600x400/deploy.prototxt
#detection_model_trained_file
#argv2=./faster_rcnn/detector_old/googlenet_faster_rcnn_iter_48000.caffemodel
#argv2=./models/SmallVGGNet/deepv_car_person/SSD_500x300/SmallVGG_deepv_car_person_SSD_500x300_iter_250000.caffemodel
argv2=./models/SmallVGGNet/deepv_car_person/SSD_600x400/SmallVGG_deepv_car_person_SSD_600x400_iter_200000.caffemodel
#input_image_list
#argv3=../test_data/videos/test.koda
#argv3=/mnt/ssd1/zdb/zdb/data/deepv/detection/deepv_full/train
#argv3=/mnt/data1/zdb/work/caffe_for_frcnn/caffe/run_test_scripts/old_frcnn_car/test.input
argv3=./test
#argv4 save output
argv4=./ssd_result
#argv5 global conf
argv5=0.5



echo "$CAFFE $argv1 $argv2 $argv3 $argv4 $argv5 > log_detection"
$CAFFE $argv1 $argv2 $argv3 $argv4 $argv5 > log_detection 

