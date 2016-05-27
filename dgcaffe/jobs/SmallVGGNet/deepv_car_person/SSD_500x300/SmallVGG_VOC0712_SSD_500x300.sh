cd /mnt/data1/zdb/work/caffe_for_frcnn/gaoyuan/caffe
./build/tools/caffe train \
--solver="models/SmallVGGNet/deepv_car_person/SSD_500x300/solver.prototxt" \
--gpu 0,1,2,3 2>&1 | tee jobs/SmallVGGNet/deepv_car_person/SSD_500x300/SmallVGG_VOC0712_SSD_500x300.log
