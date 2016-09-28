cd /mnt/data1/zdb/work/caffe_for_frcnn/gaoyuan/caffe
./build/tools/caffe train \
--solver="models/VGGNet/deepv_car_person/SSD_500x500/solver.prototxt" \
--snapshot="models/VGGNet/deepv_car_person/SSD_500x500/VGG_VOC0712_SSD_500x500_iter_60000.solverstate" \
--gpu 0,1,2,3 2>&1 | tee jobs/VGGNet/deepv_car_person/SSD_500x500/VGG_VOC0712_SSD_500x500.log
