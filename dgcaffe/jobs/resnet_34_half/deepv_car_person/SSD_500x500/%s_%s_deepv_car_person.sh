cd /mnt/data1/zdb/work/caffe_for_frcnn/gaoyuan/caffe
./build/tools/caffe train \
--solver="models/resnet_34_half/deepv_car_person/SSD_500x500/solver.prototxt" \
--weights="models/resnet_34_half/resnet_34_half_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/resnet_34_half/deepv_car_person/SSD_500x500/%s_%s_deepv_car_person.log
