//
// Created by jiajaichen on 16-6-20.
//

#include "accelerate.h"
namespace dg{
Accelerate::Accelerate(){

        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0);

    net_.reset(new Net<float>("models/accelerate/deploy.prototxt", TEST));
    net_->CopyTrainedLayersFrom("models/accelerate/test.caffemodel");


    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();
    /*  const vector<boost::shared_ptr<Layer<float> > > & layers = net_->layers();
      const vector<vector<Blob<float>* > > & bottom_vecs = net_->bottom_vecs();
      const vector<vector<Blob<float>* > > & top_vecs = net_->top_vecs();
      for(int i = 0; i < layers.size(); ++i) {
          layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      }
  */
    device_setted_ = false;

}

Accelerate::~Accelerate() {

}



}