/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_marker_classifier_processor.h"
namespace dg {

VehicleMarkerClassifierProcessor::VehicleMarkerClassifierProcessor()
          : Processor() {
     CaffeConfig dConfig;
     dConfig.model_file =
               "models/marker_model/frcnn_train_iter_100000.caffemodel";
     dConfig.deploy_file = "models/marker_model/tiny_deploy.prototxt";

     dConfig.is_model_encrypt = false;
     dConfig.batch_size = 1;
     MarkerCaffeClassifier::MarkerConfig mConfig;
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::MOT, 0.6));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::Belt, 0.8));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::Global, 0.8));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::Accessories, 0.8));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::Others, 0.8));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::TissueBox, 0.8));
     mConfig.marker_confidence.insert(
               make_pair<int, float>(MarkerCaffeClassifier::SunVisor, 0.8));

     classifier_ = new MarkerCaffeClassifier(dConfig, mConfig);

     CaffeConfig cConfig;
     cConfig.model_file =
               "models/window_model/car_python_mini_alex_256_5_iter_25000.caffemodel";
     cConfig.deploy_file = "models/window_model/window_deploy.prototxt";
     cConfig.is_model_encrypt = false;
     cConfig.batch_size = 1;

     detector_ = new WindowCaffeDetector(cConfig);

}

VehicleMarkerClassifierProcessor::~VehicleMarkerClassifierProcessor() {
     if (classifier_)
          delete classifier_;
}

void VehicleMarkerClassifierProcessor::Update(FrameBatch *frameBatch) {
     DLOG(INFO)<<"start marker processor"<<endl;

     DLOG(INFO)<<"start window detection"<<endl;

     beforeUpdate(frameBatch);

     vector<Detection> crops = detector_->DetectBatch(resized_images_,
                                                      images_);
     DLOG(INFO)<<"window crops result"<<endl;
     for (int i = 0; i < objs_.size(); i++) {
          Vehicle *v = (Vehicle*) objs_[i];
          v->set_window(crops[i]);
     }

     vector<Mat> images;
     for (int i = 0; i < crops.size(); i++) {
          Mat img = resized_images_[i](crops[i].box);
          images.push_back(img);

     }

     vector<vector<Detection> > pred = classifier_->ClassifyAutoBatch(images);
     DLOG(INFO)<<"marker pred result"<<endl;
     for (int i = 0; i < pred.size(); i++) {
          Vehicle *v = (Vehicle*) objs_[i];
          v->set_markers(pred[i]);
     }
     Proceed(frameBatch);

}

void VehicleMarkerClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {
     objs_ = frameBatch->objects(OPERATION_VEHICLE_MARKER);
     for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
               ++itr) {
          Object *obj = *itr;

          if (obj->type() == OBJECT_CAR) {

               Vehicle *v = (Vehicle*) obj;

               DLOG(INFO)<< "Put vehicle images to be marker classified: " << obj->id() << endl;
               resized_images_.push_back(v->resized_image());
               images_.push_back(v->image());

          } else {
               delete obj;
               itr = objs_.erase(itr);
               DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
          }
     }
}
bool VehicleMarkerClassifierProcessor::checkStatus(Frame *frame) {
     return true;
}
}
