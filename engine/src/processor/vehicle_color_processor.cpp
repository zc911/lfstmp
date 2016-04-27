/*
 * vehicle_color_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajiachen
 */

#include "vehicle_color_processor.h"
namespace dg {

VehicleColorProcessor::VehicleColorProcessor() {
     CaffeConfig config;
     config.model_file = "models/color/zf_q_iter_70000.caffemodel";
     config.deploy_file = "models/color/deploy.prototxt";

     config.is_model_encrypt = false;
     config.batch_size = 1;
     classifier_ = new VehicleCaffeClassifier(config);
}

VehicleColorProcessor::~VehicleColorProcessor() {
     if (classifier_)
          delete classifier_;
}

void VehicleColorProcessor::Update(FrameBatch *frameBatch) {
     DLOG(INFO)<<"Start detect frame: "<< endl;
     vector<Mat> images = this->vehicles_resized_mat(frameBatch);

     vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(
               images);

     SortPrediction(result);

     for(int i=0;i<objs_.size();i++) {
          Vehicle *v = (Vehicle*) objs_[i];
          Vehicle::Color color;
          if(result[i].size()<0) {
               continue;
          }
          color.class_id=result[i][0].first;
          color.confidence=result[i][0].second;
          v->set_color(color);
     }

     Proceed(frameBatch);

}

bool VehicleColorProcessor::checkOperation(Frame *frame) {
     return true;
}

bool VehicleColorProcessor::checkStatus(Frame *frame) {
     return true;
}

vector<Mat> VehicleColorProcessor::vehicles_resized_mat(
          FrameBatch *frameBatch) {
     vector<cv::Mat> vehicleMat;
     objs_ = frameBatch->objects();
     for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
               ++itr) {
          Object *obj = *itr;

          if (obj->type() == OBJECT_CAR) {

               Vehicle *v = (Vehicle*) obj;

               DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
               vehicleMat.push_back(v->resized_image());

          } else {
               delete obj;
               itr = objs_.erase(itr);
               DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
          }
     }
     return vehicleMat;
}

}

