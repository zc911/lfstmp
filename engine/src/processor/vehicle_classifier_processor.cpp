#include "vehicle_classifier_processor.h"

namespace dg {

VehicleClassifierProcessor::VehicleClassifierProcessor() {
    CaffeConfig config;
    config.model_file =
            "models/classify/car_python_mini_alex_256_0_iter_70000.caffemodel";
    config.deploy_file = "models/classify/deploy_256.prototxt";

    config.is_model_encrypt = false;
    config.batch_size = 1;
    classifier_ = new VehicleCaffeClassifier(config);
}

VehicleClassifierProcessor::~VehicleClassifierProcessor() {
    if (classifier_)
        delete classifier_;
}



void VehicleClassifierProcessor::Update(FrameBatch *frameBatch) {

     DLOG(INFO)<<"Start vehicle classify frame: "<< endl;
     vector<Mat> images = this->vehicles_resized_mat(frameBatch);

     vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(
               images);


     for(int i=0;i<objs_.size();i++) {
          if(result[i].size()<0) {
               continue;
          }
          vector<Prediction> pre = result[i];
          Vehicle *v = (Vehicle*) objs_[i];
          Prediction max = MaxPrediction(result[i]);
          v->set_class_id(max.first);
          v->set_confidence(max.second);
     }

     Proceed(frameBatch);

}

bool VehicleClassifierProcessor::checkOperation(Frame *frame) {
    return frame->operation().Check(OPERATION_VEHICLE_STYLE);
}
bool VehicleClassifierProcessor::checkStatus(Frame *frame) {
    return true;
}

vector<Mat>  VehicleClassifierProcessor::vehicles_resized_mat(
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
