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

    beforeUpdate(frameBatch);

    vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(
            images_);

    for(int i=0;i<objs_.size();i++) {
        Vehicle *v = (Vehicle*) objs_[i];
        Vehicle::Color color;
        if(result[i].size()<0) {
            continue;
        }
        Prediction max = MaxPrediction(result[i]);

        color.class_id=max.first;
        color.confidence=max.second;
        v->set_color(color);
    }

    Proceed(frameBatch);

}

void VehicleColorProcessor::beforeUpdate(FrameBatch *frameBatch) {
    images_ = this->vehicles_resized_mat(frameBatch);
}

bool VehicleColorProcessor::checkStatus(Frame *frame) {
    return true;
}

vector<Mat> VehicleColorProcessor::vehicles_resized_mat(
        FrameBatch *frameBatch) {
    vector<cv::Mat> vehicleMat;
    objs_ = frameBatch->collect_objects(OPERATION_VEHICLE_COLOR);
    for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
            ++itr) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;

            DLOG(INFO)<< "Put vehicle images to be color classified: " << obj->id() << endl;
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

