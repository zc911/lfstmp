/*
 * vehicle_color_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajiachen
 */

#include "vehicle_color_processor.h"
namespace dg {

VehicleColorProcessor::VehicleColorProcessor(
        const vector<VehicleCaffeClassifier::VehicleCaffeConfig> &configs) {

    for (int i = 0; i < configs.size(); i++) {

        VehicleCaffeClassifier *classifier = new VehicleCaffeClassifier(
                configs[i]);

        classifiers_.push_back(classifier);

    }
}

VehicleColorProcessor::~VehicleColorProcessor() {
    for (int i = 0; i < classifiers_.size(); i++) {
        delete classifiers_[i];
    }
    classifiers_.clear();
}

void VehicleColorProcessor::Update(FrameBatch *frameBatch) {
    DLOG(INFO)<<"Start detect frame: "<< endl;

    beforeUpdate(frameBatch);
    vector<vector<Prediction> > result;
    for_each(classifiers_.begin(),classifiers_.end(),[&](VehicleCaffeClassifier *elem) {
                auto tmpPred=elem->ClassifyAutoBatch(images_);
                vote(tmpPred,result,classifiers_.size());

            });

    //set results

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
            itr = objs_.erase(itr);
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
    return vehicleMat;
}

}

