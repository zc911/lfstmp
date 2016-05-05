/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_marker_classifier_processor.h"
namespace dg {

VehicleMarkerClassifierProcessor::VehicleMarkerClassifierProcessor(
        WindowCaffeDetector::WindowCaffeConfig & wConfig,
        MarkerCaffeClassifier::MarkerConfig &mConfig)
        : Processor() {

    classifier_ = new MarkerCaffeClassifier( mConfig);

    detector_ = new WindowCaffeDetector(wConfig);

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
    DLOG(INFO)<<"marker pred result: " << pred.size() <<endl;
    for (int i = 0; i < pred.size(); i++) {
        Vehicle *v = (Vehicle*) objs_[i];
        v->set_markers(pred[i]);

    }
    objs_.clear();
    Proceed(frameBatch);

}

void VehicleMarkerClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {
    objs_.clear();
    resized_images_.clear();
    images_.clear();

    objs_ = frameBatch->collect_objects(OPERATION_VEHICLE_MARKER);
    for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
            ++itr) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;

            DLOG(INFO)<< "Put vehicle images to be marker classified: " << obj->id() << endl;
            resized_images_.push_back(v->resized_image());
            images_.push_back(v->image());

        } else {
            itr = objs_.erase(itr);
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }
    }

}
bool VehicleMarkerClassifierProcessor::checkStatus(Frame *frame) {
    return true;
}
}
