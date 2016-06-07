/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_marker_classifier_processor.h"
#include "processor_helper.h"

namespace dg {

VehicleMarkerClassifierProcessor::VehicleMarkerClassifierProcessor(
    WindowCaffeDetector::WindowCaffeConfig &wConfig,
    MarkerCaffeClassifier::MarkerConfig &mConfig)
    : Processor() {

    classifier_ = new MarkerCaffeClassifier(mConfig);
    detector_ = new WindowCaffeDetector(wConfig);

}

VehicleMarkerClassifierProcessor::~VehicleMarkerClassifierProcessor() {
    if (classifier_)
        delete classifier_;

    images_.clear();
    resized_images_.clear();
}

bool VehicleMarkerClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start marker and window processor" << endl;

    vector<Detection> crops = detector_->DetectBatch(resized_images_,
                                                     images_);

    for (int i = 0; i < objs_.size(); i++) {
        Vehicle *v = (Vehicle *) objs_[i];
        v->set_window(crops[i]);
    }

    vector<Mat> images;
    for (int i = 0; i < crops.size(); i++) {
        Mat img = resized_images_[i](crops[i].box);
        images.push_back(img);

    }

    vector<vector<Detection> > pred = classifier_->ClassifyAutoBatch(images);
    for (int i = 0; i < pred.size(); i++) {
        Vehicle *v = (Vehicle *) objs_[i];
        v->set_markers(pred[i]);

    }
    objs_.clear();
    return true;
}

bool VehicleMarkerClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if RELEASE
    if(performance_>20000) {
        if(!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    objs_.clear();
    resized_images_.clear();
    images_.clear();

    objs_ = frameBatch->CollectObjects(OPERATION_VEHICLE_MARKER);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle *) obj;

            resized_images_.push_back(v->resized_image());
            images_.push_back(v->image());
            ++itr;

        } else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }

    return true;

}

bool VehicleMarkerClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
