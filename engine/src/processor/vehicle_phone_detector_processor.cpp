/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_phone_detector_processor.h"
#include "processor_helper.h"
#include "string_util.h"

namespace dg {

VehiclePhoneClassifierProcessor::VehiclePhoneClassifierProcessor(
    VehicleCaffeDetectorConfig &mConfig)
    : Processor() {

    detector_ = new PhoneCaffeSsdDetector(mConfig);

    marker_target_min_ = mConfig.target_min_size;
    marker_target_max_ = mConfig.target_max_size;

}
VehiclePhoneClassifierProcessor::~VehiclePhoneClassifierProcessor() {

    if (detector_) {
        delete detector_;
    }
    images_.clear();
}

bool VehiclePhoneClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start marker and window processor" << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start marker processor" << endl;
    vector<vector<Prediction> > preds;
    detector_->DetectBatch(images_, preds);

    for (int i = 0; i < objs_.size(); i++) {
        int value;
        Vehicle *v = (Vehicle *) objs_[i];
        Vehicler *vr = (Vehicler *) v->child(OBJECT_DRIVER);
        if (!vr) {
            vr = new Vehicler(OBJECT_DRIVER);
            v->set_vehicler(vr);
        }
        LOG(INFO)<<preds[i].size();
        if (preds[i].size() > 0) {
            vr->set_vehicler_attr(Vehicler::Phone, Vehicler::Yes);
        }
        else {
            vr->set_vehicler_attr(Vehicler::Phone, Vehicler::No);
        }

    }
    objs_.clear();

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish marker and window processor" << frameBatch->id() << endl;
    return true;
}

bool VehiclePhoneClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    objs_.clear();
    images_.clear();

    vector<Object *> objs = frameBatch->CollectObjects(OPERATION_VEHICLE_MARKER);
    vector<Object *>::iterator itr = objs.begin();
    while (itr != objs.end()) {
        Object *obj = *itr;
        if (obj->type() == OBJECT_CAR) {

            for (int i = 0; i < obj->children().size(); i++) {
                Object *obj_child = obj->children()[i];
                if (obj_child->type() == OBJECT_WINDOW) {
                    Window *w = (Window *) obj->children()[i];
                    images_.push_back(w->phone_image());
                    performance_++;
                    objs_.push_back(obj);

                }
            }

        } else {
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << " " << endl;
        }
        ++itr;

    }

    return true;

}

bool VehiclePhoneClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
