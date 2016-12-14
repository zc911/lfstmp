/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_phone_detector_processor.h"
#include "processor_helper.h"
#include "string_util.h"
#include "algorithm_def.h"

using namespace dgvehicle;
namespace dg {

VehiclePhoneClassifierProcessor::VehiclePhoneClassifierProcessor(float threshold)
    : Processor(), threshold_(threshold) {

    detector_ =
        AlgorithmFactory::GetInstance()->CreateProcessorInstance(AlgorithmProcessorType::c_phoneCaffeSsdDetector);

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
    detector_->BatchProcess(images_, preds);

    for (int i = 0; i < objs_.size(); i++) {
        if (preds[i].size() > 0) {
            if (preds[i][0].second < threshold_) {
                VLOG(VLOG_RUNTIME_DEBUG)
                << "Phone detection confidence lower than threshold " << preds[i][0].second << ":" << threshold_
                    << endl;
                continue;
            }

            Vehicle *v = (Vehicle *) objs_[i];
            Vehicler *vr = (Vehicler *) v->child(OBJECT_DRIVER);
            if (!vr) {
                vr = new Vehicler(OBJECT_DRIVER);
                v->set_vehicler(vr);
            }
            vr->set_vehicler_attr(Vehicler::Phone, preds[i][0].second);
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

    vector<Object *> objs = frameBatch->CollectObjects(OPERATION_DRIVER_PHONE);
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
