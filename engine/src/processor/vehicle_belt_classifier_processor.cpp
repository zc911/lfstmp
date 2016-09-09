/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_belt_classifier_processor.h"
#include "processor_helper.h"
#include "string_util.h"

namespace dg {

VehicleBeltClassifierProcessor::VehicleBeltClassifierProcessor(
    VehicleBeltConfig &mConfig)
    : Processor() {

    belt_classifier_ = new CaffeBeltClassifier(mConfig);

    marker_target_min_ = mConfig.target_min_size;
    marker_target_max_ = mConfig.target_max_size;
    is_driver=mConfig.is_driver;

}
VehicleBeltClassifierProcessor::~VehicleBeltClassifierProcessor() {

    if (belt_classifier_) {
        delete belt_classifier_;
    }
    images_.clear();
}

bool VehicleBeltClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start belt and window processor" << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start belt processor" << endl;
    vector<vector<Prediction> > preds = belt_classifier_->ClassifyAutoBatch(images_);

    for (int i = 0; i < objs_.size(); i++) {
        int value;
        Vehicle *v = (Vehicle *) objs_[i];

        if (is_driver) {
            Vehicler *vr = (Vehicler *) v->child(OBJECT_DRIVER);
            if (!vr) {
                vr = new Vehicler(OBJECT_DRIVER);
                v->set_vehicler(vr);
            }
            switch (preds[i][0].first) {
            case 0:
                                value = Vehicler::No;

                break;
            case 1:
                value = Vehicler::NotSure;
                break;
            case 2:
                            value = Vehicler::No;

                break;
            }
            if(preds[i][0].second<0.9)
                value=Vehicler::NotSure;
            vr->set_vehicler_attr(Vehicler::Belt, value);

        } else {
            if (preds[i][0].first == 1)
                continue;
            Vehicler *vr = (Vehicler *) v->child(OBJECT_CODRIVER);
            preds[i][0].first == 1;
            continue;
            if (!vr) {
                vr = new Vehicler(OBJECT_DRIVER);
                v->set_vehicler(vr);
            }
            switch (preds[i][0].first) {
            case 0:
                            value = Vehicler::No;

                break;
            case 2:
                value = Vehicler::NotSure;
                break;
            case 3:
                                value = Vehicler::Yes;

                break;
            }
            if(preds[i][0].second<0.9)
                value=Vehicler::NotSure;
            vr->set_vehicler_attr(Vehicler::Belt, value);

        }

    }
    objs_.clear();

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish marker and window processor" << frameBatch->id() << endl;
    return true;
}

bool VehicleBeltClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {

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
                    images_.push_back(w->resized_image());
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

bool VehicleBeltClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
