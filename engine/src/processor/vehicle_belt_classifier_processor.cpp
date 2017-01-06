/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "vehicle_belt_classifier_processor.h"
#include "processor_helper.h"
#include "util/caffe_helper.h"

using namespace dgvehicle;
namespace dg {

VehicleBeltClassifierProcessor::VehicleBeltClassifierProcessor(
    float threshold, bool driver)
    : Processor() {

    belt_classifier_ = AlgorithmFactory::GetInstance()->CreateBeltClassifier(driver);
    threshold_ = threshold;
    is_driver_ = driver;

}
VehicleBeltClassifierProcessor::~VehicleBeltClassifierProcessor() {

    if (belt_classifier_) {
        delete belt_classifier_;
    }
    images_.clear();
}

float GetConfidence(const vector<Prediction>& predictions) {
    float value = 0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i].first == 1 || predictions[i].second < 0) {
            continue;
        }
        value += predictions[i].second;
    }
    return value;
}

bool VehicleBeltClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start belt processor " << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start belt processor" << frameBatch->id() << endl;
    vector<vector<Prediction> > preds;
    belt_classifier_->BatchProcess(images_, preds);

    for (int i = 0; i < objs_.size(); i++) {
        float value;
        Vehicle *v = (Vehicle *) objs_[i];

        if (preds[i][0].second < threshold_) {
            VLOG(VLOG_RUNTIME_DEBUG)
            << "Belt detection confidence is lower than threshold " << preds[i][0].second << ":" << threshold_ << endl;
            continue;
        }

        switch (preds[i][0].first) {
            case BELT_LABLE_NO: {
                ObjectType driverType = OBJECT_DRIVER;
                if (!is_driver_) {
                    driverType = OBJECT_CODRIVER;
                }
                float confidence = GetConfidence(preds[i]);
                Vehicler *vr = (Vehicler *) v->child(driverType);
                if (!vr) {
                    vr = new Vehicler(driverType);
                    Detection detection;
                    GetPassengerDetection(detections_[i], detection, is_driver_);
                    detection.confidence = confidence;
                    vr->set_detection(detection);
                    v->set_vehicler(vr);
                } else {
                    Detection detection = vr->detection();
                    detection.confidence = confidence;
                    vr->set_detection(detection);
                }
                value = preds[i][0].second;
                vr->set_vehicler_attr(Vehicler::NoBelt, value);
                break;
            }
            default:
                break;
        }

    }
    objs_.clear();

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish Start belt processor " << frameBatch->id() << endl;
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
    detections_.clear();
    vector<Object *> objs;
    if (is_driver_) {
        objs = frameBatch->CollectObjects(OPERATION_DRIVER_BELT);
    } else {
        objs = frameBatch->CollectObjects(OPERATION_CODRIVER_BELT);
    }
    vector<Object *>::iterator itr = objs.begin();
    while (itr != objs.end()) {
        Object *obj = *itr;
        if (obj->type() == OBJECT_CAR) {
            for (int i = 0; i < obj->children().size(); i++) {
                Object *obj_child = obj->children()[i];
                if (obj_child->type() == OBJECT_WINDOW) {
                    Window *w = (Window *) obj->children()[i];
                    images_.push_back(w->resized_image());
                    detections_.push_back(w->detection());
                    performance_++;
                    objs_.push_back(obj);

                }
            }

        } else {
        }
        ++itr;

    }

    return true;

}

bool VehicleBeltClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
