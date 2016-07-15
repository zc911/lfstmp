/*
 * car_feature_extract_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: chenzhen
 */

#include "car_feature_extract_processor.h"
#include "processor_helper.h"
#include "log/log_val.h"
namespace dg {

CarFeatureExtractProcessor::CarFeatureExtractProcessor() {
    extractor_ = new CarFeatureExtractor();
}

CarFeatureExtractProcessor::~CarFeatureExtractProcessor() {
    if (extractor_)
        delete extractor_;
}
void CarFeatureExtractProcessor::extract(vector<Object *> &objs) {
    for (int i = 0; i < objs.size(); ++i) {
        Object *obj = objs[i];
        Vehicle *v = static_cast<Vehicle *>(obj);
        CarRankFeature feature;
        extractor_->ExtractDescriptor(v->image(), feature);
        v->set_feature(feature);
    }
}

bool CarFeatureExtractProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start feature extract(Batch). " << endl;
    extract(vehicle_to_processed_);
    vehicle_to_processed_.clear();
    return true;
}

bool CarFeatureExtractProcessor::beforeUpdate(FrameBatch *frameBatch) {
    #if DEBUG
    #else
        if(performance_>RECORD_UNIT) {
            if(!RecordFeaturePerformance()) {
                return false;
            }
        }
    #endif

    vehicle_to_processed_.clear();
    vehicle_to_processed_ = frameBatch->CollectObjects(
        OPERATION_VEHICLE_FEATURE_VECTOR);

    for (vector<Object *>::iterator itr = vehicle_to_processed_.begin();
         itr != vehicle_to_processed_.end();) {
        if ((*itr)->type() != OBJECT_CAR) {
            itr = vehicle_to_processed_.erase(itr);
        } else {
            itr++;
        }
    }
    return true;
}
bool CarFeatureExtractProcessor::RecordFeaturePerformance() {
    return RecordPerformance(FEATURE_CAR_EXTRACT, performance_);

}
} /* namespace dg */

