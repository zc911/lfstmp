/*
 * car_feature_extract_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: chenzhen
 */

#include "car_feature_extract_processor.h"

namespace dg {

CarFeatureExtractProcessor::CarFeatureExtractProcessor() {
    extractor_ = new CarFeatureExtractor();
}

CarFeatureExtractProcessor::~CarFeatureExtractProcessor() {
    if (extractor_)
        delete extractor_;
}
void CarFeatureExtractProcessor::extract(vector<Object*> &objs) {
    for (int i = 0; i < objs.size(); ++i) {
        Object *obj = objs[i];
        Vehicle *v = static_cast<Vehicle*>(obj);
        CarRankFeature feature;
        extractor_->ExtractDescriptor(v->image(), feature);
        v->set_feature(feature);
    }
}
void CarFeatureExtractProcessor::Update(Frame *frame) {
    DLOG(INFO)<< "Start feature extract. " << endl;
    extract(vehicle_to_processed_);
    DLOG(INFO)<< "End feature extract. " << endl;
    Proceed(frame);
}

void CarFeatureExtractProcessor::Update(FrameBatch *frameBatch) {
    DLOG(INFO)<< "Start feature extract(Batch). " << endl;
    beforeUpdate(frameBatch);
    extract(vehicle_to_processed_);
    vehicle_to_processed_.clear();
    DLOG(INFO)<< "End feature extract(Batch). " << endl;
    Proceed(frameBatch);
}

void CarFeatureExtractProcessor::beforeUpdate(FrameBatch *frameBatch) {
    vehicle_to_processed_.clear();
    vehicle_to_processed_ = frameBatch->collect_objects(
            OPERATION_VEHICLE_FEATURE_VECTOR);

    for (vector<Object*>::iterator itr = vehicle_to_processed_.begin();
            itr != vehicle_to_processed_.end();) {
        if ((*itr)->type() != OBJECT_CAR) {
            itr = vehicle_to_processed_.erase(itr);
        } else {
            itr++;
        }
    }
}

} /* namespace dg */

