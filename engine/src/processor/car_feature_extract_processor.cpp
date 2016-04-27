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
        if (obj->type() == OBJECT_CAR) {
            Vehicle *v = (Vehicle*) obj;
            v->image();
            CarRankFeature feature;
            extractor_->ExtractDescriptor(v->image(), feature);
            v->set_feature(feature);
        }
    }
}
void CarFeatureExtractProcessor::Update(Frame *frame) {
    extract(frame->objects());
    Proceed(frame);
}

virtual void Update(FrameBatch *frameBatch) {
    for (int i = 0; i < frameBatch->frames().size(); ++i) {
        extract(frameBatch->frames()[i]->objects());
    }
    Proceed(frameBatch);
}

} /* namespace dg */

