/*============================================================================
 * File Name   : pedestrian_classifier_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : Jul 1, 2016 8:42:35 AM
 * Description : 
 * ==========================================================================*/

#include "pedestrian_classifier_processor.h"

#include "processor_helper.h"

namespace dg {
PedestrianClassifierProcessor::PedestrianClassifierProcessor(
    PedestrianClassifier::PedestrianConfig &config) {
    classifier_ = new PedestrianClassifier(config);
}

PedestrianClassifierProcessor::~PedestrianClassifierProcessor() {
    if (classifier_)
        delete classifier_;

    images_.clear();
}

bool PedestrianClassifierProcessor::process(FrameBatch *frameBatch) {
    VLOG(VLOG_RUNTIME_DEBUG) << "Start pedestrian processor" << endl;
    VLOG(VLOG_SERVICE) << "Start pedestrian processor" << endl;
    for (int i = 0; i < images_.size(); i++) {
        if (images_[i].cols == 0 || images_[i].rows == 0) {
            return false;
        }
    }

    std::vector<std::vector<PedestrianClassifier::PedestrianAttribute>> attrc = classifier_->BatchClassify(images_);

    for (int i = 0; i < objs_.size(); i++) {
        Pedestrian *p = (Pedestrian *) objs_[i];
        std::vector<PedestrianClassifier::PedestrianAttribute> attrs_i = attrc[i];
        std::vector<Pedestrian::Attr> attrs_o;
        for (int j = 0; j < attrs_i.size(); j++) {
            Pedestrian::Attr attr;
            attr.index = attrs_i[j].index;
            attr.tagname = attrs_i[j].tagname;
            attr.confidence = attrs_i[j].confidence;
            attr.threshold_lower = attrs_i[j].threshold_lower;
            attr.threshold_upper = attrs_i[j].threshold_upper;
            attrs_o.push_back(attr);
        }
        p->set_attrs(attrs_o);
        std::vector<Pedestrian::Attr> a = p->attrs();
    }

    return false;
}

bool PedestrianClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if DEBUG
#else
    if(performance_>RECORD_UNIT)
    {
        if(!RecordFeaturePerformance())
        {
            return false;
        }
    }
#endif
    objs_.clear();
    images_.clear();

    objs_ = frameBatch->CollectObjects(OPERATION_VEHICLE_PEDESTRIAN_ATTR);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_PEDESTRIAN) {

            Pedestrian *p = (Pedestrian *) obj;
            images_.push_back(p->image());
            ++itr;
            performance_++;

        }
        else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of pedestrian: " << obj->id() << endl;
        }
    }

    return true;

}

bool PedestrianClassifierProcessor::RecordFeaturePerformance() {
    return RecordPerformance(FEATURE_CAR_MARK, performance_);
}

}
