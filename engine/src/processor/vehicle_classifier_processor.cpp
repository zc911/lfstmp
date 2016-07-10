#include "vehicle_classifier_processor.h"

namespace dg {

VehicleClassifierProcessor::VehicleClassifierProcessor(
    const vector<VehicleCaffeClassifier::VehicleCaffeConfig> &configs) {

    for (int i = 0; i < configs.size(); i++) {

        VehicleCaffeClassifier *classifier = new VehicleCaffeClassifier(
            configs[i]);

        classifiers_.push_back(classifier);

    }

}

VehicleClassifierProcessor::~VehicleClassifierProcessor() {
    for (int i = 0; i < classifiers_.size(); i++) {
        delete classifiers_[i];
    }
    classifiers_.clear();
    objs_.clear();
    images_.clear();
}

bool VehicleClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start vehicle classify process" << endl;

    vector<vector<Prediction> > result;

 /*   for_each(classifiers_.begin(), classifiers_.end(), [&](VehicleCaffeClassifier *elem) {
      auto tmpPred = elem->ClassifyAutoBatch(images_);
      vote(tmpPred, result, classifiers_.size());
    });*/
    for(auto *elem:classifiers_){
        auto tmpPred = elem->ClassifyAutoBatch(images_);
        vote(tmpPred, result, classifiers_.size());

    }

    //set results
    for (int i = 0; i < objs_.size(); i++) {
        if (result[i].size() < 0) {
            continue;
        }
        vector<Prediction> pre = result[i];
        Vehicle *v = (Vehicle *) objs_[i];
        Prediction max = MaxPrediction(result[i]);
        v->set_class_id(max.first);
        v->set_confidence(max.second);
    }

    return true;
}

bool VehicleClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {


    images_.clear();
    vehiclesResizedMat(frameBatch);
    return true;
}

void VehicleClassifierProcessor::vehiclesResizedMat(FrameBatch *frameBatch) {

    images_.clear();
    objs_.clear();
    objs_ = frameBatch->CollectObjects(OPERATION_VEHICLE_STYLE);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;
        //collect car objects
        if (obj->type() == OBJECT_CAR) {
            Vehicle *v = (Vehicle *) obj;
            images_.push_back(v->resized_image());
            ++itr;
        } else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
}

bool VehicleClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_PEDESTRIAN_ATTR, performance_);

}

}
