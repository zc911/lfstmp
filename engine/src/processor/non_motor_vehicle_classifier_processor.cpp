#include "non_motor_vehicle_classifier_processor.h"

namespace dg {

using namespace dgvehicle;

NonMotorVehicleClassifierProcessor::NonMotorVehicleClassifierProcessor() {
    nonMotorVehicleClassifier = AlgorithmFactory::GetInstance()->CreateNonMotorVehicleClassifier();
}

NonMotorVehicleClassifierProcessor::~NonMotorVehicleClassifierProcessor() {
    if (nonMotorVehicleClassifier) {
        delete nonMotorVehicleClassifier;
        nonMotorVehicleClassifier = NULL;
    }

    for (size_t i = 0; i < objs_.size(); ++i) {
        delete objs_[i];
        objs_[i] = NULL;
    }
    objs_.clear();
    images_.clear();
}

bool NonMotorVehicleClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start non-motor vehicle classify process" << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start non-motor vehicle classify processor" << endl;
    for (int i = 0; i < images_.size(); i++) {
        if (images_[i].cols == 0 || images_[i].rows == 0) {
            return false;
        }
    }

    vector<vector<NonMotorAttribute> > results;

    nonMotorVehicleClassifier->BatchClassify(images_, results);

    for (size_t i = 0; i < objs_.size(); ++i) {
        NonMotorVehicle *nmVehicle = static_cast<NonMotorVehicle *>(objs_[i]);
        vector<NonMotorVehicle::Attr> attr_;
        for (size_t j = 0; j < results[i].size(); ++j) {
            NonMotorVehicle::Attr attr;
            attr.index = results[i][j].idx;
            attr.tagname = results[i][j].name;
            attr.confidence = (Confidence) results[i][j].confidence;
            attr.threshold_upper = results[i][j].thresh_high;
            attr.threshold_lower = results[i][j].thresh_low;
            attr.mappingId = results[i][j].mappingId;
            attr.categoryId = results[i][j].categoryId;
            attr_.push_back(attr);
        }
        nmVehicle->attrs().clear();
        nmVehicle->attrs() = attr_;
    }
    return true;

}

bool NonMotorVehicleClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {


#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
/*    for (size_t i = 0; i < objs_.size(); ++i) {
        delete objs_[i];
        objs_[i] = NULL;
    } */
    objs_.clear();
    images_.clear();

    objs_ = frameBatch->CollectObjects(OPERATION_NON_VEHICLE_ATTR);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;
        if (obj->type() == OBJECT_BICYCLE || obj->type() == OBJECT_TRICYCLE) {

            NonMotorVehicle *p = (NonMotorVehicle *) obj;
            images_.push_back(p->image());
            ++itr;
            performance_++;
        }
        else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of non-motor vehicle: " << obj->id() << endl;
        }
    }
    vehiclesResizedMat(frameBatch);
    return true;
}

void NonMotorVehicleClassifierProcessor::vehiclesResizedMat(FrameBatch *frameBatch) {
/**
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
            performance_++;

        } else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
**/
}

bool NonMotorVehicleClassifierProcessor::RecordFeaturePerformance() {
    return true;
}

}
