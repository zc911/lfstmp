#include "vehicle_classifier_processor.h"
#include <boost/algorithm/string/split.hpp>
#include "algorithm_def.h"
#include "util/caffe_helper.h"
#include "string_util.h"


using namespace dgvehicle;
namespace dg {

VehicleClassifierProcessor::VehicleClassifierProcessor(string &mappingFilePath, bool encrypt) {

    AlgorithmFactory::GetInstance()->CreateBatchProcessor(AlgorithmProcessorType::c_vehicleCaffeClassifier,
                                                          classifiers_,
                                                          encrypt);

    initIsHeadRepo(mappingFilePath);
}

VehicleClassifierProcessor::~VehicleClassifierProcessor() {
    for (int i = 0; i < classifiers_.size(); i++) {
        delete classifiers_[i];
    }
    classifiers_.clear();
    objs_.clear();
    images_.clear();
}

void VehicleClassifierProcessor::initIsHeadRepo(string &mapFilePath) {
    ifstream input(mapFilePath);

    int max = 0;
    vector<std::pair<int, int>> pairs;
    for (string line; std::getline(input, line);) {
        vector<string> tokens;
        boost::iter_split(tokens, line, boost::first_finder(","));
        assert(tokens.size() == 10);

        int index = parseInt(tokens[0]);
        if (index > max)
            max = index;
        int isHead = parseInt(tokens[3]);
        pairs.push_back(std::pair<int, int>(index, isHead));
    }

    vehicle_is_head_repo_.resize(max + 1);
    for (const std::pair<int, int> &p : pairs) {
        vehicle_is_head_repo_[p.first] = p.second;
    }

}

bool VehicleClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start vehicle classify process" << frameBatch->id() << endl;

    vector<vector<Prediction> > result;

    for (auto *elem : classifiers_) {

        std::vector<vector<Prediction>> tmpPred;
        elem->BatchProcess(images_, tmpPred);

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
        Vehicle::VehiclePose pose;
        if (vehicle_is_head_repo_.size() >= max.first && vehicle_is_head_repo_[max.first] == 0) {
            pose = Vehicle::VEHICLE_POSE_TAIL;
        } else {
            pose = Vehicle::VEHICLE_POSE_HEAD;
        }
        v->set_pose(pose);
    }

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish vehicle classify process" << frameBatch->id() << endl;
    return true;
}

bool VehicleClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {


#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
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
            performance_++;

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
