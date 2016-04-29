#include "vehicle_classifier_processor.h"

namespace dg {

int vote(vector<vector<Prediction> > &src, vector<vector<Prediction> > &dst,
         int factor) {
    if (src.size() > dst.size()) {
        for (int i = 0; i < src.size(); i++) {
            vector<Prediction> tmpSrc = src[i];
            vector<Prediction> tmpDst;
            for (int j = 0; j < tmpSrc.size(); j++) {
                tmpDst.push_back(
                        pair<int, float>(tmpSrc[j].first,
                                         tmpSrc[j].second / factor));
            }
            dst.push_back(tmpDst);
        }
        return 1;
    }
    for (int i = 0; i < src.size(); i++) {

        vector<Prediction> tmpSrc = src[i];
        vector<Prediction> tmpDst = dst[i];
        if (tmpSrc.size() != tmpDst.size()) {
            return -1;
        }
        for (int j = 0; j < tmpSrc.size(); j++) {
            tmpDst[j].second += tmpSrc[j].second / factor;
        }
        dst[i] = tmpDst;
    }
    return 1;
}
VehicleClassifierProcessor::VehicleClassifierProcessor() {
    classifiers_size_ = 1;

    for (int i = 0; i < classifiers_size_; i++) {
        CaffeConfig config;
//        config.model_file = "models/car_style/front_day_" + to_string(i)
//                + "/car_python_mini_alex_256_" + to_string(i)
//                + "_iter_70000.caffemodel";
//        config.deploy_file = "models/car_style/front_day_" + to_string(i)
//                + "/deploy_256.prototxt";

        config.model_file = "models/classify/car_python_mini_alex_256_0_iter_70000.caffemodel";
        config.deploy_file = "models/classify/deploy_256.prototxt";
        config.is_model_encrypt = false;
        config.batch_size = 1;

        VehicleCaffeClassifier *classifier = new VehicleCaffeClassifier(config);

        classifiers_.push_back(classifier);

    }

}

VehicleClassifierProcessor::~VehicleClassifierProcessor() {
    for (int i = 0; i < classifiers_.size(); i++) {
        delete classifiers_[i];
    }
    classifiers_.clear();
}

void VehicleClassifierProcessor::Update(FrameBatch *frameBatch) {

    DLOG(INFO)<<"Start vehicle classify frame: "<< endl;

    beforeUpdate(frameBatch);
    vector<vector<Prediction> > result;
    for_each(classifiers_.begin(),classifiers_.end(),[&](VehicleCaffeClassifier *elem) {
                auto tmpPred=elem->ClassifyAutoBatch(images_);
                vote(tmpPred,result,classifiers_size_);

            });

    for(int i=0;i<objs_.size();i++) {
        if(result[i].size()<0) {
            continue;
        }
        vector<Prediction> pre = result[i];
        Vehicle *v = (Vehicle*) objs_[i];
        Prediction max = MaxPrediction(result[i]);
        v->set_class_id(max.first);
        v->set_confidence(max.second);
    }

    Proceed(frameBatch);

}

void VehicleClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {
    images_.clear();
    images_ = vehicles_resized_mat(frameBatch);
}
bool VehicleClassifierProcessor::checkStatus(Frame *frame) {
    return true;
}

vector<Mat> VehicleClassifierProcessor::vehicles_resized_mat(
        FrameBatch *frameBatch) {
    vector<cv::Mat> vehicleMat;
    objs_.clear();
    objs_ = frameBatch->collect_objects(OPERATION_VEHICLE_STYLE);
    for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
            ++itr) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;

            DLOG(INFO)<< "Put vehicle images to be type classified: " << obj->id() << endl;
            vehicleMat.push_back(v->resized_image());

        } else {
            delete obj;
            itr = objs_.erase(itr);
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
    return vehicleMat;
}
}
