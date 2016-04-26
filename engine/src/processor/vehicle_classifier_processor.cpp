#include "vehicle_classifier_processor.h"

namespace dg {

VehicleClassifierProcessor::VehicleClassifierProcessor() {
    CaffeConfig config;
    config.model_file =
            "models/classify/car_python_mini_alex_256_0_iter_70000.caffemodel";
    config.deploy_file = "models/classify/deploy_256.prototxt";

    config.is_model_encrypt = false;
    config.batch_size = 1;
    classifier_ = new VehicleCaffeClassifier(config);
}

VehicleClassifierProcessor::~VehicleClassifierProcessor() {
    if (classifier_)
        delete classifier_;
}

void VehicleClassifierProcessor::Update(Frame *frame) {

    vector<Mat> images;
    vector<Object*> objects = frame->objects();

    for (int i = 0; i < objects.size(); ++i) {

        Object *obj = objects[i];
        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;
            DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
            images.push_back(v->image());

        } else {
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }

    }

    vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(images);

    SortPrediction(result);

    cout << "Classify result: " << endl;
    for (int i = 0; i < 6; ++i) {
        cout << "Class: " << result[0][i].first << " , Conf: "
             << result[0][i].second << endl;
    }
}

void VehicleClassifierProcessor::Update(FrameBatch *frameBatch) {

}

bool VehicleClassifierProcessor::checkOperation(Frame *frame) {
    return true;
}
bool VehicleClassifierProcessor::checkStatus(Frame *frame) {
    return true;
}

}
