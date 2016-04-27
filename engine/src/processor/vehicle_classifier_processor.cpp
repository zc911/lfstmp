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

    if (!checkOperation(frame)) {
        DLOG(INFO)<< "OPERATION_VEHICLE_STYLE disable." << endl;
        return Proceed(frame);
    }

    vector<Mat> images;
    vector<Object*> objects = frame->objects();

    for (int i = 0; i < objects.size(); ++i) {

        Object *obj = objects[i];
        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;
            DLOG(INFO)<< "Put vehicle images into vector to be classified: " << obj->id() << endl;
            images.push_back(v->image());

        } else {
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }

    }

    vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(images);

    if (result.size() != objects.size()) {
        DLOG(ERROR)<< "Classification results size is different from object size. " << endl;
        frame->set_status(FRAME_STATUS_ERROR);
        frame->set_error_msg("Classification results size is different from object size.");
        return;
    }

    for (int i = 0; i < result.size(); ++i) {
        vector<Prediction> pre = result[i];
        Vehicle *v = (Vehicle*) objects[i];
        Prediction max = MaxPrediction(result[0]);
        v->set_class_id(max.first);
        v->set_confidence(max.second);

    }

    //Proceed(frame);
}

void VehicleClassifierProcessor::Update(FrameBatch *frameBatch) {

    for (int i = 0; i < frameBatch->frames().size(); i++) {
        Frame *frame = frameBatch->frames()[i];
        Update(frame);
    }
    Proceed(frameBatch);

}

bool VehicleClassifierProcessor::checkOperation(Frame *frame) {
    return frame->operation().Check(OPERATION_VEHICLE_STYLE);
}
bool VehicleClassifierProcessor::checkStatus(Frame *frame) {
    return true;
}

}
