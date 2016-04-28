#include "vehicle_multi_type_detector_processor.h"

namespace dg {
VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(
        int batch_size, int gpu_id, int rescale, bool is_model_encrypt)
        : Processor() {

    CaffeConfig config;
    config.batch_size = batch_size;
    config.is_model_encrypt = is_model_encrypt;

    if (config.is_model_encrypt) {
        config.model_file = MODEL_FILE;
        config.deploy_file = DEPLOY_FILE;
    } else {
        config.model_file = MODEL_FILE_EN;
        config.deploy_file = DEPLOY_FILE_EN;
    }

    config.use_gpu = true;
    config.gpu_id = gpu_id;
    config.rescale = rescale;
    detector_ = new VehicleMultiTypeDetector(config);
}

VehicleMultiTypeDetectorProcessor::~VehicleMultiTypeDetectorProcessor() {

}

void VehicleMultiTypeDetectorProcessor::Update(FrameBatch *frameBatch) {

    beforeUpdate(frameBatch);

    for (int i = 0; i < frameBatch->frames().size(); i++) {
        Frame *frame = frameBatch->frames()[i];

        if (!frame->operation().Check(OPERATION_VEHICLE_DETECT)) {
            DLOG(INFO)<<"frame :"<<frame->id()<<" doesn't need to be detected"<<endl;
        }

        DLOG(INFO)<< "Start detect frame: " << frame->id() << endl;
        Mat data = frame->payload()->data();
        vector<Detection> detections = detector_->Detect(data);
        int id = 0;
        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection detection = *itr;
            Object *obj;
            if (1) {
                Vehicle *v = new Vehicle(OBJECT_CAR);
                obj = static_cast<Object*>(v);
                obj->set_id(id++);
                Mat roi = Mat(data, detection.box);
                v->set_image(roi);
            }

            obj->set_detection(detection);
            frame->put_object(obj);
            print(detection);
        }
        DLOG(INFO)<<frame->objects().size()<<" "<<detections.size()<<" cars are detected in frame "<<frame->id()<<endl;
    }
    Proceed(frameBatch);

}

void VehicleMultiTypeDetectorProcessor::beforeUpdate(FrameBatch *frameBatch) {

}
bool VehicleMultiTypeDetectorProcessor::checkStatus(Frame *frame) {
    return true;
}
}
