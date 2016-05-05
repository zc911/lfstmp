#include "vehicle_multi_type_detector_processor.h"

namespace dg {

VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(const VehicleMultiTypeDetector::VehicleMultiTypeConfig &config)
        : Processor() {

    detector_ = new VehicleMultiTypeDetector(config);
}

// TODO complete construction
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
        DLOG(INFO)<<data.cols<<"data"<<endl;
        vector<Detection> detections = detector_->Detect(data);
        int id = 0;
        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection detection = *itr;
            Object *obj = NULL;

            // TODO check object type
            if (detection.id == DETECTION_CAR) {
                Vehicle *v = new Vehicle(OBJECT_CAR);
                Mat roi = Mat(data, detection.box);
                v->set_image(roi);
                v->set_id(id++);
                obj = static_cast<Object*>(v);
            } else {

            }
            if (obj) {
                obj->set_detection(detection);
                frame->put_object(obj);
            }

        }
        DLOG(INFO)<< frame->objects().size() << " cars are detected in frame "<<frame->id() << endl;
    }
    Proceed(frameBatch);

}

void VehicleMultiTypeDetectorProcessor::beforeUpdate(FrameBatch *frameBatch) {

}
bool VehicleMultiTypeDetectorProcessor::checkStatus(Frame *frame) {
    return true;
}
}
