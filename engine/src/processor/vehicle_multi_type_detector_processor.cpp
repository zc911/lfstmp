#include "vehicle_multi_type_detector_processor.h"

namespace dg {

VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(
        const VehicleMultiTypeDetector::VehicleMultiTypeConfig &config)
        : Processor() {

    detector_ = new VehicleMultiTypeDetector(config);
    base_id_ = 0;
}

// TODO complete construction
VehicleMultiTypeDetectorProcessor::~VehicleMultiTypeDetectorProcessor() {
    if (detector_)
        delete detector_;
}

bool VehicleMultiTypeDetectorProcessor::process(FrameBatch *frameBatch) {

    for (int i = 0; i < frameBatch->frames().size(); i++) {
        Frame *frame = frameBatch->frames()[i];

        if (!frame->operation().Check(OPERATION_VEHICLE_DETECT)) {

            DLOG(INFO)<<"Frame :"<<frame->id()<<" doesn't need to be detected"<<endl;
            continue;
        }

        DLOG(INFO)<< "Start detect frame: " << frame->id() << endl;
        Mat data = frame->payload()->data();

        if (data.rows == 0 || data.cols == 0) {
            LOG(ERROR)<< "Frame data is NULL: " << frame->id() << endl;
            continue;
        }

        vector<Detection> detections = detector_->Detect(data);
        int id = 0;
        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection detection = *itr;
            Object *obj = NULL;

            // TODO other detection type?
            if (detection.id == DETECTION_CAR) {
                Vehicle *v = new Vehicle(OBJECT_CAR);
                Mat roi = Mat(data, detection.box);
                if (roi.rows == 0 || roi.cols == 0) {
                    continue;
                }
                v->set_image(roi);
                v->set_id(base_id_ + id++);
                obj = static_cast<Object*>(v);
            }

            if (obj) {
                obj->set_detection(detection);
                frame->put_object(obj);
            }

        }
        DLOG(INFO)<< frame->objects().size() << " cars are detected in frame "<<frame->id() << endl;
    }
    return true;
}

}
