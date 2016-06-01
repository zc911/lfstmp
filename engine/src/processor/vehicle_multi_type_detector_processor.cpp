#include "vehicle_multi_type_detector_processor.h"
#include "model/model.h"
#include "processor_helper.h"
namespace dg {

VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(
    const VehicleCaffeDetector::VehicleCaffeDetectorConfig &config)
    : Processor() {

    detector_ = new VehicleCaffeDetector(config);
    base_id_ = 0;
}

// TODO complete construction
VehicleMultiTypeDetectorProcessor::~VehicleMultiTypeDetectorProcessor() {
    if (detector_)
        delete detector_;
}

bool VehicleMultiTypeDetectorProcessor::process(FrameBatch *frameBatch) {
LOG(INFO)<<"start detector"<<endl;
    vector<Mat> images;
    vector<vector<Detection> > detect_results;

    for (int i = 0; i < frameBatch->frames().size(); i++) {
        Frame *frame = frameBatch->frames()[i];

        if (!frame->operation().Check(OPERATION_VEHICLE_DETECT)) {

            DLOG(INFO) << "Frame :" << frame->id() << " doesn't need to be detected" << endl;
            continue;
        }

        DLOG(INFO) << "Start detect frame: " << frame->id() << endl;
        Mat data = frame->payload()->data();

        if (data.rows == 0 || data.cols == 0) {
            LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
            continue;
        }

        images.push_back(frame->payload()->data());
    }

    detector_->DetectBatch(images, detect_results);

    if (detect_results.size() < frameBatch->frames().size()) {
        LOG(ERROR) << "Detection results size not equals to frame batch size: " << detect_results.size() << "-"
            << frameBatch->frames().size() << endl;
        return false;
    }
    int id = 0;

    for (int i = 0; i < frameBatch->frames().size(); ++i) {

        Frame *frame = frameBatch->frames()[i];
        vector<Detection> &imageDetection = detect_results[i];

        for (int j = 0; j < imageDetection.size(); ++j) {
            Detection d = imageDetection[j];
            if(!roiFilter(frame->get_rois(),d.box))
                continue;
            Object *obj = NULL;
            if (d.id == DETECTION_PEDESTRIAN) {
                // if is pedestrain
                Pedestrain *p = new Pedestrain();
                p->set_detection(d);
                p->set_id(base_id_ + id++);
                obj = static_cast<Object *>(p);

            } else if (d.id != DETECTION_UNKNOWN) {
                ObjectType objectType;


                if (d.id == DETECTION_CAR)
                    objectType = OBJECT_CAR;
                else if (d.id == DETECTION_BICYCLE)
                    objectType = OBJECT_BICYCLE;
                else if (d.id == DETECTION_TRICYCLE)
                    objectType = OBJECT_TRICYCLE;
                else
                    objectType = OBJECT_UNKNOWN;

                Vehicle *v = new Vehicle(objectType);
                Mat roi = CutImage(frame->payload()->data(), d.box);


                if (roi.rows == 0 || roi.cols == 0) {
                    continue;
                }
                v->set_image(roi);
                v->set_id(base_id_ + id++);
                obj = static_cast<Object *>(v);
            }

            if (obj) {
                obj->set_detection(d);
                frame->put_object(obj);
            }

        }
    }


//    vector<Detection> detections = detector_->DetectBatch(data);
//    int id = 0;
//    for (vector<Detection>::iterator itr = detections.begin();
//         itr != detections.end(); ++itr) {
//        Detection detection = *itr;
//        Object *obj = NULL;
//
//        // TODO other detection type?
//        if (detection.id == DETECTION_CAR) {
//            Vehicle *v = new Vehicle(OBJECT_CAR);
//            Mat roi = Mat(data, detection.box);
//            if (roi.rows == 0 || roi.cols == 0) {
//                continue;
//            }
//            v->set_image(roi);
//            v->set_id(base_id_ + id++);
//            obj = static_cast<Object *>(v);
//        }
//
//        if (obj) {
//            obj->set_detection(detection);
//            frame->put_object(obj);
//        }
//
//    }
//    DLOG(INFO) << frame->objects().size() << " cars are detected in frame " << frame->id() << endl;
    return true;
}


bool VehicleMultiTypeDetectorProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if RELEASE
    if(performance_>20000) {
        if(!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif

    return true;
}

bool VehicleMultiTypeDetectorProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_DETECTION, performance_);

}

}
