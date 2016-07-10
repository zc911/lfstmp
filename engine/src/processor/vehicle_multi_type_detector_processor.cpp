#include <alg/detector/detector.h>
#include "alg/detector/vehicle_caffe_detector.h"
#include "vehicle_multi_type_detector_processor.h"
#include "processor_helper.h"

namespace dg {

VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(
    const VehicleCaffeDetectorConfig &config)
    : Processor(), config_(config) {
    if (config_.car_only) {
        car_only_detector_ = new CarOnlyCaffeDetector(config);
    } else {
        vehicle_detector_ = new VehicleCaffeDetector(config);
    }

    base_id_ = 0;
}

// TODO complete construction
VehicleMultiTypeDetectorProcessor::~VehicleMultiTypeDetectorProcessor() {
    if (vehicle_detector_)
        delete vehicle_detector_;

    if (car_only_detector_)
        delete car_only_detector_;
}

bool VehicleMultiTypeDetectorProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start detector" << endl;

    vector<int> frameIds;
    vector<Mat> images;
    vector<vector<Detection> > detect_results;

    for (int i = 0; i < frameBatch->frames().size(); i++) {
        Frame *frame = frameBatch->frames()[i];

        if (!frame->operation().Check(OPERATION_VEHICLE_DETECT)) {

            DLOG(INFO) << "Frame :" << frame->id() << " doesn't need to be detected" << endl;
            continue;
        }

        Mat data = frame->payload()->data();

        if (data.rows == 0 || data.cols == 0) {
            LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
            continue;
        }
        frameIds.push_back(i);
        images.push_back(frame->payload()->data());
    }

    if (images.size() == 0) {
        return true;
    }

    if (frameIds.size() != images.size()) {
        LOG(ERROR) << "Frame id size not equals to images size" << endl;
        return false;
    }

    if (config_.car_only) {
        car_only_detector_->DetectBatch(images, detect_results);
    } else {
        vehicle_detector_->DetectBatch(images, detect_results);
    }

    if (detect_results.size() < images.size()) {
        LOG(ERROR) << "Detection results size not equals to frame batch size: " << detect_results.size() << "-"
            << frameBatch->frames().size() << endl;
        return false;
    }

    int id = 0;
    for (int i = 0; i < frameIds.size(); ++i) {

        int frameId = frameIds[i];
        vector<Detection> &imageDetection = detect_results[i];

        Frame *frame = frameBatch->frames()[frameId];

        for (int j = 0; j < imageDetection.size(); ++j) {
            Detection d = imageDetection[j];
            if (!roiFilter(frame->get_rois(), d.box))
                continue;
            Object *obj = NULL;
            if (d.id == DETECTION_PEDESTRIAN) {
                // if is pedestrain
                Pedestrian *p = new Pedestrian();
                Mat roi = CutImage(frame->payload()->data(), d.box);

                if (roi.rows == 0 || roi.cols == 0) {
                    continue;
                }
                p->set_image(roi);
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
