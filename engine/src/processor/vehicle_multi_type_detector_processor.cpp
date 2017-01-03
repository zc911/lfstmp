#include "vehicle_multi_type_detector_processor.h"
#include "processor_helper.h"
#include "algorithm_def.h"
#include "util/convert_util.h"

using namespace dgvehicle;

namespace dg {

using namespace AlgorithmProcessorType;

VehicleMultiTypeDetectorProcessor::VehicleMultiTypeDetectorProcessor(bool car_only, bool accelate)
    : Processor(), car_only_(car_only) , car_only_detector_(NULL), car_only_confirm_(NULL), vehicle_detector_(NULL) {
    if (car_only_) {
        car_only_detector_ = AlgorithmFactory::GetInstance()->CreateMultiTypeDetector(c_carOnlyCaffeDetector, accelate);
        car_only_confirm_ = AlgorithmFactory::GetInstance()->CreateMultiTypeDetector(c_carOnlyConfirmCaffeDetector, accelate);
    } else {
        vehicle_detector_ = AlgorithmFactory::GetInstance()->CreateMultiTypeDetector(c_vehicleCaffeDetector, accelate);
    }

    base_id_ = 0;
    threshold_ = 0.0f;
}

// TODO complete construction
VehicleMultiTypeDetectorProcessor::~VehicleMultiTypeDetectorProcessor() {
    if (vehicle_detector_)
        delete vehicle_detector_;

    if (car_only_confirm_)
        delete car_only_confirm_;

    if (car_only_detector_)
        delete car_only_detector_;
}

bool VehicleMultiTypeDetectorProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start detector: " << frameBatch->id() << endl;

    vector<int> frameIds;
    vector<Mat> images;
    vector<vector<dgvehicle::Detection> > detect_results;

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
        performance_++;
    }

    if (images.size() == 0) {
        return true;
    }

    if (frameIds.size() != images.size()) {
        LOG(ERROR) << "Frame id size not equals to images size" << endl;
        return false;
    }

    if (car_only_) {
        VLOG(VLOG_RUNTIME_DEBUG) << "Car only detection and confirm. " << endl;
        car_only_detector_->BatchProcess(images, detect_results);
        car_only_confirm_->BatchProcess(images, detect_results);

    } else {
        VLOG(VLOG_RUNTIME_DEBUG) << "Multi detection " << endl;
        vehicle_detector_->BatchProcess(images, detect_results);
    }

    if (detect_results.size() < images.size()) {
        LOG(ERROR) << "Detection results size not equals to frame batch size: " << detect_results.size() << "-"
                   << frameBatch->frames().size() << endl;
        return false;
    }

    int id = 0;
    for (int i = 0; i < frameIds.size(); ++i) {

        int frameId = frameIds[i];
        vector<dgvehicle::Detection> &imageDetection = detect_results[i];

        Frame *frame = frameBatch->frames()[frameId];

        for (int j = 0; j < imageDetection.size(); ++j) {

            if (imageDetection[j].confidence < threshold_) {
                VLOG(VLOG_RUNTIME_DEBUG)
                << "Detection confidence is low than threshold " << imageDetection[j].confidence << ":" << threshold_
                    << endl;
                continue;
            }

            Detection d = ConvertDgvehicleDetection(imageDetection[j]);
            if (!roiFilter(frame->get_rois(), d.box()))
                continue;
            Object *obj = NULL;
            if (d.id == DETECTION_PEDESTRIAN) {
                // if is pedestrain
                Pedestrian *p = new Pedestrian();
                Mat roi = CutImage(frame->payload()->data(), d.box());

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

                Mat roi = CutImage(frame->payload()->data(), d.box());

                if (roi.rows == 0 || roi.cols == 0) {
                    continue;
                }

                if (objectType == OBJECT_BICYCLE || objectType == OBJECT_TRICYCLE) {
                    NonMotorVehicle *v = new NonMotorVehicle(objectType);
                    v->set_image(roi);
                    v->set_id(base_id_ + id++);
                    obj = static_cast<Object *>(v);
                } else {
                    Vehicle *v = new Vehicle(objectType);
                    v->set_image(roi);
                    v->set_id(base_id_ + id++);
                    // set pose head in default
                    v->set_pose(Vehicle::VEHICLE_POSE_HEAD);
                    obj = static_cast<Object *>(v);
                }
            }

            if (obj) {
                //     Mat tmp = frame->payload()->data();
                //   rectangle(tmp,d.box,Scalar(255,0,0));

                obj->set_detection(d);
                frame->put_object(obj);
            }

        }

    }
    VLOG(VLOG_RUNTIME_DEBUG) << "finish detector: " << frameBatch->id() << endl;
    return true;
}


bool VehicleMultiTypeDetectorProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if DEBUG
#else

    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
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
