/*
 * vehicle_window_detector_processor.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: jiajaichen
 */

#include "vehicle_window_detector_processor.h"
#include "processor_helper.h"
#include "util/caffe_helper.h"
#include "string_util.h"
#include "algorithm_def.h"
#include "util/convert_util.h"


using namespace dgvehicle;
namespace dg {
VehicleWindowDetectorProcessor::VehicleWindowDetectorProcessor()
    : Processor() {

    ssd_window_detector_ =
        AlgorithmFactory::GetInstance()->CreateProcessorInstance(AlgorithmProcessorType::c_windowCaffeSsdDetector);

}

VehicleWindowDetectorProcessor::~VehicleWindowDetectorProcessor() {
    if (ssd_window_detector_) {
        delete ssd_window_detector_;
    }
    images_.clear();
    resized_images_.clear();
}

bool VehicleWindowDetectorProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start window processor" << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start marker and window processor " << frameBatch->id() << endl;

    float costtime, diff;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    vector<vector<dgvehicle::Detection> > crops;
    vector<Window> windows;
    ssd_window_detector_->BatchProcess(images_, crops);

    int target_row = 256;
    int target_col = 384;

    for (int i = 0; i < crops.size(); i++) {
        if (crops[i].size() <= 0){
            continue;
        }

        int xmin = crops[i][0].box.x;
        int ymin = crops[i][0].box.y;
        int xmax = crops[i][0].box.x + crops[i][0].box.width;
        int ymax = crops[i][0].box.y + crops[i][0].box.height;
        int cxmin, cymin, cpxmin, cpymin;
        cv::Mat image = images_[i].clone();
        cv::Mat phone_image = images_[i].clone();

        Mat phone_img = crop_phone_image(phone_image, xmin, ymin, xmax, ymax, &cpxmin, &cpymin);
        Mat img = crop_image(image, xmin, ymin, xmax, ymax, &cxmin, &cymin);
        Mat resized_img = crop_image(image, xmin, ymin, xmax, ymax);

        vector<float> params;
        params.push_back(cxmin);
        params.push_back(cymin);
        int tymin;
        int tymax;
        float ratio = 0.15;
        show_enlarged_box(images_[i], image, xmin, ymin, xmax, ymax, &tymin, &tymax, ratio);

        params.push_back(tymin);
        params.push_back(tymax);

        params.push_back(img.rows * 1.0 / target_row);
        params.push_back(img.cols * 1.0 / target_col);
        vector<Rect> fob = forbidden_area(xmin, ymin, xmax, ymax);

        resize(phone_img, phone_img, Size(512, 256));
        resize(img, img, Size(target_col, target_row));

        Window *window = new Window(img, fob, params);
        window->set_detection(ConvertDgvehicleDetection(crops[i][0]));
        window->set_resized_img(resized_img);
        window->set_phone_img(phone_img);
        Vehicle *v = (Vehicle *) objs_[i];
        v->set_window(window);
    }

    gettimeofday(&end, NULL);
    diff = ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec)
        / 1000.f;
    VLOG(VLOG_PROCESS_COST) << "Window cost: " << diff << endl;
    objs_.clear();

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish window processor" << frameBatch->id() << endl;
    return true;
}

bool VehicleWindowDetectorProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    objs_.clear();
    images_.clear();

    objs_ = frameBatch->CollectObjects(
        OPERATION_VEHICLE_MARKER | OPERATION_DRIVER_PHONE | OPERATION_DRIVER_BELT | OPERATION_CODRIVER_BELT);

    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle *) obj;
            // Only the detected vehicle with HEAD pose
            if (v->pose() == Vehicle::VEHICLE_POSE_HEAD) {
                images_.push_back(v->image());
                performance_++;
            }
            ++itr;

        } else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }

    return true;

}

bool VehicleWindowDetectorProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
