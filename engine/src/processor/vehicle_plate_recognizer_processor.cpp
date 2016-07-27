/*
 * vehicle_plate_recognizer_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajiachen
 */
#include "vehicle_plate_recognizer_processor.h"
#include "processor_helper.h"
namespace dg {

PlateRecognizerProcessor::PlateRecognizerProcessor(
    const PlateRecognizer::PlateConfig &pConfig) {
    enable_sharpen_ = pConfig.isSharpen;
    recognizer_ = PlateRecognizer::Instance(pConfig);
}
PlateRecognizerProcessor::~PlateRecognizerProcessor() {
    if (recognizer_)
        delete recognizer_;
    images_.clear();
}

bool PlateRecognizerProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start plate recognize processor " << endl;

    if (images_.size() != objs_.size()) {
        LOG(ERROR) << "Image size not equal to vehicle size. " << endl;
        return false;
    }

    vector<Vehicle::Plate> results = recognizer_->RecognizeBatch(images_);
    for (int i = 0; i < images_.size(); i++) {
        Vehicle *v = (Vehicle *) objs_[i];
        vector<Vehicle::Plate> plates;
        plates.push_back(results[i]);
        v->set_plates(plates);
    }
    return true;
}

bool PlateRecognizerProcessor::beforeUpdate(FrameBatch *frameBatch) {
    filterVehicle(frameBatch);
#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    this->filterVehicle(frameBatch);
    performance_ += frameBatch->batch_size();
    return true;
}

void PlateRecognizerProcessor::sharpenImage(const cv::Mat &image,
                                            cv::Mat &result) {
    Mat tmp;
    float wRate = (float) image.cols / 350.0;
    float hRate = (float) image.rows / 290.0;
    if ((wRate < 1) || (hRate < 1)) {
        if (wRate < hRate) {
            resize(image, tmp, Size(350, (int) ((float) image.rows / wRate)));
        } else {
            resize(image, tmp, Size((int) ((float) image.cols / hRate), 290));
        }
    } else {
        tmp = image;
    }

//创建并初始化滤波模板
    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
    kernel.at<float>(1, 1) = 5.0;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    kernel.at<float>(2, 1) = -1.0;

    result.create(image.size(), image.type());

//对图像进行滤波
    cv::filter2D(image, result, image.depth(), kernel);
}
void PlateRecognizerProcessor::filterVehicle(FrameBatch *frameBatch) {
    objs_.clear();
    images_.clear();
    objs_ = frameBatch->CollectObjects(OPERATION_VEHICLE_PLATE);
    vector<Object *>::iterator itr = objs_.begin();
    while (itr != objs_.end()) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {
            Vehicle *v = (Vehicle *) obj;

            if (enable_sharpen_) {
                Mat result;
                sharpenImage(v->image(), result);
                images_.push_back(result);
            } else {
                images_.push_back(v->image());

            }
            itr++;

        } else {
            itr = objs_.erase(itr);
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << endl;
        }
    }

}
bool PlateRecognizerProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_PLATE, performance_);
}

}

