/*
 * vehicle_plate_recognizer_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajiachen
 */
#include "vehicle_plate_recognizer_processor.h"

namespace dg {

PlateRecognizerProcessor::PlateRecognizerProcessor() {
    PlateRecognizer::PlateConfig pConfig;
    pConfig.LocalProvince = "";

    pConfig.OCR = 1;
    pConfig.PlateLocate = 5;

    recognizer_ = new PlateRecognizer(pConfig);
}

PlateRecognizerProcessor::~PlateRecognizerProcessor() {
    if (recognizer_)
        delete recognizer_;
}
void PlateRecognizerProcessor::Update(FrameBatch *frameBatch) {
    DLOG(INFO)<<"Start plate recognize processor "<< endl;

    beforeUpdate(frameBatch);

    for(int i=0;i<images_.size();i++) {
        Vehicle *v = (Vehicle*) objs_[i];
        Mat tmp = images_[i];
        Vehicle::Plate pred = recognizer_->Recognize(tmp);
        v->set_plate(pred);
    }
    Proceed(frameBatch);
}

void PlateRecognizerProcessor::beforeUpdate(FrameBatch *frameBatch) {
    images_.clear();
    images_ = this->vehicles_mat(frameBatch);
}
bool PlateRecognizerProcessor::checkStatus(Frame *frame) {
    return true;
}
void PlateRecognizerProcessor::sharpenImage(const cv::Mat &image,
                                            cv::Mat &result) {
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
vector<Mat> PlateRecognizerProcessor::vehicles_mat(FrameBatch *frameBatch) {
    vector<cv::Mat> vehicleMat;
    objs_.clear();
    objs_ = frameBatch->collect_objects(OPERATION_VEHICLE_PLATE);

    for (vector<Object *>::iterator itr = objs_.begin(); itr != objs_.end();
            ++itr) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;

            DLOG(INFO)<< "Put vehicle images to be plate recognized: " << obj->id() << endl;
            vehicleMat.push_back(v->image());

        } else {
            delete obj;
            itr = objs_.erase(itr);
            DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
        }
    }
    return vehicleMat;
}

}

