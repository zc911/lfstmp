/*
 * vehicle_plate_recognizer_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajiachen
 */
#include "vehicle_plate_recognizer_processor.h"

namespace dg{

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
    DLOG(INFO)<<"Start detect frame: "<< endl;
    vector<Mat> vehicles = this->vehicles_mat(frameBatch);

    for(int i=0;i<vehicles.size();i++) {
        Vehicle *v = (Vehicle*) objs_[i];
        Mat tmp = vehicles[i];
        Vehicle::Plate pred = recognizer_->Recognize(tmp);
        DLOG(INFO)<<"plate number "<<pred.plate_num<<endl;
        v->set_plate(pred);
    }
    Proceed(frameBatch);
}

bool PlateRecognizerProcessor::checkOperation(Frame *frame) {
    return true;
}
bool PlateRecognizerProcessor::checkStatus(Frame *frame) {
    return true;
}
void PlateRecognizerProcessor::sharpenImage(const cv::Mat &image, cv::Mat &result) {
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
vector<Mat > PlateRecognizerProcessor::vehicles_mat(FrameBatch *frameBatch) {
    vector<cv::Mat> vehicleMat;
    objs_ = frameBatch->objects();
    for (vector<Object *>::iterator itr = objs_.begin();
            itr != objs_.end(); ++itr) {
        Object *obj = *itr;

        if (obj->type() == OBJECT_CAR) {

            Vehicle *v = (Vehicle*) obj;

            DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
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

