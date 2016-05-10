/*============================================================================
 * File Name   : face_detect_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年3月2日 下午1:53:19
 * Description : 
 * ==========================================================================*/

#include "processor/face_detect_processor.h"

namespace dg {

FaceDetectProcessor::FaceDetectProcessor(
        FaceDetector::FaceDetectorConfig config) {
    //Initialize face detection caffe model and arguments
    DLOG(INFO)<< "Start loading face detector model" << std::endl;

    //Initialize face detector
    detector_ = new FaceDetector(config);
    DLOG(INFO) << "Face detector has been initialized" << std::endl;
}

FaceDetectProcessor::~FaceDetectProcessor() {
    delete detector_;
}

void FaceDetectProcessor::Update(Frame *frame) {
    vector<Mat> imgs;
    imgs.push_back(frame->payload()->data());
    vector<vector<Detection>> boxes_in = detector_->Detect(imgs);

    for (size_t bbox_id = 0; bbox_id < boxes_in[0].size(); bbox_id++) {
        Detection detection = boxes_in[0][bbox_id];
        Face *face = new Face(bbox_id, detection, detection.confidence);
        frame->put_object(face);
    }
    Proceed(frame);
}

// TODO change to "real" batch
void FaceDetectProcessor::Update(FrameBatch *frameBatch) {
    for (int i = 0; i < frameBatch->frames().size(); ++i) {
        Frame *frame = frameBatch->frames()[i];
        vector<Mat> imgs;
        imgs.push_back(frame->payload()->data());
        vector<vector<Detection>> boxes_in = detector_->Detect(imgs);

        for (size_t bbox_id = 0; bbox_id < boxes_in[0].size(); bbox_id++) {
            Detection detection = boxes_in[0][bbox_id];
            Face *face = new Face(bbox_id, detection, detection.confidence);
            frame->put_object(face);
        }
    }
    Proceed(frameBatch);
}

} /* namespace dg */
