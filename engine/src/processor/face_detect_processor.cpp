/*============================================================================
 * File Name   : face_detect_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年3月2日 下午1:53:19
 * Description :
 * ==========================================================================*/

#include "processor/face_detect_processor.h"
#include "processor_helper.h"
#include "debug_util.h"

namespace dg {

FaceDetectProcessor::FaceDetectProcessor(
    FaceDetectorConfig config, int method) {
    //Initialize face detection caffe model and arguments
    DLOG(INFO) << "Start loading face detector model" << std::endl;

    //Initialize face detector
    switch (method) {
        case 1:
            detector_ = new FaceCaffeDetector(config);
            break;
        case 2:
            detector_ = new FaceSsdDetector(config);
            LOG(INFO) << "SSD";
            break;
    }
    base_id_ = 5000;
    DLOG(INFO) << "Face detector has been initialized" << std::endl;
}
FaceDetectProcessor::FaceDetectProcessor(
    FaceDlibDetector::FaceDetectorConfig config) {
    //Initialize face detection caffe model and arguments
    DLOG(INFO) << "Start loading face detector model" << std::endl;

    //Initialize face detector
    detector_ = new FaceDlibDetector(config);
    base_id_ = 5000;
    DLOG(INFO) << "Face detector has been initialized" << std::endl;
}

FaceDetectProcessor::~FaceDetectProcessor() {
    if (detector_)
        delete detector_;
}


static void noDetectionButFeature(Frame *frame) {
    Mat data = frame->payload()->data();
    if (data.rows == 0 || data.cols == 0) {
        LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
        return;
    }
    Detection det;
    det.id = DETECTION_FACE;
    det.box = cv::Rect(0, 0, data.cols, data.rows);
    Face *face = new Face(0, det, 1.0);
    face->set_image(data);
    frame->put_object((Object *) face);
}


bool FaceDetectProcessor::process(Frame *frame) {

    if (!frame->operation().Check(OPERATION_FACE_DETECTOR)) {
        VLOG(VLOG_RUNTIME_DEBUG) << "Frame " << frame->id() << "does not need face detect" << endl;

        if (frame->operation().Check(OPERATION_FACE_FEATURE_VECTOR)) {
            noDetectionButFeature(frame);
            return false;
        }


        return false;
    }
    Mat data = frame->payload()->data();

    if (data.rows == 0 || data.cols == 0) {
        LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
        return false;
    }


    vector<Mat> imgs;
    imgs.push_back(data);

    vector<vector<Detection> > boxes_in;
    detector_->Detect(imgs, boxes_in);

    for (size_t bbox_id = 0; bbox_id < boxes_in[0].size(); bbox_id++) {

        Detection detection = boxes_in[0][bbox_id];
        Face *face = new Face(base_id_ + bbox_id, detection,
                              detection.confidence);
        cv::Mat data = frame->payload()->data();
        cv::Mat image = data(detection.box);
        face->set_image(image);
        frame->put_object(face);
    }
}

bool FaceDetectProcessor::process(FrameBatch *frameBatch) {

    vector<Mat> imgs;
    vector<int> frameIds;

    for (int i = 0; i < frameBatch->frames().size(); ++i) {

        Frame *frame = frameBatch->frames()[i];
        if (!frame->operation().Check(OPERATION_FACE_DETECTOR)) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Frame " << frame->id() << "does not need face detect"
                << endl;
            if (frame->operation().Check(OPERATION_FACE_FEATURE_VECTOR)) {
                noDetectionButFeature(frame);
            }
            continue;
        }

        Mat data = frame->payload()->data();

        if (data.rows == 0 || data.cols == 0) {
            LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
            continue;
        }
        frameIds.push_back(i);

        imgs.push_back(data);
        performance_++;

    }
    vector<vector<Detection>> boxes_in;

    struct timeval start, finish;
    gettimeofday(&start, NULL);
    detector_->Detect(imgs, boxes_in);
    gettimeofday(&finish, NULL);
    VLOG(VLOG_PROCESS_COST) << "Faces detection costs: " << TimeCostInMs(start, finish) << endl;


    for (int i = 0; i < frameIds.size(); ++i) {
        int frameId = frameIds[i];
        Frame *frame = frameBatch->frames()[frameId];
        for (size_t bbox_id = 0; bbox_id < boxes_in[i].size(); bbox_id++) {
            Detection detection = boxes_in[i][bbox_id];
            Face *face = new Face(base_id_ + bbox_id, detection,
                                  detection.confidence);
            cv::Mat data = frame->payload()->data();
            cv::Mat image = data(detection.box);
            face->set_image(image);
            frame->put_object(face);
        }
    }

    return true;
}
bool FaceDetectProcessor::beforeUpdate(FrameBatch *frameBatch) {
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
bool FaceDetectProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_FACE_DETECTION, performance_);

}
} /* namespace dg */
