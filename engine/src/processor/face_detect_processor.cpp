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
#include "algorithm_def.h"
#include "util/convert_util.h"

using namespace dgvehicle;
namespace dg {

FaceDetectProcessor::FaceDetectProcessor() {
    //Initialize face detection caffe model and arguments
    DLOG(INFO) << "Start loading face detector model" << std::endl;

    //Initialize face detector
    detector_ = AlgorithmFactory::GetInstance()->CreateProcessorInstance(AlgorithmProcessorType::c_faceDetector);
    base_id_ = 5000;
    DLOG(INFO) << "Face detector has been initialized" << std::endl;
}

FaceDetectProcessor::~FaceDetectProcessor() {
    if (detector_)
        delete detector_;
}

bool FaceDetectProcessor::process(Frame *frame) {

    if (!frame->operation().Check(OPERATION_FACE_DETECTOR)) {
        VLOG(VLOG_RUNTIME_DEBUG) << "Frame " << frame->id() << "does not need face detect" << endl;
        return false;
    }
    Mat data = frame->payload()->data();

    if (data.rows == 0 || data.cols == 0) {
        LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
        return false;
    }


    vector<Mat> imgs;
    imgs.push_back(data);

    vector<vector<dgvehicle::Detection> > boxes_in;
    detector_->BatchProcess(imgs, boxes_in);

    for (size_t bbox_id = 0; bbox_id < boxes_in[0].size(); bbox_id++) {

        Detection detection = ConvertDgvehicleDetection(boxes_in[0][bbox_id]);
        Face *face = new Face(base_id_ + bbox_id, detection, detection.confidence);
        cv::Mat data = frame->payload()->data();
        cv::Mat image = data(detection.box);
        face->set_image(image);
        frame->put_object(face);
    }
}

// TODO change to "real" batch
bool FaceDetectProcessor::process(FrameBatch *frameBatch) {

    for (int i = 0; i < frameBatch->frames().size(); ++i) {

        Frame *frame = frameBatch->frames()[i];
        if (!frame->operation().Check(OPERATION_FACE_DETECTOR)) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Frame " << frame->id() << "does not need face detect"
                                     << endl;
            continue;
        }

        Mat data = frame->payload()->data();

        if (data.rows == 0 || data.cols == 0) {
            LOG(ERROR) << "Frame data is NULL: " << frame->id() << endl;
            continue;
        }

        vector<Mat> imgs;
        imgs.push_back(data);
        performance_++;

        vector<vector<dgvehicle::Detection> > boxes_in;
        detector_->BatchProcess(imgs, boxes_in);

        for (size_t bbox_id = 0; bbox_id < boxes_in[0].size(); bbox_id++) {
            Detection detection = ConvertDgvehicleDetection(boxes_in[0][bbox_id]);
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
