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
//#include "dgface/detector/det_dlib.h"
//#include "dgface/detector/det_rpn.h"
//#include "dgface/detector/det_ssd.h"
//#include "dgface/detector/det_fcn.h"

namespace dg {

FaceDetectProcessor::FaceDetectProcessor(
    FaceDetectorConfig config, DetectMethod method) {
    //Initialize face detection caffe model and arguments
    DLOG(INFO) << "Start loading face detector model" << std::endl;

    //Initialize face detector
    switch (method) {
        case DetectMethod::DlibMethod:
            LOG(FATAL) << "Not implemented DLIB Detection" << endl;
            exit(-1);
            break;
        case DetectMethod::RpnMethod: {
            LOG(FATAL) << "Not implemented RPN Detection" << endl;
            exit(-1);
            break;
        }
        case DetectMethod::SsdMethod: {
            LOG(INFO) << "Create SSD face detector" << endl;
            detector_ = DGFace::create_detector_with_config(DGFace::det_method::SSD, config.model_dir,
                                                            config.gpu_id, config.is_model_encrypt, config.batch_size);
            break;
        }
        case DetectMethod::FcnMethod: {
            LOG(INFO) << "Create FCN face detector" << endl;
            detector_ = DGFace::create_detector_with_config(DGFace::det_method::FCN, config.model_dir,
                                                            config.gpu_id, config.is_model_encrypt, config.batch_size);
            break;
        }
    }
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
    det.set_box(cv::Rect(0, 0, data.cols, data.rows));
    Face *face = new Face(0, det, 1.0);
    face->set_image(data);
    frame->put_object((Object *) face);
}


bool FaceDetectProcessor::process(Frame *frame) {
}

static bool BoxCmp(const Detection &d1, const Detection &d2) {
    return d1.box().area() > d2.box().area();
}

// TODO change to "real" batch
bool FaceDetectProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start face detector " << endl;

    vector<Mat> imgs;
    vector<int> frameIds;

    for (int i = 0; i < frameBatch->frames().size(); ++i) {

        Frame *frame = frameBatch->frames()[i];
        if (!frame->operation().Check(OPERATION_FACE_DETECT)) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Frame " << frame->id() << "does not need face detect"
                << endl;
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
    if (imgs.size() == 0)
        return true;
    vector<vector<Detection>> boxes_in;
    vector<DGFace::DetectResult> detect_result;

    detector_->detect(imgs, detect_result);

    DetectResult2Detection(detect_result, boxes_in);

    for (int i = 0; i < frameIds.size(); ++i) {
        int frameId = frameIds[i];
        Frame *frame = frameBatch->frames()[frameId];
        sort(boxes_in[i].begin(), boxes_in[i].end(), BoxCmp);

        for (size_t bbox_id = 0; bbox_id < boxes_in[i].size(); bbox_id++) {
            Detection detection = boxes_in[i][bbox_id];
            Face *face = new Face(base_id_ + bbox_id, detection,
                                  detection.confidence);
            VLOG(VLOG_RUNTIME_DEBUG) << "Create a face object: " << face->id() << " detection: " << detection << endl;
            cv::Mat data = frame->payload()->data();
            cv::Mat image = data(detection.box());
            face->set_full_image(data);
            face->set_image(image);
            frame->put_object(face);

        }
    }
    return true;
}

int FaceDetectProcessor::DetectResult2Detection(const vector<DGFace::DetectResult> &detect_results,
                                                vector<vector<Detection> > &detections) {
    for (auto detect_result : detect_results) {
        vector<Detection> detection_tmp;
        for (auto box : detect_result.boundingBox) {
            Detection d;
            d.set_rotated_box(box.second);
            d.confidence = (Confidence) box.first;
            detection_tmp.push_back(d);
        }
        detections.push_back(detection_tmp);
    }

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
