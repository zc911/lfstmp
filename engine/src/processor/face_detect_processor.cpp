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
namespace dg {

FaceDetectProcessor::FaceDetectProcessor(
    FaceDetectorConfig config, int method) {
    //Initialize face detection caffe model and arguments
    DLOG(INFO) << "Start loading face detector model" << std::endl;
    //Initialize face detector
    switch (method) {
    case DlibMethod:
        detector_ = new DGFace::DlibDetector(config.img_scale_max, config.img_scale_min);

        break;
    case RpnMethod: {
        size_t stride = 16;
        size_t max_per_img = 100;
        vector<float> area = {576, 1152, 2304, 4608, 9216, 18432, 36864};
        vector<float> ratio = {1};
        vector<float> mean = {128, 128, 128};
        LOG(INFO)<<config.use_gpu<<" "<<config.gpu_id;
        detector_ = new DGFace::RpnDetector(config.img_scale_max,
                                            config.img_scale_min,
                                            config.deploy_file, config.model_file, "conv_face_16_cls",
                                            "conv_face_16_reg", area,
                                            ratio, mean, config.confidence, max_per_img,
                                            stride, config.scale, config.use_gpu,config.gpu_id);
        break;
    }
    case SsdMethod: {

        vector<float> mean = {104, 117, 123};
        LOG(INFO)<<config.use_gpu<<" "<<config.gpu_id;

        detector_ = new DGFace::SSDDetector(config.img_scale_max,
                                            config.img_scale_min,
                                            config.deploy_file, config.model_file, mean, config.confidence, config.scale, config.use_gpu,config.gpu_id);
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

bool FaceDetectProcessor::process(Frame *frame) {
    /*   LOG(INFO) << "HA";

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
       }*/
}

// TODO change to "real" batch
bool FaceDetectProcessor::process(FrameBatch *frameBatch) {
    vector<Mat> imgs;
    vector<int> frameIds;

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
        frameIds.push_back(i);

        imgs.push_back(data);
        performance_++;

    }
    vector<vector<Detection>> boxes_in;
    vector<DGFace::DetectResult> detect_result;
        LOG(INFO)<<"detect begin";

    detector_->detect(imgs, detect_result);

    DetectResult2Detection(detect_result, boxes_in);

    for (int i = 0; i < frameIds.size(); ++i) {
        int frameId = frameIds[i];
        Frame *frame = frameBatch->frames()[frameId];
        for (size_t bbox_id = 0; bbox_id < boxes_in[i].size(); bbox_id++) {
            Detection detection = boxes_in[i][bbox_id];
            Face *face = new Face(base_id_ + bbox_id, detection,
                                  detection.confidence);
            cv::Mat data = frame->payload()->data();
            cv::Mat image = data(detection.box);
            string name = to_string(bbox_id)+to_string(frameId) + "face.jpg";
            imwrite(name, image);
            face->set_image(image);
            frame->put_object(face);
        }
    }
    LOG(INFO)<<"detect end";
    return true;
}
int FaceDetectProcessor::DetectResult2Detection(const vector<DGFace::DetectResult> &detect_results, vector< vector<Detection> > &detections) {
    for (auto detect_result : detect_results) {
        vector<Detection> detection_tmp;
        LOG(INFO) << detect_result.boundingBox.size();
        for (auto box : detect_result.boundingBox) {
            Detection d;
            d.box = box.second;
            d.confidence = (Confidence)box.first;
            detection_tmp.push_back(d);
            LOG(INFO) << d;
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
