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
  case DlibMethod:
    detector_ = new DGFace::DlibDetector(config.img_scale_max, config.img_scale_min);
    detect_type_ = "";
    break;
  case RpnMethod: {
    size_t stride = 16;
    size_t max_per_img = 100;
    vector<float> area = {576, 1152, 2304, 4608, 9216, 18432, 36864};
    vector<float> ratio = {1};
    vector<float> mean = {128, 128, 128};
    string clsname = "conv_face_16_cls";
    string regname = "conv_face_16_reg";

    detector_ = new DGFace::RpnDetector(config.img_scale_max,
                                        config.img_scale_min,
                                        config.deploy_file, config.model_file, clsname,
                                        regname, area,
                                        ratio, mean, config.confidence, max_per_img,
                                        stride, config.scale, config.use_gpu, config.gpu_id);
    detect_type_ = "rpn";
    break;
  }
  case SsdMethod: {

    vector<float> mean = {104, 117, 123};
    detector_ = new DGFace::SSDDetector(config.img_scale_max,
                                        config.img_scale_min,
                                        config.deploy_file, config.model_file, mean, config.confidence, config.scale, config.use_gpu, config.gpu_id);
    detect_type_ = "ssd";
    break;
  }
  case FcnMethod: {
    detector_ = new DGFace::FcnDetector(config.img_scale_max, config.img_scale_min, config.deploy_file, config.model_file, config.gpu_id);
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
  det.box = cv::Rect(0, 0, data.cols, data.rows);
  Face *face = new Face(0, det, 1.0);
  face->set_image(data);
  frame->put_object((Object *) face);
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
static bool BoxCmp(const Detection &d1,const Detection &d2){
    return d1.box.area()>d2.box.area();
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
  if (imgs.size() == 0)
    return true;
  vector<vector<Detection>> boxes_in;
  vector<DGFace::DetectResult> detect_result;

  detector_->detect(imgs, detect_result);
  DetectResult2Detection(detect_result, boxes_in);
  vector<vector<Rect>> enlarge_boxes;
  //enlarge_box(boxes_in, enlarge_boxes);
  for (int i = 0; i < frameIds.size(); ++i) {
    int frameId = frameIds[i];
    Frame *frame = frameBatch->frames()[frameId];
    sort(boxes_in[i].begin(),boxes_in[i].end(),BoxCmp);

    /* if (boxes_in[i].size() == 0)
       continue;
     int size = 0, index_id = 0;
     for (size_t bbox_id = 0; bbox_id < boxes_in[i].size(); bbox_id++) {
       Detection detection = boxes_in[i][bbox_id];
       if (size < detection.box.area()) {
         index_id = bbox_id;
         size=detection.box.area();
       }
     }
     Detection detection = boxes_in[i][index_id];
     Face *face = new Face(base_id_ + 0, detection,
                           detection.confidence);
     cv::Mat data = frame->payload()->data();
     cv::Mat image = data;

     face->set_image(image);
     frame->put_object(face);*/
    for (size_t bbox_id = 0; bbox_id < boxes_in[i].size(); bbox_id++) {
      Detection detection = boxes_in[i][bbox_id];
      Face *face = new Face(base_id_ + bbox_id, detection,
                            detection.confidence);
      cv::Mat data = frame->payload()->data();
      cv::Mat image = data;
    //   string name = to_string(bbox_id)+to_string(frameId) + "face.jpg";
     //  imwrite(name, image);
      face->set_image(image);
      frame->put_object(face);
    }
  }
  return true;
}
void FaceDetectProcessor::enlarge_box(vector<vector<Detection>> boxes, vector<vector<Rect>> &enlarge_boxes) {
  if (detect_type_ == "")
    return;

  enlarge_boxes.resize(boxes.size());
  for (int i = 0; i < enlarge_boxes.size(); i++) {
    for (auto bbox : boxes[i]) {
      Rect adjust_box = bbox.box;
      Rect reverse_box;
      if (detect_type_ == "ssd") {
        const float h_rate = 0.42;
        reverse_box.height = adjust_box.height / (1 - h_rate);
        reverse_box.y = adjust_box.y - reverse_box.height * h_rate;

        const float w_rate = 0.12;
        reverse_box.width = adjust_box.width / (1 - w_rate * 2);
        reverse_box.x = adjust_box.x - reverse_box.width * w_rate;
      } else if (detect_type_ == "rpn") {
        const float h_rate = 0.32;
        reverse_box.height = adjust_box.height / (1 - h_rate);
        reverse_box.y = adjust_box.y - reverse_box.height * h_rate;

        const float w_rate = 0.16;
        reverse_box.width = adjust_box.width / (1 - w_rate * 2);
        reverse_box.x = adjust_box.x - reverse_box.width * w_rate;
      }
      enlarge_boxes[i].push_back(reverse_box);
    }
  }
}
int FaceDetectProcessor::DetectResult2Detection(const vector<DGFace::DetectResult> &detect_results, vector< vector<Detection> > &detections) {
  for (auto detect_result : detect_results) {
    vector<Detection> detection_tmp;
    for (auto box : detect_result.boundingBox) {
      Detection d;
      d.box = box.second;
      d.confidence = (Confidence)box.first;
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
