/*
 * faster_rcnn_detector.h
 *
 *  Created on: 11/04/2016
 *      Author: chenzhen
 */

#ifndef FASTER_RCNN_DETECTOR_H_
#define FASTER_RCNN_DETECTOR_H_

#include <string>

#include <opencv2/opencv.hpp>
#include "caffe/caffe.hpp"

#include "caffe_config.h"
#include "caffe_detector.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class FasterRcnnDetector : public CaffeDetector {
 public:

    FasterRcnnDetector(const CaffeConfig &config);
    ~FasterRcnnDetector();

    vector<Detection> Detect(const cv::Mat &img);
    vector<vector<Detection>> DetectBatch(const vector<cv::Mat> &img);

 private:
    struct Bbox {
        float confidence;
        Rect rect;
        bool deleted;
        int cls_id;
    };

    void forward(vector<cv::Mat> imgs, vector<Blob<float>*> &outputs);
    void getDetection(vector<Blob<float>*>& outputs,
                      vector<struct Bbox> &final_vbbox);

    void nms(vector<struct Bbox>& p, float threshold);
    static bool mycmp(struct Bbox b1, struct Bbox b2);
    void bboxTransformInvClip(Blob<float>* roi, Blob<float>* cls,
                              Blob<float>* reg, Blob<float>* im_info_layer,
                              vector<struct Bbox> &vbbox);
    int num_channels_;
    int batch_size_;
    vector<float> pixel_means_;
    int scale_;
    float conf_thres_;
    string layer_name_rois_;
    string layer_name_score_;
    string layer_name_bbox_;
    int sliding_window_stride_;
    int max_per_img_;
};

}

#endif /* FASTER_RCNN_DETECTOR_H_ */
