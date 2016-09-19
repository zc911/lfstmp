//
// Created by jiajaichen on 16-8-5.
//

#ifndef PROJECT_WINDOW_CAFFE_SSD_DETECTOR_H
#define PROJECT_WINDOW_CAFFE_SSD_DETECTOR_H

#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dgcaffe/caffe/caffe.hpp"
#include "model/basic.h"
#include "alg/detector/detector.h"
#include "alg/caffe_config.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class WindowCaffeSsdDetector: public Detector {

public:

    WindowCaffeSsdDetector(const VehicleCaffeDetectorConfig &config);
    virtual ~WindowCaffeSsdDetector();
    virtual int DetectBatch(vector<cv::Mat> &img,
                            vector<vector<Detection> > &detect_results);

protected:
    vector<Blob<float>*> PredictBatch(const vector<Mat> &imgs);
    void Fullfil(vector<cv::Mat> &img, vector<Blob<float> *> &outputs, vector<vector<Detection> > &detect_results);

private:
    boost::shared_ptr<caffe::Net<float>> net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleCaffeDetectorConfig caffe_config_;
    bool use_gpu_;
    int gpu_id_;
    int batch_size_;
    float threshold_;
    int target_col_;
    int target_row_;
#ifdef SHOW_VIS
    vector<Scalar> color_;
    vector<string> tags_;
#endif

};
}

#endif //PROJECT_WINDOW_CAFFE_SSD_DETECTOR_H
