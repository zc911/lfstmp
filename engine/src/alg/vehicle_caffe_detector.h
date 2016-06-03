/*
 * caffe_detector.h

 *
 *  Created on: Apr 18, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_CAFFE_DETECTOR_H_
#define SRC_ALG_CAFFE_DETECTOR_H_

#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dgcaffe/caffe/caffe.hpp"
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class VehicleCaffeDetector {

public:
    typedef struct {
        bool is_model_encrypt = false;
        int batch_size = 1;
        int target_min_size = 600;
        int target_max_size = 1000;
        int gpu_id = 0;
        bool use_gpu = true;
        string deploy_file;
        string model_file;
        float threshold = 0.5;
    } VehicleCaffeDetectorConfig;


    VehicleCaffeDetector(const VehicleCaffeDetectorConfig &config);
    virtual ~VehicleCaffeDetector();
    virtual int DetectBatch(vector<cv::Mat> &img,
                            vector<vector<Detection> > &detect_results);

protected:
    vector<Blob<float> *> PredictBatch(const vector<Mat> &imgs);
    void Fullfil(vector<cv::Mat> &img,
                 vector<Blob<float> *> &outputs,
                 vector<vector<Detection> > &detect_results);

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
#ifdef SHOW_VIS
    vector<Scalar> color_;
    vector<string> tags_;
#endif

};
}
#endif /* ENGINE_SRC_ALG_CAFFE_DETECTOR_H_ */
